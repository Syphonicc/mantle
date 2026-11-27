//! Cut-cell volume mesh of the fluid region (outside the soup) inside the octree root cube.
//!
//! * Corner signs come from the octree (inside = solid). Cut points are the exact edge hits; multi-hit edges take the
//!   hit at which the signed winding, accumulated from the fluid endpoint, crosses 0.5. Sign-changing edges without a hit
//!   (holes in dirty soups) get the edge midpoint and are counted (`unresolved_edges`).
//! * Sliver control: a cut point within `eps * edge` of a corner snaps to it. Snapping to a solid corner just moves the
//!   cut point there; snapping to a fluid corner turns that corner solid (the surface now passes through it), so every
//!   cut point finally sits on the solid end of its edge or strictly inside it.
//! * Uncut fluid leaves are hexahedra. A face towards a finer leaf is split into the four sub-faces; any lattice vertex
//!   hanging on a face edge is inserted into that face polygon, so every cell stays a closed polyhedron.
//! * Cut leaves (mixed signs, all neighbours the same size thanks to the octree conformity buffer): marching squares on
//!   each cube face; ambiguous (diagonal) faces take the sign of the face centre (FWN), decided once per face key so both
//!   cells agree. Chords of the face polygons, reversed and chained, give the cut (wall) polygons; connected components
//!   of the polygon set become separate cells.
//! * With `Clip::Shell` (octree clipped against the offset surface `s = delta`) nothing changes in the cell construction:
//!   signs and cut points come from the octree, the ambiguous-face centre is fluid iff psi > 0, and the clipped patch is
//!   named `shell`. `shell_surface` extracts the shell patch (points, normals, nearest wall data, polygons) for the layer step.
//! * Faces are deduplicated by their sorted vertex set. Patches: `wall` = cut polygons, `farfield` = faces on the root
//!   cube, `orphan` = unmatched interior faces (a defect, expected 0).

use crate::bvh::Bvh;
use crate::dexel::Tri;
use crate::fwn::Fwn;
use crate::octree::{vertex_key, Leaf, Octree};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

pub const NONE: u32 = u32::MAX;
pub const INTERNAL: u8 = 0;
pub const WALL: u8 = 1;
pub const FARFIELD: u8 = 2;
pub const ORPHAN: u8 = 3;

#[derive(Default)]
pub struct Fx(u64);
impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u8(&mut self, x: u8) {
        self.write_u64(x as u64);
    }
    fn write_u32(&mut self, x: u32) {
        self.write_u64(x as u64);
    }
    fn write_usize(&mut self, x: usize) {
        self.write_u64(x as u64);
    }
    fn write_u64(&mut self, x: u64) {
        let h = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
        self.0 = h ^ (h >> 29);
    }
}
pub type Map<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

type V3 = [f64; 3];
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Polygon face stored with the outward orientation of its owner cell.
#[derive(Clone, Debug)]
pub struct Face {
    pub verts: Vec<u32>,
    pub owner: u32,
    pub neigh: u32,
    pub patch: u8,
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub faces: Vec<u32>,
    /// the 8 corner points in SU2/VTK hexahedron order when the cell is an uncut leaf with six plain quad faces
    pub hex: Option<[u32; 8]>,
    /// edge of the leaf the cell was cut from (largest member after agglomeration)
    pub h: f64,
    /// cell was cut by the surface (or contains such a cell after agglomeration)
    pub cut: bool,
    /// number of original cells merged into this one
    pub n_orig: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub leaves: usize,
    pub hex_cells: usize,
    pub poly_cells: usize,
    pub cut_leaves: usize,
    pub cut_cells: usize,
    pub multi_component_cells: usize,
    pub unresolved_edges: usize,
    pub multi_hit_nonfirst: usize,
    pub snapped_points: usize,
    pub flipped_corners: usize,
    pub ambiguous_faces: usize,
    pub ambiguous_connected: usize,
    pub dropped_faces: usize,
    pub open_wall_chains: usize,
    pub unclosed_cells: usize,
    pub degenerate_cells: usize,
    pub nonmanifold_faces: usize,
    pub orphan_faces: usize,
    pub min_vol_ratio: f64,
    /// agglomeration: min volume ratio before merging, cells removed / merged cells created, bad cells before / after /
    /// unmergeable in the last pass, passes run
    pub min_vol_ratio_before: f64,
    pub cells_merged: usize,
    pub merge_groups: usize,
    pub bad_cells_before: usize,
    pub bad_cells_after: usize,
    pub unmergeable_bad_cells: usize,
    pub merge_passes: usize,
    pub q_before: Quality,
    pub q_after: Quality,
    pub n_nonpositive_cells: usize,
    pub volume: f64,
    /// total volume with the OpenFOAM definition (area-weighted face centres, pyramids about the mean face centre)
    pub volume_foam_def: f64,
    pub t_edges: f64,
    pub t_faces: f64,
    pub t_cells: f64,
    pub t_final: f64,
    pub t_merge: f64,
}

pub struct CutMesh {
    pub points: Vec<V3>,
    pub faces: Vec<Face>,
    pub cells: Vec<Cell>,
    pub stats: Stats,
    /// name of the clipped-surface patch (`WALL` faces): `wall` or `shell`
    pub patch_name: &'static str,
    /// cells that are native SU2 prisms (13) / pyramids (14): cell id -> (type, nodes in SU2 order); only set by the layer step
    pub native: Map<u32, (u8, [u32; 8])>,
}

pub struct Options {
    /// cut points closer than `eps * edge length` to a corner snap to it
    pub eps: f64,
    /// agglomerate cut cells with volume < `merge_tau * h^3` (and concave / bad-weight cut cells); 0 disables
    pub merge_tau: f64,
}
impl Default for Options {
    fn default() -> Options {
        Options { eps: 0.05, merge_tau: 0.1 }
    }
}

/// Face-plane concavity tolerance in units of the cell edge (calibrated so that only truly concave cut cells, not
/// mild wall curvature, are flagged: 1e-3 flags ~6900 of 15k cut cells at L=7, 2e-2 flags a few hundred).
const CONCAVE_TOL: f64 = 0.02;
/// Preferred maximum OpenFOAM face skewness of a merged cell (checkMesh flags > 4); candidates above it are used only
/// when no other neighbour yields a valid cell.
const SKEW_MAX: f64 = 3.0;

/// Cell / face quality counters (OpenFOAM-style definitions, cut cells only for the cell flags).
#[derive(Clone, Debug, Default)]
pub struct Quality {
    /// bad cut cells: volume < tau h^3, concave by the face-plane test (tolerance 0.02 h), a low-weight face, failing the OpenFOAM face-pyramid/tet test, or without a valid SU2 apex
    pub bad: usize,
    pub small: usize,
    pub concave: usize,
    /// internal faces with min(w, 1-w) < 0.05 (all faces) and with cell volume ratio < 0.01
    pub low_weight_faces: usize,
    pub vol_ratio_faces: usize,
    /// concave (same test) cut cells that are agglomerates (not counted as bad: merging more cannot fix them)
    pub concave_merged: usize,
    /// cut cells failing OpenFOAM's face-pyramid / face-tet checks about their cell centre
    pub bad_foam_tets: usize,
    /// cut cells without a valid SU2 apex (not star-shaped): bad, merged like the others
    pub not_star_shaped: usize,
}

#[derive(Clone, Copy)]
struct EdgeInfo {
    pt: V3,
    /// 0 none, 1 snapped to `a`, 2 snapped to `b`
    snap: u8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tag {
    Wall,
    Cube,
    Boundary,
}

/// Corners (cube corner index: bit0 x, bit1 y, bit2 z) of face (axis `a`, side `s`) in right-hand order about the outward normal.
pub(crate) fn ring(a: usize, s: usize) -> [usize; 4] {
    let (u, v) = ((a + 1) % 3, (a + 2) % 3);
    let idx = |du: usize, dv: usize| (s << a) | (du << u) | (dv << v);
    if s == 1 {
        [idx(0, 0), idx(1, 0), idx(1, 1), idx(0, 1)]
    } else {
        [idx(0, 0), idx(0, 1), idx(1, 1), idx(1, 0)]
    }
}

pub(crate) fn geom(l: &Leaf, ml: u8) -> ([u32; 3], u32) {
    let s = ml - l.level;
    ([l.i << s, l.j << s, l.k << s], 1u32 << s)
}

fn amb_key(lat: &[[u32; 3]; 4], a: usize) -> (u64, u64) {
    let mut mn = [u32::MAX; 3];
    let mut mx = [0u32; 3];
    for p in lat {
        for d in 0..3 {
            mn[d] = mn[d].min(p[d]);
            mx[d] = mx[d].max(p[d]);
        }
    }
    let e = mx[(a + 1) % 3] - mn[(a + 1) % 3];
    (vertex_key(mn[0], mn[1], mn[2]), a as u64 | (e as u64) << 2)
}

struct Bld<'a> {
    oc: &'a Octree,
    vk: &'a [u64],
    es: Vec<bool>,
    os: Vec<bool>,
    edges: Map<(u64, u64), EdgeInfo>,
    centre: Map<(u64, u64), bool>,
    pts: Vec<V3>,
    pmap: Map<(u64, u64), u32>,
    faces: Vec<Face>,
    fmap: Map<Vec<u32>, u32>,
    cells: Vec<Cell>,
    st: Stats,
    amin2: f64,
}

impl Bld<'_> {
    fn corner_id(&mut self, key: u64) -> u32 {
        if let Some(&p) = self.pmap.get(&(key, u64::MAX)) {
            return p;
        }
        let id = self.pts.len() as u32;
        self.pts.push(self.oc.vertex_pos(key));
        self.pmap.insert((key, u64::MAX), id);
        id
    }
    fn lat_id(&mut self, c: [u32; 3]) -> u32 {
        self.corner_id(vertex_key(c[0], c[1], c[2]))
    }
    /// lattice midpoint of a face edge if it is an octree vertex (hanging node)
    fn mid(&self, p: [u32; 3], q: [u32; 3]) -> Option<[u32; 3]> {
        if (0..3).any(|d| (p[d] + q[d]) % 2 != 0) {
            return None;
        }
        let m = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2];
        self.vk.binary_search(&vertex_key(m[0], m[1], m[2])).is_ok().then_some(m)
    }
    /// point id of the cut on the sign-changing (effective signs) edge `(ka, kb)`, `ka` the low end
    fn cut_id(&mut self, ka: u64, kb: u64, ia: usize, ib: usize) -> u32 {
        let (solid_key, solid_idx) = if self.es[ia] { (ka, ia) } else { (kb, ib) };
        if !self.os[solid_idx] {
            return self.corner_id(solid_key); // fluid corner turned solid: cut point is that corner
        }
        let e = self.edges[&(ka, kb)];
        match e.snap {
            1 => self.corner_id(ka),
            2 => self.corner_id(kb),
            _ => {
                if let Some(&p) = self.pmap.get(&(ka, kb)) {
                    return p;
                }
                let id = self.pts.len() as u32;
                self.pts.push(e.pt);
                self.pmap.insert((ka, kb), id);
                id
            }
        }
    }
    /// remove repeated vertices; `None` if fewer than 3 remain or the area vanishes
    fn clean(&self, mut v: Vec<u32>) -> Option<Vec<u32>> {
        v.dedup();
        while v.len() > 1 && v[0] == *v.last().unwrap() {
            v.pop();
        }
        if v.len() < 3 {
            return None;
        }
        let p0 = self.pts[v[0] as usize];
        let mut n = [0.0; 3];
        for t in 1..v.len() - 1 {
            let c = cross(sub(self.pts[v[t] as usize], p0), sub(self.pts[v[t + 1] as usize], p0));
            for d in 0..3 {
                n[d] += c[d];
            }
        }
        if dot(n, n).sqrt() <= 2.0 * self.amin2 {
            return None;
        }
        Some(v)
    }
    fn add_face(&mut self, verts: Vec<u32>, tag: Tag, cell: u32) -> u32 {
        if tag == Tag::Wall {
            self.faces.push(Face { verts, owner: cell, neigh: NONE, patch: WALL });
            return self.faces.len() as u32 - 1;
        }
        let mut key = verts.clone();
        key.sort_unstable();
        if let Some(&f) = self.fmap.get(&key) {
            let fc = &mut self.faces[f as usize];
            if fc.neigh == NONE && fc.owner != cell && fc.patch == ORPHAN {
                fc.neigh = cell;
                fc.patch = INTERNAL;
                return f;
            }
            self.st.nonmanifold_faces += 1;
            self.faces.push(Face { verts, owner: cell, neigh: NONE, patch: ORPHAN });
            return self.faces.len() as u32 - 1;
        }
        let patch = if tag == Tag::Boundary { FARFIELD } else { ORPHAN };
        self.faces.push(Face { verts, owner: cell, neigh: NONE, patch });
        let f = self.faces.len() as u32 - 1;
        self.fmap.insert(key, f);
        f
    }
    fn neighbour(&self, l: &Leaf, a: usize, s: usize) -> Option<usize> {
        let mut c = [l.i as i64, l.j as i64, l.k as i64];
        c[a] += if s == 1 { 1 } else { -1 };
        self.oc.leaf_containing(l.level, c[0], c[1], c[2])
    }

    /// Uncut fluid leaf: hexahedron, faces split towards finer neighbours, hanging vertices inserted.
    fn uncut_cell(&mut self, l: &Leaf) {
        let ml = self.oc.max_level;
        let (base, e) = geom(l, ml);
        let coord = |n: usize| [base[0] + e * (n as u32 & 1), base[1] + e * ((n as u32 >> 1) & 1), base[2] + e * ((n as u32 >> 2) & 1)];
        let ci = self.cells.len() as u32;
        let mut fl: Vec<u32> = Vec::with_capacity(6);
        for a in 0..3 {
            for s in 0..2 {
                let r = ring(a, s);
                let lat: [[u32; 3]; 4] = std::array::from_fn(|t| coord(r[t]));
                let nb = self.neighbour(l, a, s);
                let tag = if nb.is_none() { Tag::Boundary } else { Tag::Cube };
                let quads: Vec<[[u32; 3]; 4]> = match nb {
                    Some(x) if self.oc.leaves()[x].level > l.level => {
                        let m: [[u32; 3]; 4] = std::array::from_fn(|t| std::array::from_fn(|d| (lat[t][d] + lat[(t + 1) % 4][d]) / 2));
                        let c: [u32; 3] = std::array::from_fn(|d| (lat[0][d] + lat[2][d]) / 2);
                        vec![[lat[0], m[0], c, m[3]], [m[0], lat[1], m[1], c], [c, m[1], lat[2], m[2]], [m[3], c, m[2], lat[3]]]
                    }
                    _ => vec![lat],
                };
                for q in quads {
                    let mut ids: Vec<u32> = Vec::with_capacity(8);
                    for t in 0..4 {
                        ids.push(self.lat_id(q[t]));
                        if let Some(m) = self.mid(q[t], q[(t + 1) % 4]) {
                            ids.push(self.lat_id(m));
                        }
                    }
                    fl.push(self.add_face(ids, tag, ci));
                }
            }
        }
        let mut hex = None;
        if fl.len() == 6 && fl.iter().all(|&f| self.faces[f as usize].verts.len() == 4) {
            let c: Vec<u32> = (0..8).map(|n| self.lat_id(coord(n))).collect();
            hex = Some([c[0], c[1], c[3], c[2], c[4], c[5], c[7], c[6]]);
            self.st.hex_cells += 1;
        } else {
            self.st.poly_cells += 1;
        }
        self.cells.push(Cell { faces: fl, hex, h: self.oc.leaf_size(l), cut: false, n_orig: 1 });
    }

    /// Cut leaf: clipped polyhedron; returns the number of cells emitted (components).
    fn cut_cell(&mut self, l: &Leaf, vi: [usize; 8]) -> usize {
        let ml = self.oc.max_level;
        let (base, e) = geom(l, ml);
        let coord = |n: usize| [base[0] + e * (n as u32 & 1), base[1] + e * ((n as u32 >> 1) & 1), base[2] + e * ((n as u32 >> 2) & 1)];
        let es: [bool; 8] = std::array::from_fn(|n| self.es[vi[n]]);
        let ck: [u64; 8] = std::array::from_fn(|n| self.vk[vi[n]]);
        let mut polys: Vec<(Vec<u32>, Tag)> = Vec::new();
        let mut chords: Vec<(u32, u32)> = Vec::new();
        for a in 0..3 {
            for s in 0..2 {
                let r = ring(a, s);
                let tag = if self.neighbour(l, a, s).is_none() { Tag::Boundary } else { Tag::Cube };
                // node kinds: 0 fluid corner / hanging vertex, 1 solid corner, 2 exit (fluid -> solid cut), 3 entry (solid -> fluid cut)
                let mut nodes: Vec<(u32, u8)> = Vec::with_capacity(12);
                for t in 0..4 {
                    let (n0, n1) = (r[t], r[(t + 1) % 4]);
                    let id0 = self.corner_id(ck[n0]);
                    nodes.push((id0, es[n0] as u8));
                    if es[n0] != es[n1] {
                        let (lo, hi) = if n0 < n1 { (n0, n1) } else { (n1, n0) };
                        let pid = self.cut_id(ck[lo], ck[hi], vi[lo], vi[hi]);
                        nodes.push((pid, if es[n0] { 3 } else { 2 }));
                    } else if !es[n0] {
                        if let Some(m) = self.mid(coord(n0), coord(n1)) {
                            let id = self.lat_id(m);
                            nodes.push((id, 0));
                        }
                    }
                }
                let mut f: Vec<(u32, u8)> = nodes.into_iter().filter(|x| x.1 != 1).collect();
                let ncut = f.iter().filter(|x| x.1 >= 2).count();
                if ncut == 0 {
                    if f.len() >= 3 {
                        polys.push((f.iter().map(|x| x.0).collect(), tag));
                    }
                    continue;
                }
                let p = f.iter().position(|x| x.1 == 3).expect("entry");
                f.rotate_left(p);
                let mut segs: Vec<Vec<(u32, u8)>> = Vec::new();
                let mut cur: Vec<(u32, u8)> = Vec::new();
                for x in f {
                    cur.push(x);
                    if x.1 == 2 {
                        segs.push(std::mem::take(&mut cur));
                    }
                }
                let lat: [[u32; 3]; 4] = std::array::from_fn(|t| coord(r[t]));
                let connected = segs.len() == 2 && {
                    self.st.ambiguous_faces += 1;
                    let c = *self.centre.get(&amb_key(&lat, a)).unwrap_or(&false);
                    self.st.ambiguous_connected += c as usize;
                    c
                };
                if connected {
                    let n = segs.len();
                    for k in 0..n {
                        let (u, v) = (segs[k].last().unwrap().0, segs[(k + 1) % n][0].0);
                        if u != v {
                            chords.push((u, v));
                        }
                    }
                    polys.push((segs.iter().flatten().map(|x| x.0).collect(), tag));
                } else {
                    for sg in segs {
                        let (u, v) = (sg.last().unwrap().0, sg[0].0);
                        if u != v {
                            chords.push((u, v));
                        }
                        polys.push((sg.iter().map(|x| x.0).collect(), tag));
                    }
                }
            }
        }
        // cut (wall) polygons: reversed chords chained into cycles
        let rev: Vec<(u32, u32)> = chords.iter().map(|&(u, v)| (v, u)).collect();
        let mut out: Map<u32, Vec<usize>> = Map::default();
        for (i, &(u, _)) in rev.iter().enumerate() {
            out.entry(u).or_default().push(i);
        }
        let mut used = vec![false; rev.len()];
        for e0 in 0..rev.len() {
            if used[e0] {
                continue;
            }
            let start = rev[e0].0;
            let mut verts = vec![start];
            let mut cur = e0;
            let mut closed = false;
            loop {
                used[cur] = true;
                let to = rev[cur].1;
                if to == start {
                    closed = true;
                    break;
                }
                verts.push(to);
                match out.get(&to).and_then(|v| v.iter().find(|&&q| !used[q])) {
                    Some(&q) => cur = q,
                    None => break,
                }
            }
            if !closed {
                self.st.open_wall_chains += 1;
                continue;
            }
            polys.push((verts, Tag::Wall));
        }
        // clean up, drop degenerate polygons
        let mut keep: Vec<(Vec<u32>, Tag)> = Vec::with_capacity(polys.len());
        for (v, t) in polys {
            match self.clean(v) {
                Some(v) => keep.push((v, t)),
                None => self.st.dropped_faces += 1,
            }
        }
        // connected components through shared edges
        let np = keep.len();
        let mut par: Vec<usize> = (0..np).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        let mut em: Map<(u32, u32), usize> = Map::default();
        for (pi, (v, _)) in keep.iter().enumerate() {
            for t in 0..v.len() {
                let (x, y) = (v[t], v[(t + 1) % v.len()]);
                let k = (x.min(y), x.max(y));
                match em.get(&k) {
                    Some(&q) => {
                        let (ra, rb) = (find(&mut par, pi), find(&mut par, q));
                        par[ra] = rb;
                    }
                    None => {
                        em.insert(k, pi);
                    }
                }
            }
        }
        let mut groups: Map<usize, Vec<usize>> = Map::default();
        let mut order: Vec<usize> = Vec::new();
        for pi in 0..np {
            let r = find(&mut par, pi);
            groups.entry(r).or_insert_with(|| {
                order.push(r);
                Vec::new()
            });
            groups.get_mut(&r).unwrap().push(pi);
        }
        let h = self.oc.leaf_size(l);
        let mut emitted = 0;
        for r in order {
            let g = &groups[&r];
            if g.len() < 4 {
                self.st.degenerate_cells += 1;
                continue;
            }
            let mut bal: Map<(u32, u32), i32> = Map::default();
            for &pi in g {
                let v = &keep[pi].0;
                for t in 0..v.len() {
                    let (x, y) = (v[t], v[(t + 1) % v.len()]);
                    *bal.entry((x.min(y), x.max(y))).or_insert(0) += if x < y { 1 } else { -1 };
                }
            }
            if bal.values().any(|&b| b != 0) {
                self.st.unclosed_cells += 1;
            }
            let ci = self.cells.len() as u32;
            let mut fl = Vec::with_capacity(g.len());
            for &pi in g {
                let (v, t) = &keep[pi];
                fl.push(self.add_face(v.clone(), *t, ci));
            }
            self.cells.push(Cell { faces: fl, hex: None, h, cut: true, n_orig: 1 });
            emitted += 1;
        }
        if emitted > 1 {
            self.st.multi_component_cells += 1;
        }
        emitted
    }
}

impl CutMesh {
    /// Faces of cell `c` with the outward orientation of that cell.
    pub fn cell_polys(&self, c: usize) -> Vec<Vec<u32>> {
        self.cells[c]
            .faces
            .iter()
            .map(|&f| {
                let fc = &self.faces[f as usize];
                let mut v = fc.verts.clone();
                if fc.owner as usize != c {
                    v.reverse();
                }
                v
            })
            .collect()
    }

    /// Total volume with OpenFOAM's primitiveMesh definition (equal to `cell_volume` sums only for planar faces).
    pub fn foam_volume(&self) -> f64 {
        self.foam_geom().1.iter().map(|c| c.1).sum()
    }

    /// OpenFOAM primitiveMesh geometry: per face (area vector, centre), per cell (centre, volume).
    pub fn foam_geom(&self) -> (Vec<(V3, V3)>, Vec<(V3, f64)>) {
        let nc = self.cells.len();
        let mut fc: Vec<(V3, V3)> = Vec::with_capacity(self.faces.len()); // (area vector, centre)
        for f in &self.faces {
            let p: Vec<V3> = f.verts.iter().map(|&v| self.points[v as usize]).collect();
            let n = p.len();
            if n == 3 {
                let s = cross(sub(p[1], p[0]), sub(p[2], p[0])).map(|x| 0.5 * x);
                fc.push((s, [0, 1, 2].map(|d| (p[0][d] + p[1][d] + p[2][d]) / 3.0)));
                continue;
            }
            let c0 = {
                let mut c = [0.0; 3];
                for q in &p {
                    for d in 0..3 {
                        c[d] += q[d] / n as f64;
                    }
                }
                c
            };
            let tri = |i: usize| {
                let (a, b) = (p[i], p[(i + 1) % n]);
                (cross(sub(b, a), sub(c0, a)), [0, 1, 2].map(|d| a[d] + b[d] + c0[d]))
            };
            let mut sn = [0.0; 3];
            for i in 0..n {
                let t = tri(i).0;
                for d in 0..3 {
                    sn[d] += t[d];
                }
            }
            let l = dot(sn, sn).sqrt();
            let hat = sn.map(|x| x / l.max(1e-300));
            let (mut sa, mut sac) = (0.0, [0.0; 3]);
            for i in 0..n {
                let (t, c) = tri(i);
                let a = dot(t, hat);
                sa += a;
                for d in 0..3 {
                    sac[d] += a * c[d];
                }
            }
            fc.push((sn.map(|x| 0.5 * x), sac.map(|x| x / 3.0 / sa)));
        }
        let mut est = vec![([0.0; 3], 0usize); nc];
        for (i, f) in self.faces.iter().enumerate() {
            for c in [f.owner, f.neigh] {
                if c != NONE {
                    for d in 0..3 {
                        est[c as usize].0[d] += fc[i].1[d];
                    }
                    est[c as usize].1 += 1;
                }
            }
        }
        let mut cg = vec![([0.0; 3], 0.0f64); nc];
        let mut cvol = vec![0.0f64; nc];
        for (i, f) in self.faces.iter().enumerate() {
            let (s, c) = fc[i];
            for (cell, sign) in [(f.owner, 1.0), (f.neigh, -1.0)] {
                if cell == NONE {
                    continue;
                }
                let e = est[cell as usize];
                let ec = e.0.map(|x| x / e.1 as f64);
                let pv = sign * dot(s, sub(c, ec)) / 3.0;
                cvol[cell as usize] += pv;
                let pc = [0, 1, 2].map(|d| 0.75 * c[d] + 0.25 * ec[d]);
                for d in 0..3 {
                    cg[cell as usize].0[d] += pv * pc[d];
                }
                cg[cell as usize].1 += pv;
            }
        }
        let cells = (0..nc)
            .map(|c| {
                let e = est[c];
                let ec = e.0.map(|x| x / e.1.max(1) as f64);
                let ctr = if cg[c].1.abs() > 1e-300 { cg[c].0.map(|x| x / cg[c].1) } else { ec };
                (ctr, cvol[c])
            })
            .collect();
        (fc, cells)
    }

    /// Volume of cell `c` from the closed surface triangulated with face-centre fans (vertex average).
    pub fn cell_volume(&self, c: usize) -> f64 {
        let polys = self.cell_polys(c);
        let (mut o, mut n) = ([0.0; 3], 0.0);
        for p in &polys {
            for &v in p {
                for d in 0..3 {
                    o[d] += self.points[v as usize][d];
                }
                n += 1.0;
            }
        }
        let o = o.map(|x| x / n);
        let mut vol = 0.0;
        for p in &polys {
            let mut cf = [0.0; 3];
            for &v in p {
                for d in 0..3 {
                    cf[d] += self.points[v as usize][d];
                }
            }
            let cf = cf.map(|x| x / p.len() as f64);
            for t in 0..p.len() {
                let (a, b) = (self.points[p[t] as usize], self.points[p[(t + 1) % p.len()] as usize]);
                let nrm = cross(sub(a, cf), sub(b, cf));
                vol += dot(nrm, sub(cf, o)) / 6.0;
            }
        }
        vol
    }
}

impl CutMesh {
    /// Recompute volume statistics (`volume`, `min_vol_ratio`, `volume_foam_def`, non-positive cells).
    pub fn update_volume_stats(&mut self) {
        let (mut minr, mut tot, mut neg) = (f64::INFINITY, 0.0, 0usize);
        for c in 0..self.cells.len() {
            let v = self.cell_volume(c);
            let h = self.cells[c].h;
            minr = minr.min(v / (h * h * h));
            tot += v;
            neg += (v <= 0.0) as usize;
        }
        self.stats.min_vol_ratio = minr;
        self.stats.volume = tot;
        self.stats.volume_foam_def = self.foam_volume();
        self.stats.n_nonpositive_cells = neg;
    }

    /// Drop unused points and renumber (faces and hex node lists).
    fn compact_points(&mut self) {
        let mut remap = vec![NONE; self.points.len()];
        let mut np: Vec<V3> = Vec::with_capacity(self.points.len());
        for f in self.faces.iter_mut() {
            for v in f.verts.iter_mut() {
                if remap[*v as usize] == NONE {
                    remap[*v as usize] = np.len() as u32;
                    np.push(self.points[*v as usize]);
                }
                *v = remap[*v as usize];
            }
        }
        for c in self.cells.iter_mut() {
            if let Some(h) = c.hex.as_mut() {
                for v in h.iter_mut() {
                    *v = remap[*v as usize];
                }
            }
        }
        self.points = np;
    }

    /// Largest distance by which a vertex of the polyhedron (faces given as (face, cell is owner)) lies beyond the plane
    /// of one of its faces (face-plane concavity test with OpenFOAM face centres / normals).
    fn concavity(&self, fl: &[(u32, bool)], fg: &[(V3, V3)]) -> f64 {
        let mut mx = 0.0f64;
        for &(f, own) in fl {
            let (s, ctr) = fg[f as usize];
            let sl = dot(s, s).sqrt().max(1e-300);
            let sg = if own { 1.0 } else { -1.0 };
            let n = s.map(|x| sg * x / sl);
            for &(g, _) in fl {
                for &v in &self.faces[g as usize].verts {
                    mx = mx.max(dot(n, sub(self.points[v as usize], ctr)));
                }
            }
        }
        mx
    }

    /// OpenFOAM face-pyramid and face-tet checks of a polyhedron (faces as (face, cell is owner)) about its OpenFOAM
    /// cell centre: every face pyramid and every (edge, face centre, cell centre) tet must have positive volume.
    fn foam_cell_ok(&self, fl: &[(u32, bool)], fg: &[(V3, V3)]) -> bool {
        self.foam_cell_check(fl, fg).is_some()
    }

    /// Like `foam_cell_ok` but returns the OpenFOAM cell centre when the checks pass.
    fn foam_cell_check(&self, fl: &[(u32, bool)], fg: &[(V3, V3)]) -> Option<V3> {
        let mut est = [0.0; 3];
        for &(f, _) in fl {
            for d in 0..3 {
                est[d] += fg[f as usize].1[d] / fl.len() as f64;
            }
        }
        let (mut cc, mut vs) = ([0.0; 3], 0.0);
        for &(f, own) in fl {
            let (s, c) = fg[f as usize];
            let sg = if own { 1.0 } else { -1.0 };
            let pv = sg * dot(s, sub(c, est)) / 3.0;
            for d in 0..3 {
                cc[d] += pv * (0.75 * c[d] + 0.25 * est[d]);
            }
            vs += pv;
        }
        if vs <= 0.0 {
            return None;
        }
        let cc = cc.map(|x| x / vs);
        for &(f, own) in fl {
            let (s, c) = fg[f as usize];
            let sg = if own { 1.0 } else { -1.0 };
            if sg * dot(s, sub(c, cc)) <= 0.0 {
                return None;
            }
            let v = &self.faces[f as usize].verts;
            let n = v.len();
            for t in 0..n {
                let (a, b) = (self.points[v[t] as usize], self.points[v[(t + 1) % n] as usize]);
                let (a, b) = if own { (a, b) } else { (b, a) };
                if dot(cross(sub(b, a), sub(c, a)), sub(cc, a)) >= 0.0 {
                    return None;
                }
            }
        }
        Some(cc)
    }

    /// OpenFOAM face skewness (checkMesh limit 4) of every face of the union cell `un` (faces `fl`, cell centre `cc`); the
    /// neighbour centres come from `cg`. Returns the maximum.
    fn skew_max(&self, fl: &[(u32, bool)], un: &[u32], cc: V3, fg: &[(V3, V3)], cg: &[(V3, f64)]) -> f64 {
        let mut mx = 0.0f64;
        for &(f, _) in fl {
            let fc = &self.faces[f as usize];
            let (s, c) = fg[f as usize];
            let cpf = sub(c, cc);
            let (sv, fd0) = if fc.neigh == NONE {
                let sl = dot(s, s).sqrt().max(1e-300);
                let nrm = s.map(|x| x / sl);
                let d = nrm.map(|x| x * dot(nrm, cpf));
                let k = dot(s, cpf) / dot(s, d).abs().max(1e-300) * dot(s, d).signum();
                (sub(cpf, d.map(|x| x * k)), 0.4 * dot(d, d).sqrt())
            } else {
                let other = if un.contains(&fc.owner) { fc.neigh } else { fc.owner };
                let d = sub(cg[other as usize].0, cc);
                let sd = if un.contains(&fc.owner) { 1.0 } else { -1.0 };
                let d = d.map(|x| x * sd); // owner -> neighbour direction of the stored face
                let cpf = if sd > 0.0 { cpf } else { sub(c, cg[other as usize].0) };
                let den = dot(s, cpf);
                let den = if den.abs() < 1e-300 { 1e-300 } else { den };
                (sub(cpf.map(|x| x * dot(s, d) / den), d), 0.2 * dot(d, d).sqrt())
            };
            let sl = dot(sv, sv).sqrt();
            let mut fd = fd0 + 1e-300;
            if sl > 0.0 {
                let h = sv.map(|x| x / sl);
                for &v in &fc.verts {
                    fd = fd.max(dot(h, sub(self.points[v as usize], c)).abs());
                }
            }
            mx = mx.max(sl / fd);
        }
        mx
    }

    /// Flags of bad cut cells (see `Quality`), cell volumes and the quality counters.
    pub fn quality(&self, tau: f64) -> (Vec<bool>, Vec<f64>, Quality) {
        let nc = self.cells.len();
        let vol: Vec<f64> = (0..nc).map(|c| self.cell_volume(c)).collect();
        let (fg, cg) = self.foam_geom();
        let mut q = Quality::default();
        let mut bad = vec![false; nc];
        let mut small = vec![false; nc];
        let mut concave = vec![false; nc];
        let mut badtet = vec![false; nc];
        let mut nostar = vec![false; nc];
        for (c, cell) in self.cells.iter().enumerate() {
            if !cell.cut {
                continue;
            }
            small[c] = vol[c] < tau * cell.h * cell.h * cell.h;
            let fl: Vec<(u32, bool)> = cell.faces.iter().map(|&f| (f, self.faces[f as usize].owner as usize == c)).collect();
            concave[c] = self.concavity(&fl, &fg) > CONCAVE_TOL * cell.h;
            badtet[c] = !self.foam_cell_ok(&fl, &fg);
            let polys: Vec<(&[u32], bool)> = fl.iter().map(|&(f, o)| (self.faces[f as usize].verts.as_slice(), o)).collect();
            nostar[c] = !crate::su2::polys_decomposable(&self.points, &polys);
        }
        for (i, f) in self.faces.iter().enumerate() {
            if f.neigh == NONE {
                continue;
            }
            let (o, n) = (f.owner as usize, f.neigh as usize);
            let (s, fc) = fg[i];
            let a = dot(s, sub(fc, cg[o].0)).abs();
            let b = dot(s, sub(cg[n].0, fc)).abs();
            let w = b / (a + b).max(1e-300);
            if w.min(1.0 - w) < 0.05 {
                q.low_weight_faces += 1;
                let pick = if vol[o] <= vol[n] { [o, n] } else { [n, o] };
                if let Some(&c) = pick.iter().find(|&&c| self.cells[c].cut) {
                    bad[c] = true;
                }
            }
            if vol[o].min(vol[n]) < 0.01 * vol[o].max(vol[n]) {
                q.vol_ratio_faces += 1;
            }
        }
        for c in 0..nc {
            q.small += small[c] as usize;
            q.concave += (concave[c] && self.cells[c].n_orig == 1) as usize;
            q.concave_merged += (concave[c] && self.cells[c].n_orig > 1) as usize;
            bad[c] |= small[c] || badtet[c] || nostar[c] || (concave[c] && self.cells[c].n_orig == 1);
            q.bad_foam_tets += badtet[c] as usize;
            q.not_star_shaped += nostar[c] as usize;
            q.bad += bad[c] as usize;
        }
        (bad, vol, q)
    }

    /// True when the outward directed edges of the polygons cancel pairwise (closed polyhedron).
    fn polys_closed(faces: &[(&[u32], bool)]) -> bool {
        let mut m: Map<(u32, u32), i32> = Map::default();
        for &(v, own) in faces {
            let n = v.len();
            for t in 0..n {
                let (a, b) = (v[t], v[(t + 1) % n]);
                let (a, b) = if own { (a, b) } else { (b, a) };
                if a < b {
                    *m.entry((a, b)).or_insert(0) += 1;
                } else {
                    *m.entry((b, a)).or_insert(0) -= 1;
                }
            }
        }
        m.values().all(|&x| x == 0)
    }

    /// Merge bad cut cells (volume < tau h^3, concave, or low-weight face) into their best face neighbour: maximise
    /// shared area times neighbour volume, preferring neighbours that are not bad themselves. The merged cell is the
    /// union polyhedron (shared faces removed, all other faces untouched). A merge is accepted only if the union stays
    /// closed, has at most `cap` original cells and admits a valid SU2 apex decomposition (`su2::polys_decomposable`);
    /// otherwise the next neighbour is tried. At most `passes` passes (each re-evaluates the flags).
    pub fn agglomerate(&mut self, tau: f64, passes: usize, cap: u32) {
        let n_before = self.cells.len();
        let (_, _, q0) = self.quality(tau);
        self.stats.q_before = q0.clone();
        self.stats.bad_cells_before = q0.bad;
        self.stats.min_vol_ratio_before = self.stats.min_vol_ratio;
        let mut unmerge = 0;
        for _ in 0..passes {
            let (bad, vol, q) = self.quality(tau);
            if q.bad == 0 {
                unmerge = 0;
                break;
            }
            self.stats.merge_passes += 1;
            let (merged, um) = self.merge_pass(&bad, &vol, cap);
            unmerge = um;
            if merged == 0 {
                break;
            }
        }
        self.compact_points();
        self.update_volume_stats();
        let (_, _, q1) = self.quality(tau);
        self.stats.bad_cells_after = q1.bad;
        self.stats.q_after = q1;
        self.stats.unmergeable_bad_cells = unmerge;
        self.stats.cells_merged = n_before - self.cells.len();
        self.stats.merge_groups = self.cells.iter().filter(|c| c.n_orig > 1).count();
        let hex = self.cells.iter().filter(|c| c.hex.is_some()).count();
        self.stats.poly_cells = self.cells.len() - hex;
        self.stats.hex_cells = hex;
        self.stats.cut_cells = self.cells.iter().filter(|c| c.cut).count();
    }

    /// One agglomeration pass; returns (number of merges, bad cells left unmerged for lack of a valid partner).
    fn merge_pass(&mut self, bad: &[bool], vol: &[f64], cap: u32) -> (usize, usize) {
        let nc = self.cells.len();
        let (fg, cg) = self.foam_geom();
        let mut parent: Vec<u32> = (0..nc as u32).collect();
        let mut members: Vec<Vec<u32>> = (0..nc as u32).map(|c| vec![c]).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        let mut order: Vec<usize> = (0..nc).filter(|&c| bad[c]).collect();
        order.sort_by(|&a, &b| vol[a].partial_cmp(&vol[b]).unwrap());
        let (mut merges, mut unmergeable) = (0usize, 0usize);
        for c in order {
            let rc = find(&mut parent, c as u32);
            if members[rc as usize].len() > 1 {
                continue; // already part of a merge made in this pass
            }
            // neighbouring groups: root -> (shared area, a neighbouring cell id)
            let mut nb: Vec<(u32, f64, usize)> = Vec::new();
            for &f in &self.cells[c].faces {
                let fc = &self.faces[f as usize];
                if fc.neigh == NONE {
                    continue;
                }
                let o = if fc.owner as usize == c { fc.neigh } else { fc.owner };
                let r = find(&mut parent, o);
                let a = dot(fg[f as usize].0, fg[f as usize].0).sqrt();
                if let Some(e) = nb.iter_mut().find(|e| e.0 == r) {
                    e.1 += a;
                    if bad[o as usize] && !bad[e.2] {
                        e.2 = o as usize;
                    }
                } else {
                    nb.push((r, a, o as usize));
                }
            }
            let gvol = |mem: &[u32]| mem.iter().map(|&m| vol[m as usize]).sum::<f64>();
            let key: Vec<(bool, f64, u32)> = nb
                .iter()
                .map(|&(r, a, o)| (bad[o] && members[r as usize].len() == 1, a * gvol(&members[r as usize]), r))
                .collect();
            let mut idx: Vec<usize> = (0..nb.len()).collect();
            idx.sort_by(|&a, &b| key[a].0.cmp(&key[b].0).then(key[b].1.partial_cmp(&key[a].1).unwrap()));
            // candidates in preference order; accept the first valid union that is nearly convex and low-skew, else the
            // first low-skew one, else the valid union with the smallest skewness
            let mut valid: Vec<(f64, u32, Vec<u32>)> = Vec::new();
            let mut pick: Option<usize> = None;
            for &i in &idx {
                let r = key[i].2;
                let mut un: Vec<u32> = members[rc as usize].clone();
                un.extend_from_slice(&members[r as usize]);
                if un.iter().map(|&m| self.cells[m as usize].n_orig).sum::<u32>() > cap {
                    continue;
                }
                // faces of the union: those with an end outside it
                let mut fl: Vec<u32> = Vec::new();
                for &m in &un {
                    for &f in &self.cells[m as usize].faces {
                        let fc = &self.faces[f as usize];
                        let inside = |x: u32| x != NONE && un.contains(&x);
                        if !(inside(fc.owner) && inside(fc.neigh)) {
                            fl.push(f);
                        }
                    }
                }
                let fo: Vec<(u32, bool)> = fl.iter().map(|&f| (f, un.contains(&self.faces[f as usize].owner))).collect();
                let Some(cc) = self.foam_cell_check(&fo, &fg) else { continue };
                let polys: Vec<(&[u32], bool)> = fo.iter().map(|&(f, o)| (self.faces[f as usize].verts.as_slice(), o)).collect();
                if !Self::polys_closed(&polys) || !crate::su2::polys_decomposable(&self.points, &polys) {
                    continue;
                }
                let skew = self.skew_max(&fo, &un, cc, &fg, &cg);
                let hmax = un.iter().map(|&m| self.cells[m as usize].h).fold(0.0, f64::max);
                let convex = self.concavity(&fo, &fg) <= CONCAVE_TOL * hmax;
                valid.push((skew, r, un));
                if convex && skew <= SKEW_MAX {
                    pick = Some(valid.len() - 1);
                    break;
                }
            }
            if pick.is_none() {
                pick = valid.iter().position(|v| v.0 <= SKEW_MAX).or_else(|| {
                    (0..valid.len()).min_by(|&a, &b| valid[a].0.partial_cmp(&valid[b].0).unwrap())
                });
            }
            let mut done = false;
            if let Some(k) = pick {
                let (_, r, un) = valid.swap_remove(k);
                parent[r as usize] = rc;
                members[rc as usize] = un;
                members[r as usize] = Vec::new();
                merges += 1;
                done = true;
            }
            if !done {
                unmergeable += 1;
            }
        }
        if merges == 0 {
            return (0, unmergeable);
        }
        // apply: group representative = smallest member
        let mut newid = vec![NONE; nc];
        let mut rep: Vec<u32> = Vec::new(); // new id -> representative old id
        let mut root_new: Map<u32, u32> = Map::default();
        for c in 0..nc as u32 {
            let r = find(&mut parent, c);
            let id = *root_new.entry(r).or_insert_with(|| {
                rep.push(r);
                rep.len() as u32 - 1
            });
            newid[c as usize] = id;
        }
        let mut group_of: Vec<Vec<u32>> = vec![Vec::new(); rep.len()];
        for c in 0..nc as u32 {
            group_of[newid[c as usize] as usize].push(c);
        }
        // face renumbering: drop faces internal to one group
        let mut fnew = vec![NONE; self.faces.len()];
        let mut faces: Vec<Face> = Vec::with_capacity(self.faces.len());
        for (i, f) in self.faces.iter().enumerate() {
            let (o, n) = (newid[f.owner as usize], if f.neigh == NONE { NONE } else { newid[f.neigh as usize] });
            if n != NONE && o == n {
                continue;
            }
            fnew[i] = faces.len() as u32;
            faces.push(Face { verts: f.verts.clone(), owner: o, neigh: n, patch: f.patch });
        }
        let mut cells: Vec<Cell> = Vec::with_capacity(rep.len());
        for g in &group_of {
            if g.len() == 1 {
                let c = &self.cells[g[0] as usize];
                cells.push(Cell { faces: c.faces.iter().map(|&f| fnew[f as usize]).collect(), hex: c.hex, h: c.h, cut: c.cut, n_orig: c.n_orig });
            } else {
                let mut fl: Vec<u32> = Vec::new();
                let (mut h, mut cut, mut no) = (0.0f64, false, 0);
                for &m in g {
                    let c = &self.cells[m as usize];
                    fl.extend(c.faces.iter().map(|&f| fnew[f as usize]).filter(|&f| f != NONE));
                    h = h.max(c.h);
                    cut |= c.cut;
                    no += c.n_orig;
                }
                cells.push(Cell { faces: fl, hex: None, h, cut, n_orig: no });
            }
        }
        self.faces = faces;
        self.cells = cells;
        (merges, unmergeable)
    }
}

/// The clipped patch of a shell-clipped mesh: per vertex the point, the shell normal (unit, from the nearest wall point to
/// the vertex, away from the wall), the nearest wall point and its triangle; polygons are the patch faces re-oriented so
/// their normals point away from the wall (the cells' outward orientation points towards it).
pub struct ShellSurface {
    pub pts: Vec<V3>,
    pub nrm: Vec<V3>,
    pub wall: Vec<V3>,
    pub tri: Vec<u32>,
    pub polys: Vec<Vec<u32>>,
}

pub fn shell_surface(m: &CutMesh, tris: &[Tri]) -> ShellSurface {
    let bvh = Bvh::new(tris);
    let mut remap: Map<u32, u32> = Map::default();
    let mut ss = ShellSurface { pts: vec![], nrm: vec![], wall: vec![], tri: vec![], polys: vec![] };
    for f in m.faces.iter().filter(|f| f.patch == WALL) {
        let mut poly: Vec<u32> = Vec::with_capacity(f.verts.len());
        for &v in f.verts.iter().rev() {
            let id = *remap.entry(v).or_insert_with(|| {
                let p = m.points[v as usize];
                let (d, q, tri) = bvh.closest(p);
                let mut n = sub(p, q);
                if d > 1e-300 {
                    n = n.map(|x| x / d);
                } else {
                    let tr = &tris[tri as usize];
                    n = cross(sub(tr[1], tr[0]), sub(tr[2], tr[0]));
                    let nl = dot(n, n).sqrt();
                    n = n.map(|x| x / nl);
                }
                ss.pts.push(p);
                ss.nrm.push(n);
                ss.wall.push(q);
                ss.tri.push(tri);
                ss.pts.len() as u32 - 1
            });
            poly.push(id);
        }
        ss.polys.push(poly);
    }
    ss
}

/// Write [`ShellSurface`] as binary: u64 nverts, nverts * (3 f64 point, 3 f64 normal, 3 f64 wall point, u32 tri), u64 npolys,
/// per polygon u32 nv then nv u32 indices (little endian).
pub fn write_shell_surface(ss: &ShellSurface, path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(&(ss.pts.len() as u64).to_le_bytes())?;
    for i in 0..ss.pts.len() {
        for v in [ss.pts[i], ss.nrm[i], ss.wall[i]] {
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        w.write_all(&ss.tri[i].to_le_bytes())?;
    }
    w.write_all(&(ss.polys.len() as u64).to_le_bytes())?;
    for p in &ss.polys {
        w.write_all(&(p.len() as u32).to_le_bytes())?;
        for &v in p {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()
}

/// Build the fluid cut-cell mesh from a (buffered) octree with signs and cut edges.
pub fn build(oc: &Octree, tris: &[Tri], opt: &Options) -> CutMesh {
    let mut st = Stats::default();
    let t = Instant::now();
    let vk = oc.vertices();
    let nv = vk.len();
    let vidx = |k: u64| vk.binary_search(&k).expect("vertex");
    let os: Vec<bool> = vk.iter().map(|&k| oc.corner_sign(k)).collect();
    let ml = oc.max_level;
    let leaves = oc.leaves();
    let ckey = |l: &Leaf| -> [u64; 8] {
        let (b, e) = geom(l, ml);
        std::array::from_fn(|n| vertex_key(b[0] + e * (n as u32 & 1), b[1] + e * ((n as u32 >> 1) & 1), b[2] + e * ((n as u32 >> 2) & 1)))
    };
    let lv: Vec<[u32; 8]> = leaves
        .iter()
        .map(|l| {
            let ck = ckey(l);
            std::array::from_fn(|n| vidx(ck[n]) as u32)
        })
        .collect();

    // --- cut points per sign-changing edge
    let bvh = Bvh::new(tris);
    let mut edges: Map<(u64, u64), EdgeInfo> = Map::default();
    for c in oc.cut_edges() {
        let mut pt = c.point;
        if c.n_hits > 1 {
            let (pa, pb) = (oc.vertex_pos(c.a), oc.vertex_pos(c.b));
            let fluid_a = !os[vidx(c.a)];
            let mut hits = bvh.segment_hits(pa, pb);
            let d = if fluid_a { sub(pb, pa) } else { sub(pa, pb) };
            if !fluid_a {
                hits.reverse();
            }
            let (mut w, mut pick) = (0i32, 0usize);
            for (q, &(_, tri)) in hits.iter().enumerate() {
                let tr = &tris[tri as usize];
                let dd = dot(cross(sub(tr[1], tr[0]), sub(tr[2], tr[0])), d);
                w += if dd > 0.0 { -1 } else if dd < 0.0 { 1 } else { 0 };
                if w.abs() >= 1 {
                    pick = q;
                    break;
                }
            }
            if pick != 0 {
                st.multi_hit_nonfirst += 1;
            }
            let t = hits[pick].0;
            pt = [0, 1, 2].map(|k| pa[k] + t * (pb[k] - pa[k]));
        }
        edges.insert((c.a, c.b), EdgeInfo { pt, snap: 0 });
    }
    for (li, l) in leaves.iter().enumerate() {
        let _ = l;
        let v = &lv[li];
        for n in 0..8usize {
            for ax in 0..3 {
                if n & (1 << ax) != 0 {
                    continue;
                }
                let m = n | (1 << ax);
                if os[v[n] as usize] != os[v[m] as usize] {
                    let key = (vk[v[n] as usize], vk[v[m] as usize]);
                    edges.entry(key).or_insert_with(|| {
                        st.unresolved_edges += 1;
                        let (pa, pb) = (oc.vertex_pos(key.0), oc.vertex_pos(key.1));
                        EdgeInfo { pt: [0, 1, 2].map(|k| 0.5 * (pa[k] + pb[k])), snap: 0 }
                    });
                }
            }
        }
    }
    let mut flipped = vec![false; nv];
    for (&(a, b), e) in edges.iter_mut() {
        let (pa, pb) = (oc.vertex_pos(a), oc.vertex_pos(b));
        let len = dot(sub(pb, pa), sub(pb, pa)).sqrt();
        let (da, db) = (dot(sub(e.pt, pa), sub(e.pt, pa)).sqrt(), dot(sub(e.pt, pb), sub(e.pt, pb)).sqrt());
        if da <= opt.eps * len && da <= db {
            e.snap = 1;
        } else if db <= opt.eps * len {
            e.snap = 2;
        }
        if e.snap != 0 {
            st.snapped_points += 1;
            let end = vidx(if e.snap == 1 { a } else { b });
            if !os[end] {
                flipped[end] = true;
            }
        }
    }
    st.flipped_corners = flipped.iter().filter(|&&x| x).count();
    let es: Vec<bool> = (0..nv).map(|i| os[i] || flipped[i]).collect();
    st.t_edges = t.elapsed().as_secs_f64();

    // --- ambiguous cube faces: one decision per face key from the face centre
    let t = Instant::now();
    let mut cents: Map<(u64, u64), V3> = Map::default();
    for (li, l) in leaves.iter().enumerate() {
        let v = &lv[li];
        let s: [bool; 8] = std::array::from_fn(|n| es[v[n] as usize]);
        if s.iter().all(|&x| x) || s.iter().all(|&x| !x) {
            continue;
        }
        let (base, e) = geom(l, ml);
        for a in 0..3 {
            for sd in 0..2 {
                let r = ring(a, sd);
                if s[r[0]] == s[r[2]] && s[r[1]] == s[r[3]] && s[r[0]] != s[r[1]] {
                    let lat: [[u32; 3]; 4] = std::array::from_fn(|t| [base[0] + e * (r[t] as u32 & 1), base[1] + e * ((r[t] as u32 >> 1) & 1), base[2] + e * ((r[t] as u32 >> 2) & 1)]);
                    let key = amb_key(&lat, a);
                    cents.entry(key).or_insert_with(|| {
                        let p0 = oc.vertex_pos(vertex_key(lat[0][0], lat[0][1], lat[0][2]));
                        let p2 = oc.vertex_pos(vertex_key(lat[2][0], lat[2][1], lat[2][2]));
                        [0, 1, 2].map(|k| 0.5 * (p0[k] + p2[k]))
                    });
                }
            }
        }
    }
    let mut centre: Map<(u64, u64), bool> = Map::default();
    if !cents.is_empty() {
        let keys: Vec<(u64, u64)> = cents.keys().cloned().collect();
        let pts: Vec<V3> = keys.iter().map(|k| cents[k]).collect();
        let w = Fwn::new(tris).winding_many(&pts);
        for (q, (k, w)) in keys.into_iter().zip(w).enumerate() {
            // fluid = psi > 0: outside the soup and (shell) farther than delta from it
            let fluid = w.abs() <= 0.5 && oc.clip.delta().is_none_or(|dl| bvh.closest_within(pts[q], dl).is_none());
            centre.insert(k, fluid);
        }
    }
    st.t_faces = t.elapsed().as_secs_f64();

    // --- cells
    let t = Instant::now();
    let u = oc.size / (1u64 << ml) as f64;
    let mut b = Bld { oc, vk, es, os, edges, centre, pts: Vec::new(), pmap: Map::default(), faces: Vec::new(), fmap: Map::default(), cells: Vec::new(), st, amin2: 1e-12 * u * u };
    for (li, l) in leaves.iter().enumerate() {
        let v = lv[li];
        let nsolid = v.iter().filter(|&&x| b.es[x as usize]).count();
        if nsolid == 8 {
            continue;
        }
        if nsolid == 0 {
            b.uncut_cell(l);
        } else {
            b.st.cut_leaves += 1;
            let n = b.cut_cell(l, std::array::from_fn(|n| v[n] as usize));
            b.st.cut_cells += n;
        }
    }
    b.st.leaves = leaves.len();
    b.st.t_cells = t.elapsed().as_secs_f64();

    // --- finalize: orphan count, point compaction, volumes
    let t = Instant::now();
    let Bld { pts, mut faces, mut cells, mut st, .. } = b;
    st.orphan_faces = faces.iter().filter(|f| f.patch == ORPHAN).count();
    let mut remap = vec![NONE; pts.len()];
    let mut points: Vec<V3> = Vec::with_capacity(pts.len());
    for f in faces.iter_mut() {
        for v in f.verts.iter_mut() {
            if remap[*v as usize] == NONE {
                remap[*v as usize] = points.len() as u32;
                points.push(pts[*v as usize]);
            }
            *v = remap[*v as usize];
        }
    }
    for c in cells.iter_mut() {
        if let Some(h) = c.hex.as_mut() {
            for v in h.iter_mut() {
                *v = remap[*v as usize];
            }
        }
    }
    let mut mesh = CutMesh { points, faces, cells, stats: st, patch_name: oc.clip.patch_name(), native: Default::default() };
    mesh.update_volume_stats();
    mesh.stats.min_vol_ratio_before = mesh.stats.min_vol_ratio;
    let tm = Instant::now();
    if opt.merge_tau > 0.0 {
        mesh.agglomerate(opt.merge_tau, 3, 4);
    }
    mesh.stats.t_merge = tm.elapsed().as_secs_f64();
    mesh.stats.t_final = t.elapsed().as_secs_f64();
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::{Clip, SignMode, SizeField};
    use std::f64::consts::PI;

    fn sphere(c: V3, r: f64) -> Vec<Tri> {
        let (nu, nv) = (48usize, 24usize);
        let p = |i: usize, j: usize| {
            let (th, ph) = (PI * i as f64 / nv as f64, 2.0 * PI * (j % nu) as f64 / nu as f64);
            [c[0] + r * th.sin() * ph.cos(), c[1] + r * th.sin() * ph.sin(), c[2] + r * th.cos()]
        };
        let mut t: Vec<Tri> = Vec::new();
        for i in 0..nv {
            for j in 0..nu {
                if i > 0 {
                    t.push([p(i, j), p(i + 1, j), p(i, j + 1)]);
                }
                if i < nv - 1 {
                    t.push([p(i, j + 1), p(i + 1, j), p(i + 1, j + 1)]);
                }
            }
        }
        t
    }

    #[test]
    fn sphere_mesh_is_closed_and_has_right_volume() {
        let (c, r) = ([0.05, 0.02, -0.03], 0.5);
        let tris = sphere(c, r);
        let sf = SizeField::new(2.0 / 32.0, 0.3, 2.0 / 8.0);
        let oc = Octree::build_with_buffer(&tris, [-1.0; 3], 2.0, 5, &sf, 0, SignMode::Dexel);
        let m = build(&oc, &tris, &Options::default());
        let s = &m.stats;
        println!("cells {} (hex {} poly {} cut {}) vol {} min_ratio {:.3e} orphan {} unclosed {}", m.cells.len(), s.hex_cells, s.poly_cells, s.cut_cells, s.volume, s.min_vol_ratio, s.orphan_faces, s.unclosed_cells);
        assert_eq!(s.orphan_faces, 0);
        assert_eq!(s.unclosed_cells, 0);
        assert_eq!(s.nonmanifold_faces, 0);
        assert_eq!(s.n_nonpositive_cells, 0);
        let exact = 8.0 - 4.0 / 3.0 * PI * r * r * r;
        assert!((s.volume - exact).abs() < 0.02 * exact, "volume {} vs {exact}", s.volume);
        // writers: SU2 elements all positive; polyMesh face counts consistent
        let dir = std::env::temp_dir().join(format!("rcore_cutmesh_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s2 = crate::su2::write_su2(&m, &dir.join("mesh.su2")).unwrap();
        assert_eq!(s2.nonpositive, 0);
        assert_eq!(s2.n_hex, s.hex_cells);
        let (ni, patches) = crate::foam::write_polymesh(&m, &dir.join("polyMesh")).unwrap();
        assert_eq!(ni + patches.iter().map(|p| p.1).sum::<usize>(), m.faces.len());
        std::fs::remove_dir_all(&dir).unwrap();
        // every internal face has two cells, owners differ
        for f in &m.faces {
            assert!(f.patch != INTERNAL || (f.neigh != NONE && f.neigh != f.owner));
        }
    }

    #[test]
    fn wall_mode_counts_unchanged() {
        // recorded before the shell clipping was introduced
        let (c, r) = ([0.05, 0.02, -0.03], 0.5);
        let tris = sphere(c, r);
        let sf = SizeField::new(2.0 / 32.0, 0.3, 2.0 / 8.0);
        let oc = Octree::build_clipped(&tris, [-1.0; 3], 2.0, 5, &sf, 0, SignMode::Dexel, Clip::Wall);
        let m = build(&oc, &tris, &Options::default());
        assert_eq!((m.cells.len(), m.stats.hex_cells, m.stats.poly_cells, m.stats.cut_cells), (10188, 8333, 1855, 1005));
        assert_eq!(format!("{:.12}", m.stats.volume), "7.484578156114");
        assert_eq!(m.patch_name, "wall");
    }

    #[test]
    fn shell_mesh_closed_surface_oriented() {
        let (c, r) = ([0.05, 0.02, -0.03], 0.5);
        let tris = sphere(c, r);
        let sf = SizeField::new(2.0 / 32.0, 0.3, 2.0 / 8.0);
        let delta = 4.0 * 2.0 / 32.0;
        let oc = Octree::build_clipped(&tris, [-1.0; 3], 2.0, 5, &sf, 0, SignMode::Dexel, Clip::Shell { delta });
        let m = build(&oc, &tris, &Options { eps: 0.05, merge_tau: 0.0 });
        let s = &m.stats;
        println!("shell: cells {} (hex {} poly {} cut {}) vol {} min_ratio {:.3e}", m.cells.len(), s.hex_cells, s.poly_cells, s.cut_cells, s.volume, s.min_vol_ratio);
        assert_eq!((s.orphan_faces, s.unclosed_cells, s.nonmanifold_faces, s.n_nonpositive_cells), (0, 0, 0, 0));
        assert_eq!(m.patch_name, "shell");
        let exact = 8.0 - 4.0 / 3.0 * PI * (r + delta).powi(3);
        assert!((s.volume - exact).abs() < 0.03 * exact, "volume {} vs {exact}", s.volume);
        let ss = shell_surface(&m, &tris);
        assert!(!ss.polys.is_empty() && ss.polys.len() == m.faces.iter().filter(|f| f.patch == WALL).count());
        let bvh = Bvh::new(&tris);
        for i in 0..ss.pts.len() {
            let n = ss.nrm[i];
            assert!((dot(n, n).sqrt() - 1.0).abs() < 1e-12);
            let d = sub(ss.pts[i], ss.wall[i]);
            assert!(dot(n, d) > 0.0);
            assert!((bvh.closest(ss.pts[i]).0 - dot(d, d).sqrt()).abs() < 1e-12);
            assert!((dot(d, d).sqrt() - delta).abs() < 0.06 * delta, "vertex off the shell");
        }
        // polygons: Newell normal points away from the wall (agrees with the vertex normals)
        for p in &ss.polys {
            let p0 = ss.pts[p[0] as usize];
            let mut nn = [0.0; 3];
            for t in 1..p.len() - 1 {
                let a = cross(sub(ss.pts[p[t] as usize], p0), sub(ss.pts[p[t + 1] as usize], p0));
                for k in 0..3 {
                    nn[k] += a[k];
                }
            }
            let vn = p.iter().fold([0.0; 3], |a, &v| [0, 1, 2].map(|k| a[k] + ss.nrm[v as usize][k]));
            assert!(dot(nn, vn) > 0.0, "polygon oriented towards the wall");
        }
        let dir = std::env::temp_dir().join(format!("rcore_shell_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_shell_surface(&ss, &dir.join("shell_surface.bin")).unwrap();
        let sz = std::fs::metadata(dir.join("shell_surface.bin")).unwrap().len() as usize;
        assert_eq!(sz, 8 + ss.pts.len() * 76 + 8 + ss.polys.iter().map(|p| 4 + 4 * p.len()).sum::<usize>());
        let s2 = crate::su2::write_su2(&m, &dir.join("mesh.su2")).unwrap();
        assert_eq!(s2.nonpositive, 0);
        let (ni, patches) = crate::foam::write_polymesh(&m, &dir.join("polyMesh")).unwrap();
        assert_eq!(patches[0].0, "shell");
        assert_eq!(ni + patches.iter().map(|p| p.1).sum::<usize>(), m.faces.len());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn agglomeration_preserves_volume_and_closedness() {
        let (c, r) = ([0.05, 0.02, -0.03], 0.5);
        let tris = sphere(c, r);
        let sf = SizeField::new(2.0 / 32.0, 0.3, 2.0 / 8.0);
        let oc = Octree::build_with_buffer(&tris, [-1.0; 3], 2.0, 5, &sf, 0, SignMode::Dexel);
        let mut m = build(&oc, &tris, &Options { eps: 0.05, merge_tau: 0.0 });
        let (n0, v0) = (m.cells.len(), (0..m.cells.len()).map(|c| m.cell_volume(c)).sum::<f64>());
        m.agglomerate(0.3, 3, 4);
        let v1: f64 = (0..m.cells.len()).map(|c| m.cell_volume(c)).sum();
        println!("cells {n0} -> {} merged groups {} volume {v0} -> {v1}", m.cells.len(), m.stats.merge_groups);
        assert!(m.cells.len() < n0 && m.stats.merge_groups > 0, "nothing merged");
        assert_eq!(n0 - m.cells.len(), m.stats.cells_merged);
        assert!((v1 - v0).abs() <= 1e-12 * v0.abs(), "volume {v0} -> {v1}");
        let mut seen_merged = 0;
        for c in 0..m.cells.len() {
            let polys = m.cell_polys(c);
            let mut sum = [0.0; 3];
            let mut mag = 0.0;
            for p in &polys {
                let p0 = m.points[p[0] as usize];
                for t in 1..p.len() - 1 {
                    let a = cross(sub(m.points[p[t] as usize], p0), sub(m.points[p[t + 1] as usize], p0));
                    for d in 0..3 {
                        sum[d] += 0.5 * a[d];
                    }
                    mag += 0.5 * dot(a, a).sqrt();
                }
            }
            assert!(dot(sum, sum).sqrt() <= 1e-12 * mag.max(1e-300), "cell {c} not closed: {sum:?}");
            let fo: Vec<(&[u32], bool)> = m.cells[c].faces.iter().map(|&f| (m.faces[f as usize].verts.as_slice(), m.faces[f as usize].owner as usize == c)).collect();
            assert!(CutMesh::polys_closed(&fo), "cell {c} edges do not cancel");
            seen_merged += (m.cells[c].n_orig > 1) as usize;
        }
        assert_eq!(seen_merged, m.stats.merge_groups);
        for f in &m.faces {
            assert!(f.patch != INTERNAL || (f.neigh != NONE && f.neigh != f.owner));
        }
        let dir = std::env::temp_dir().join(format!("rcore_merge_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s2 = crate::su2::write_su2(&m, &dir.join("mesh.su2")).unwrap();
        assert_eq!(s2.nonpositive, 0);
        let (ni, patches) = crate::foam::write_polymesh(&m, &dir.join("polyMesh")).unwrap();
        assert_eq!(ni + patches.iter().map(|p| p.1).sum::<usize>(), m.faces.len());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
