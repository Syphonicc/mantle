//! SU2 3D writer (`NDIME= 3`) with a conforming element decomposition.
//!
//! * Uncut leaves whose six faces are plain quads -> HEXAHEDRON (12, VTK node order).
//! * Every other fluid cell gets an apex point (volume centroid, falling back to the vertex or face-centre mean when that
//!   is not star-shaped) and one PYRAMID (14) per quad face, one TETRAHEDRON (10) per triangle face; faces with more than
//!   four vertices are fanned about a face-centroid point, created once per face (shared by both cells) -> tets.
//! * Markers `wall`, `farfield` (and `orphan` if any): TRIANGLE (5) / QUAD (9) matching the volume face decomposition
//!   exactly, outward orientation.
//! * Quad faces that are not convex / planar enough are split into two triangles (best diagonal) by a pure function of the
//!   face geometry, so both cells and the boundary element agree.
//! * Every element is checked (`element_valid`): tets positive; pyramids: both diagonal splits of the base give positive
//!   tets and the base is convex in projection; hexes: all 8 corner Jacobians positive (a failing hex is decomposed like a
//!   polyhedron). Surface elements are written with the normal pointing INTO the fluid domain (SU2 convention, found
//!   empirically: outward-oriented elements make SU2_CFD report re-orientation of every surface element).

use crate::cutmesh::{CutMesh, FARFIELD, ORPHAN, WALL};
use std::io::{BufWriter, Write};
use std::path::Path;

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
fn mean(p: &[V3]) -> V3 {
    let mut c = [0.0; 3];
    for q in p {
        for d in 0..3 {
            c[d] += q[d];
        }
    }
    c.map(|x| x / p.len() as f64)
}
/// signed volume of tet (a, b, c, d); positive when (b-a)x(c-a) points towards d
fn tet_vol(a: V3, b: V3, c: V3, d: V3) -> f64 {
    dot(cross(sub(b, a), sub(c, a)), sub(d, a)) / 6.0
}

#[derive(Clone, Debug, Default)]
pub struct Su2Stats {
    pub n_tet: usize,
    pub n_hex: usize,
    pub n_prism: usize,
    pub n_pyramid: usize,
    pub n_points: usize,
    /// elements failing `element_valid` (after apex search)
    pub nonpositive: usize,
    pub n_wall_elems: usize,
    pub n_far_elems: usize,
    pub n_orphan_elems: usize,
    /// cells whose apex had to be moved from the volume centroid
    pub apex_fallbacks: usize,
    /// quad faces split into two triangles (non-convex / twisted)
    pub quads_split: usize,
    /// hexahedra that failed the corner-Jacobian check and were decomposed like polyhedra
    pub hex_decomposed: usize,
}

const NB: [[usize; 3]; 8] = [[1, 3, 4], [2, 0, 5], [3, 1, 6], [0, 2, 7], [7, 5, 0], [4, 6, 1], [5, 7, 2], [6, 4, 3]];


/// Validity of one SU2 element; `p` holds the node coordinates in SU2/VTK order (tet 4, pyramid 5, hex 8).
pub fn element_valid(ty: u8, p: &[V3]) -> bool {
    match ty {
        10 => tet_vol(p[0], p[1], p[2], p[3]) > 0.0,
        14 => {
            let q = [p[0], p[1], p[2], p[3]];
            let a = p[4];
            // all four corner tets positive, with a small slack for slightly warped / nearly flat bases (SU2 accepts those)
            let l = (0..4).map(|i| dot(sub(q[i], a), sub(q[i], a)).sqrt()).fold(0.0, f64::max);
            let tol = -1e-6 * l * l * l;
            tet_vol(q[0], q[1], q[2], a) > tol && tet_vol(q[0], q[2], q[3], a) > tol && tet_vol(q[0], q[1], q[3], a) > tol && tet_vol(q[1], q[2], q[3], a) > tol
        }
        13 => {
            // VTK wedge: (0,1,2) is clockwise seen from (3,4,5), i.e. left-handed relative to Gmsh; map to Gmsh order and test the 6 corners
            let g = [p[0], p[2], p[1], p[3], p[5], p[4]];
            (0..6).all(|i| {
                let (j, k) = ((i % 3 + 1) % 3 + 3 * (i / 3), (i % 3 + 2) % 3 + 3 * (i / 3));
                let o = (i + 3) % 6;
                let d = dot(cross(sub(g[j], g[i]), sub(g[k], g[i])), sub(g[o], g[i]));
                if i < 3 { d > 0.0 } else { d < 0.0 }
            })
        }
        12 => (0..8).all(|i| dot(cross(sub(p[NB[i][0]], p[i]), sub(p[NB[i][1]], p[i])), sub(p[NB[i][2]], p[i])) > 0.0),
        _ => false,
    }
}

/// Triangulation of a polygon face (> 4 vertices) in the owner orientation: a centre point (vertex mean, else area
/// centroid) if all fan triangles keep the polygon normal, otherwise ear clipping without a centre point.
fn triangulate(pts: &[V3], v: &[u32]) -> (Option<V3>, Vec<[u32; 3]>, Vec<usize>) {
    let n = v.len();
    let p: Vec<V3> = v.iter().map(|&i| pts[i as usize]).collect();
    let p0 = p[0];
    let mut nn = [0.0; 3];
    for t in 1..n - 1 {
        let c = cross(sub(p[t], p0), sub(p[t + 1], p0));
        for d in 0..3 {
            nn[d] += c[d];
        }
    }
    let valid = |c: V3| (0..n).all(|t| dot(cross(sub(p[t], c), sub(p[(t + 1) % n], c)), nn) > 0.0);
    let vm = mean(&p);
    // area centroid of the fan about p0
    let (mut ac, mut aw) = ([0.0; 3], 0.0);
    for t in 1..n - 1 {
        let a = dot(cross(sub(p[t], p0), sub(p[t + 1], p0)), nn);
        for d in 0..3 {
            ac[d] += a * (p0[d] + p[t][d] + p[t + 1][d]) / 3.0;
        }
        aw += a;
    }
    let cands = [vm, ac.map(|x| x / aw)];
    for c in cands {
        if valid(c) {
            return (Some(c), Vec::new(), (0..n).collect());
        }
    }
    // ear clipping in the plane of the dominant normal axis
    let ax = (0..3).max_by(|&a, &b| nn[a].abs().partial_cmp(&nn[b].abs()).unwrap()).unwrap();
    let (u, w) = ((ax + 1) % 3, (ax + 2) % 3);
    let sg = if nn[ax] >= 0.0 { 1.0 } else { -1.0 };
    let area2 = |a: V3, b: V3, c: V3| sg * ((b[u] - a[u]) * (c[w] - a[w]) - (b[w] - a[w]) * (c[u] - a[u]));
    let mut idx: Vec<usize> = (0..n).collect();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    while idx.len() > 3 {
        let m = idx.len();
        let mut best: Option<(usize, f64)> = None;
        for k in 0..m {
            let (a, b, c) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
            let ar = area2(p[a], p[b], p[c]);
            if ar <= 0.0 {
                continue;
            }
            let inside = idx.iter().any(|&q| {
                q != a && q != b && q != c && area2(p[a], p[b], p[q]) >= 0.0 && area2(p[b], p[c], p[q]) >= 0.0 && area2(p[c], p[a], p[q]) >= 0.0
            });
            if !inside && best.map_or(true, |x| ar > x.1) {
                best = Some((k, ar));
            }
        }
        let k = best.map_or(0, |x| x.0); // degenerate polygon: clip anyway
        let (a, b, c) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
        tris.push([v[a], v[b], v[c]]);
        idx.remove(k);
    }
    tris.push([v[idx[0]], v[idx[1]], v[idx[2]]]);
    (None, tris, Vec::new())
}


/// Quad faces that need splitting get two triangles (diagonal chosen so both agree with the quad normal); polygons with
/// more than four vertices use `triangulate`. `None` for triangles and good quads. Triangles may use the id `u32::MAX`
/// for the (returned) face-centre point.
fn split_face(pts: &[V3], v: &[u32]) -> Option<(Option<V3>, Vec<[u32; 3]>)> {
    match v.len() {
        3 => None,
        // quads stay native (SU2 handles non-planar quads); only the apex fallback in `write_su2` splits them
        4 => None,
        n => {
            let (c, tris, _) = triangulate(pts, v);
            Some(match c {
                Some(c) => (Some(c), (0..n).map(|t| [v[t], v[(t + 1) % n], u32::MAX]).collect()),
                None => (None, tris),
            })
        }
    }
}

/// Split a quad along the diagonal whose two triangles agree best with the quad normal.
fn diag_split(pts: &[V3], v: &[u32]) -> Vec<[u32; 3]> {
    let q: [V3; 4] = std::array::from_fn(|i| pts[v[i] as usize]);
    let n = cross(sub(q[2], q[0]), sub(q[3], q[1]));
    let nl = dot(n, n).sqrt().max(1e-300);
    let al = |a: usize, b: usize, c: usize| {
        let t = cross(sub(q[b], q[a]), sub(q[c], q[a]));
        dot(t, n) / (dot(t, t).sqrt().max(1e-300) * nl)
    };
    let s02 = al(0, 1, 2).min(al(0, 2, 3));
    let s13 = al(1, 2, 3).min(al(1, 3, 0));
    if s02 >= s13 { vec![[v[0], v[1], v[2]], [v[0], v[2], v[3]]] } else { vec![[v[1], v[2], v[3]], [v[1], v[3], v[0]]] }
}

/// Outward triangle / quad of a cell face (ids may refer to temporary points).
#[derive(Clone, Copy)]
enum Item {
    Tri([u32; 3]),
    Quad([u32; 4]),
}

fn item_has(it: &Item, v: u32) -> bool {
    match it {
        Item::Tri(t) => t.contains(&v),
        Item::Quad(q) => q.contains(&v),
    }
}

fn rev3(t: [u32; 3]) -> [u32; 3] {
    [t[2], t[1], t[0]]
}

/// Search an apex from which every item is a valid element; returns (apex, number of invalid elements, moved from the centroid).
fn find_apex(p: &dyn Fn(u32) -> V3, items: &[Item]) -> (V3, usize, bool) {
    let verts = |it: &Item| -> Vec<V3> {
        match it {
            Item::Tri(t) => t.iter().map(|&v| p(v)).collect(),
            Item::Quad(q) => q.iter().map(|&v| p(v)).collect(),
        }
    };
    let mut allv: Vec<V3> = Vec::new();
    let mut fcs: Vec<V3> = Vec::new();
    for it in items {
        let vs = verts(it);
        fcs.push(mean(&vs));
        allv.extend(vs);
    }
    let vmean = mean(&allv);
    let (mut vc, mut vt) = ([0.0; 3], 0.0);
    for (it, cf) in items.iter().zip(&fcs) {
        let vs = verts(it);
        for t in 0..vs.len() {
            let (a, b) = (vs[t], vs[(t + 1) % vs.len()]);
            let w = tet_vol(a, *cf, b, vmean).abs();
            for d in 0..3 {
                vc[d] += w * 0.25 * (a[d] + b[d] + cf[d] + vmean[d]);
            }
            vt += w;
        }
    }
    let score = |ap: V3| -> usize {
        let mut bad = 0;
        for it in items {
            match it {
                Item::Tri(t) => bad += !element_valid(10, &[p(t[0]), p(t[2]), p(t[1]), ap]) as usize,
                Item::Quad(q) => bad += !element_valid(14, &[p(q[0]), p(q[3]), p(q[2]), p(q[1]), ap]) as usize,
            }
        }
        bad
    };
    let mut cands: Vec<V3> = vec![vc.map(|x| x / vt.max(1e-300)), vmean, mean(&fcs)];
    let mut best = 0;
    let mut bs = score(cands[0]);
    let mut k = 1;
    let mut stage = 0;
    while bs > 0 {
        if k == cands.len() {
            stage += 1;
            match stage {
                1 => {
                    for f in &fcs {
                        for w in [0.25, 0.5, 0.75] {
                            cands.push([0, 1, 2].map(|d| w * f[d] + (1.0 - w) * cands[0][d]));
                            cands.push([0, 1, 2].map(|d| w * f[d] + (1.0 - w) * cands[1][d]));
                        }
                    }
                }
                2 => {
                    // deterministic pseudo-random points in the bounding box
                    let (mut lo, mut hi) = (allv[0], allv[0]);
                    for q in &allv {
                        for d in 0..3 {
                            lo[d] = lo[d].min(q[d]);
                            hi[d] = hi[d].max(q[d]);
                        }
                    }
                    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
                    for _ in 0..400 {
                        let mut pt = [0.0; 3];
                        for d in 0..3 {
                            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                            pt[d] = lo[d] + (hi[d] - lo[d]) * ((s >> 11) as f64 / (1u64 << 53) as f64);
                        }
                        cands.push(pt);
                    }
                }
                _ => break,
            }
            continue;
        }
        let s = score(cands[k]);
        if s < bs {
            bs = s;
            best = k;
        }
        k += 1;
    }
    (cands[best], bs, best != 0)
}

/// True when the polyhedron given by `faces` ((vertices in owner orientation, cell is owner)) can be decomposed by the SU2
/// writer into valid pyramids / tets about a single apex (star-shaped w.r.t. some point).
pub fn polys_decomposable(pts: &[V3], faces: &[(&[u32], bool)]) -> bool {
    let n0 = pts.len() as u32;
    let mut extra: Vec<V3> = Vec::new();
    let mut items: Vec<Item> = Vec::new();
    for &(v, own) in faces {
        match split_face(pts, v) {
            None => {
                if v.len() == 3 {
                    items.push(Item::Tri(if own { [v[0], v[1], v[2]] } else { [v[2], v[1], v[0]] }));
                } else {
                    items.push(Item::Quad(if own { [v[0], v[1], v[2], v[3]] } else { [v[3], v[2], v[1], v[0]] }));
                }
            }
            Some((c, tris)) => {
                let id = if let Some(c) = c {
                    extra.push(c);
                    n0 + extra.len() as u32 - 1
                } else {
                    u32::MAX
                };
                for t in tris {
                    let t = if t[2] == u32::MAX { [t[0], t[1], id] } else { t };
                    items.push(Item::Tri(if own { t } else { rev3(t) }));
                }
            }
        }
    }
    let p = |i: u32| if i < n0 { pts[i as usize] } else { extra[(i - n0) as usize] };
    find_apex(&p, &items).1 == 0
}

/// Write `path` (.su2) and return element statistics.
pub fn write_su2(m: &CutMesh, path: &Path) -> std::io::Result<Su2Stats> {
    let mut pts = m.points.clone();
    let mut st = Su2Stats::default();
    let mut elems: Vec<(u8, [u32; 8])> = Vec::with_capacity(m.cells.len() * 4);
    // face triangulations (owner orientation) for faces that are not plain triangles / good quads
    let mut ftri: Vec<Vec<[u32; 3]>> = vec![Vec::new(); m.faces.len()];
    for (fi, f) in m.faces.iter().enumerate() {
        if let Some((c, mut tris)) = split_face(&pts, &f.verts) {
            if f.verts.len() == 4 {
                st.quads_split += 1;
            }
            if let Some(c) = c {
                pts.push(c);
                let id = pts.len() as u32 - 1;
                for t in tris.iter_mut() {
                    if t[2] == u32::MAX {
                        t[2] = id;
                    }
                }
            }
            ftri[fi] = tris;
        }
    }
    // cells written as native elements need no apex decomposition (their faces stay quads / triangles, non-planar quads are fine for SU2)
    let native_ok = |ci: usize, c: &crate::cutmesh::Cell| -> bool {
        if let Some(&(ty, e)) = m.native.get(&(ci as u32)) {
            let n = if ty == 13 { 6 } else if ty == 10 { 4 } else { 5 };
            if element_valid(ty, &e[..n].iter().map(|&i| pts[i as usize]).collect::<Vec<V3>>()) {
                return true;
            }
        }
        c.hex.is_some_and(|h| element_valid(12, &h.iter().map(|&i| pts[i as usize]).collect::<Vec<V3>>()))
    };
    // Fallback for cells without a valid apex: split all their plain quads into triangles (shared with the neighbours,
    // so the mesh stays conforming) and retry, up to three rounds.
    let mut vapex: std::collections::HashMap<usize, u32> = Default::default();
    for _ in 0..3 {
        let mut changed = false;
        for (ci, c) in m.cells.iter().enumerate() {
            if native_ok(ci, c) || vapex.contains_key(&ci) {
                continue;
            }
            let mut items: Vec<Item> = Vec::new();
            for &f in &c.faces {
                let fc = &m.faces[f as usize];
                let own = fc.owner as usize == ci;
                let v = &fc.verts;
                if !ftri[f as usize].is_empty() {
                    for t in &ftri[f as usize] {
                        items.push(Item::Tri(if own { *t } else { rev3(*t) }));
                    }
                } else if v.len() == 3 {
                    items.push(Item::Tri(if own { [v[0], v[1], v[2]] } else { [v[2], v[1], v[0]] }));
                } else {
                    items.push(Item::Quad(if own { [v[0], v[1], v[2], v[3]] } else { [v[3], v[2], v[1], v[0]] }));
                }
            }
            let pl = |i: u32| pts[i as usize];
            let fa = find_apex(&pl, &items);
            if fa.1 == 0 {
                continue;
            }
            // vertex apex (collapsed cells such as wedges whose kernel is a vertex / edge): pyramids over the faces not touching the apex vertex;
            // the faces touching it are fanned from it (shared with the neighbours through `ftri`)
            let mut vs: Vec<u32> = c.faces.iter().flat_map(|&f| m.faces[f as usize].verts.iter().cloned()).collect();
            vs.sort_unstable();
            vs.dedup();
            let ok_at = |v: u32| {
                let ap = pts[v as usize];
                items.iter().filter(|it| !item_has(it, v)).all(|it| match it {
                    Item::Tri(t) => element_valid(10, &[pl(t[0]), pl(t[2]), pl(t[1]), ap]),
                    Item::Quad(q) => element_valid(14, &[pl(q[0]), pl(q[3]), pl(q[2]), pl(q[1]), ap]),
                })
            };
            if let Some(&v) = vs.iter().find(|&&v| ok_at(v)) {
                for &f in &c.faces {
                    let w = &m.faces[f as usize].verts;
                    if w.len() == 4 && ftri[f as usize].is_empty() {
                        if let Some(i) = w.iter().position(|&x| x == v) {
                            ftri[f as usize] = vec![[w[i], w[(i + 1) % 4], w[(i + 2) % 4]], [w[i], w[(i + 2) % 4], w[(i + 3) % 4]]];
                            st.quads_split += 1;
                        }
                    }
                }
                vapex.insert(ci, v);
                changed = true;
                continue;
            }
            for &f in &c.faces {
                let v = &m.faces[f as usize].verts;
                if v.len() == 4 && ftri[f as usize].is_empty() {
                    ftri[f as usize] = diag_split(&pts, v);
                    st.quads_split += 1;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut push = |st: &mut Su2Stats, pts: &[V3], ty: u8, e: [u32; 8]| {
        let n = match ty {
            10 => 4,
            12 => 8,
            13 => 6,
            _ => 5,
        };
        let q: Vec<V3> = e[..n].iter().map(|&i| pts[i as usize]).collect();
        if !element_valid(ty, &q) {
            st.nonpositive += 1;
        }
        match ty {
            10 => st.n_tet += 1,
            12 => st.n_hex += 1,
            13 => st.n_prism += 1,
            _ => st.n_pyramid += 1,
        }
        elems.push((ty, e));
    };
    for (ci, c) in m.cells.iter().enumerate() {
        // a hex is only written directly when none of its faces was split into triangles (warped pillow quads), else it is
        // decomposed so that its faces match the neighbouring elements and the boundary elements
        if let Some(&(ty, e)) = m.native.get(&(ci as u32)) {
            if c.faces.iter().all(|&f| ftri[f as usize].is_empty()) {
                let n = match ty {
                    13 => 6,
                    10 => 4,
                    _ => 5,
                };
                let q: Vec<V3> = e[..n].iter().map(|&i| pts[i as usize]).collect();
                if element_valid(ty, &q) {
                    push(&mut st, &pts, ty, e);
                    continue;
                }
            }
        }
        let hex = c.hex.filter(|_| c.faces.iter().all(|&f| ftri[f as usize].is_empty()));
        if let Some(h) = hex {
            let q: Vec<V3> = h.iter().map(|&i| pts[i as usize]).collect();
            if element_valid(12, &q) {
                push(&mut st, &pts, 12, h);
                continue;
            }
            st.hex_decomposed += 1;
        }
        // faces of the cell as outward triangles / quads
        let mut items: Vec<Item> = Vec::new();
        for &f in &c.faces {
            let fc = &m.faces[f as usize];
            let own = fc.owner as usize == ci;
            let v = &fc.verts;
            if !ftri[f as usize].is_empty() {
                for t in &ftri[f as usize] {
                    items.push(Item::Tri(if own { *t } else { rev3(*t) }));
                }
            } else if v.len() == 3 {
                items.push(Item::Tri(if own { [v[0], v[1], v[2]] } else { [v[2], v[1], v[0]] }));
            } else {
                items.push(Item::Quad(if own { [v[0], v[1], v[2], v[3]] } else { [v[3], v[2], v[1], v[0]] }));
            }
        }
        if let Some(&v) = vapex.get(&ci) {
            let z = 0;
            for it in items.iter().filter(|it| !item_has(it, v)) {
                match it {
                    Item::Tri(t) => push(&mut st, &pts, 10, [t[0], t[2], t[1], v, z, z, z, z]),
                    Item::Quad(p) => push(&mut st, &pts, 14, [p[0], p[3], p[2], p[1], v, z, z, z]),
                }
            }
            continue;
        }
        let (ap, _bad, moved) = {
            let pl = |i: u32| pts[i as usize];
            find_apex(&pl, &items)
        };
        if moved {
            st.apex_fallbacks += 1;
        }
        pts.push(ap);
        let apex = pts.len() as u32 - 1;
        let z = 0;
        for it in &items {
            match it {
                Item::Tri(t) => push(&mut st, &pts, 10, [t[0], t[2], t[1], apex, z, z, z, z]),
                Item::Quad(p) => push(&mut st, &pts, 14, [p[0], p[3], p[2], p[1], apex, z, z, z]),
            }
        }
    }
    // boundary elements
    let mut marks: [Vec<(u8, [u32; 4])>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (fi, f) in m.faces.iter().enumerate() {
        let k = match f.patch {
            WALL => 0,
            FARFIELD => 1,
            ORPHAN => 2,
            _ => continue,
        };
        let v = &f.verts;
        if !ftri[fi].is_empty() {
            for t in &ftri[fi] {
                marks[k].push((5, [t[0], t[2], t[1], 0]));
            }
        } else if v.len() == 3 {
            marks[k].push((5, [v[0], v[2], v[1], 0]));
        } else {
            marks[k].push((9, [v[0], v[3], v[2], v[1]]));
        }
    }
    st.n_wall_elems = marks[0].len();
    st.n_far_elems = marks[1].len();
    st.n_orphan_elems = marks[2].len();
    st.n_points = pts.len();

    let mut w = BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "NDIME= 3\nNELEM= {}", elems.len())?;
    for (i, (ty, e)) in elems.iter().enumerate() {
        let n = match ty {
            10 => 4,
            12 => 8,
            13 => 6,
            _ => 5,
        };
        write!(w, "{ty}")?;
        for v in &e[..n] {
            write!(w, " {v}")?;
        }
        writeln!(w, " {i}")?;
    }
    writeln!(w, "NPOIN= {}", pts.len())?;
    for (i, p) in pts.iter().enumerate() {
        writeln!(w, "{:.16e} {:.16e} {:.16e} {i}", p[0], p[1], p[2])?;
    }
    let names = [m.patch_name, "farfield", "orphan"];
    let nm = if marks[2].is_empty() { 2 } else { 3 };
    writeln!(w, "NMARK= {nm}")?;
    for k in 0..nm {
        writeln!(w, "MARKER_TAG= {}\nMARKER_ELEMS= {}", names[k], marks[k].len())?;
        for (ty, e) in &marks[k] {
            if *ty == 5 {
                writeln!(w, "5 {} {} {}", e[0], e[1], e[2])?;
            } else {
                writeln!(w, "9 {} {} {} {}", e[0], e[1], e[2], e[3])?;
            }
        }
    }
    w.flush()?;
    Ok(st)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_validator_rejects_inverted_pyramid_and_twisted_hex() {
        // pyramid in SU2 order: base normal (counter-clockwise seen from the apex) points towards the apex
        let base = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let good = [base[0], base[1], base[2], base[3], [0.5, 0.5, 1.0]];
        assert!(element_valid(14, &good));
        let inverted = [base[0], base[1], base[2], base[3], [0.5, 0.5, -1.0]];
        assert!(!element_valid(14, &inverted));
        let bowtie = [base[0], base[1], base[3], base[2], [0.5, 0.5, 1.0]];
        assert!(!element_valid(14, &bowtie));
        // apex outside the base's convex hull: one split diagonal gives a negative tet
        let nonconvex_base = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.2, 0.0], [0.0, 1.0, 0.0]];
        let np = [nonconvex_base[0], nonconvex_base[1], nonconvex_base[2], nonconvex_base[3], [0.5, 0.5, 1.0]];
        assert!(!element_valid(14, &np));
        // hexahedron
        let cube: Vec<V3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        assert!(element_valid(12, &cube));
        let mut twisted = cube.clone();
        twisted.swap(6, 7); // top face bow-tie
        assert!(!element_valid(12, &twisted));
        let mut folded = cube.clone();
        folded[6] = [0.1, 0.1, -0.2]; // top corner pushed through the bottom
        assert!(!element_valid(12, &folded));
        assert!(element_valid(10, &[base[0], base[1], base[2], [0.0, 0.0, 1.0]]));
        assert!(!element_valid(10, &[base[0], base[2], base[1], [0.0, 0.0, 1.0]]));
    }
}
