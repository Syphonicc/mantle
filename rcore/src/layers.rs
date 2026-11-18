//! Boundary layers between the wall and the offset shell of the hex-core mesher.
//!
//! * Columns: every shell vertex `v` (patch `shell` of the hex core) gets a straight column from its nearest wall point `w`
//!   (recomputed with the BVH) to `v`. Wall points closer than `1e-9 size` are merged into one wall node (convex edges / corners of the
//!   wall: all columns of a fan start at the same node); columns of different wall nodes are checked for crossings.
//! * Node distribution: `n` layers, first height `dy0`, growth `r`; per column `r` is re-solved so that the stack spans exactly `|v - w|`
//!   (`r` clamped to [1, 1.5]; when clamped the first height of that column deviates from `dy0`: `r = 1` gives `|v-w|/n`, `r = 1.5` gives
//!   a first height `|v-w| 0.5 / (1.5^n - 1)`). `n` is constant over the whole mesh.
//! * Cells: one hexahedron per shell quad and layer (`[L0 L1 L2 L3 U0 U1 U2 U3]`, quad normal pointing away from the wall). Merged wall nodes
//!   collapse bottom edges: two opposite collapsed edges give a prism (SU2 13), a collapsed bottom face a pyramid (14, apex on the wall),
//!   anything else (one or three collapsed edges) a general polyhedron (decomposed by the SU2 writer, native polyhedron in OpenFOAM).
//!   Faces with fewer than three distinct nodes are dropped. Above layer 0 all cells are hexahedra.
//! * Patches: the layer-0 bottom faces form patch `wall`; the shell faces of the pillow become internal faces between the pillow hexes and
//!   the top layer. The result is a new `CutMesh` (`patch_name` `wall`, prisms / pyramids registered in `CutMesh::native`).

use crate::bvh::Bvh;
use crate::cutmesh::{Cell, CutMesh, Face, Map, INTERNAL, NONE, ORPHAN, WALL};
use crate::dexel::Tri;
use crate::hexcore::{hex_sj_min, HexCore};
use crate::tree::par_map;
use std::time::Instant;

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
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

#[derive(Clone, Debug)]
pub struct LayerOptions {
    pub n: usize,
    /// nominal first-cell height
    pub dy0: f64,
    pub growth: f64,
    /// wall points within `snap * local shell spacing` are merged into one wall node (0 = exact coincidence only)
    pub snap: f64,
    /// layer-0 cells with a scaled Jacobian below this (or non-positive) get the wall nodes of their most collapsed bottom edge merged;
    /// repeated until no such cell remains (<= 8 rounds). 0 = off
    pub repair_sj: f64,
}

/// First height such that `n` layers with growth `r` span `h`.
pub fn fit_dy0(h: f64, n: usize, r: f64) -> f64 {
    if (r - 1.0).abs() < 1e-12 { h / n as f64 } else { h * (r - 1.0) / (r.powi(n as i32) - 1.0) }
}

/// Heights of the column nodes (`n + 1` values from 0 to exactly `h`) and (growth used, first height, clamped).
fn column_heights(h: f64, n: usize, dy0: f64, _growth_nominal: f64) -> (Vec<f64>, f64, f64, bool) {
    let sum = |r: f64, d: f64| if (r - 1.0).abs() < 1e-12 { d * n as f64 } else { d * (r.powi(n as i32) - 1.0) / (r - 1.0) };
    let (r, d, clamped);
    // solve r with the nominal first height
    if sum(1.0, dy0) >= h {
        r = 1.0;
        d = h / n as f64;
        clamped = sum(1.0, dy0) > h * (1.0 + 1e-12);
    } else if sum(1.5, dy0) <= h {
        r = 1.5;
        d = fit_dy0(h, n, 1.5);
        clamped = sum(1.5, dy0) < h * (1.0 - 1e-12);
    } else {
        let (mut lo, mut hi) = (1.0, 1.5);
        for _ in 0..100 {
            let m = 0.5 * (lo + hi);
            if sum(m, dy0) < h { lo = m } else { hi = m }
        }
        r = 0.5 * (lo + hi);
        d = dy0;
        clamped = false;
    }
    let mut s: Vec<f64> = (0..=n).map(|k| if (r - 1.0).abs() < 1e-12 { d * k as f64 } else { d * (r.powi(k as i32) - 1.0) / (r - 1.0) }).collect();
    let last = s[n];
    for x in s.iter_mut() {
        *x *= h / last; // exact span (removes the bisection residual)
    }
    s[n] = h;
    (s, r, d, clamped)
}

/// Closest distance between segments p1q1 and p2q2 (Ericson); returns (distance, s, t) with the parameters in [0, 1].
fn seg_seg(p1: V3, q1: V3, p2: V3, q2: V3) -> (f64, f64, f64) {
    let (d1, d2, r) = (sub(q1, p1), sub(q2, p2), sub(p1, p2));
    let (a, e, f) = (dot(d1, d1), dot(d2, d2), dot(d2, r));
    let (s, t);
    if a <= 1e-300 && e <= 1e-300 {
        (s, t) = (0.0, 0.0);
    } else if a <= 1e-300 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if e <= 1e-300 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let den = a * e - b * b;
            let mut ss = if den > 1e-300 { ((b * f - c * e) / den).clamp(0.0, 1.0) } else { 0.0 };
            let mut tt = (b * ss + f) / e;
            if tt < 0.0 {
                tt = 0.0;
                ss = (-c / a).clamp(0.0, 1.0);
            } else if tt > 1.0 {
                tt = 1.0;
                ss = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t) = (ss, tt);
        }
    }
    let (c1, c2) = ([0, 1, 2].map(|d| p1[d] + d1[d] * s), [0, 1, 2].map(|d| p2[d] + d2[d] * t));
    (norm(sub(c1, c2)), s, t)
}

#[derive(Clone, Debug, Default)]
pub struct LayerStats {
    pub layers: usize,
    pub dy0: f64,
    pub growth: f64,
    pub snap: f64,
    pub repair_sj: f64,
    pub repair_rounds: usize,
    pub repair_merges: usize,
    pub columns: usize,
    pub hex: usize,
    pub prism: usize,
    pub pyramid: usize,
    pub poly: usize,
    pub wall_faces: usize,
    pub wall_tris: usize,
    pub wall_nodes: usize,
    pub merged_wall_nodes: usize,
    pub shell_quads: usize,
    /// cells dropped because their shell quad is degenerate (two columns coincide on the shell) - must be 0
    pub degenerate_quads: usize,
    pub sj_min: f64,
    pub sj_min_hex: f64,
    pub sj_min_prism: f64,
    pub sj_min_pyramid: f64,
    pub n_below_0p1: usize,
    pub n_below_0p2: usize,
    pub n_nonpositive: usize,
    pub nonpositive_poly: usize,
    pub first_min: f64,
    pub first_median: f64,
    pub first_max: f64,
    pub first_rel_dev_max: f64,
    pub columns_clamped: usize,
    pub growth_min: f64,
    pub growth_max: f64,
    pub height_min: f64,
    pub height_max: f64,
    pub stack_err_max: f64,
    /// shell edges whose two columns (different wall nodes) come closer than 1e-6 |v-w| somewhere
    pub column_crossings: usize,
    pub orient_mismatch: usize,
    pub nonmanifold: usize,
    pub t_layers: f64,
}

impl LayerStats {
    pub fn json(&self) -> String {
        let fin = |x: f64| if x.is_finite() { x } else { f64::NAN };
        let (sj_min_hex, sj_min_prism, sj_min_pyramid) = (fin(self.sj_min_hex), fin(self.sj_min_prism), fin(self.sj_min_pyramid));
        format!(
            "{{\"layers\": {}, \"dy0\": {:e}, \"growth\": {}, \"snap\": {}, \"repair_sj\": {}, \"repair_rounds\": {}, \"repair_merges\": {}, \"columns\": {}, \"layer_cells\": {{\"hex\": {}, \"prism\": {}, \"pyramid\": {}, \"poly\": {}}}, \"wall_faces\": {}, \"wall_tris\": {}, \"wall_nodes\": {}, \"merged_wall_nodes\": {}, \"shell_quads\": {}, \"degenerate_quads\": {}, \"layer_sj_min\": {:e}, \"layer_sj_min_by_type\": {{\"hex\": {:e}, \"prism\": {:e}, \"pyramid\": {:e}}}, \"n_layer_sj_below_0p1\": {}, \"n_layer_sj_below_0p2\": {}, \"n_layer_nonpositive\": {}, \"nonpositive_poly\": {}, \"first_cell_height\": {{\"min\": {:e}, \"median\": {:e}, \"max\": {:e}, \"max_rel_dev_from_dy0\": {:e}}}, \"columns_clamped\": {}, \"growth_used\": [{:.6}, {:.6}], \"column_height\": [{:e}, {:e}], \"stack_err_max\": {:e}, \"column_crossings\": {}, \"orient_mismatch\": {}, \"layer_nonmanifold\": {}, \"t_layers\": {:.3}}}",
            self.layers, self.dy0, self.growth, self.snap, self.repair_sj, self.repair_rounds, self.repair_merges, self.columns, self.hex, self.prism, self.pyramid, self.poly, self.wall_faces, self.wall_tris, self.wall_nodes, self.merged_wall_nodes, self.shell_quads, self.degenerate_quads, self.sj_min, sj_min_hex, sj_min_prism, sj_min_pyramid, self.n_below_0p1, self.n_below_0p2, self.n_nonpositive, self.nonpositive_poly, self.first_min, self.first_median, self.first_max, self.first_rel_dev_max, self.columns_clamped, self.growth_min, self.growth_max, self.height_min, self.height_max, self.stack_err_max, self.column_crossings, self.orient_mismatch, self.nonmanifold, self.t_layers
        )
        .replace("NaN", "null")
    }
}

pub struct LayerMesh {
    pub mesh: CutMesh,
    pub stats: LayerStats,
    /// mesh point ids of the column nodes: `nodes[i * (layers + 1) + k]` for shell vertex `i`, level `k` (0 = wall node, `layers` = shell vertex)
    pub nodes: Vec<u32>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Hex,
    /// gmsh-order prism nodes
    Prism([u32; 6]),
    /// base (normal towards the apex) + apex
    Pyramid([u32; 5]),
    Poly,
}

fn classify(l: [u32; 4], u: [u32; 4]) -> Kind {
    let udist = (0..4).all(|i| (i + 1..4).all(|j| u[i] != u[j]));
    if !udist || (0..4).any(|i| u.contains(&l[i])) {
        return Kind::Poly;
    }
    let ldist = (0..4).all(|i| (i + 1..4).all(|j| l[i] != l[j]));
    if ldist {
        return Kind::Hex;
    }
    if l[0] == l[1] && l[1] == l[2] && l[2] == l[3] {
        return Kind::Pyramid([u[3], u[2], u[1], u[0], l[0]]);
    }
    if l[0] == l[1] && l[2] == l[3] && l[0] != l[2] {
        return Kind::Prism([l[0], u[0], u[1], l[2], u[3], u[2]]);
    }
    if l[1] == l[2] && l[3] == l[0] && l[1] != l[3] {
        return Kind::Prism([l[1], u[1], u[2], l[3], u[0], u[3]]);
    }
    Kind::Poly
}

/// Scaled Jacobian of a gmsh-order prism (min over the 6 corners; the ideal right prism with an equilateral base gives 0.866).
fn prism_sj(g: &[V3; 6]) -> f64 {
    let mut m = f64::INFINITY;
    for i in 0..6 {
        let (b, c) = (3 * (i / 3), i % 3);
        let (j, k, o) = (b + (c + 1) % 3, b + (c + 2) % 3, (i + 3) % 6);
        let (e1, e2, e3) = (sub(g[j], g[i]), sub(g[k], g[i]), sub(g[o], g[i]));
        let l = norm(e1) * norm(e2) * norm(e3);
        let d = dot(cross(e1, e2), e3) * if i < 3 { 1.0 } else { -1.0 };
        m = m.min(if l > 0.0 { d / l } else { -1.0 });
    }
    m
}

/// Scaled Jacobian of a pyramid (base corners, base normal towards the apex) combined with the 4 apex-fan tet checks.
fn pyramid_sj(p: &[V3; 5]) -> f64 {
    let mut m = f64::INFINITY;
    for i in 0..4 {
        let (e1, e2, e3) = (sub(p[(i + 1) % 4], p[i]), sub(p[(i + 3) % 4], p[i]), sub(p[4], p[i]));
        let l = norm(e1) * norm(e2) * norm(e3);
        m = m.min(if l > 0.0 { dot(cross(e1, e2), e3) / l } else { -1.0 });
    }
    m
}

struct FaceBuilder {
    faces: Vec<Face>,
    fmap: Map<[u32; 4], u32>,
    mismatch: usize,
    nonmanifold: usize,
}

impl FaceBuilder {
    /// Add (or complete) a face given as outward vertices of `cell`; returns the face id.
    fn add(&mut self, v: &[u32], cell: u32, patch: u8) -> u32 {
        let mut key = [u32::MAX; 4];
        let mut tmp: Vec<u32> = v.to_vec();
        tmp.sort_unstable();
        if v.len() <= 4 {
            key[..v.len()].copy_from_slice(&tmp);
        } else {
            unreachable!()
        }
        if let Some(&f) = self.fmap.get(&key) {
            let fc = &mut self.faces[f as usize];
            if fc.patch == INTERNAL && fc.neigh == NONE {
                // orientation: the second cell must see the reversed cycle
                let n = v.len();
                let s = fc.verts.iter().position(|&x| x == v[0]);
                let ok = s.is_some_and(|s| (0..n).all(|t| fc.verts[(s + n - t) % n] == v[t]));
                if !ok {
                    self.mismatch += 1;
                }
                fc.neigh = cell;
                return f;
            }
            self.nonmanifold += 1;
            self.faces.push(Face { verts: v.to_vec(), owner: cell, neigh: NONE, patch: ORPHAN });
            return self.faces.len() as u32 - 1;
        }
        self.faces.push(Face { verts: v.to_vec(), owner: cell, neigh: NONE, patch });
        let f = self.faces.len() as u32 - 1;
        self.fmap.insert(key, f);
        f
    }
}

/// Outward faces of the (possibly collapsed) hexahedron `[l, u]` with consecutive duplicate nodes removed; faces with fewer than three
/// distinct nodes are dropped. Entry 0 is the bottom (wall side) face.
fn hex_faces(l: [u32; 4], u: [u32; 4]) -> Vec<(bool, Vec<u32>)> {
    let mut raw: Vec<Vec<u32>> = vec![vec![l[3], l[2], l[1], l[0]], vec![u[0], u[1], u[2], u[3]]];
    for i in 0..4 {
        let j = (i + 1) % 4;
        raw.push(vec![l[i], l[j], u[j], u[i]]);
    }
    raw.into_iter()
        .enumerate()
        .filter_map(|(fi, mut v)| {
            let mut t = 0;
            while t < v.len() && v.len() > 1 {
                if v[t] == v[(t + 1) % v.len()] {
                    v.remove(t);
                } else {
                    t += 1;
                }
            }
            (v.len() >= 3).then_some((fi == 0, v))
        })
        .collect()
}

#[derive(Default, Clone)]
struct Col {
    w: V3,
    tri: u32,
}

/// Build the boundary-layer mesh on top of a hex-core mesh. `hc.shell` must carry the mesh ids (`mesh_pid`, `mesh_face`).
/// Degenerate layer-0 cells are repaired by merging wall nodes (`LayerOptions::repair_sj`), see there.
pub fn build(hc: &HexCore, tris: &[Tri], opt: &LayerOptions) -> LayerMesh {
    let t0 = Instant::now();
    let bvh = Bvh::new(tris);
    let mut forced: Vec<(u32, u32)> = Vec::new();
    let mut best: Option<LayerMesh> = None;
    let (mut rounds, mut stale) = (0, 0);
    loop {
        let (mut lm, pairs) = build_pass(hc, &bvh, opt, &forced);
        lm.stats.repair_rounds = rounds;
        lm.stats.repair_merges = forced.len();
        // keep the best round (fewest invalid cells, then largest minimum Jacobian): repairs may help one place and hurt another
        let better = best.as_ref().is_none_or(|b| (lm.stats.n_nonpositive, -lm.stats.sj_min.max(-1.0)) < (b.stats.n_nonpositive, -b.stats.sj_min.max(-1.0)));
        if better {
            stale = 0;
            best = Some(lm);
        } else {
            stale += 1;
        }
        if pairs.is_empty() || rounds >= 12 || stale >= 3 || opt.repair_sj <= 0.0 || best.as_ref().unwrap().stats.n_nonpositive == 0 && best.as_ref().unwrap().stats.sj_min >= opt.repair_sj && stale > 0 {
            let mut b = best.unwrap();
            b.stats.t_layers = t0.elapsed().as_secs_f64();
            return b;
        }
        forced.extend(pairs);
        rounds += 1;
    }
}

fn build_pass(hc: &HexCore, bvh: &Bvh, opt: &LayerOptions, forced: &[(u32, u32)]) -> (LayerMesh, Vec<(u32, u32)>) {
    let t0 = Instant::now();
    let mut bad_pairs: Vec<(u32, u32)> = Vec::new();
    let (sh, base) = (&hc.shell, &hc.mesh);
    let n = opt.n.max(1);
    let nv = sh.pts.len();
    let nq = sh.quads.len();
    let mut st = LayerStats { layers: n, dy0: opt.dy0, growth: opt.growth, snap: opt.snap, repair_sj: opt.repair_sj, columns: nv, shell_quads: nq, ..Default::default() };
    assert_eq!(sh.mesh_pid.len(), nv, "shell quads carry no mesh ids");
    let cols: Vec<Col> = par_map(&sh.pts, |&p| {
        let (_, w, tri) = bvh.closest(p);
        Col { w, tri }
    });
    let _ = cols.iter().map(|c| c.tri).count();
    // size scale for tolerances
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in &sh.pts {
        for d in 0..3 {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    let size = (0..3).map(|d| hi[d] - lo[d]).fold(0.0f64, f64::max);
    let tol = 1e-9 * size;
    // Wall-node merging. Columns joined by a shell edge (a, b) share a wall node when their wall points are closer than
    // max(1e-9 size, snap * |v_a - v_b|): on a convex edge / corner the nearest points of a whole fan coincide (up to the noise left by
    // the pillow optimiser, which snap absorbs), while in smooth regions the feet are about |v_a - v_b| / (1 + kappa delta) apart.
    // The shared node is the cluster centroid projected back onto the wall (so it stays on the soup). snap = 0: exact coincidence only.
    let mut parent: Vec<u32> = (0..nv as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    {
        let mut seen: Map<(u32, u32), ()> = Map::default();
        for q in &sh.quads {
            for t in 0..4 {
                let (a, b) = (q[t], q[(t + 1) % 4]);
                if seen.insert((a.min(b), a.max(b)), ()).is_some() {
                    continue;
                }
                let (pa, pb) = (a as usize, b as usize);
                let lim = tol.max(opt.snap * norm(sub(sh.pts[pa], sh.pts[pb])));
                if norm(sub(cols[pa].w, cols[pb].w)) <= lim {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    if ra != rb {
                        parent[ra.max(rb) as usize] = ra.min(rb);
                    }
                }
            }
        }
    }
    for &(a, b) in forced {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent[ra.max(rb) as usize] = ra.min(rb);
        }
    }
    // exactly coincident wall points of columns that are not shell neighbours (e.g. a corner reached from two sides): sort sweep
    {
        let mut ord: Vec<u32> = (0..nv as u32).collect();
        ord.sort_by(|&a, &b| cols[a as usize].w[0].partial_cmp(&cols[b as usize].w[0]).unwrap());
        for ia in 0..nv {
            let a = ord[ia] as usize;
            for ib in ia + 1..nv {
                let b = ord[ib] as usize;
                if cols[b].w[0] - cols[a].w[0] > tol {
                    break;
                }
                if norm(sub(cols[a].w, cols[b].w)) <= tol {
                    let (ra, rb) = (find(&mut parent, a as u32), find(&mut parent, b as u32));
                    if ra != rb {
                        parent[ra.max(rb) as usize] = ra.min(rb);
                    }
                }
            }
        }
    }
    let mut wid = vec![u32::MAX; nv];
    let mut wnodes: Vec<V3> = Vec::new();
    let mut root_id: Map<u32, u32> = Map::default();
    let mut members: Vec<Vec<u32>> = Vec::new();
    for i in 0..nv {
        let r = find(&mut parent, i as u32);
        let id = *root_id.entry(r).or_insert_with(|| {
            members.push(Vec::new());
            members.len() as u32 - 1
        });
        members[id as usize].push(i as u32);
        wid[i] = id;
    }
    for m in &members {
        if m.len() == 1 {
            wnodes.push(cols[m[0] as usize].w);
        } else {
            let mut c = [0.0; 3];
            for &i in m {
                for d in 0..3 {
                    c[d] += cols[i as usize].w[d] / m.len() as f64;
                }
            }
            let (_, w, _) = bvh.closest(c);
            wnodes.push(w);
        }
    }
    let mut cols = cols;
    for i in 0..nv {
        cols[i].w = wnodes[wid[i] as usize];
    }
    let _ = &cols;
    st.wall_nodes = wnodes.len();
    st.merged_wall_nodes = nv - wnodes.len();

    // column heights and node coordinates
    let np0 = base.points.len() as u32;
    let nw = wnodes.len() as u32;
    let mut pts: Vec<V3> = base.points.clone();
    pts.extend_from_slice(&wnodes);
    let mut firsts: Vec<f64> = Vec::with_capacity(nv);
    let (mut gmin, mut gmax, mut hmin, mut hmax) = (f64::INFINITY, 0.0f64, f64::INFINITY, 0.0f64);
    let mut dev = 0.0f64;
    for i in 0..nv {
        let (v, w) = (sh.pts[i], cols[i].w);
        let dv = sub(v, w);
        let h = norm(dv);
        let (s, r, d, clamped) = column_heights(h, n, opt.dy0, opt.growth);
        st.columns_clamped += clamped as usize;
        gmin = gmin.min(r);
        gmax = gmax.max(r);
        hmin = hmin.min(h);
        hmax = hmax.max(h);
        dev = dev.max((d - opt.dy0).abs() / opt.dy0);
        for k in 1..n {
            let f = s[k] / h;
            pts.push([0, 1, 2].map(|c| w[c] + dv[c] * f));
        }
        firsts.push(s[1] - s[0]);
    }
    st.growth_min = gmin;
    st.growth_max = gmax;
    st.height_min = hmin;
    st.height_max = hmax;
    st.first_rel_dev_max = dev;
    // measured first-cell heights (wall node to first column node)
    {
        let mut meas: Vec<f64> = (0..nv)
            .map(|i| {
                let p1 = if n > 1 { pts[(np0 + nw) as usize + i * (n - 1)] } else { sh.pts[i] };
                norm(sub(p1, wnodes[wid[i] as usize]))
            })
            .collect();
        meas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        st.first_min = meas[0];
        st.first_median = meas[nv / 2];
        st.first_max = meas[nv - 1];
        let _ = firsts;
    }
    let node = |i: usize, k: usize| -> u32 {
        if k == 0 {
            np0 + wid[i]
        } else if k == n {
            sh.mesh_pid[i]
        } else {
            np0 + nw + (i * (n - 1) + k - 1) as u32
        }
    };
    // column crossing check over unique shell edges
    {
        let mut seen: Map<(u32, u32), ()> = Map::default();
        for q in &sh.quads {
            for t in 0..4 {
                let (a, b) = (q[t], q[(t + 1) % 4]);
                let key = (a.min(b), a.max(b));
                if seen.insert(key, ()).is_some() || wid[a as usize] == wid[b as usize] {
                    continue;
                }
                let (a, b) = (a as usize, b as usize);
                let (d, _, _) = seg_seg(cols[a].w, sh.pts[a], cols[b].w, sh.pts[b]);
                let h = norm(sub(sh.pts[a], cols[a].w)).min(norm(sub(sh.pts[b], cols[b].w)));
                // adjacent columns of different wall nodes start apart; touching somewhere means a crossing
                if d < 1e-6 * h {
                    st.column_crossings += 1;
                }
            }
        }
    }

    // cells
    let nc0 = base.cells.len() as u32;
    let mut fb = FaceBuilder { faces: base.faces.clone(), fmap: Map::default(), mismatch: 0, nonmanifold: 0 };
    let mut cells: Vec<Cell> = base.cells.clone();
    let mut native: Map<u32, (u8, [u32; 8])> = Map::default();
    let mut kinds: Vec<Kind> = Vec::with_capacity(nq * n);
    let (mut sjm, mut sjh, mut sjp, mut sjy) = (f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::INFINITY);
    for (qi, q) in sh.quads.iter().enumerate() {
        let qv = [q[0] as usize, q[1] as usize, q[2] as usize, q[3] as usize];
        if (0..4).any(|i| (i + 1..4).any(|j| qv[i] == qv[j])) {
            st.degenerate_quads += 1;
        }
        for k in 0..n {
            let l: [u32; 4] = std::array::from_fn(|i| node(qv[i], k));
            let u: [u32; 4] = std::array::from_fn(|i| node(qv[i], k + 1));
            let ci = nc0 + (qi * n + k) as u32;
            let kind = classify(l, u);
            let mut fl: Vec<u32> = Vec::with_capacity(6);
            for (bottom, v) in hex_faces(l, u) {
                let top = !bottom && v.len() == 4 && v == vec![u[0], u[1], u[2], u[3]];
                if top && k == n - 1 {
                    let f = sh.mesh_face[qi];
                    let fc = &mut fb.faces[f as usize];
                    fc.patch = INTERNAL;
                    fc.neigh = ci;
                    fl.push(f);
                    // the pillow face is oriented towards the wall; the top of this layer cell is its reverse
                    let rv: Vec<u32> = fc.verts.iter().rev().cloned().collect();
                    let s = rv.iter().position(|&x| x == v[0]);
                    if !s.is_some_and(|s| (0..4).all(|t| rv[(s + t) % 4] == v[t])) {
                        fb.mismatch += 1;
                    }
                    continue;
                }
                let patch = if bottom && k == 0 { WALL } else { INTERNAL };
                if patch == WALL {
                    if v.len() == 3 {
                        st.wall_tris += 1;
                    }
                    st.wall_faces += 1;
                }
                fl.push(fb.add(&v, ci, patch));
            }
            let hh = (sh.pts[qv[0]][0] - sh.pts[qv[1]][0]).abs().max(1e-300);
            let _ = hh;
            let mut hex = None;
            let sj = match kind {
                Kind::Hex => {
                    let ids = [l[0], l[1], l[2], l[3], u[0], u[1], u[2], u[3]];
                    hex = Some(ids);
                    let p: [V3; 8] = ids.map(|i| pts[i as usize]);
                    let s = hex_sj_min(&p);
                    st.hex += 1;
                    sjh = sjh.min(s);
                    s
                }
                Kind::Prism(g) => {
                    let p: [V3; 6] = g.map(|i| pts[i as usize]);
                    let s = prism_sj(&p);
                    native.insert(ci, (13, [g[0], g[2], g[1], g[3], g[5], g[4], 0, 0]));
                    st.prism += 1;
                    sjp = sjp.min(s);
                    s
                }
                Kind::Pyramid(pp) => {
                    let p: [V3; 5] = pp.map(|i| pts[i as usize]);
                    let s = pyramid_sj(&p);
                    native.insert(ci, (14, [pp[0], pp[1], pp[2], pp[3], pp[4], 0, 0, 0]));
                    st.pyramid += 1;
                    sjy = sjy.min(s);
                    s
                }
                Kind::Poly => {
                    st.poly += 1;
                    f64::NAN
                }
            };
            kinds.push(kind);
            if opt.repair_sj > 0.0 && !sj.is_nan() && (sj <= 0.0 || (k == 0 && sj < opt.repair_sj)) {
                repair_pick(k, &qv, &wid, &cols, sh, forced, &mut bad_pairs);
            }
            if !sj.is_nan() {
                sjm = sjm.min(sj);
                st.n_below_0p1 += (sj < 0.1) as usize;
                st.n_below_0p2 += (sj < 0.2) as usize;
                st.n_nonpositive += (sj <= 0.0) as usize;
            }
            let h = sub(sh.pts[qv[0]], cols[qv[0]].w);
            cells.push(Cell { faces: fl, hex, h: (norm(h) / n as f64).max(1e-300), cut: false, n_orig: 1 });
        }
    }
    st.sj_min_hex = sjh;
    st.sj_min_prism = sjp;
    st.sj_min_pyramid = sjy;
    st.orient_mismatch = fb.mismatch;
    st.nonmanifold = fb.nonmanifold;
    let mut mesh = CutMesh { points: pts, faces: fb.faces, cells, stats: Default::default(), patch_name: "wall", native };
    mesh.stats = base.stats.clone();
    mesh.stats.hex_cells = base.stats.hex_cells + st.hex;
    mesh.stats.poly_cells = base.stats.poly_cells + st.prism + st.pyramid + st.poly;
    mesh.stats.orphan_faces = mesh.faces.iter().filter(|f| f.patch == ORPHAN).count();
    mesh.update_volume_stats();
    // polyhedral layer cells are validated by volume
    for (i, kd) in kinds.iter().enumerate() {
        if *kd == Kind::Poly {
            let ci = nc0 as usize + i;
            let vol = mesh.cell_volume(ci);
            if vol <= 0.0 && opt.repair_sj > 0.0 {
                let q = &sh.quads[i / n];
                repair_pick(i % n, &[q[0] as usize, q[1] as usize, q[2] as usize, q[3] as usize], &wid, &cols, sh, forced, &mut bad_pairs);
            }
            if vol <= 0.0 {
                st.nonpositive_poly += 1;
                st.n_nonpositive += 1;
            }
            sjm = sjm.min(if vol <= 0.0 { -1.0 } else { f64::INFINITY });
        }
    }
    st.sj_min = sjm;
    // total stack thickness check: sum of the column node spacings equals |v - w|
    {
        let mut e = 0.0f64;
        for i in 0..nv {
            let mut tot = 0.0;
            let mut prev = wnodes[wid[i] as usize];
            for k in 1..=n {
                let p = mesh.points[node(i, k) as usize];
                tot += norm(sub(p, prev));
                prev = p;
            }
            let h = norm(sub(sh.pts[i], cols[i].w));
            e = e.max((tot - h).abs() / h);
        }
        st.stack_err_max = e;
    }
    st.t_layers = t0.elapsed().as_secs_f64();
    bad_pairs.sort_unstable();
    bad_pairs.dedup();
    let nodes: Vec<u32> = (0..nv).flat_map(|i| (0..=n).map(move |k| (i, k))).map(|(i, k)| node(i, k)).collect();
    (LayerMesh { mesh, stats: st, nodes }, bad_pairs)
}

/// Repair choice for a bad layer cell. A layer-0 cell first gets the wall nodes of its collapsed-looking bottom edges (foot distance
/// < 0.8 shell edge) merged, else its most collapsed one (each pair once). A bad cell that cannot be helped that way (or sits above
/// layer 0, i.e. columns twisted between levels) gets all four wall nodes merged: the stack of that quad becomes a pyramid frustum whose
/// cross-sections are homothetic copies of the shell quad, which is valid whenever the quad is (the apex is the projected foot centroid).
fn repair_pick(k: usize, qv: &[usize; 4], wid: &[u32], cols: &[Col], sh: &crate::hexcore::ShellQuads, forced: &[(u32, u32)], pairs: &mut Vec<(u32, u32)>) {
    let before = pairs.len();
    if k == 0 {
        for t in 0..4 {
            let (a, b) = (qv[t], qv[(t + 1) % 4]);
            if wid[a] == wid[b] {
                continue;
            }
            let r = norm(sub(cols[a].w, cols[b].w)) / norm(sub(sh.pts[a], sh.pts[b])).max(1e-300);
            let p = (a.min(b) as u32, a.max(b) as u32);
            if r < 0.8 && !forced.contains(&p) {
                pairs.push(p);
            }
        }
        if pairs.len() == before {
            if let Some(p) = best_pair(qv, wid, cols, sh) {
                if !forced.contains(&p) {
                    pairs.push(p);
                }
            }
        }
    }
    if pairs.len() == before {
        // only for a local fan: feet spread over at most 1.2 shell edges (concave creases have feet much further apart; there the
        // apex would sit far below the quad and the pyramid frustum is not valid)
        let (mut spread, mut len) = (0.0f64, 0.0f64);
        for i in 0..4 {
            len = len.max(norm(sub(sh.pts[qv[i]], sh.pts[qv[(i + 1) % 4]])));
            for j in i + 1..4 {
                spread = spread.max(norm(sub(cols[qv[i]].w, cols[qv[j]].w)));
            }
        }
        if spread > 1.2 * len {
            return;
        }
        for t in 0..4 {
            let (a, b) = (qv[t], qv[(t + 1) % 4]);
            let p = (a.min(b) as u32, a.max(b) as u32);
            if wid[a] != wid[b] && !forced.contains(&p) {
                pairs.push(p);
            }
        }
    }
}

/// Shell edge of the quad whose two columns have different wall nodes and the smallest foot distance relative to the shell edge length.
fn best_pair(qv: &[usize; 4], wid: &[u32], cols: &[Col], sh: &crate::hexcore::ShellQuads) -> Option<(u32, u32)> {
    let mut best: Option<(f64, (u32, u32))> = None;
    for t in 0..4 {
        let (a, b) = (qv[t], qv[(t + 1) % 4]);
        if wid[a] == wid[b] {
            continue;
        }
        let r = norm(sub(cols[a].w, cols[b].w)) / norm(sub(sh.pts[a], sh.pts[b])).max(1e-300);
        if best.is_none_or(|x| r < x.0) {
            best = Some((r, (a.min(b) as u32, a.max(b) as u32)));
        }
    }
    best.map(|x| x.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cutmesh::Map;
    use crate::hexcore::{self, Options};
    use crate::octree::{Clip, Octree, SignMode, SizeField};
    use std::f64::consts::PI;

    fn sphere(c: V3, r: f64) -> Vec<Tri> {
        let (nu, nv) = (96usize, 48usize);
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

    fn cube(lo: f64, hi: f64) -> Vec<Tri> {
        let p = |i: usize, j: usize, k: usize| [if i == 1 { hi } else { lo }, if j == 1 { hi } else { lo }, if k == 1 { hi } else { lo }];
        let quads = [
            [p(0, 0, 0), p(0, 1, 0), p(0, 1, 1), p(0, 0, 1)],
            [p(1, 0, 0), p(1, 0, 1), p(1, 1, 1), p(1, 1, 0)],
            [p(0, 0, 0), p(0, 0, 1), p(1, 0, 1), p(1, 0, 0)],
            [p(0, 1, 0), p(1, 1, 0), p(1, 1, 1), p(0, 1, 1)],
            [p(0, 0, 0), p(1, 0, 0), p(1, 1, 0), p(0, 1, 0)],
            [p(0, 0, 1), p(0, 1, 1), p(1, 1, 1), p(1, 0, 1)],
        ];
        quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).collect()
    }

    fn check_closed(m: &CutMesh) {
        assert_eq!((m.stats.orphan_faces, m.stats.nonmanifold_faces, m.stats.n_nonpositive_cells), (0, 0, 0));
        for f in &m.faces {
            assert_eq!(f.neigh == NONE, f.patch != INTERNAL, "internal face without neighbour / boundary face with one");
        }
        for ci in 0..m.cells.len() {
            let mut e: Map<(u32, u32), i32> = Map::default();
            for p in m.cell_polys(ci) {
                for t in 0..p.len() {
                    let (a, b) = (p[t], p[(t + 1) % p.len()]);
                    *e.entry((a.min(b), a.max(b))).or_insert(0) += if a < b { 1 } else { -1 };
                }
            }
            assert!(e.values().all(|&v| v == 0), "cell {ci} not closed");
        }
    }

    fn core(tris: &[Tri], org: f64, size: f64, l: u8) -> (HexCore, f64) {
        let hs = size / (1u64 << l) as f64;
        let delta = 4.0 * hs;
        let oc = Octree::build_clipped(tris, [org; 3], size, l, &SizeField::new(hs, 0.3, size / 8.0), 0, SignMode::Dexel, Clip::Shell { delta });
        (hexcore::build(&oc, tris, &Options { delta, ..Options::default() }), delta)
    }

    #[test]
    fn sphere_layers() {
        let tris = sphere([0.0, 0.0, 0.0], 0.6);
        let (hc, delta) = core(&tris, -1.5, 3.2, 6);
        let (nl, r) = (8usize, 1.2);
        let dy0 = fit_dy0(delta, nl, r);
        let lm = build(&hc, &tris, &LayerOptions { n: nl, dy0, growth: r, snap: 0.4, repair_sj: 0.05 });
        let (s, m) = (&lm.stats, &lm.mesh);
        println!("layers: {}", s.json());
        assert_eq!(s.n_nonpositive, 0);
        assert!(s.sj_min > 0.0);
        // the faceted soup has (tiny) convex edges: a few wall nodes merge, hence a few prisms / polyhedra and fewer wall faces
        assert_eq!(s.hex + s.prism + s.pyramid + s.poly, hc.shell.quads.len() * nl);
        assert!(s.hex as f64 > 0.99 * (hc.shell.quads.len() * nl) as f64);
        assert!(s.wall_faces <= hc.shell.quads.len() && s.wall_faces as f64 > 0.99 * hc.shell.quads.len() as f64);
        assert_eq!((s.orient_mismatch, s.nonmanifold, s.column_crossings, s.degenerate_quads), (0, 0, 0, 0));
        // columns whose wall node was not merged: first cell within 1 % of dy0, stack spans |v - w| = delta
        let bvh = Bvh::new(&tris);
        let (mut unmerged, mut bad_first) = (0usize, 0usize);
        for i in 0..hc.shell.pts.len() {
            let nd = &lm.nodes[i * (nl + 1)..(i + 1) * (nl + 1)];
            let (foot, top) = (m.points[nd[0] as usize], hc.shell.pts[i]);
            assert_eq!(nd[nl], hc.shell.mesh_pid[i]);
            let w = bvh.closest(top).1;
            if norm(sub(foot, w)) > 1e-9 {
                continue;
            }
            unmerged += 1;
            let tot: f64 = (0..nl).map(|k| norm(sub(m.points[nd[k + 1] as usize], m.points[nd[k] as usize]))).sum();
            assert!((tot - delta).abs() < 1e-9 * delta && (norm(sub(top, foot)) - delta).abs() < 1e-9, "stack thickness");
            let f = norm(sub(m.points[nd[1] as usize], foot));
            bad_first += ((f - dy0).abs() > 0.01 * dy0) as usize;
        }
        assert!(unmerged as f64 > 0.97 * hc.shell.pts.len() as f64, "unmerged {unmerged}");
        assert_eq!(bad_first, 0, "first cell height off by more than 1 %");
        assert!(s.stack_err_max < 1e-12, "stack err {}", s.stack_err_max);
        check_closed(m);
        let nwall = m.faces.iter().filter(|f| f.patch == WALL).count();
        assert_eq!(nwall, s.wall_faces);
        assert!(nwall as f64 > 0.99 * hc.shell.quads.len() as f64);
        for ci in hc.mesh.cells.len()..m.cells.len() {
            assert!(m.cell_volume(ci) > 0.0);
        }
        // volume grows by the band between wall and shell
        let band = 4.0 / 3.0 * PI * ((0.6 + delta).powi(3) - 0.6f64.powi(3));
        let added = m.stats.volume - hc.mesh.stats.volume;
        assert!((added - band).abs() < 0.1 * band, "band {added} vs {band}");
        let dir = std::env::temp_dir().join(format!("rcore_layers_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s2 = crate::su2::write_su2(m, &dir.join("mesh.su2")).unwrap();
        println!("su2: {:?}", s2);
        assert!(s2.nonpositive <= 2, "su2 invalid elements {}", s2.nonpositive);
        assert!(s2.n_wall_elems >= s.wall_faces);
        let (_, patches) = crate::foam::write_polymesh(m, &dir.join("polyMesh")).unwrap();
        assert_eq!(patches[0].0, "wall");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cube_layers() {
        let tris = cube(-0.5, 0.5);
        let (hc, delta) = core(&tris, -1.0, 2.0, 5);
        let (nl, r) = (6usize, 1.2);
        let dy0 = fit_dy0(delta, nl, r);
        let lm = build(&hc, &tris, &LayerOptions { n: nl, dy0, growth: r, snap: 0.4, repair_sj: 0.05 });
        let (s, m) = (&lm.stats, &lm.mesh);
        println!("layers cube: {}", s.json());
        assert!(s.prism > 0 && s.pyramid > 0, "no wedges / pyramids at the cube edges / corners");
        assert!(s.merged_wall_nodes > 0);
        assert_eq!(s.n_nonpositive, 0, "invalid layer cells");
        assert_eq!((s.orient_mismatch, s.nonmanifold), (0, 0));
        check_closed(m);
        for ci in hc.mesh.cells.len()..m.cells.len() {
            assert!(m.cell_volume(ci) > 0.0);
        }
        let dir = std::env::temp_dir().join(format!("rcore_layers_cube_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s2 = crate::su2::write_su2(m, &dir.join("mesh.su2")).unwrap();
        assert_eq!(s2.nonpositive, 0);
        assert!(s2.n_prism > 0 && s2.n_pyramid > 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn column_heights_span_and_clamp() {
        let (s, r, d, c) = column_heights(1.0, 10, fit_dy0(1.0, 10, 1.25), 1.25);
        assert!((r - 1.25).abs() < 1e-9 && !c && (s[10] - 1.0).abs() < 1e-15 && (s[1] - d).abs() < 1e-12);
        let (s, r, _, c) = column_heights(1.0, 10, 0.5, 1.25); // far too thick: uniform
        assert!(r == 1.0 && c && (s[1] - 0.1).abs() < 1e-12);
        let (_, r, d, c) = column_heights(1.0, 3, 1e-4, 1.25); // too thin: r = 1.5, first height raised
        assert!(r == 1.5 && c && (d - fit_dy0(1.0, 3, 1.5)).abs() < 1e-12);
    }

    #[test]
    fn prism_pyramid_orientation() {
        // hex with two collapsed opposite bottom edges: prism with positive volume in both conventions
        let l = [0, 0, 1, 1];
        let u = [2, 3, 4, 5];
        let Kind::Prism(g) = classify(l, u) else { panic!() };
        let p: Vec<V3> = vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, 0.0, 1.0], [0.5, 0.0, 1.0], [0.5, 1.0, 1.0], [-0.5, 1.0, 1.0]];
        let gp: [V3; 6] = g.map(|i| p[i as usize]);
        assert!(prism_sj(&gp) > 0.3);
        let vtk = [g[0], g[2], g[1], g[3], g[5], g[4]].map(|i| p[i as usize]);
        assert!(crate::su2::element_valid(13, &vtk));
        let bad = [vtk[0], vtk[2], vtk[1], vtk[3], vtk[5], vtk[4]];
        assert!(!crate::su2::element_valid(13, &bad));
        assert!(matches!(classify([0, 0, 0, 0], [1, 2, 3, 4]), Kind::Pyramid(_)));
        assert!(matches!(classify([0, 0, 1, 2], [3, 4, 5, 6]), Kind::Poly));
    }
}
