//! Projection + pillow hex-core mesher for the shell-first pipeline (replaces cut cells at the offset shell).
//!
//! * Core = octree leaves whose 8 corners all satisfy `psi > gamma * h_leaf` (psi = signed distance to the soup minus delta, fluid
//!   side positive; the octree must be built with `Clip::Shell { delta }` and the conformity buffer).
//! * The core boundary is cleaned until it is a 2-manifold quad surface of unsplit, conforming quads: core cells with a boundary
//!   quad that has a hanging vertex on an edge, or whose face towards a finer neighbour is only partly core (split boundary face), are
//!   removed; so are the cells of non-manifold boundary edges (more than two quads; the owner with the smallest margin) and of
//!   non-manifold boundary vertices (boundary quads around the vertex form several fans; all fans but the largest).
//! * Normal cleanup: every boundary vertex `v` has the projection target `t = w + delta (v - w)/|v - w|` (`w` = nearest wall
//!   point, i.e. the exact closest point of the shell). If the angle between `t - v` and the area-weighted vertex normal exceeds
//!   `theta_max`, or `|t - v|` exceeds `max_proj_h` cell edges, all leaves touching `v` are removed. Boundary quads whose pillow
//!   hex has a corner scaled Jacobian <= `sj_remove` remove their owner. Manifold cleanup and normal cleanup alternate (<= `max_rounds`).
//! * Pillow layer: one hexahedron (a, b, c, d, a', b', c', d') per boundary quad with `v' = t`.
//! * Smoothing: tangential Laplacian of the projected vertices followed by re-projection onto the shell, a move is rejected when it
//!   lowers the minimum corner scaled Jacobian of the incident pillow hexes; optional move of the core-boundary vertices a fraction
//!   towards their targets (same acceptance test on the pillow hexes only).
//! * Assembly as a `CutMesh` (core hexes; cells with 2:1 split faces are polyhedra with hanging vertices inserted into the
//!   face polygons as in `cutmesh`), patches `shell` (= the `WALL` patch: the projected quads) and `farfield`, so `foam` / `su2` are
//!   reused unchanged. `ShellQuads` / `write_shell_quads` export the projected quad surface for the layer step.

use crate::bvh::Bvh;
use crate::cutmesh::{geom, ring, Cell, CutMesh, Face, Map, FARFIELD, INTERNAL, NONE, ORPHAN, WALL};
use crate::dexel::Tri;
use crate::octree::{vertex_key, Octree};
use crate::tree::par_map;
use std::thread;
use std::time::Instant;

type V3 = [f64; 3];
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
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

/// hexahedron corner neighbours in VTK order (as in `su2`)
const NB: [[usize; 3]; 8] = [[1, 3, 4], [2, 0, 5], [3, 1, 6], [0, 2, 7], [7, 5, 0], [4, 6, 1], [5, 7, 2], [6, 4, 3]];

/// Minimum over the 8 corners of the scaled Jacobian det[e1 e2 e3] / (|e1||e2||e3|) of a VTK-ordered hexahedron.
pub fn hex_sj_min(p: &[V3; 8]) -> f64 {
    let mut m = f64::INFINITY;
    for i in 0..8 {
        let e = [sub(p[NB[i][0]], p[i]), sub(p[NB[i][1]], p[i]), sub(p[NB[i][2]], p[i])];
        let l = norm(e[0]) * norm(e[1]) * norm(e[2]);
        m = m.min(if l > 0.0 { dot(cross(e[0], e[1]), e[2]) / l } else { -1.0 });
    }
    m
}

/// Histogram bins of the minimum scaled Jacobian: < 0, [0, 0.2), [0.2, 0.5), [0.5, 0.8), >= 0.8.
fn sj_bin(s: f64) -> usize {
    if s < 0.0 {
        0
    } else if s < 0.2 {
        1
    } else if s < 0.5 {
        2
    } else if s < 0.8 {
        3
    } else {
        4
    }
}

pub struct Options {
    /// offset-shell distance (must equal the octree's `Clip::Shell` delta)
    pub delta: f64,
    /// core margin: corners need `psi > gamma * h_leaf`
    pub gamma: f64,
    /// maximum angle (degrees) between the projection direction and the boundary vertex normal
    pub theta_max: f64,
    /// maximum projection distance in leaf edges
    pub max_proj_h: f64,
    pub smooth_iters: usize,
    /// fraction by which core-boundary vertices are moved towards their targets (0 = off)
    pub core_move: f64,
    pub max_rounds: usize,
    /// pillow hexes with min corner scaled Jacobian <= this remove their owner cell
    pub sj_remove: f64,
    /// cap on cells removed by the normal / Jacobian criteria (fraction of the initial core); removal of cells in a concave
    /// crease only moves the front back (and into the coarse graded region, where hanging nodes cascade), so it is bounded
    pub max_removed_frac: f64,
    /// remove all core leaves touching a bad vertex (else only the owners of its boundary quads)
    pub remove_all_incident: bool,
    /// max Gauss-Seidel sweeps of the local pillow optimiser (0 = off)
    pub opt_sweeps: usize,
    /// optimiser may move core-boundary vertices (within 0.5 h of the lattice)
    pub opt_core: bool,
    /// optimiser may also move one ring of core-interior vertices
    pub opt_ring: bool,
    /// try 0.25 / 0.5 fractions towards the targets as initial guess of core-boundary vertices
    pub opt_init: bool,
    /// objective cap: vertices whose incident hexes all have SJ >= this are left alone and the min is not pushed beyond it
    pub opt_cap: f64,
    /// pattern-search stops when the step falls below this fraction of h
    pub opt_min_step: f64,
}

impl Default for Options {
    fn default() -> Options {
        Options { delta: 0.0, gamma: 0.3, theta_max: 60.0, max_proj_h: 2.5, smooth_iters: 5, core_move: 0.0, max_rounds: 5, sj_remove: 0.0, max_removed_frac: 0.03, remove_all_incident: false, opt_sweeps: 30, opt_core: true, opt_ring: false, opt_init: true, opt_cap: 0.5, opt_min_step: 0.02 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HcStats {
    pub leaves: usize,
    /// leaves passing the margin criterion before any cleanup
    pub core_initial: usize,
    /// leaves with all corners psi > 0 that fail the gamma margin
    pub removed_margin: usize,
    /// leaves with at least one corner not psi > 0 (solid, band, shell side)
    pub removed_nonfluid: usize,
    pub removed_nonmanifold: usize,
    /// boundary quads with a hanging vertex on an edge (owner removed)
    pub removed_hanging: usize,
    /// cells whose face towards a finer neighbour is only partly core (split boundary face)
    pub removed_split: usize,
    pub removed_normal: usize,
    /// margin-removed fluid leaves added back to the core (repair of pinched boundary vertices)
    pub added_fill: usize,
    pub removed_jacobian: usize,
    pub rounds: usize,
    pub manifold_iters: usize,
    pub core_cells: usize,
    pub core_boundary_quads: usize,
    pub core_boundary_verts: usize,
    /// boundary edges with exactly one quad (shell touching the domain boundary), final
    pub open_edges: usize,
    pub nonmanifold_edges_final: usize,
    pub nonmanifold_verts_final: usize,
    /// vertices violating the normal / distance test and quads with non-positive pillow Jacobian after the last round
    pub normal_violations_final: usize,
    pub jacobian_violations_final: usize,
    pub pillow_hexes: usize,
    /// core faces split 2:1 towards a finer core neighbour
    pub core_split_faces: usize,
    pub core_hex: usize,
    pub core_poly: usize,
    pub sj_min_before: f64,
    pub sj_min_after: f64,
    pub sj_mean: f64,
    pub n_sj_below_0p2: usize,
    pub n_sj_below_0: usize,
    pub n_sj_below_0p1: usize,
    pub n_sj_below_0p3: usize,
    pub sj_hist_before: [usize; 5],
    pub sj_hist_after: [usize; 5],
    pub max_proj_dist_over_h: f64,
    pub smooth_accepted: usize,
    pub smooth_rejected: usize,
    /// smoothing steps that moved a vertex to its vertex-normal ray target (repair candidate)
    pub smooth_ray_moves: usize,
    /// boundary vertices whose target is the vertex-normal ray instead of the closest point
    pub ray_targets: usize,
    pub post_rounds: usize,
    /// a repair step stopped: no trial (remove / fill) reduced the number of bad vertices / hexes, or the change budget was exceeded
    pub budget_hit: bool,
    post_budget_hit: bool,
    /// vertices moved off the shell (along the vertex normal, towards the core) by the last-resort Jacobian repair
    pub unprojected: usize,
    pub core_move_rejected: usize,
    pub stage_initial: SjStage,
    pub stage_smooth: SjStage,
    pub stage_opt: SjStage,
    pub opt_sweeps: usize,
    pub moved_core_vertices: usize,
    pub core_sj_min: f64,
    pub t_opt: f64,
    pub t_core: f64,
    pub t_manifold: f64,
    pub t_normal: f64,
    pub t_pillow: f64,
    pub t_smooth: f64,
    pub t_assemble: f64,
}

/// Projected shell quads for the layer step (points on the shell, shell normals pointing away from the wall, nearest wall
/// points and triangles); quads are oriented with normals pointing away from the wall.
pub struct ShellQuads {
    pub pts: Vec<V3>,
    pub nrm: Vec<V3>,
    pub wall: Vec<V3>,
    pub tri: Vec<u32>,
    pub quads: Vec<[u32; 4]>,
    /// mesh point id of every shell vertex / mesh face id (patch `shell`) of every quad (filled by the assembly; for the layer step)
    pub mesh_pid: Vec<u32>,
    pub mesh_face: Vec<u32>,
}

pub struct HexCore {
    pub mesh: CutMesh,
    pub shell: ShellQuads,
    pub stats: HcStats,
}

#[derive(Clone, Copy, Default)]
struct Tgt {
    t: V3,
    w: V3,
    tri: u32,
}

/// Exact closest point of the offset shell for a point outside the band: `t = w + delta (x - w)/|x - w|`.
fn project(bvh: &Bvh, x: V3, delta: f64) -> Tgt {
    let (d, w, tri) = bvh.closest(x);
    let t = if d > 1e-300 { add(w, scale(sub(x, w), delta / d)) } else { x };
    Tgt { t, w, tri }
}

/// Boundary quad of the core: owner leaf, lattice corners in right-hand order about the outward (out of the core) normal.
#[derive(Clone, Copy)]
struct BQ {
    leaf: u32,
    key: [u64; 4],
    h: f64,
}

type Out = (Vec<BQ>, Vec<u64>, Vec<V3>, Vec<V3>, Vec<[u32; 4]>, usize);

struct Scan {
    bq: Vec<BQ>,
    hang: Vec<u32>,
    split: Vec<u32>,
}

#[derive(Default)]
struct ManRes {
    rem: Vec<u32>,
    open_edges: usize,
    nm_edges: usize,
    nm_verts: usize,
}

#[derive(Default)]
struct GeomRes {
    /// sorted boundary vertex keys with chosen targets and area-weighted vertex normals
    keys: Vec<u64>,
    tg: Vec<Tgt>,
    nrm: Vec<V3>,
    normal_rem: Vec<u32>,
    /// non-core fluid leaves (all corners psi > 0) touching bad vertices: filling them is the dual repair
    fill: Vec<u32>,
    bad_vertices: usize,
    ray_targets: usize,
    max_proj_h: f64,
    hmax: Vec<f64>,
}

/// A closest-point target is only accepted if its direction makes more than this cosine with every incident quad normal (else the
/// pillow hex would be folded at that corner).
const FACE_COS_MIN: f64 = 0.0;

/// First intersection of the ray `v + s dir` with the shell (sphere tracing on psi = d - delta), `s <= smax`.
fn ray_target(bvh: &Bvh, v: V3, dir: V3, delta: f64, smax: f64) -> Option<Tgt> {
    let mut s = 0.0;
    for _ in 0..200 {
        let x = add(v, scale(dir, s));
        let (d, w, tri) = bvh.closest(x);
        let g = d - delta;
        if g.abs() < 1e-12 {
            return Some(Tgt { t: x, w, tri });
        }
        s += g;
        if !(0.0..=smax).contains(&s) {
            return None;
        }
    }
    None
}

fn par_ranges<R: Send>(n: usize, f: impl Fn(std::ops::Range<usize>) -> R + Sync) -> Vec<R> {
    let nt = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let per = n.div_ceil(nt * 4).max(1);
    thread::scope(|sc| {
        let hs: Vec<_> = (0..n)
            .step_by(per)
            .map(|s| {
                let f = &f;
                sc.spawn(move || f(s..(s + per).min(n)))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

struct Work<'a> {
    oc: &'a Octree,
    bvh: &'a Bvh,
    delta: f64,
    lv: Vec<[u32; 8]>,
    core: Vec<bool>,
    /// number of core leaves touching each octree vertex
    vcount: Vec<u32>,
    /// min over corners of psi / h per leaf
    cmin: Vec<f64>,
    cache: Map<u64, Tgt>,
}

impl Work<'_> {
    fn remove(&mut self, li: u32) -> bool {
        let li = li as usize;
        if !self.core[li] {
            return false;
        }
        self.core[li] = false;
        for &v in &self.lv[li] {
            self.vcount[v as usize] -= 1;
        }
        true
    }

    /// lattice midpoint of the edge `p q` if it is a vertex of a current core leaf (hanging node)
    fn mid(&self, p: [u32; 3], q: [u32; 3]) -> Option<[u32; 3]> {
        if (0..3).any(|d| (p[d] + q[d]) % 2 != 0) {
            return None;
        }
        let m = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2];
        match self.oc.vertices().binary_search(&vertex_key(m[0], m[1], m[2])) {
            Ok(i) if self.vcount[i] > 0 => Some(m),
            _ => None,
        }
    }

    fn neighbour(&self, l: &crate::octree::Leaf, a: usize, s: usize) -> Option<usize> {
        let mut c = [l.i as i64, l.j as i64, l.k as i64];
        c[a] += if s == 1 { 1 } else { -1 };
        self.oc.leaf_containing(l.level, c[0], c[1], c[2])
    }

    /// Boundary quads of the core (faces towards non-core leaves) plus leaves to remove: boundary quads with a hanging vertex
    /// and leaves whose face towards finer neighbours is partly non-core.
    fn scan(&self) -> Scan {
        let oc = self.oc;
        let (leaves, ml) = (oc.leaves(), oc.max_level);
        let parts = par_ranges(leaves.len(), |rg| {
            let (mut bq, mut hang, mut split) = (Vec::new(), Vec::new(), Vec::new());
            for li in rg {
                if !self.core[li] {
                    continue;
                }
                let l = leaves[li];
                let (base, e) = geom(&l, ml);
                let coord = |n: usize| [base[0] + e * (n as u32 & 1), base[1] + e * ((n as u32 >> 1) & 1), base[2] + e * ((n as u32 >> 2) & 1)];
                for a in 0..3 {
                    for s in 0..2 {
                        let Some(nb) = self.neighbour(&l, a, s) else { continue };
                        if leaves[nb].level > l.level {
                            let (u, v) = ((a + 1) % 3, (a + 2) % 3);
                            let lc = [l.i as i64, l.j as i64, l.k as i64];
                            let mut bad = false;
                            for dv in 0..2i64 {
                                for du in 0..2i64 {
                                    let mut f = [2 * lc[0], 2 * lc[1], 2 * lc[2]];
                                    f[a] = if s == 1 { 2 * lc[a] + 2 } else { 2 * lc[a] - 1 };
                                    f[u] += du;
                                    f[v] += dv;
                                    match oc.leaf_containing(l.level + 1, f[0], f[1], f[2]) {
                                        Some(x) if self.core[x] => {}
                                        _ => bad = true,
                                    }
                                }
                            }
                            if bad {
                                split.push(li as u32);
                            }
                        } else if !self.core[nb] {
                            let r = ring(a, s);
                            let lat: [[u32; 3]; 4] = std::array::from_fn(|t| coord(r[t]));
                            if (0..4).any(|t| self.mid(lat[t], lat[(t + 1) % 4]).is_some()) {
                                hang.push(li as u32);
                            }
                            let key = lat.map(|c| vertex_key(c[0], c[1], c[2]));
                            bq.push(BQ { leaf: li as u32, key, h: oc.leaf_size(&l) });
                        }
                    }
                }
            }
            (bq, hang, split)
        });
        let mut sc = Scan { bq: Vec::new(), hang: Vec::new(), split: Vec::new() };
        for (b, h, s) in parts {
            sc.bq.extend(b);
            sc.hang.extend(h);
            sc.split.extend(s);
        }
        sc
    }

    /// Non-manifold boundary edges / vertices of the quad set and the leaves to remove for them.
    fn manifold_scan(&self, bq: &[BQ]) -> ManRes {
        let mut r = ManRes::default();
        let mut ev: Vec<(u64, u64, u32)> = Vec::with_capacity(bq.len() * 4);
        let mut vv: Vec<(u64, u32, u64, u64)> = Vec::with_capacity(bq.len() * 4);
        for (qi, q) in bq.iter().enumerate() {
            for t in 0..4 {
                let (a, b) = (q.key[t], q.key[(t + 1) % 4]);
                ev.push((a.min(b), a.max(b), qi as u32));
                vv.push((a, qi as u32, q.key[(t + 3) % 4], b));
            }
        }
        ev.sort_unstable();
        vv.sort_unstable();
        let mut i = 0;
        while i < ev.len() {
            let mut j = i;
            while j < ev.len() && ev[j].0 == ev[i].0 && ev[j].1 == ev[i].1 {
                j += 1;
            }
            match j - i {
                1 => r.open_edges += 1,
                2 => {}
                _ => {
                    r.nm_edges += 1;
                    let best = (i..j).map(|k| bq[ev[k].2 as usize].leaf).min_by(|&a, &b| self.cmin[a as usize].partial_cmp(&self.cmin[b as usize]).unwrap()).unwrap();
                    r.rem.push(best);
                }
            }
            i = j;
        }
        let mut i = 0;
        while i < vv.len() {
            let mut j = i;
            while j < vv.len() && vv[j].0 == vv[i].0 {
                j += 1;
            }
            let g = &vv[i..j];
            let n = g.len();
            let mut par: Vec<usize> = (0..n).collect();
            fn find(p: &mut Vec<usize>, mut x: usize) -> usize {
                while p[x] != x {
                    p[x] = p[p[x]];
                    x = p[x];
                }
                x
            }
            for x in 0..n {
                for y in x + 1..n {
                    if g[x].2 == g[y].2 || g[x].2 == g[y].3 || g[x].3 == g[y].2 || g[x].3 == g[y].3 {
                        let (rx, ry) = (find(&mut par, x), find(&mut par, y));
                        par[rx] = ry;
                    }
                }
            }
            let roots: Vec<usize> = (0..n).map(|x| find(&mut par, x)).collect();
            let mut sizes: Map<usize, usize> = Map::default();
            for &rt in &roots {
                *sizes.entry(rt).or_insert(0) += 1;
            }
            if sizes.len() > 1 {
                r.nm_verts += 1;
                let big = *sizes.iter().max_by_key(|(k, c)| (**c, std::cmp::Reverse(**k))).unwrap().0;
                for x in 0..n {
                    if roots[x] != big {
                        r.rem.push(bq[g[x].1 as usize].leaf);
                    }
                }
            }
            i = j;
        }
        r.rem.sort_unstable();
        r.rem.dedup();
        r
    }

    /// Repeat hanging / split / non-manifold removal until none remains (capped); returns the final boundary quads.
    fn cleanup_manifold(&mut self, st: &mut HcStats) -> Vec<BQ> {
        for _ in 0..200 {
            let sc = self.scan();
            if !sc.hang.is_empty() || !sc.split.is_empty() {
                for &l in &sc.hang {
                    st.removed_hanging += self.remove(l) as usize;
                }
                for &l in &sc.split {
                    st.removed_split += self.remove(l) as usize;
                }
                st.manifold_iters += 1;
                continue;
            }
            let m = self.manifold_scan(&sc.bq);
            if m.rem.is_empty() {
                return sc.bq;
            }
            for &l in &m.rem {
                st.removed_nonmanifold += self.remove(l) as usize;
            }
            st.manifold_iters += 1;
        }
        self.scan().bq
    }

    fn add(&mut self, li: u32) -> bool {
        let li = li as usize;
        if self.core[li] {
            return false;
        }
        self.core[li] = true;
        for &v in &self.lv[li] {
            self.vcount[v as usize] += 1;
        }
        true
    }

    /// All leaves touching the lattice vertex `key` (found through the 8 finest cells around it) whose core flag equals `core`.
    fn leaves_at_vertex(&self, key: u64, core: bool, out: &mut Vec<u32>) {
        let c = crate::octree::vertex_coords(key);
        let ml = self.oc.max_level;
        for n in 0..8i64 {
            let (i, j, k) = (c[0] as i64 - 1 + (n & 1), c[1] as i64 - 1 + ((n >> 1) & 1), c[2] as i64 - 1 + ((n >> 2) & 1));
            if let Some(x) = self.oc.leaf_containing(ml, i, j, k) {
                if self.core[x] == core {
                    out.push(x as u32);
                }
            }
        }
    }

    /// Apply one repair round for the bad vertices of `g` (mode 0: remove cells, 1: fill pinched leaves), then clean up.
    fn apply_repair(&mut self, st: &mut HcStats, g: &GeomRes, mode: u8) -> Vec<BQ> {
        if mode == 0 {
            for &l in &g.normal_rem {
                st.removed_normal += self.remove(l) as usize;
            }
        } else {
            for &l in &g.fill {
                st.added_fill += self.add(l) as usize;
            }
        }
        self.cleanup_manifold(st)
    }

    /// Targets and normal / projection-distance test of the boundary vertices. The target is the closest-point projection
    /// (`t = w + delta (v - w)/|v - w|`) if its direction is within `theta_max` of the area-weighted vertex normal, within
    /// `max_proj_h` cell edges and makes a positive cosine with every incident quad normal. Otherwise (typically near concave
    /// creases, where the closest point jumps between surface parts) the ray along the vertex normal onto the shell is tried;
    /// vertices without an acceptable target are bad and all leaves touching them are listed for removal.
    fn geom_pass(&mut self, bq: &[BQ], opt: &Options) -> GeomRes {
        let oc = self.oc;
        let mut keys: Vec<u64> = bq.iter().flat_map(|q| q.key).collect();
        keys.sort_unstable();
        keys.dedup();
        let missing: Vec<u64> = keys.iter().copied().filter(|k| !self.cache.contains_key(k)).collect();
        let (bvh, delta) = (self.bvh, self.delta);
        let res = par_map(&missing, |&k| project(bvh, oc.vertex_pos(k), delta));
        for (k, t) in missing.into_iter().zip(res) {
            self.cache.insert(k, t);
        }
        let idx = |k: u64| keys.binary_search(&k).unwrap();
        let pos: Vec<V3> = keys.iter().map(|&k| oc.vertex_pos(k)).collect();
        let mut nrm = vec![[0.0; 3]; keys.len()];
        let mut hmax = vec![0.0f64; keys.len()];
        let mut fn_: Vec<Vec<V3>> = vec![Vec::new(); keys.len()];
        let mut owners: Vec<Vec<u32>> = vec![Vec::new(); keys.len()];
        for q in bq {
            let p: [V3; 4] = q.key.map(|k| pos[idx(k)]);
            let n = scale(cross(sub(p[2], p[0]), sub(p[3], p[1])), 0.5);
            let nl = norm(n);
            for t in 0..4 {
                let i = idx(q.key[t]);
                nrm[i] = add(nrm[i], n);
                hmax[i] = hmax[i].max(q.h);
                fn_[i].push(scale(n, 1.0 / nl.max(1e-300)));
                owners[i].push(q.leaf);
            }
        }
        let cos_max = (opt.theta_max.to_radians()).cos();
        let face_ok = |i: usize, d: V3| {
            let dl = norm(d);
            dl > 0.0 && fn_[i].iter().all(|&f| dot(d, f) / dl > FACE_COS_MIN)
        };
        let mut g = GeomRes { tg: vec![Tgt::default(); keys.len()], ..Default::default() };
        let mut need_ray: Vec<usize> = Vec::new();
        for (i, &k) in keys.iter().enumerate() {
            let tc = self.cache[&k];
            let d = sub(tc.t, pos[i]);
            let (dl, nl) = (norm(d), norm(nrm[i]));
            let ok = nl > 1e-300 && dl <= opt.max_proj_h * hmax[i] && dl > 0.0 && dot(d, nrm[i]) / (dl * nl) >= cos_max && face_ok(i, d);
            if ok {
                g.tg[i] = tc;
            } else {
                need_ray.push(i);
            }
        }
        let rays: Vec<Option<Tgt>> = par_map(&need_ray, |&i| {
            let nl = norm(nrm[i]);
            if nl < 1e-300 {
                return None;
            }
            let nh = scale(nrm[i], 1.0 / nl);
            let dc = sub(self.cache[&keys[i]].t, pos[i]);
            let ch = scale(dc, 1.0 / norm(dc).max(1e-300));
            // directions: vertex normal, then blends with the closest-point direction
            for (a, b) in [(1.0, 0.0), (1.0, 1.0), (2.0, 1.0), (1.0, 2.0)] {
                let d = add(scale(nh, a), scale(ch, b));
                let dl = norm(d);
                if dl < 1e-12 {
                    continue;
                }
                if let Some(r) = ray_target(bvh, pos[i], scale(d, 1.0 / dl), delta, opt.max_proj_h * hmax[i]) {
                    if face_ok(i, sub(r.t, pos[i])) {
                        return Some(r);
                    }
                }
            }
            None
        });
        let mut bad_leaves: Vec<u32> = Vec::new();
        let mut fill: Vec<u32> = Vec::new();
        for (&i, r) in need_ray.iter().zip(rays) {
            match r {
                Some(t) => {
                    g.tg[i] = t;
                    g.ray_targets += 1;
                }
                None => {
                    g.bad_vertices += 1;
                    g.tg[i] = self.cache[&keys[i]];
                    let mut f = Vec::new();
                    self.leaves_at_vertex(keys[i], false, &mut f);
                    fill.extend(f.into_iter().filter(|&l| self.cmin[l as usize] > 0.0));
                    if opt.remove_all_incident {
                        self.leaves_at_vertex(keys[i], true, &mut bad_leaves);
                    } else {
                        bad_leaves.extend_from_slice(&owners[i]);
                    }
                }
            }
        }
        for i in 0..keys.len() {
            g.max_proj_h = g.max_proj_h.max(norm(sub(g.tg[i].t, pos[i])) / hmax[i]);
        }
        bad_leaves.sort_unstable();
        bad_leaves.dedup();
        fill.sort_unstable();
        fill.dedup();
        g.fill = fill;
        g.normal_rem = bad_leaves;
        g.nrm = nrm;
        g.hmax = hmax;
        g.keys = keys;
        g
    }
}

/// Face / point assembly (faces deduplicated by their sorted vertex set, as in `cutmesh`).
struct Asm<'a> {
    oc: &'a Octree,
    pts: Vec<V3>,
    pmap: Map<u64, u32>,
    faces: Vec<Face>,
    fmap: Map<Vec<u32>, u32>,
    nonmanifold: usize,
}

const T_CUBE: u8 = 0;
const T_FAR: u8 = 1;
const T_SHELL: u8 = 2;

impl Asm<'_> {
    fn lat_id(&mut self, c: [u32; 3]) -> u32 {
        let key = vertex_key(c[0], c[1], c[2]);
        if let Some(&p) = self.pmap.get(&key) {
            return p;
        }
        let id = self.pts.len() as u32;
        self.pts.push(self.oc.vertex_pos(key));
        self.pmap.insert(key, id);
        id
    }
    fn add_face(&mut self, verts: Vec<u32>, tag: u8, cell: u32) -> u32 {
        if tag == T_SHELL {
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
            self.nonmanifold += 1;
            self.faces.push(Face { verts, owner: cell, neigh: NONE, patch: ORPHAN });
            return self.faces.len() as u32 - 1;
        }
        let patch = if tag == T_FAR { FARFIELD } else { ORPHAN };
        self.faces.push(Face { verts, owner: cell, neigh: NONE, patch });
        let f = self.faces.len() as u32 - 1;
        self.fmap.insert(key, f);
        f
    }
}

fn hex_min(h: &[u32; 4], x: &[V3], p: &[V3]) -> f64 {
    hex_sj_min(&[x[h[0] as usize], x[h[1] as usize], x[h[2] as usize], x[h[3] as usize], p[h[0] as usize], p[h[1] as usize], p[h[2] as usize], p[h[3] as usize]])
}

/// Min / mean / counts below 0.1, 0.2, 0.3 of the pillow hex scaled Jacobians (one optimisation stage).
#[derive(Clone, Copy, Debug, Default)]
pub struct SjStage {
    pub min: f64,
    pub mean: f64,
    pub n_lt_0p1: usize,
    pub n_lt_0p2: usize,
    pub n_lt_0p3: usize,
}

fn sj_stage(hexq: &[[u32; 4]], xs: &[V3], ps: &[V3]) -> SjStage {
    let mut s = SjStage { min: f64::INFINITY, ..Default::default() };
    let mut sum = 0.0;
    for h in hexq {
        let v = hex_min(h, xs, ps);
        s.min = s.min.min(v);
        sum += v;
        s.n_lt_0p1 += (v < 0.1) as usize;
        s.n_lt_0p2 += (v < 0.2) as usize;
        s.n_lt_0p3 += (v < 0.3) as usize;
    }
    s.mean = sum / hexq.len().max(1) as f64;
    s
}

/// Cap of the per-vertex objective  and floor of core cells.
const CORE_FLOOR: f64 = 0.5;

fn gather(p: &[V3], c: &[u32; 8]) -> [V3; 8] {
    std::array::from_fn(|n| p[c[n] as usize])
}

/// (min SJ, sum SJ, min core SJ) over the cells `inc` (cells with index >= `npil` are core cells).
fn local_score(p: &[V3], cells: &[[u32; 8]], npil: usize, inc: &[u32]) -> (f64, f64, f64) {
    let (mut mn, mut sum, mut cm) = (f64::INFINITY, 0.0, f64::INFINITY);
    for &c in inc {
        let s = hex_sj_min(&gather(p, &cells[c as usize]));
        mn = mn.min(s);
        sum += s;
        if c as usize >= npil {
            cm = cm.min(s);
        }
    }
    (mn, sum, cm)
}

/// Is `a` a better (admissible) score than `b`: core cells keep SJ >= `CORE_FLOOR`; lexicographic (min capped at `OPT_CAP`, sum).
fn better(a: (f64, f64, f64), b: (f64, f64, f64), cap: f64) -> bool {
    if a.2 < CORE_FLOOR.min(b.2) {
        return false;
    }
    let (am, bm) = (a.0.min(cap), b.0.min(cap));
    am > bm + 1e-12 || (am >= bm - 1e-12 && a.1 > b.1 + 1e-4)
}

#[derive(Default)]
struct OptRes {
    sweeps: usize,
    moved_core: usize,
    core_sj_min: f64,
}

/// Local optimisation of the pillow layer: Gauss-Seidel sweeps of a derivative-free pattern search per vertex. Shell vertices move
/// tangentially and are re-projected exactly; core-boundary vertices (and optionally one ring of core-interior vertices) move freely
/// within `0.5 h` of their lattice position as long as all incident core hexes keep SJ >= 0.5. Vertices of polyhedral (split-face)
/// core cells, and the hanging nodes on their faces, never move.
#[allow(clippy::too_many_arguments)]
fn optimise(w: &Work, bvh: &Bvh, delta: f64, bkeys: &[u64], hexq: &[[u32; 4]], hmax: &[f64], xs: &mut [V3], ps: &mut [V3], opt: &Options) -> OptRes {
    let (oc, vk, leaves, ml) = (w.oc, w.oc.vertices(), w.oc.leaves(), w.oc.max_level);
    let nbv = bkeys.len();
    let mut bidx = vec![u32::MAX; vk.len()];
    for (i, &k) in bkeys.iter().enumerate() {
        if let Ok(j) = vk.binary_search(&k) {
            bidx[j] = i as u32;
        }
    }
    // vertices of polyhedral core leaves (corners, hanging nodes, split-face centres) are fixed
    let mut fixed = vec![false; vk.len()];
    for li in 0..leaves.len() {
        if !w.core[li] {
            continue;
        }
        let l = leaves[li];
        let (base, e) = geom(&l, ml);
        let coord = |n: usize| [base[0] + e * (n as u32 & 1), base[1] + e * ((n as u32 >> 1) & 1), base[2] + e * ((n as u32 >> 2) & 1)];
        let (mut poly, mut extra): (bool, Vec<[u32; 3]>) = (false, Vec::new());
        for a in 0..3 {
            for s in 0..2 {
                let r = ring(a, s);
                let lat: [[u32; 3]; 4] = std::array::from_fn(|t| coord(r[t]));
                if let Some(nb) = w.neighbour(&l, a, s) {
                    if leaves[nb].level > l.level {
                        poly = true;
                        for t in 0..4 {
                            extra.push(std::array::from_fn(|d| (lat[t][d] + lat[(t + 1) % 4][d]) / 2));
                        }
                        extra.push(std::array::from_fn(|d| (lat[0][d] + lat[2][d]) / 2));
                    }
                }
                for t in 0..4 {
                    if let Some(m) = w.mid(lat[t], lat[(t + 1) % 4]) {
                        poly = true;
                        extra.push(m);
                    }
                }
            }
        }
        if poly {
            for n in 0..8 {
                fixed[w.lv[li][n] as usize] = true;
            }
            for m in extra {
                if let Ok(j) = vk.binary_search(&vertex_key(m[0], m[1], m[2])) {
                    fixed[j] = true;
                }
            }
        }
    }
    // ring-1 core-interior vertices
    let mut ring_of: Map<u32, u32> = Map::default();
    let mut ring_v: Vec<u32> = Vec::new();
    if opt.opt_ring {
        for li in 0..leaves.len() {
            if w.core[li] && w.lv[li].iter().any(|&v| bidx[v as usize] != u32::MAX) {
                for &v in &w.lv[li] {
                    if bidx[v as usize] == u32::MAX && !fixed[v as usize] && !ring_of.contains_key(&v) {
                        ring_of.insert(v, ring_v.len() as u32);
                        ring_v.push(v);
                    }
                }
            }
        }
    }
    let nr = ring_v.len();
    let nmov = 2 * nbv + nr;
    let mut p: Vec<V3> = Vec::with_capacity(nmov + 8);
    p.extend_from_slice(xs);
    p.extend_from_slice(ps);
    p.extend(ring_v.iter().map(|&v| oc.vertex_pos(vk[v as usize])));
    let x0: Vec<V3> = (0..nmov).map(|i| if i < nbv { oc.vertex_pos(bkeys[i]) } else if i >= 2 * nbv { p[i] } else { [0.0; 3] }).collect();
    let mut fidx: Map<u32, u32> = Map::default();
    let mut cells: Vec<[u32; 8]> = hexq.iter().map(|h| [h[0], h[1], h[2], h[3], nbv as u32 + h[0], nbv as u32 + h[1], nbv as u32 + h[2], nbv as u32 + h[3]]).collect();
    let npil = cells.len();
    let mut ch: Vec<f64> = Vec::new();
    for li in 0..leaves.len() {
        if !w.core[li] || !w.lv[li].iter().any(|&v| bidx[v as usize] != u32::MAX || ring_of.contains_key(&v)) {
            continue;
        }
        let mut c = [0u32; 8];
        for n in 0..8 {
            let v = w.lv[li][n];
            c[n] = if bidx[v as usize] != u32::MAX {
                bidx[v as usize]
            } else if let Some(&r) = ring_of.get(&v) {
                (2 * nbv) as u32 + r
            } else {
                *fidx.entry(v).or_insert_with(|| {
                    p.push(oc.vertex_pos(vk[v as usize]));
                    p.len() as u32 - 1
                })
            };
        }
        cells.push([c[0], c[1], c[3], c[2], c[4], c[5], c[7], c[6]]);
        ch.push(oc.leaf_size(&leaves[li]));
    }
    let mut inc: Vec<Vec<u32>> = vec![Vec::new(); nmov];
    let mut hloc = vec![f64::INFINITY; nmov];
    for (ci, c) in cells.iter().enumerate() {
        for &v in c {
            if (v as usize) < nmov {
                inc[v as usize].push(ci as u32);
                if ci >= npil {
                    hloc[v as usize] = hloc[v as usize].min(ch[ci - npil]);
                }
            }
        }
    }
    let movable: Vec<bool> = (0..nmov)
        .map(|i| {
            if i < nbv {
                opt.opt_core && !fixed[vk.binary_search(&bkeys[i]).unwrap_or(0)] && hloc[i].is_finite()
            } else if i < 2 * nbv {
                true
            } else {
                hloc[i].is_finite()
            }
        })
        .collect();
    for i in nbv..2 * nbv {
        hloc[i] = hmax[i - nbv];
    }
    let lim = |i: usize, c: V3| -> V3 {
        let d = sub(c, x0[i]);
        let (dl, mx) = (norm(d), 0.5 * hloc[i]);
        if dl > mx { add(x0[i], scale(d, mx / dl)) } else { c }
    };
    let mut res = OptRes::default();
    // initial guess: core-boundary vertices a fraction towards their projection targets (per vertex whichever is better locally)
    if opt.opt_init && opt.opt_core {
        for i in 0..nbv {
            if !movable[i] {
                continue;
            }
            let mut cur = local_score(&p, &cells, npil, &inc[i]);
            let mut bestp = p[i];
            for f in [0.25, 0.5] {
                p[i] = lim(i, add(x0[i], scale(sub(p[nbv + i], x0[i]), f)));
                let sc = local_score(&p, &cells, npil, &inc[i]);
                if better(sc, cur, opt.opt_cap) {
                    cur = sc;
                    bestp = p[i];
                }
            }
            p[i] = bestp;
        }
    }
    // pattern directions
    let s3 = 1.0 / 3f64.sqrt();
    let mut dirs3: Vec<V3> = vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
    for sx in [-s3, s3] {
        for sy in [-s3, s3] {
            for sz in [-s3, s3] {
                dirs3.push([sx, sy, sz]);
            }
        }
    }
    let mut active = vec![true; nmov];
    for sweep in 0..opt.opt_sweeps {
        let mut any = false;
        for v in 0..nmov {
            if !active[v] {
                continue;
            }
            active[v] = false;
            if !movable[v] {
                continue;
            }
            let mut cur = local_score(&p, &cells, npil, &inc[v]);
            if cur.0 >= opt.opt_cap {
                continue;
            }
            let shell = v >= nbv && v < 2 * nbv;
            let h = hloc[v];
            let (mut step, mut moved, mut it) = (0.25 * h, false, 0);
            while step >= opt.opt_min_step * h && it < 60 {
                it += 1;
                let old = p[v];
                let mut cands: Vec<V3> = Vec::with_capacity(14);
                if shell {
                    let (d, wp, _) = bvh.closest(old);
                    let nl = norm(sub(old, wp));
                    let n = if d > 1e-300 && nl > 1e-300 { scale(sub(old, wp), 1.0 / nl) } else { [0.0, 0.0, 1.0] };
                    let a = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
                    let e1 = cross(n, a);
                    let e1 = scale(e1, 1.0 / norm(e1));
                    let e2 = cross(n, e1);
                    for k in 0..8 {
                        let th = std::f64::consts::FRAC_PI_4 * k as f64;
                        let dv = add(scale(e1, th.cos()), scale(e2, th.sin()));
                        cands.push(project(bvh, add(old, scale(dv, step)), delta).t);
                    }
                } else {
                    for d in &dirs3 {
                        cands.push(lim(v, add(old, scale(*d, step))));
                    }
                }
                let mut best: Option<(V3, (f64, f64, f64))> = None;
                for c in cands {
                    p[v] = c;
                    let sc = local_score(&p, &cells, npil, &inc[v]);
                    if better(sc, best.map_or(cur, |b| b.1), opt.opt_cap) {
                        best = Some((c, sc));
                    }
                }
                match best {
                    Some((c, sc)) => {
                        p[v] = c;
                        cur = sc;
                        moved = true;
                    }
                    None => {
                        p[v] = old;
                        step *= 0.5;
                    }
                }
            }
            if moved {
                any = true;
                for &ci in &inc[v] {
                    for &u in &cells[ci as usize] {
                        if (u as usize) < nmov {
                            active[u as usize] = true;
                        }
                    }
                }
                active[v] = false;
            }
        }
        res.sweeps = sweep + 1;
        if !any {
            break;
        }
    }
    xs.copy_from_slice(&p[..nbv]);
    ps.copy_from_slice(&p[nbv..2 * nbv]);
    res.moved_core = (0..nbv).filter(|&i| norm(sub(p[i], x0[i])) > 1e-12 * oc.size).count() + (0..nr).filter(|&r| norm(sub(p[2 * nbv + r], x0[2 * nbv + r])) > 1e-12 * oc.size).count();
    res.core_sj_min = cells[npil..].iter().map(|c| hex_sj_min(&gather(&p, c))).fold(1.0, f64::min);
    res
}

/// Build the hex-core + pillow mesh from a shell-clipped, buffered octree (`Octree::build_clipped` with `Clip::Shell`).
pub fn build(oc: &Octree, tris: &[Tri], opt: &Options) -> HexCore {
    let delta = oc.clip.delta().expect("hexcore needs an octree built with Clip::Shell");
    let mut st = HcStats::default();
    let t = Instant::now();
    let bvh = Bvh::new(tris);
    let (leaves, ml) = (oc.leaves(), oc.max_level);
    let vk = oc.vertices();
    st.leaves = leaves.len();

    // --- core selection
    let psi: Vec<f64> = par_map(vk, |&k| if oc.corner_sign(k) { f64::NEG_INFINITY } else { bvh.closest(oc.vertex_pos(k)).0 - delta });
    let lv: Vec<[u32; 8]> = par_map(&(0..leaves.len()).collect::<Vec<usize>>(), |&li| {
        let (b, e) = geom(&leaves[li], ml);
        std::array::from_fn(|n| vk.binary_search(&vertex_key(b[0] + e * (n as u32 & 1), b[1] + e * ((n as u32 >> 1) & 1), b[2] + e * ((n as u32 >> 2) & 1))).expect("vertex") as u32)
    });
    let mut core = vec![false; leaves.len()];
    let mut cmin = vec![0.0; leaves.len()];
    let mut vcount = vec![0u32; vk.len()];
    for (li, l) in leaves.iter().enumerate() {
        let h = oc.leaf_size(l);
        let pmin = lv[li].iter().map(|&v| psi[v as usize]).fold(f64::INFINITY, f64::min);
        cmin[li] = pmin / h;
        if pmin > opt.gamma * h {
            core[li] = true;
            for &v in &lv[li] {
                vcount[v as usize] += 1;
            }
            st.core_initial += 1;
        } else if pmin > 0.0 {
            st.removed_margin += 1;
        } else {
            st.removed_nonfluid += 1;
        }
    }
    let mut w = Work { oc, bvh: &bvh, delta, lv, core, vcount, cmin, cache: Map::default() };
    st.t_core = t.elapsed().as_secs_f64();

    // --- cleanup rounds (manifold, normal) -> pillow hexes -> smoothing; pillow hexes whose min corner scaled Jacobian is
    // <= sj_remove remove their owner cell and the pass is repeated (<= max_rounds times)
    let (mut t_man, mut t_nrm, mut t_pil, mut t_smo) = (0.0, 0.0, 0.0, 0.0);
    let mut post = 0usize;
    let budget = (opt.max_removed_frac * st.core_initial as f64) as usize;
    let mut allow = true;
    let core0 = w.core.clone();
    let mut best: Option<(Out, usize, Vec<bool>, Vec<u32>, HcStats)> = None;
    let (mut fails, mut mode) = (0usize, 1u8);
    loop {
        let mut rounds = 0usize;
        let t = Instant::now();
        let mut bq = w.cleanup_manifold(&mut st);
        t_man += t.elapsed().as_secs_f64();
        let (bq, g) = loop {
            let t = Instant::now();
            let g = w.geom_pass(&bq, opt);
            t_nrm += t.elapsed().as_secs_f64();
            if g.bad_vertices == 0 || rounds >= opt.max_rounds || !allow {
                break (bq, g);
            }
            // Two trial repairs of the bad vertices: remove cells (spec) or fill pinched trenches (dual). A trial is kept only
            // when it strictly reduces the number of bad vertices and the total change stays within the budget; cell removal in
            // concave creases otherwise moves the front back and cascades through hanging nodes of the graded region.
            let snap = (w.core.clone(), w.vcount.clone(), st.clone());
            let mut best: Option<(usize, u8)> = None;
            for mode in 0..2u8 {
                let t = Instant::now();
                let nbq = w.apply_repair(&mut st, &g, mode);
                t_man += t.elapsed().as_secs_f64();
                let g2 = w.geom_pass(&nbq, opt);
                let changed = w.core.iter().zip(&core0).filter(|(a, b)| a != b).count();
                if g2.bad_vertices < g.bad_vertices && changed <= budget && best.is_none_or(|b| g2.bad_vertices < b.0) {
                    best = Some((g2.bad_vertices, mode));
                }
                (w.core, w.vcount, st) = (snap.0.clone(), snap.1.clone(), snap.2.clone());
            }
            let Some((_, mode)) = best else {
                st.budget_hit = true;
                allow = false;
                continue;
            };
            let t = Instant::now();
            bq = w.apply_repair(&mut st, &g, mode);
            t_man += t.elapsed().as_secs_f64();
            st.rounds += 1;
            rounds += 1;
        };
        st.max_proj_dist_over_h = g.max_proj_h;
        st.normal_violations_final = g.bad_vertices;
        st.ray_targets = g.ray_targets;
        let m = w.manifold_scan(&bq);
        (st.open_edges, st.nonmanifold_edges_final, st.nonmanifold_verts_final) = (m.open_edges, m.nm_edges, m.nm_verts);
        st.core_boundary_quads = bq.len();

        // pillow hexes: boundary vertices X (core side) and projected vertices P
        let t = Instant::now();
        let bkeys = g.keys.clone();
        let nbv = bkeys.len();
        st.core_boundary_verts = nbv;
        let mut xs: Vec<V3> = bkeys.iter().map(|&k| oc.vertex_pos(k)).collect();
        let mut ps: Vec<V3> = g.tg.iter().map(|t| t.t).collect();
        let hexq: Vec<[u32; 4]> = bq.iter().map(|q| q.key.map(|k| bkeys.binary_search(&k).unwrap() as u32)).collect();
        let mut inc: Vec<Vec<u32>> = vec![Vec::new(); nbv];
        let mut nbrs: Vec<Vec<u32>> = vec![Vec::new(); nbv];
        for (qi, h) in hexq.iter().enumerate() {
            for t in 0..4 {
                inc[h[t] as usize].push(qi as u32);
                nbrs[h[t] as usize].push(h[(t + 1) % 4]);
                nbrs[h[t] as usize].push(h[(t + 3) % 4]);
            }
        }
        for n in nbrs.iter_mut() {
            n.sort_unstable();
            n.dedup();
        }
        let hist = |xs: &[V3], ps: &[V3]| -> (f64, [usize; 5], f64) {
            let (mut mn, mut hi, mut sum) = (f64::INFINITY, [0usize; 5], 0.0);
            for h in &hexq {
                let s = hex_min(h, xs, ps);
                mn = mn.min(s);
                hi[sj_bin(s)] += 1;
                sum += s;
            }
            (mn, hi, sum / hexq.len().max(1) as f64)
        };
        let local = |i: usize, xs: &[V3], ps: &[V3]| inc[i].iter().map(|&q| hex_min(&hexq[q as usize], xs, ps)).fold(f64::INFINITY, f64::min);
        let (b_min, b_hist, _) = hist(&xs, &ps);
        st.sj_min_before = b_min;
        st.stage_initial = sj_stage(&hexq, &xs, &ps);
        st.sj_hist_before = b_hist;
        t_pil += t.elapsed().as_secs_f64();

        // quality improvement
        let t = Instant::now();
        if opt.core_move > 0.0 {
            for i in 0..nbv {
                let (old, xo) = (local(i, &xs, &ps), xs[i]);
                xs[i] = add(xo, scale(sub(ps[i], xo), opt.core_move));
                if local(i, &xs, &ps) < old {
                    xs[i] = xo;
                    st.core_move_rejected += 1;
                }
            }
        }
        let idxs: Vec<usize> = (0..nbv).collect();
        // repair candidate: ray along the vertex normal onto the shell (fixed during smoothing)
        let ray: Vec<Option<Tgt>> = if opt.smooth_iters > 0 {
            par_map(&idxs, |&i| {
                let nl = norm(g.nrm[i]);
                if nl < 1e-300 {
                    return None;
                }
                ray_target(&bvh, xs[i], scale(g.nrm[i], 1.0 / nl), delta, opt.max_proj_h * g.hmax[i])
            })
        } else {
            Vec::new()
        };
        for _ in 0..opt.smooth_iters {
            let cand: Vec<V3> = par_map(&idxs, |&i| {
                let nb = &nbrs[i];
                let avg = scale(nb.iter().fold([0.0; 3], |a, &j| add(a, ps[j as usize])), 1.0 / nb.len().max(1) as f64);
                project(&bvh, avg, delta).t
            });
            for i in 0..nbv {
                let (old, po) = (local(i, &xs, &ps), ps[i]);
                ps[i] = cand[i];
                let mut cur = local(i, &xs, &ps);
                if cur < old {
                    ps[i] = po;
                    cur = old;
                    st.smooth_rejected += 1;
                } else {
                    st.smooth_accepted += 1;
                }
                if let Some(r) = &ray[i] {
                    let keep = ps[i];
                    ps[i] = r.t;
                    if local(i, &xs, &ps) > cur + 1e-3 {
                        st.smooth_ray_moves += 1;
                    } else {
                        ps[i] = keep;
                    }
                }
            }
        }
        st.stage_smooth = sj_stage(&hexq, &xs, &ps);
        let t_o = Instant::now();
        st.core_sj_min = 1.0;
        if opt.opt_sweeps > 0 {
            let r = optimise(&w, &bvh, delta, &bkeys, &hexq, &g.hmax, &mut xs, &mut ps, opt);
            (st.opt_sweeps, st.moved_core_vertices, st.core_sj_min) = (r.sweeps, r.moved_core, r.core_sj_min);
        }
        st.t_opt = t_o.elapsed().as_secs_f64();
        st.stage_opt = sj_stage(&hexq, &xs, &ps);
        // last resort: vertices of hexes with min corner scaled Jacobian <= sj_remove move along the vertex normal towards the core
        // (off the shell) to the best local Jacobian
        let mut unproj = vec![false; nbv];
        for _ in 0..10 {
            let mut vs: Vec<usize> = hexq.iter().filter(|h| hex_min(h, &xs, &ps) <= opt.sj_remove).flat_map(|h| h.iter().map(|&v| v as usize)).collect();
            if vs.is_empty() {
                break;
            }
            vs.sort_unstable();
            vs.dedup();
            for i in vs {
                let nl = norm(g.nrm[i]);
                if nl < 1e-300 {
                    continue;
                }
                let nh = scale(g.nrm[i], 1.0 / nl);
                let s0 = norm(sub(ps[i], xs[i])).max(1e-9);
                let mut best = (local(i, &xs, &ps), ps[i]);
                for f in [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.08] {
                    ps[i] = add(xs[i], scale(nh, s0 * f));
                    let l = local(i, &xs, &ps);
                    if l > best.0 {
                        best = (l, ps[i]);
                        unproj[i] = true;
                    }
                }
                ps[i] = best.1;
            }
        }
        st.unprojected = unproj.iter().filter(|&&u| u).count();
        let (a_min, a_hist, a_mean) = hist(&xs, &ps);
        st.sj_min_after = a_min;
        st.sj_hist_after = a_hist;
        st.sj_mean = a_mean;
        st.n_sj_below_0 = a_hist[0];
        st.n_sj_below_0p2 = a_hist[0] + a_hist[1];
        (st.n_sj_below_0p1, st.n_sj_below_0p3) = (hexq.iter().filter(|h| hex_min(h, &xs, &ps) < 0.1).count(), hexq.iter().filter(|h| hex_min(h, &xs, &ps) < 0.3).count());
        st.pillow_hexes = hexq.len();
        t_smo += t.elapsed().as_secs_f64();
        let badq: Vec<usize> = (0..hexq.len()).filter(|&q| hex_min(&hexq[q], &xs, &ps) <= opt.sj_remove).collect();
        st.jacobian_violations_final = badq.len();
        let out = (bq, bkeys, xs, ps, hexq, nbv);
        // keep the best pass; a modification that does not reduce the number of bad hexes is undone
        let improved = best.as_ref().is_none_or(|b| badq.len() < b.1);
        if improved {
            best = Some((out, badq.len(), w.core.clone(), w.vcount.clone(), st.clone()));
            fails = 0;
        } else {
            fails += 1;
            let b = best.as_ref().unwrap();
            (w.core, w.vcount, st) = (b.2.clone(), b.3.clone(), b.4.clone());
            if fails >= 2 {
                break;
            }
            mode ^= 1;
        }
        let b = best.as_ref().unwrap();
        if b.1 == 0 || post >= opt.max_rounds || st.post_budget_hit {
            break;
        }
        // modify around the bad hexes of the best pass: fill pinched margin-removed leaves (mode 1) or remove the owners (mode 0)
        let (bo, bqs) = (&b.0, (0..b.0 .4.len()).filter(|&q| hex_min(&b.0 .4[q], &b.0 .2, &b.0 .3) <= opt.sj_remove).collect::<Vec<usize>>());
        let mut changed = false;
        for &q in &bqs {
            if mode == 0 {
                st.removed_jacobian += w.remove(bo.0[q].leaf) as usize;
            } else {
                for k in bo.0[q].key {
                    let mut f = Vec::new();
                    w.leaves_at_vertex(k, false, &mut f);
                    for l in f {
                        if w.cmin[l as usize] > 0.0 {
                            st.added_fill += w.add(l) as usize;
                        }
                    }
                }
            }
            changed = true;
        }
        let _ = w.cleanup_manifold(&mut st);
        post += 1;
        if !changed || w.core.iter().zip(&core0).filter(|(a, b)| a != b).count() > budget {
            st.budget_hit = true;
            st.post_budget_hit = true;
        }
    };
    let (bq, bkeys, xs, ps, hexq, bn) = {
        let b = best.take().unwrap();
        (w.core, w.vcount, st) = (b.2, b.3, b.4);
        b.0
    };
    let _ = bn;
    st.post_rounds = post;
    st.t_manifold = t_man;
    st.t_normal = t_nrm;
    st.t_pillow = t_pil;
    st.core_cells = w.core.iter().filter(|&&c| c).count();
    let t = Instant::now();
    let fin: Vec<Tgt> = par_map(&ps, |&p| {
        let (_, wp, tri) = bvh.closest(p);
        Tgt { t: p, w: wp, tri }
    });
    let mut shell = ShellQuads {
        pts: ps.clone(),
        nrm: fin
            .iter()
            .map(|f| {
                let d = sub(f.t, f.w);
                let l = norm(d);
                if l > 1e-300 { scale(d, 1.0 / l) } else { [0.0, 0.0, 1.0] }
            })
            .collect(),
        wall: fin.iter().map(|f| f.w).collect(),
        tri: fin.iter().map(|f| f.tri).collect(),
        quads: hexq.iter().map(|h| [h[3], h[2], h[1], h[0]]).collect(),
        mesh_pid: Vec::new(),
        mesh_face: Vec::new(),
    };
    st.t_smooth = t_smo + t.elapsed().as_secs_f64();

    // --- assemble the volume mesh
    let t = Instant::now();
    let mut asm = Asm { oc, pts: Vec::new(), pmap: Map::default(), faces: Vec::new(), fmap: Map::default(), nonmanifold: 0 };
    let mut cells: Vec<Cell> = Vec::with_capacity(st.core_cells + hexq.len());
    for li in 0..leaves.len() {
        if !w.core[li] {
            continue;
        }
        let l = leaves[li];
        let (base, e) = geom(&l, ml);
        let coord = |n: usize| [base[0] + e * (n as u32 & 1), base[1] + e * ((n as u32 >> 1) & 1), base[2] + e * ((n as u32 >> 2) & 1)];
        let ci = cells.len() as u32;
        let mut fl: Vec<u32> = Vec::with_capacity(6);
        for a in 0..3 {
            for s in 0..2 {
                let r = ring(a, s);
                let lat: [[u32; 3]; 4] = std::array::from_fn(|t| coord(r[t]));
                let nb = w.neighbour(&l, a, s);
                let (tag, quads): (u8, Vec<[[u32; 3]; 4]>) = match nb {
                    None => (T_FAR, vec![lat]),
                    Some(x) if leaves[x].level > l.level => {
                        st.core_split_faces += 1;
                        let m: [[u32; 3]; 4] = std::array::from_fn(|t| std::array::from_fn(|d| (lat[t][d] + lat[(t + 1) % 4][d]) / 2));
                        let c: [u32; 3] = std::array::from_fn(|d| (lat[0][d] + lat[2][d]) / 2);
                        (T_CUBE, vec![[lat[0], m[0], c, m[3]], [m[0], lat[1], m[1], c], [c, m[1], lat[2], m[2]], [m[3], c, m[2], lat[3]]])
                    }
                    _ => (T_CUBE, vec![lat]),
                };
                for q in quads {
                    let mut ids: Vec<u32> = Vec::with_capacity(8);
                    for t in 0..4 {
                        ids.push(asm.lat_id(q[t]));
                        if let Some(m) = w.mid(q[t], q[(t + 1) % 4]) {
                            ids.push(asm.lat_id(m));
                        }
                    }
                    fl.push(asm.add_face(ids, tag, ci));
                }
            }
        }
        let mut hex = None;
        if fl.len() == 6 && fl.iter().all(|&f| asm.faces[f as usize].verts.len() == 4) {
            let c: Vec<u32> = (0..8).map(|n| asm.lat_id(coord(n))).collect();
            hex = Some([c[0], c[1], c[3], c[2], c[4], c[5], c[7], c[6]]);
            st.core_hex += 1;
        } else {
            st.core_poly += 1;
        }
        cells.push(Cell { faces: fl, hex, h: oc.leaf_size(&l), cut: false, n_orig: 1 });
    }
    let cid: Vec<u32> = bkeys.iter().map(|&k| asm.lat_id(crate::octree::vertex_coords(k))).collect();
    let pid: Vec<u32> = ps
        .iter()
        .map(|&p| {
            asm.pts.push(p);
            asm.pts.len() as u32 - 1
        })
        .collect();
    for (i, &c) in cid.iter().enumerate() {
        asm.pts[c as usize] = xs[i];
    }
    shell.mesh_pid = pid.clone();
    for (qi, h) in hexq.iter().enumerate() {
        let ci = cells.len() as u32;
        let ca: [u32; 4] = h.map(|v| cid[v as usize]);
        let pb: [u32; 4] = h.map(|v| pid[v as usize]);
        let fl = vec![
            asm.add_face(vec![ca[3], ca[2], ca[1], ca[0]], T_CUBE, ci),
            asm.add_face(vec![ca[0], ca[1], pb[1], pb[0]], T_CUBE, ci),
            asm.add_face(vec![ca[1], ca[2], pb[2], pb[1]], T_CUBE, ci),
            asm.add_face(vec![ca[2], ca[3], pb[3], pb[2]], T_CUBE, ci),
            asm.add_face(vec![ca[3], ca[0], pb[0], pb[3]], T_CUBE, ci),
            asm.add_face(vec![pb[0], pb[1], pb[2], pb[3]], T_SHELL, ci),
        ];
        shell.mesh_face.push(fl[5]);
        cells.push(Cell { faces: fl, hex: Some([ca[0], ca[1], ca[2], ca[3], pb[0], pb[1], pb[2], pb[3]]), h: bq[qi].h, cut: true, n_orig: 1 });
    }
    let mut mesh = CutMesh { points: asm.pts, faces: asm.faces, cells, stats: Default::default(), patch_name: "shell", native: Default::default() };
    mesh.stats.leaves = leaves.len();
    mesh.stats.hex_cells = st.core_hex + hexq.len();
    mesh.stats.poly_cells = st.core_poly;
    mesh.stats.nonmanifold_faces = asm.nonmanifold;
    mesh.stats.orphan_faces = mesh.faces.iter().filter(|f| f.patch == ORPHAN).count();
    mesh.update_volume_stats();
    st.t_assemble = t.elapsed().as_secs_f64();
    HexCore { mesh, shell, stats: st }
}

/// Write [`ShellQuads`] as binary: u64 nverts, nverts * (3 f64 point, 3 f64 shell normal, 3 f64 wall point, u32 tri), u64 nquads,
/// nquads * 4 u32 (little endian; quads oriented away from the wall).
pub fn write_shell_quads(s: &ShellQuads, path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(&(s.pts.len() as u64).to_le_bytes())?;
    for i in 0..s.pts.len() {
        for v in [s.pts[i], s.nrm[i], s.wall[i]] {
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        w.write_all(&s.tri[i].to_le_bytes())?;
    }
    w.write_all(&(s.quads.len() as u64).to_le_bytes())?;
    for q in &s.quads {
        for &v in q {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()
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
    fn sphere_hexcore_pillow() {
        let (c, r, size) = ([0.05, 0.02, -0.03], 0.5, 2.0);
        let tris = sphere(c, r);
        let hs = size / 32.0;
        let delta = 4.0 * hs;
        let oc = Octree::build_clipped(&tris, [-1.0; 3], size, 5, &SizeField::new(hs, 0.3, size / 8.0), 0, SignMode::Dexel, Clip::Shell { delta });
        let hc = build(&oc, &tris, &Options { delta, ..Options::default() });
        let (s, m) = (&hc.stats, &hc.mesh);
        println!(
            "hexcore: leaves {} core_init {} margin {} nonmanifold {} hang {} split {} normal {} jac {} rounds {} core {} quads {} pillow {} sj {:.3}->{:.3} mean {:.3} proj/h {:.2}",
            s.leaves, s.core_initial, s.removed_margin, s.removed_nonmanifold, s.removed_hanging, s.removed_split, s.removed_normal, s.removed_jacobian, s.rounds, s.core_cells, s.core_boundary_quads, s.pillow_hexes, s.sj_min_before, s.sj_min_after, s.sj_mean, s.max_proj_dist_over_h
        );
        assert!(s.pillow_hexes > 0 && s.pillow_hexes == s.core_boundary_quads);
        assert_eq!((s.nonmanifold_edges_final, s.nonmanifold_verts_final, s.open_edges), (0, 0, 0));
        assert_eq!(s.n_sj_below_0, 0);
        assert!(s.sj_min_after >= s.sj_min_before - 1e-12);
        // pillow cells: positive volume, projected vertices on the shell
        let pillow: Vec<usize> = (0..m.cells.len()).filter(|&i| m.cells[i].cut).collect();
        assert_eq!(pillow.len(), s.pillow_hexes);
        for &i in &pillow {
            assert!(m.cell_volume(i) > 0.0, "pillow volume");
        }
        let bvh = Bvh::new(&tris);
        for p in &hc.shell.pts {
            assert!((bvh.closest(*p).0 - delta).abs() < 1e-9 * size, "vertex off the shell");
        }
        // closed mesh: every internal face has two cells, every cell is closed (directed edges pair up)
        assert_eq!((m.stats.orphan_faces, m.stats.nonmanifold_faces, m.stats.n_nonpositive_cells), (0, 0, 0));
        for f in &m.faces {
            assert_eq!(f.neigh == NONE, f.patch != INTERNAL, "internal face without neighbour");
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
        let nshell = m.faces.iter().filter(|f| f.patch == WALL).count();
        assert_eq!(nshell, s.pillow_hexes);
        let exact = size.powi(3) - 4.0 / 3.0 * PI * (r + delta).powi(3);
        assert!((m.stats.volume - exact).abs() < 0.03 * exact, "volume {} vs {exact}", m.stats.volume);
        // writers
        let dir = std::env::temp_dir().join(format!("rcore_hexcore_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s2 = crate::su2::write_su2(m, &dir.join("mesh.su2")).unwrap();
        assert_eq!(s2.nonpositive, 0);
        let (ni, patches) = crate::foam::write_polymesh(m, &dir.join("polyMesh")).unwrap();
        assert_eq!(patches[0].0, "shell");
        assert_eq!(ni + patches.iter().map(|p| p.1).sum::<usize>(), m.faces.len());
        write_shell_quads(&hc.shell, &dir.join("shell_quads.bin")).unwrap();
        let sz = std::fs::metadata(dir.join("shell_quads.bin")).unwrap().len() as usize;
        assert_eq!(sz, 8 + hc.shell.pts.len() * 76 + 8 + hc.shell.quads.len() * 16);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn optimiser_monotone_and_on_shell() {
        let (c, r, size) = ([0.05, 0.02, -0.03], 0.5, 2.0);
        let tris = sphere(c, r);
        let hs = size / 32.0;
        let delta = 4.0 * hs;
        let oc = Octree::build_clipped(&tris, [-1.0; 3], size, 5, &SizeField::new(hs, 0.3, size / 8.0), 0, SignMode::Dexel, Clip::Shell { delta });
        let off = build(&oc, &tris, &Options { delta, opt_sweeps: 0, ..Options::default() });
        let on = build(&oc, &tris, &Options { delta, opt_sweeps: 30, ..Options::default() });
        let (a, b) = (&off.stats, &on.stats);
        println!("opt: sj_min {:.3} -> {:.3} (smooth stage {:.3}), core min {:.3}, moved {}", a.sj_min_after, b.sj_min_after, b.stage_smooth.min, b.core_sj_min, b.moved_core_vertices);
        assert!(b.stage_opt.min >= b.stage_smooth.min - 1e-9, "optimiser decreased the global min SJ");
        assert!(b.sj_min_after >= a.sj_min_after - 1e-9);
        assert!(b.stage_opt.mean >= b.stage_smooth.mean - 1e-3 || b.stage_opt.min > b.stage_smooth.min);
        assert!(b.core_sj_min >= 0.5 - 1e-9, "core cell SJ {}", b.core_sj_min);
        assert_eq!(b.n_sj_below_0, 0);
        let bvh = Bvh::new(&tris);
        for p in &on.shell.pts {
            assert!((bvh.closest(*p).0 - delta).abs() < 1e-9 * size, "vertex off the shell");
        }
        assert_eq!(on.mesh.stats.n_nonpositive_cells, 0);
    }

    #[test]
    fn hex_sj_orientation() {
        let cube: [V3; 8] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]];
        assert!((hex_sj_min(&cube) - 1.0).abs() < 1e-14);
        let mut flipped = cube;
        flipped.swap(0, 4);
        flipped.swap(1, 5);
        flipped.swap(2, 6);
        flipped.swap(3, 7);
        assert!(hex_sj_min(&flipped) < 0.0);
    }
}
