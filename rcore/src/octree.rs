//! Adaptive linear octree over a triangle soup, the bulk-fill stage of the shell-first mesher.
//!
//! * Domain: axis-aligned cube (`origin`, `size`), max level `L <= 20`. Leaves are `(level, i, j, k)`;
//!   vertices live on the integer lattice of level `L` (`vertex_key` packs x | y << 21 | z << 42).
//! * Size field: `h(x) = min(h_surf + grade * d(x), h_max)`, `d` = unsigned distance to the soup
//!   (optionally `h_surf` per nearest triangle). A leaf is refined while its edge exceeds `h` evaluated
//!   conservatively (`d_min = max(0, d(centre) - half_diagonal)`); leaves that may touch the surface
//!   (`d(centre) <= half_diagonal`) are refined down to `h_surf`. Refinement is parallel per level.
//! * Strict 2:1 balance over faces, edges and corners (26-neighbourhood), descending-level sweep.
//! * Corner signs: behind `classify_points_fwn`; tri-dexel rays first (`classify_points`), FWN only as fallback.
//! * Hermite cut data: every unique leaf edge whose endpoints differ in sign gets the first exact
//!   segment/soup intersection (point, triangle id, unit normal) plus the hit count.

use crate::bvh::Bvh;
use crate::dexel::Tri;
use crate::fwn::Fwn;
use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};
use std::thread;
use std::time::Instant;

pub const MAX_LEVEL: u8 = 20;

/// Fast multiplicative hasher for packed integer keys.
#[derive(Default)]
struct Fx(u64);
impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, x: u64) {
        let h = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
        self.0 = h ^ (h >> 29);
    }
}
type Set = HashSet<u64, BuildHasherDefault<Fx>>;

fn spread(mut x: u64) -> u64 {
    x &= 0x1f_ffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    x
}
fn compact(mut x: u64) -> u64 {
    x &= 0x1249249249249249;
    x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
    x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
    x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
    x = (x ^ (x >> 16)) & 0x1f00000000ffff;
    x = (x ^ (x >> 32)) & 0x1f_ffff;
    x
}
/// 63-bit Morton code (x in bits 0,3,..; y in 1,4,..; z in 2,5,..) of 21-bit integer coordinates.
pub fn morton_encode(i: u32, j: u32, k: u32) -> u64 {
    spread(i as u64) | spread(j as u64) << 1 | spread(k as u64) << 2
}
pub fn morton_decode(m: u64) -> [u32; 3] {
    [compact(m) as u32, compact(m >> 1) as u32, compact(m >> 2) as u32]
}

/// Pack lattice coordinates (each <= 2^20) into a vertex key.
pub fn vertex_key(x: u32, y: u32, z: u32) -> u64 {
    x as u64 | (y as u64) << 21 | (z as u64) << 42
}
pub fn vertex_coords(key: u64) -> [u32; 3] {
    [(key & 0x1f_ffff) as u32, ((key >> 21) & 0x1f_ffff) as u32, ((key >> 42) & 0x1f_ffff) as u32]
}
fn pack(i: u32, j: u32, k: u32) -> u64 {
    i as u64 | (j as u64) << 20 | (k as u64) << 40
}
fn unpack(p: u64) -> [u32; 3] {
    [(p & 0xf_ffff) as u32, ((p >> 20) & 0xf_ffff) as u32, ((p >> 40) & 0xf_ffff) as u32]
}

/// Leaf cell: integer coordinates `(i, j, k)` at `level` (cell edge = domain size / 2^level).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Leaf {
    pub level: u8,
    pub i: u32,
    pub j: u32,
    pub k: u32,
}

impl Leaf {
    /// Morton code of the min corner on the level-`max_level` lattice (leaves sort by this).
    pub fn morton(&self, max_level: u8) -> u64 {
        let s = max_level - self.level;
        morton_encode(self.i << s, self.j << s, self.k << s)
    }
}

/// Target cell size: `h(x) = min(h_surf + grade * d(x), h_max)`; with `tri_h` the `h_surf` of the nearest triangle is used.
#[derive(Clone, Debug)]
pub struct SizeField {
    pub h_surf: f64,
    pub grade: f64,
    pub h_max: f64,
    /// optional per-triangle surface size (indexed by triangle id)
    pub tri_h: Option<Vec<f64>>,
}

impl SizeField {
    pub fn new(h_surf: f64, grade: f64, h_max: f64) -> SizeField {
        SizeField { h_surf, grade, h_max, tri_h: None }
    }
    /// surface size of triangle `tri` (global `h_surf` when no per-triangle sizes are given)
    pub fn h_surf_of(&self, tri: u32) -> f64 {
        match &self.tri_h {
            Some(v) => v[tri as usize],
            None => self.h_surf,
        }
    }
    /// target size at distance `d` from the soup, nearest triangle `tri`
    pub fn h(&self, d: f64, tri: u32) -> f64 {
        (self.h_surf_of(tri) + self.grade * d).min(self.h_max)
    }
    fn min_h_surf(&self) -> f64 {
        self.tri_h.as_ref().map_or(self.h_surf, |v| v.iter().cloned().fold(f64::INFINITY, f64::min))
    }
}

/// Does a cell (centre `c`, edge `size`) need refining? Conservative size rule plus the surface rule.
pub fn refine_needed(bvh: &Bvh, sf: &SizeField, c: [f64; 3], size: f64) -> bool {
    let hd = size * 0.5 * 3f64.sqrt();
    // beyond this distance h is capped at h_max for every triangle
    let reach = if sf.grade > 0.0 { (sf.h_max - sf.min_h_surf()) / sf.grade + hd } else { f64::INFINITY };
    let hit = if reach.is_finite() { bvh.closest_within(c, reach.max(0.0)) } else { Some(bvh.closest(c)) };
    match hit {
        None => size > sf.h_max,
        Some((d, _, tri)) => {
            if tri == crate::tree::NONE {
                return size > sf.h_max;
            }
            let h = sf.h((d - hd).max(0.0), tri);
            size > h || (d <= hd && size > sf.h_surf_of(tri))
        }
    }
}

/// Surface the octree is clipped against. `Wall`: the soup itself (inside = solid). `Shell { delta }`: the smooth offset
/// surface `psi(x) = s(x) - delta = 0` outside the soup (`s` = signed distance, negative inside); the fluid bulk is
/// `psi > 0`, everything else (solid and the layer band `0 < s <= delta`) is removed.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Clip {
    #[default]
    Wall,
    Shell { delta: f64 },
}

impl Clip {
    pub fn delta(&self) -> Option<f64> {
        match *self {
            Clip::Wall => None,
            Clip::Shell { delta } => Some(delta),
        }
    }
    pub fn is_shell(&self) -> bool {
        matches!(self, Clip::Shell { .. })
    }
    /// name of the clipped boundary patch
    pub fn patch_name(&self) -> &'static str {
        if self.is_shell() { "shell" } else { "wall" }
    }
}

/// Offset-shell implicit `psi = s - delta` helpers (distance by BVH, sign by |winding| > 0.5 for arbitrary points).
pub struct ShellFn<'a> {
    pub bvh: &'a Bvh,
    pub fwn: Fwn,
    pub delta: f64,
}

impl ShellFn<'_> {
    /// `|s(c) - delta|` given the unsigned wall distance `d`; the FWN sign is only evaluated where it matters (`d <= delta + hd`,
    /// otherwise the point is treated as outside, which only changes the size field deep inside the solid).
    pub fn dist(&self, c: [f64; 3], d: f64, hd: f64) -> f64 {
        if d <= self.delta + hd && self.fwn.winding(c).abs() > 0.5 { d + self.delta } else { (d - self.delta).abs() }
    }
    /// signed `psi(c)` with the sign from the FWN
    pub fn psi(&self, c: [f64; 3]) -> f64 {
        let d = self.bvh.closest(c).0;
        if self.fwn.winding(c).abs() > 0.5 { -d - self.delta } else { d - self.delta }
    }
}

/// [`refine_needed`] against the shell: distance-to-shell `|s - delta|` replaces distance-to-wall.
pub fn refine_needed_shell(sh: &ShellFn, sf: &SizeField, c: [f64; 3], size: f64) -> bool {
    let hd = size * 0.5 * 3f64.sqrt();
    let reach = if sf.grade > 0.0 { (sf.h_max - sf.min_h_surf()) / sf.grade + hd } else { f64::INFINITY };
    let hit = if reach.is_finite() { sh.bvh.closest_within(c, reach.max(0.0) + sh.delta) } else { Some(sh.bvh.closest(c)) };
    match hit {
        None => size > sf.h_max,
        Some((d, _, tri)) => {
            if tri == crate::tree::NONE {
                return size > sf.h_max;
            }
            let dist = sh.dist(c, d, hd);
            size > sf.h((dist - hd).max(0.0), tri) || (dist <= hd && size > sf.h_surf_of(tri))
        }
    }
}

/// Hermite data of one sign-changing leaf edge. For `Clip::Shell`: `point` = root of psi (the shell cut point), `normal` = unit
/// vector from the nearest wall point `wall_pt` to `point` (the shell normal, pointing away from the wall), `tri` = the
/// nearest wall triangle, `n_hits` = 1.
#[derive(Clone, Copy, Debug, Default)]
pub struct CutEdge {
    /// endpoint vertex keys (`a` has the smaller coordinate along the edge axis)
    pub a: u64,
    pub b: u64,
    /// first intersection with the soup, measured from `a`
    pub point: [f64; 3],
    pub t: f64,
    pub tri: u32,
    /// unit normal of the hit triangle (geometric, from vertex order)
    pub normal: [f64; 3],
    /// number of distinct intersections on the edge
    pub n_hits: u32,
    /// shell mode: nearest wall point to `point`
    pub wall_pt: [f64; 3],
    /// shell mode: root-finder iterations (trace + Illinois)
    pub iters: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    pub bvh: f64,
    pub build: f64,
    pub balance: f64,
    pub signs: f64,
    pub cuts: f64,
}

pub struct Octree {
    pub origin: [f64; 3],
    pub size: f64,
    pub max_level: u8,
    /// leaves sorted by Morton code of the min corner
    leaves: Vec<Leaf>,
    lmorton: Vec<u64>,
    /// unique leaf corner vertex keys (sorted) and their inside flags
    vkeys: Vec<u64>,
    vsign: Vec<bool>,
    cuts: Vec<CutEdge>,
    /// sign-changing edges without any exact hit (open soup / classification noise)
    pub unresolved: usize,
    /// vertices whose sign came from the FWN fallback (0 for `SignMode::Fwn`)
    pub fallback_vertices: usize,
    pub timings: Timings,
    /// clipping surface the signs / cut edges refer to
    pub clip: Clip,
    /// shell mode: max iterations used by the 1D root finder over all cut edges
    pub root_iterations_max: u32,
}

fn nthreads() -> usize {
    thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Run `f` on contiguous chunks of `items` in parallel, results in chunk order.
fn par_chunks<A: Sync, R: Send>(items: &[A], f: impl Fn(&[A]) -> R + Sync) -> Vec<R> {
    let per = items.len().div_ceil(nthreads() * 4).max(1);
    thread::scope(|sc| {
        let hs: Vec<_> = items
            .chunks(per)
            .map(|ch| {
                let f = &f;
                sc.spawn(move || f(ch))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// FWN-only inside/outside (`|w| > 0.5`); the reference path and the dexel fallback.
pub fn classify_points_fwn(fwn: &Fwn, pts: &[[f64; 3]]) -> Vec<bool> {
    fwn.winding_many(pts).iter().map(|w| w.abs() > 0.5).collect()
}

/// Which method decides the corner signs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignMode {
    Fwn,
    Dexel,
}

fn ceil_idx(g: &[f64], v: f64) -> i64 {
    let h = g[1] - g[0];
    let mut i = ((v - g[0]) / h).ceil() as i64;
    while i > 0 && (i - 1) < g.len() as i64 && g[(i - 1) as usize] >= v {
        i -= 1;
    }
    while i < g.len() as i64 && i >= 0 && g[i as usize] < v {
        i += 1;
    }
    i
}

fn floor_idx(g: &[f64], v: f64) -> i64 {
    let h = g[1] - g[0];
    let mut i = ((v - g[0]) / h).floor() as i64;
    while i >= 0 && (i as usize) < g.len() && g[i as usize] > v {
        i -= 1;
    }
    while i + 1 >= 0 && ((i + 1) as usize) < g.len() && g[(i + 1) as usize] <= v {
        i += 1;
    }
    i
}

#[derive(Clone, Copy)]
struct Cr {
    ray: u32,
    sign: i8,
    s: f64,
}

/// Per-vertex verdict of the rays parallel to `ax`: bit0 = inside (non-zero 1D winding), bit1 = ray consistent.
fn dexel_family_flags(tris: &[Tri], g: [&[f64]; 3], ax: usize, coords: &[[u32; 3]]) -> Vec<u8> {
    let (o1, o2) = match ax {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let nv = coords.len();
    // vertices grouped by ray, sorted along the ray
    let mut sv: Vec<(u64, u32)> = coords
        .iter()
        .enumerate()
        .map(|(v, c)| (((c[o1] as u64) << 42) | ((c[o2] as u64) << 21) | c[ax] as u64, v as u32))
        .collect();
    sv.sort_unstable();
    let mut rk: Vec<u64> = Vec::new(); // (a << 21 | b) per needed ray
    let mut vstart: Vec<usize> = Vec::new();
    for (p, &(k, _)) in sv.iter().enumerate() {
        let r = k >> 21;
        if rk.last() != Some(&r) {
            rk.push(r);
            vstart.push(p);
        }
    }
    vstart.push(nv);
    let nray = rk.len();
    let n1 = g[o1].len();
    let mut row = vec![0usize; n1 + 1];
    for &r in &rk {
        row[(r >> 21) as usize + 1] += 1;
    }
    for i in 0..n1 {
        row[i + 1] += row[i];
    }
    let (g1, g2) = (g[o1], g[o2]);
    let n2 = g2.len() as i64;

    // crossings on needed rays only
    let nt = nthreads();
    let chunk = tris.len().div_ceil(nt * 4).max(1);
    let (rk_r, row_r) = (&rk, &row);
    let parts: Vec<Vec<Cr>> = thread::scope(|sc| {
        let hs: Vec<_> = tris
            .chunks(chunk)
            .map(|tc| {
                sc.spawn(move || {
                    let mut out = Vec::new();
                    for t in tc {
                        let (a, b, c) = (t[0], t[1], t[2]);
                        let min1 = a[o1].min(b[o1]).min(c[o1]);
                        let max1 = a[o1].max(b[o1]).max(c[o1]);
                        let min2 = a[o2].min(b[o2]).min(c[o2]);
                        let max2 = a[o2].max(b[o2]).max(c[o2]);
                        let i0 = ceil_idx(g1, min1).max(0);
                        let i1 = floor_idx(g1, max1).min(g1.len() as i64 - 1);
                        let j0 = ceil_idx(g2, min2).max(0);
                        let j1 = floor_idx(g2, max2).min(n2 - 1);
                        if i0 > i1 || j0 > j1 {
                            continue;
                        }
                        let nrm = (b[o1] - a[o1]) * (c[o2] - a[o2]) - (b[o2] - a[o2]) * (c[o1] - a[o1]);
                        let nrm_axis = if ax == 1 { -nrm } else { nrm };
                        let sign: i8 = if nrm_axis > 0.0 { -1 } else { 1 };
                        for i in i0..=i1 {
                            let rg = &rk_r[row_r[i as usize]..row_r[i as usize + 1]];
                            if rg.is_empty() {
                                continue;
                            }
                            let x = g1[i as usize];
                            for j in j0..=j1 {
                                let key = ((i as u64) << 21) | j as u64;
                                let Ok(pos) = rg.binary_search(&key) else { continue };
                                let y = g2[j as usize];
                                let ef = |p: [f64; 3], q: [f64; 3]| (q[o1] - p[o1]) * (y - p[o2]) - (q[o2] - p[o2]) * (x - p[o1]);
                                let (e0, e1, e2) = (ef(a, b), ef(b, c), ef(c, a));
                                let inside = (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);
                                let area = e0 + e1 + e2;
                                if !inside || area == 0.0 {
                                    continue;
                                }
                                let s = (e1 * a[ax] + e2 * b[ax] + e0 * c[ax]) / area;
                                out.push(Cr { ray: (row_r[i as usize] + pos) as u32, sign, s });
                            }
                        }
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // bucket by ray
    let mut cstart = vec![0usize; nray + 1];
    for p in &parts {
        for c in p {
            cstart[c.ray as usize + 1] += 1;
        }
    }
    for r in 0..nray {
        cstart[r + 1] += cstart[r];
    }
    let mut fill = cstart.clone();
    let mut cs = vec![(0.0f64, 0i8); cstart[nray]];
    for p in &parts {
        for c in p {
            cs[fill[c.ray as usize]] = (c.s, c.sign);
            fill[c.ray as usize] += 1;
        }
    }
    drop(parts);
    // per-ray winding at each vertex, parallel over ray groups
    let mut flags_sorted = vec![0u8; nv];
    let ga = g[ax];
    let ngroups = (nt * 8).min(nray.max(1));
    let per = nray.div_ceil(ngroups).max(1);
    thread::scope(|sc| {
        let (mut fl_rest, mut cs_rest) = (&mut flags_sorted[..], &mut cs[..]);
        let sv = &sv;
        let (vstart, cstart) = (&vstart, &cstart);
        let mut r0 = 0;
        while r0 < nray {
            let r1 = (r0 + per).min(nray);
            let (fl, f2) = fl_rest.split_at_mut(vstart[r1] - vstart[r0]);
            let (cc, c2) = cs_rest.split_at_mut(cstart[r1] - cstart[r0]);
            fl_rest = f2;
            cs_rest = c2;
            let (vb, cb) = (vstart[r0], cstart[r0]);
            sc.spawn(move || {
                for r in r0..r1 {
                    let crs = &mut cc[cstart[r] - cb..cstart[r + 1] - cb];
                    crs.sort_unstable_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
                    let net: i32 = crs.iter().map(|c| c.1 as i32).sum();
                    let ok = net == 0;
                    let mut acc = 0i32;
                    let mut p = 0usize;
                    for pos in vstart[r]..vstart[r + 1] {
                        let along = (sv[pos].0 & 0x1f_ffff) as usize;
                        let z = ga[along];
                        while p < crs.len() && crs[p].0 < z {
                            acc += crs[p].1 as i32;
                            p += 1;
                        }
                        fl[pos - vb] = (acc != 0) as u8 | ((ok as u8) << 1);
                    }
                }
            });
            r0 = r1;
        }
    });
    let mut flags = vec![0u8; nv];
    for (pos, &(_, v)) in sv.iter().enumerate() {
        flags[v as usize] = flags_sorted[pos];
    }
    flags
}

/// The one place where inside/outside is decided: tri-dexel rays through every lattice line holding a
/// vertex (v3 rule of `dexel::classify`), exact-ish FWN fallback where rays are inconsistent or disagree.
/// Returns the signs and the number of fallback vertices.
fn classify_points(tris: &[Tri], origin: [f64; 3], size: f64, max_level: u8, vkeys: &[u64]) -> (Vec<bool>, usize) {
    let n = (1usize << max_level) + 1;
    let u = size / (1u64 << max_level) as f64;
    let gs: Vec<Vec<f64>> = (0..3).map(|d| (0..n).map(|i| origin[d] + i as f64 * u).collect()).collect();
    let g = [&gs[0][..], &gs[1][..], &gs[2][..]];
    let coords: Vec<[u32; 3]> = vkeys.iter().map(|&k| vertex_coords(k)).collect();
    let fl = [dexel_family_flags(tris, g, 0, &coords), dexel_family_flags(tris, g, 1, &coords), dexel_family_flags(tris, g, 2, &coords)];
    let mut inside = vec![false; vkeys.len()];
    let mut fb: Vec<u32> = Vec::new();
    for q in 0..vkeys.len() {
        let ins = [fl[0][q] & 1 != 0, fl[1][q] & 1 != 0, fl[2][q] & 1 != 0];
        let o = [fl[0][q] & 2 != 0, fl[1][q] & 2 != 0, fl[2][q] & 2 != 0];
        inside[q] = if o[2] { ins[2] } else if o[0] { ins[0] } else { ins[1] };
        let disagree = (o[0] && o[2] && ins[0] != ins[2]) || (o[1] && o[2] && ins[1] != ins[2]) || (o[0] && o[1] && ins[0] != ins[1]);
        if !(o[0] && o[1] && o[2]) || disagree {
            fb.push(q as u32);
        }
    }
    if !fb.is_empty() {
        let fwn = Fwn::new(tris);
        let pts: Vec<[f64; 3]> = fb
            .iter()
            .map(|&q| {
                let c = coords[q as usize];
                [origin[0] + c[0] as f64 * u, origin[1] + c[1] as f64 * u, origin[2] + c[2] as f64 * u]
            })
            .collect();
        let res = classify_points_fwn(&fwn, &pts);
        for (k, &q) in fb.iter().enumerate() {
            inside[q as usize] = res[k];
        }
    }
    (inside, fb.len())
}

fn oc_pos(origin: [f64; 3], size: f64, max_level: u8, key: u64) -> [f64; 3] {
    let u = size / (1u64 << max_level) as f64;
    let c = vertex_coords(key);
    [origin[0] + c[0] as f64 * u, origin[1] + c[1] as f64 * u, origin[2] + c[2] as f64 * u]
}

/// Root of `psi = d(x) - delta` on the segment `pf -> pr` (`pf` kept: `d > delta`; `pr` removed). The first root from the
/// fluid end lies before any wall crossing (the path must pass `d = delta` first), so unsigned distance is the signed one
/// up to `u_end` = first wall hit (or 1). Sphere tracing (step `psi / len`, never overshoots: psi is 1-Lipschitz) for up to
/// 12 steps, then Illinois on the remaining bracket; stops at `|psi| < 1e-10 len`, <= 60 iterations total.
/// Returns (u from `pf`, iterations), `None` if psi does not change sign on `[0, u_end]` (dirty soup).
fn shell_root(bvh: &Bvh, delta: f64, pf: [f64; 3], pr: [f64; 3]) -> Option<(f64, u32)> {
    let dir = [pr[0] - pf[0], pr[1] - pf[1], pr[2] - pf[2]];
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    let tol = 1e-10 * len;
    let at = |u: f64| [pf[0] + u * dir[0], pf[1] + u * dir[1], pf[2] + u * dir[2]];
    let psi = |u: f64| bvh.closest(at(u)).0 - delta;
    let u_end = bvh.segment_hits(pf, pr).first().map_or(1.0, |h| h.0.min(1.0));
    let mut u = 0.0;
    let mut f = psi(0.0);
    let mut it = 0u32;
    while f > tol && it < 12 {
        u = (u + f / len).min(u_end);
        f = psi(u);
        it += 1;
        if u >= u_end {
            break;
        }
    }
    if f.abs() <= tol {
        return Some((u, it));
    }
    // bracket [lo, hi]: psi(lo) > 0 (trace never overshoots), psi(hi) <= 0
    let (mut lo, mut flo) = (u, f);
    let (mut hi, mut fhi) = (u_end, if u >= u_end { f } else { psi(u_end) });
    if flo <= 0.0 {
        hi = lo;
        fhi = flo;
        lo = 0.0;
        flo = psi(0.0);
    }
    if !(flo > 0.0 && fhi <= 0.0) {
        return None;
    }
    let mut side = 0i8;
    while it < 60 {
        let x = if (flo - fhi).abs() > 0.0 { (lo * fhi - hi * flo) / (fhi - flo) } else { 0.5 * (lo + hi) };
        let x = if x.is_finite() && x > lo.min(hi) && x < lo.max(hi) { x } else { 0.5 * (lo + hi) };
        let fx = psi(x);
        it += 1;
        if fx.abs() <= tol || (hi - lo).abs() * len < 1e-14 * len {
            return Some((x, it));
        }
        if fx > 0.0 {
            lo = x;
            flo = fx;
            if side == 1 {
                fhi *= 0.5;
            }
            side = 1;
        } else {
            hi = x;
            fhi = fx;
            if side == -1 {
                flo *= 0.5;
            }
            side = -1;
        }
    }
    Some((0.5 * (lo + hi), it))
}

fn corner_keys(l: &Leaf, max_level: u8) -> [u64; 8] {
    let s = max_level - l.level;
    let (x, y, z, e) = (l.i << s, l.j << s, l.k << s, 1u32 << s);
    let mut c = [0u64; 8];
    for (n, v) in c.iter_mut().enumerate() {
        *v = vertex_key(x + (n as u32 & 1) * e, y + ((n as u32 >> 1) & 1) * e, z + ((n as u32 >> 2) & 1) * e);
    }
    c
}

impl Octree {
    /// Build from the root cell. See [`Octree::build_from`].
    pub fn build(tris: &[Tri], origin: [f64; 3], size: f64, max_level: u8, sf: &SizeField) -> Octree {
        Self::build_from(tris, origin, size, max_level, sf, 0)
    }

    /// Build starting from a uniform level `l0`: refine, 2:1 balance, corner signs, cut edges.
    pub fn build_from(tris: &[Tri], origin: [f64; 3], size: f64, max_level: u8, sf: &SizeField, l0: u8) -> Octree {
        Self::build_with(tris, origin, size, max_level, sf, l0, SignMode::Dexel)
    }

    /// [`Octree::build_from`] with an explicit corner-sign method.
    pub fn build_with(tris: &[Tri], origin: [f64; 3], size: f64, max_level: u8, sf: &SizeField, l0: u8, mode: SignMode) -> Octree {
        Self::build_impl(tris, origin, size, max_level, sf, l0, mode, false, Clip::Wall)
    }

    /// [`Octree::build_with`] plus the conformity buffer: before balancing, every leaf in the 26-neighbourhood of a
    /// surface leaf (`d(centre) <= half-diagonal`, hence every cut leaf) is refined to at least the surface leaf's level,
    /// so a cut cell and all its face/edge/corner neighbours are never coarser than it (equal size for a uniform `h_surf`).
    pub fn build_with_buffer(tris: &[Tri], origin: [f64; 3], size: f64, max_level: u8, sf: &SizeField, l0: u8, mode: SignMode) -> Octree {
        Self::build_impl(tris, origin, size, max_level, sf, l0, mode, true, Clip::Wall)
    }

    /// [`Octree::build_with_buffer`] clipped against `clip` (`Clip::Wall` gives exactly `build_with_buffer`). With
    /// `Clip::Shell` the size field, surface-leaf test, conformity buffer, corner signs (`psi > 0` = kept) and cut points
    /// (1D roots of psi) all refer to the offset shell.
    #[allow(clippy::too_many_arguments)]
    pub fn build_clipped(tris: &[Tri], origin: [f64; 3], size: f64, max_level: u8, sf: &SizeField, l0: u8, mode: SignMode, clip: Clip) -> Octree {
        Self::build_impl(tris, origin, size, max_level, sf, l0, mode, true, clip)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_impl(tris: &[Tri], origin: [f64; 3], size: f64, max_level: u8, sf: &SizeField, l0: u8, mode: SignMode, buffer: bool, clip: Clip) -> Octree {
        assert!(max_level <= MAX_LEVEL && l0 <= max_level);
        let mut tm = Timings::default();
        let t = Instant::now();
        let bvh = Bvh::new(tris);
        tm.bvh = t.elapsed().as_secs_f64();
        let shell: Option<ShellFn> = clip.delta().map(|delta| ShellFn { bvh: &bvh, fwn: Fwn::new(tris), delta });

        // --- adaptive refinement, one level at a time
        let t = Instant::now();
        let ll = max_level as usize;
        let mut sets: Vec<Set> = (0..=ll).map(|_| Set::default()).collect();
        let n0 = 1u32 << l0;
        let mut frontier: Vec<u64> = Vec::with_capacity((n0 as usize).pow(3));
        for k in 0..n0 {
            for j in 0..n0 {
                for i in 0..n0 {
                    frontier.push(pack(i, j, k));
                }
            }
        }
        for l in l0 as usize..=ll {
            let cs = size / (1u64 << l) as f64;
            let parts = par_chunks(&frontier, |ch| {
                let (mut keep, mut kids) = (Vec::new(), Vec::new());
                for &p in ch {
                    let [i, j, k] = unpack(p);
                    let c = [origin[0] + (i as f64 + 0.5) * cs, origin[1] + (j as f64 + 0.5) * cs, origin[2] + (k as f64 + 0.5) * cs];
                    let need = match &shell {
                        Some(sh) => refine_needed_shell(sh, sf, c, cs),
                        None => refine_needed(&bvh, sf, c, cs),
                    };
                    if l < ll && need {
                        for n in 0..8u32 {
                            kids.push(pack(2 * i + (n & 1), 2 * j + ((n >> 1) & 1), 2 * k + ((n >> 2) & 1)));
                        }
                    } else {
                        keep.push(p);
                    }
                }
                (keep, kids)
            });
            frontier = Vec::new();
            for (keep, kids) in parts {
                sets[l].extend(keep);
                frontier.extend(kids);
            }
        }
        tm.build = t.elapsed().as_secs_f64();

        // --- conformity buffer (optional), then 2:1 balance
        let t = Instant::now();
        if buffer {
            let surf = |c: [f64; 3], hd: f64| -> bool {
                match &shell {
                    None => bvh.closest_within(c, hd).is_some(),
                    Some(sh) => bvh.closest_within(c, hd + sh.delta).is_some_and(|(d, _, _)| sh.dist(c, d, hd) <= hd),
                }
            };
            Self::buffer(&mut sets, origin, size, &surf);
        }
        Self::balance(&mut sets);
        tm.balance = t.elapsed().as_secs_f64();

        let mut leaves: Vec<Leaf> = Vec::new();
        for (l, s) in sets.iter().enumerate() {
            for &p in s {
                let [i, j, k] = unpack(p);
                leaves.push(Leaf { level: l as u8, i, j, k });
            }
        }
        let mut oc = Self::from_leaves(origin, size, max_level, leaves);
        oc.timings = tm;

        // --- vertices and signs
        let t = Instant::now();
        oc.vkeys = oc.unique_vertices();
        match mode {
            SignMode::Fwn => {
                let pts: Vec<[f64; 3]> = oc.vkeys.iter().map(|&k| oc.vertex_pos(k)).collect();
                oc.vsign = classify_points_fwn(&Fwn::new(tris), &pts);
            }
            SignMode::Dexel => {
                let (s, nfb) = classify_points(tris, origin, size, max_level, &oc.vkeys);
                oc.vsign = s;
                oc.fallback_vertices = nfb;
            }
        }
        oc.clip = clip;
        if let Some(sh) = &shell {
            // kept (false) = psi > 0: outside the soup and farther than delta from it
            let idx: Vec<usize> = (0..oc.vkeys.len()).collect();
            let (vk, vs) = (&oc.vkeys, &oc.vsign);
            let removed: Vec<bool> = par_chunks(&idx, |ch| {
                ch.iter().map(|&q| vs[q] || bvh.closest_within(oc_pos(origin, size, max_level, vk[q]), sh.delta).is_some()).collect::<Vec<bool>>()
            })
            .into_iter()
            .flatten()
            .collect();
            oc.vsign = removed;
        }
        oc.timings.signs = t.elapsed().as_secs_f64();

        // --- cut edges
        let t = Instant::now();
        oc.compute_cuts(tris, &bvh);
        oc.timings.cuts = t.elapsed().as_secs_f64();
        oc
    }

    /// Wrap an arbitrary leaf set (no balancing, no signs/cuts): used by the checker tests.
    pub fn from_leaves(origin: [f64; 3], size: f64, max_level: u8, mut leaves: Vec<Leaf>) -> Octree {
        leaves.sort_by_key(|l| (l.morton(max_level), l.level));
        let lmorton = leaves.iter().map(|l| l.morton(max_level)).collect();
        Octree { origin, size, max_level, leaves, lmorton, vkeys: Vec::new(), vsign: Vec::new(), cuts: Vec::new(), unresolved: 0, fallback_vertices: 0, timings: Timings::default(), clip: Clip::Wall, root_iterations_max: 0 }
    }

    /// Descending-level sweep: a leaf at level l forces every leaf in its 26-neighbourhood to level >= l-1.
    fn balance(sets: &mut [Set]) {
        for l in (2..sets.len()).rev() {
            if sets[l].is_empty() {
                continue;
            }
            // a coarse leaf may need several splits against the same fine leaves: repeat until the level is clean
            let snap: Vec<u64> = sets[l].iter().cloned().collect();
            loop {
            let lim = 1i64 << l;
            let parts = par_chunks(&snap, |ch| {
                let mut out: Vec<(u8, u64)> = Vec::new();
                for &p in ch {
                    let [i, j, k] = unpack(p);
                    for dk in -1i64..=1 {
                        for dj in -1i64..=1 {
                            for di in -1i64..=1 {
                                if di == 0 && dj == 0 && dk == 0 {
                                    continue;
                                }
                                let (ni, nj, nk) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                                if ni < 0 || nj < 0 || nk < 0 || ni >= lim || nj >= lim || nk >= lim {
                                    continue;
                                }
                                for a in (0..=l).rev() {
                                    let sh = l - a;
                                    let q = pack((ni >> sh) as u32, (nj >> sh) as u32, (nk >> sh) as u32);
                                    if sets[a].contains(&q) {
                                        if a + 2 <= l {
                                            out.push((a as u8, q));
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                out
            });
            let mut split: Vec<(u8, u64)> = parts.into_iter().flatten().collect();
            split.sort_unstable();
            split.dedup();
            if split.is_empty() {
                break;
            }
            for (a, q) in split {
                let a = a as usize;
                if sets[a].remove(&q) {
                    let [i, j, k] = unpack(q);
                    for n in 0..8u32 {
                        sets[a + 1].insert(pack(2 * i + (n & 1), 2 * j + ((n >> 1) & 1), 2 * k + ((n >> 2) & 1)));
                    }
                }
            }
            }
        }
    }

    /// Conformity buffer: a surface leaf of level l forces the leaf containing each of its 26 same-level neighbour cells
    /// down to level l. Splits one level per pass; new surface children and sources that still had a violation go again.
    fn buffer(sets: &mut [Set], origin: [f64; 3], size: f64, surf: &(dyn Fn([f64; 3], f64) -> bool + Sync)) {
        let ll = sets.len() - 1;
        let is_surface = |l: usize, p: u64| -> bool {
            let cs = size / (1u64 << l) as f64;
            let [i, j, k] = unpack(p);
            let c = [origin[0] + (i as f64 + 0.5) * cs, origin[1] + (j as f64 + 0.5) * cs, origin[2] + (k as f64 + 0.5) * cs];
            surf(c, cs * 0.5 * 3f64.sqrt())
        };
        let mut active: Vec<(u8, u64)> = Vec::new();
        for l in 0..=ll {
            let snap: Vec<u64> = sets[l].iter().cloned().collect();
            for v in par_chunks(&snap, |ch| ch.iter().filter(|&&p| is_surface(l, p)).cloned().collect::<Vec<u64>>()) {
                active.extend(v.into_iter().map(|p| (l as u8, p)));
            }
        }
        while !active.is_empty() {
            let sref: &[Set] = sets;
            let parts = par_chunks(&active, |ch| {
                let (mut split, mut again) = (Vec::new(), Vec::new());
                for &(l, p) in ch {
                    let l = l as usize;
                    if !sref[l].contains(&p) {
                        continue; // split meanwhile (it was a coarse neighbour of a finer source)
                    }
                    let [i, j, k] = unpack(p);
                    let lim = 1i64 << l;
                    let mut viol = false;
                    for d in 0..27i64 {
                        let (ni, nj, nk) = (i as i64 + d % 3 - 1, j as i64 + (d / 3) % 3 - 1, k as i64 + d / 9 - 1);
                        if d == 13 || ni < 0 || nj < 0 || nk < 0 || ni >= lim || nj >= lim || nk >= lim {
                            continue;
                        }
                        for a in (0..=l).rev() {
                            let sh = l - a;
                            let q = pack((ni >> sh) as u32, (nj >> sh) as u32, (nk >> sh) as u32);
                            if sref[a].contains(&q) {
                                if a < l {
                                    split.push((a as u8, q));
                                    viol = true;
                                }
                                break;
                            }
                        }
                    }
                    if viol {
                        again.push((l as u8, p));
                    }
                }
                (split, again)
            });
            let (mut split, mut next): (Vec<(u8, u64)>, Vec<(u8, u64)>) = (Vec::new(), Vec::new());
            for (s, a) in parts {
                split.extend(s);
                next.extend(a);
            }
            split.sort_unstable();
            split.dedup();
            for (a, q) in split {
                let a = a as usize;
                if sets[a].remove(&q) {
                    let [i, j, k] = unpack(q);
                    for n in 0..8u32 {
                        let c = pack(2 * i + (n & 1), 2 * j + ((n >> 1) & 1), 2 * k + ((n >> 2) & 1));
                        sets[a + 1].insert(c);
                        if is_surface(a + 1, c) {
                            next.push((a as u8 + 1, c));
                        }
                    }
                }
            }
            next.sort_unstable();
            next.dedup();
            active = next;
        }
    }

    fn unique_vertices(&self) -> Vec<u64> {
        let ml = self.max_level;
        let parts = par_chunks(&self.leaves, |ch| {
            let mut v: Vec<u64> = ch.iter().flat_map(|l| corner_keys(l, ml)).collect();
            v.sort_unstable();
            v.dedup();
            v
        });
        let mut all: Vec<u64> = parts.into_iter().flatten().collect();
        all.sort_unstable();
        all.dedup();
        all
    }

    fn compute_cuts(&mut self, tris: &[Tri], bvh: &Bvh) {
        let ml = self.max_level;
        let (vk, vs) = (&self.vkeys, &self.vsign);
        let sign = |k: u64| vs[vk.binary_search(&k).unwrap()];
        let parts = par_chunks(&self.leaves, |ch| {
            let mut v: Vec<(u64, u64)> = Vec::new();
            for l in ch {
                let c = corner_keys(l, ml);
                let s: [bool; 8] = std::array::from_fn(|n| sign(c[n]));
                for n in 0..8usize {
                    for ax in 0..3 {
                        if n & (1 << ax) == 0 && s[n] != s[n | (1 << ax)] {
                            v.push((c[n], c[n | (1 << ax)]));
                        }
                    }
                }
            }
            v.sort_unstable();
            v.dedup();
            v
        });
        let mut edges: Vec<(u64, u64)> = parts.into_iter().flatten().collect();
        edges.sort_unstable();
        edges.dedup();
        let res = par_chunks(&edges, |ch| {
            ch.iter()
                .map(|&(ka, kb)| {
                    let (pa, pb) = (self.vertex_pos(ka), self.vertex_pos(kb));
                    if let Some(delta) = self.clip.delta() {
                        let a_removed = sign(ka);
                        let (pf, pr) = if a_removed { (pb, pa) } else { (pa, pb) };
                        return match shell_root(bvh, delta, pf, pr) {
                            None => CutEdge { a: ka, b: kb, n_hits: 0, ..Default::default() },
                            Some((u, iters)) => {
                                let p = [0, 1, 2].map(|k| pf[k] + u * (pr[k] - pf[k]));
                                let (d, q, tri) = bvh.closest(p);
                                let mut n = [0, 1, 2].map(|k| p[k] - q[k]);
                                if d > 1e-300 {
                                    n = n.map(|x| x / d);
                                } else {
                                    let tr = &tris[tri as usize];
                                    let (u, w) = ([0, 1, 2].map(|k| tr[1][k] - tr[0][k]), [0, 1, 2].map(|k| tr[2][k] - tr[0][k]));
                                    n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
                                    let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                                    n = n.map(|x| x / nl);
                                }
                                let t = if a_removed { 1.0 - u } else { u };
                                CutEdge { a: ka, b: kb, point: p, t, tri, normal: n, n_hits: 1, wall_pt: q, iters }
                            }
                        };
                    }
                    let hits = bvh.segment_hits(pa, pb);
                    match hits.first() {
                        None => CutEdge { a: ka, b: kb, n_hits: 0, ..Default::default() },
                        Some(&(t, tri)) => {
                            let tr = &tris[tri as usize];
                            let (u, w) = ([0, 1, 2].map(|k| tr[1][k] - tr[0][k]), [0, 1, 2].map(|k| tr[2][k] - tr[0][k]));
                            let mut n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
                            let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                            if nl > 0.0 {
                                n = n.map(|x| x / nl);
                            }
                            CutEdge { a: ka, b: kb, point: [0, 1, 2].map(|k| pa[k] + t * (pb[k] - pa[k])), t, tri, normal: n, n_hits: hits.len() as u32, ..Default::default() }
                        }
                    }
                })
                .collect::<Vec<_>>()
        });
        let all: Vec<CutEdge> = res.into_iter().flatten().collect();
        self.unresolved = all.iter().filter(|c| c.n_hits == 0).count();
        self.root_iterations_max = all.iter().map(|c| c.iters).max().unwrap_or(0);
        self.cuts = all.into_iter().filter(|c| c.n_hits > 0).collect();
    }

    pub fn leaves(&self) -> &[Leaf] {
        &self.leaves
    }
    /// Index of the leaf containing the min corner of cell `(level, i, j, k)` (possibly coarser, or finer if the cell is
    /// subdivided); `None` outside the root cube.
    pub fn leaf_containing(&self, level: u8, i: i64, j: i64, k: i64) -> Option<usize> {
        let lim = 1i64 << level;
        if i < 0 || j < 0 || k < 0 || i >= lim || j >= lim || k >= lim {
            return None;
        }
        let s = self.max_level - level;
        let code = morton_encode((i as u32) << s, (j as u32) << s, (k as u32) << s);
        let idx = self.lmorton.partition_point(|&m| m <= code);
        (idx > 0).then(|| idx - 1)
    }
    pub fn leaf_size(&self, l: &Leaf) -> f64 {
        self.size / (1u64 << l.level) as f64
    }
    pub fn leaf_min(&self, l: &Leaf) -> [f64; 3] {
        let h = self.leaf_size(l);
        [self.origin[0] + l.i as f64 * h, self.origin[1] + l.j as f64 * h, self.origin[2] + l.k as f64 * h]
    }
    pub fn leaf_centre(&self, l: &Leaf) -> [f64; 3] {
        let (h, m) = (self.leaf_size(l), self.leaf_min(l));
        [m[0] + 0.5 * h, m[1] + 0.5 * h, m[2] + 0.5 * h]
    }
    /// Physical position of a lattice vertex key.
    pub fn vertex_pos(&self, key: u64) -> [f64; 3] {
        let u = self.size / (1u64 << self.max_level) as f64;
        let c = vertex_coords(key);
        [self.origin[0] + c[0] as f64 * u, self.origin[1] + c[1] as f64 * u, self.origin[2] + c[2] as f64 * u]
    }
    /// Sorted unique leaf-corner vertex keys.
    pub fn vertices(&self) -> &[u64] {
        &self.vkeys
    }
    /// Inside flag of a leaf corner vertex (panics if `key` is not a leaf corner).
    pub fn corner_sign(&self, key: u64) -> bool {
        self.vsign[self.vkeys.binary_search(&key).expect("not a leaf corner vertex")]
    }
    pub fn cut_edges(&self) -> &[CutEdge] {
        &self.cuts
    }
    pub fn leaves_per_level(&self) -> Vec<usize> {
        let mut v = vec![0; self.max_level as usize + 1];
        for l in &self.leaves {
            v[l.level as usize] += 1;
        }
        v
    }

    /// Independent 2:1 checker: for every leaf and each of its 26 same-level neighbour cells, find the leaf
    /// containing that cell's min corner by binary search on the Morton-sorted leaves; a containing leaf
    /// two or more levels coarser is a violation. Returns the number of violating (leaf, neighbour) pairs.
    pub fn balance_violations(&self) -> usize {
        let ml = self.max_level;
        par_chunks(&self.leaves, |ch| {
            let mut bad = 0usize;
            for l in ch {
                let s = ml - l.level;
                let lim = 1i64 << l.level;
                for d in 0..27i64 {
                    let (di, dj, dk) = (d % 3 - 1, (d / 3) % 3 - 1, d / 9 - 1);
                    if d == 13 {
                        continue;
                    }
                    let (ni, nj, nk) = (l.i as i64 + di, l.j as i64 + dj, l.k as i64 + dk);
                    if ni < 0 || nj < 0 || nk < 0 || ni >= lim || nj >= lim || nk >= lim {
                        continue;
                    }
                    let code = morton_encode((ni as u32) << s, (nj as u32) << s, (nk as u32) << s);
                    let idx = self.lmorton.partition_point(|&m| m <= code);
                    if idx > 0 && self.leaves[idx - 1].level + 2 <= l.level {
                        bad += 1;
                    }
                }
            }
            bad
        })
        .into_iter()
        .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const C: [f64; 3] = [0.05, 0.02, -0.03];
    const R: f64 = 0.5;

    fn sphere() -> Vec<Tri> {
        let (nu, nv) = (48usize, 24usize);
        let p = |i: usize, j: usize| {
            let (th, ph) = (PI * i as f64 / nv as f64, 2.0 * PI * (j % nu) as f64 / nu as f64);
            [C[0] + R * th.sin() * ph.cos(), C[1] + R * th.sin() * ph.sin(), C[2] + R * th.cos()]
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

    const ORG: [f64; 3] = [-1.0; 3];
    const SZ: f64 = 2.0;
    const L: u8 = 6;

    fn sf() -> SizeField {
        SizeField::new(SZ / 64.0, 0.3, SZ / 8.0)
    }

    #[test]
    fn morton_roundtrip() {
        for &(i, j, k) in &[(0u32, 0u32, 0u32), (1, 2, 3), (1 << 20, (1 << 20) - 1, 12345)] {
            assert_eq!(morton_decode(morton_encode(i, j, k)), [i, j, k]);
        }
    }

    #[test]
    fn balance_checker() {
        let tris = sphere();
        let oc = Octree::build(&tris, ORG, SZ, L, &sf());
        assert_eq!(oc.balance_violations(), 0);
        // abrupt size field: balancing must actually split leaves, and the result must still be clean
        let steep = Octree::build(&tris, ORG, SZ, L, &SizeField::new(SZ / 64.0, 8.0, SZ / 4.0));
        assert_eq!(steep.balance_violations(), 0);
        // building again from the same leaves is a fixed point (balance is idempotent)
        let mut sets: Vec<Set> = (0..=L as usize).map(|_| Set::default()).collect();
        for l in oc.leaves() {
            sets[l.level as usize].insert(pack(l.i, l.j, l.k));
        }
        let before: usize = sets.iter().map(|s| s.len()).sum();
        Octree::balance(&mut sets);
        assert_eq!(before, sets.iter().map(|s| s.len()).sum::<usize>());
        // deliberately unbalanced: level-3 leaf touching a level-1 leaf only at a corner
        let mut lv = vec![];
        for n in 0..8u32 {
            lv.push(Leaf { level: 1, i: n & 1, j: (n >> 1) & 1, k: (n >> 2) & 1 });
        }
        lv.retain(|l| !(l.i == 0 && l.j == 0 && l.k == 0));
        for n in 0..8u32 {
            lv.push(Leaf { level: 2, i: n & 1, j: (n >> 1) & 1, k: (n >> 2) & 1 });
        }
        lv.retain(|l| !(l.level == 2 && l.i == 1 && l.j == 1 && l.k == 1));
        for n in 0..8u32 {
            lv.push(Leaf { level: 3, i: 2 + (n & 1), j: 2 + ((n >> 1) & 1), k: 2 + ((n >> 2) & 1) });
        }
        let bad = Octree::from_leaves(ORG, SZ, 3, lv);
        assert!(bad.balance_violations() > 0);
    }

    #[test]
    fn size_field_respected() {
        let tris = sphere();
        let bvh = Bvh::new(&tris);
        let s = sf();
        let oc = Octree::build(&tris, ORG, SZ, L, &s);
        let mut forced = 0;
        for (n, l) in oc.leaves().iter().enumerate() {
            let (c, h) = (oc.leaf_centre(l), oc.leaf_size(l));
            // conservative size requirement
            let hd = h * 0.5 * 3f64.sqrt();
            let (d, _, tri) = bvh.closest(c);
            let req = s.h((d - hd).max(0.0), tri);
            assert!(h <= req * 1.0001 || l.level == L, "leaf {l:?} size {h} > {req}");
            // not over-refined: parent not requested => a leaf >= 2 levels finer than the parent touches it
            if l.level > 0 && n % 5 == 0 {
                let par = Leaf { level: l.level - 1, i: l.i / 2, j: l.j / 2, k: l.k / 2 };
                let (pc, ph) = (oc.leaf_centre(&par), oc.leaf_size(&par));
                if !refine_needed(&bvh, &s, pc, ph) {
                    let lo = oc.leaf_min(&par);
                    let touch = oc.leaves().iter().any(|o| {
                        if o.level < l.level + 1 {
                            return false;
                        }
                        let (om, oh) = (oc.leaf_min(o), oc.leaf_size(o));
                        (0..3).all(|k| om[k] <= lo[k] + ph + 1e-12 && om[k] + oh >= lo[k] - 1e-12)
                    });
                    assert!(touch, "leaf {l:?} over-refined (parent not required, nothing finer adjacent)");
                    forced += 1;
                }
            }
        }
        eprintln!("balance-forced leaves among sampled: {forced}");
    }

    #[test]
    fn cuts_lie_on_soup_and_edge() {
        let tris = sphere();
        let bvh = Bvh::new(&tris);
        let oc = Octree::build(&tris, ORG, SZ, L, &sf());
        assert!(!oc.cut_edges().is_empty());
        for c in oc.cut_edges() {
            let (a, b) = (oc.vertex_pos(c.a), oc.vertex_pos(c.b));
            assert!(bvh.closest(c.point).0 < 1e-9 * SZ);
            let ab = [0, 1, 2].map(|k| b[k] - a[k]);
            let t = ((0..3).map(|k| (c.point[k] - a[k]) * ab[k]).sum::<f64>()) / (0..3).map(|k| ab[k] * ab[k]).sum::<f64>();
            assert!((-1e-12..=1.0 + 1e-12).contains(&t));
            let dist = (0..3).map(|k| (a[k] + t * ab[k] - c.point[k]).powi(2)).sum::<f64>().sqrt();
            assert!(dist < 1e-12 * SZ);
            assert!(c.n_hits >= 1);
            assert_ne!(oc.corner_sign(c.a), oc.corner_sign(c.b));
        }
        // closed sphere: every sign change is exactly crossed
        assert_eq!(oc.unresolved, 0);
    }

    #[test]
    fn corner_signs_match_sphere() {
        let tris = sphere();
        let oc = Octree::build(&tris, ORG, SZ, L, &sf());
        let (mut n, mut ins) = (0, 0);
        for &k in oc.vertices() {
            let p = oc.vertex_pos(k);
            let r = (0..3).map(|i| (p[i] - C[i]).powi(2)).sum::<f64>().sqrt();
            if (r - R).abs() < 0.02 * R {
                continue;
            }
            assert_eq!(oc.corner_sign(k), r < R, "vertex {p:?} r={r}");
            n += 1;
            ins += (r < R) as usize;
        }
        assert!(n > 100 && ins > 0);
    }

    #[test]
    fn buffer_property() {
        let tris = sphere();
        let bvh = Bvh::new(&tris);
        let steep = SizeField::new(SZ / 64.0, 2.0, SZ / 4.0);
        let plain = Octree::build(&tris, ORG, SZ, L, &steep);
        let buf = Octree::build_with_buffer(&tris, ORG, SZ, L, &steep, 0, SignMode::Dexel);
        assert_eq!(buf.balance_violations(), 0);
        let count_bad = |oc: &Octree| {
            let mut bad = 0;
            for l in oc.leaves() {
                let (c, h) = (oc.leaf_centre(l), oc.leaf_size(l));
                if bvh.closest_within(c, h * 0.5 * 3f64.sqrt()).is_none() {
                    continue;
                }
                for d in 0..27i64 {
                    let (di, dj, dk) = (d % 3 - 1, (d / 3) % 3 - 1, d / 9 - 1);
                    if d == 13 {
                        continue;
                    }
                    if let Some(n) = oc.leaf_containing(l.level, l.i as i64 + di, l.j as i64 + dj, l.k as i64 + dk) {
                        bad += (oc.leaves()[n].level < l.level) as usize;
                    }
                }
            }
            bad
        };
        let (bp, bb) = (count_bad(&plain), count_bad(&buf));
        println!("coarse surface-neighbour pairs: plain {bp}, buffered {bb}");
        assert!(bp > 0, "test needs a case where the buffer matters");
        assert_eq!(bb, 0);
        // every cut leaf has only same-or-finer neighbours and at least one sign change exists
        assert!(!buf.cut_edges().is_empty());
        assert_eq!(buf.unresolved, 0);
    }

    fn shell_oc(tris: &[Tri]) -> (Octree, f64) {
        let delta = 4.0 * SZ / 64.0;
        (Octree::build_clipped(tris, ORG, SZ, L, &sf(), 0, SignMode::Dexel, Clip::Shell { delta }), delta)
    }

    #[test]
    fn shell_cut_points_on_offset_surface() {
        let tris = sphere();
        let bvh = Bvh::new(&tris);
        let (oc, delta) = shell_oc(&tris);
        assert!(oc.cut_edges().len() > 500 && oc.unresolved == 0);
        let mut max_f = 0.0f64;
        for c in oc.cut_edges() {
            // exact: at distance delta from the soup
            assert!((bvh.closest(c.point).0 - delta).abs() < 1e-9 * SZ, "psi = {}", bvh.closest(c.point).0 - delta);
            // faceting tolerance: distance from the sphere centre vs R + delta
            let r = (0..3).map(|k| (c.point[k] - C[k]).powi(2)).sum::<f64>().sqrt();
            max_f = max_f.max((r - (R + delta)).abs());
            // sign change between the endpoints, point between them
            assert_ne!(oc.corner_sign(c.a), oc.corner_sign(c.b));
            let (a, b) = (oc.vertex_pos(c.a), oc.vertex_pos(c.b));
            for k in 0..3 {
                assert!(c.point[k] >= a[k].min(b[k]) - 1e-12 && c.point[k] <= a[k].max(b[k]) + 1e-12);
            }
        }
        println!("shell cut edges {} max |r-(R+delta)| {:.3e} root iterations max {}", oc.cut_edges().len(), max_f, oc.root_iterations_max);
        assert!(max_f < 3e-3, "faceting error {max_f}");
        assert!(oc.root_iterations_max <= 60);
        // the removed region contains the band: a vertex at r = R + delta/2 is removed, one at R + 2 delta kept
        for &k in oc.vertices() {
            let p = oc.vertex_pos(k);
            let r = (0..3).map(|i| (p[i] - C[i]).powi(2)).sum::<f64>().sqrt();
            if r < R + 0.8 * delta - 0.01 {
                assert!(oc.corner_sign(k), "r={r} should be removed");
            } else if r > R + 1.2 * delta + 0.01 {
                assert!(!oc.corner_sign(k), "r={r} should be kept");
            }
        }
    }

    #[test]
    fn shell_normals_unit_and_outward() {
        let tris = sphere();
        let (oc, delta) = shell_oc(&tris);
        for c in oc.cut_edges() {
            let n = c.normal;
            assert!(((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() - 1.0).abs() < 1e-12);
            let d = [0, 1, 2].map(|k| c.point[k] - c.wall_pt[k]);
            let dl = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((dl - delta).abs() < 1e-9 * SZ);
            assert!(n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0.0);
            // away from the wall = away from the sphere centre
            let rc = [0, 1, 2].map(|k| c.point[k] - C[k]);
            assert!(n[0] * rc[0] + n[1] * rc[1] + n[2] * rc[2] > 0.9 * (rc[0] * rc[0] + rc[1] * rc[1] + rc[2] * rc[2]).sqrt());
            assert_ne!(c.tri, crate::tree::NONE);
        }
    }

    fn corner_pts(oc: &Octree) -> Vec<[f64; 3]> {
        oc.vertices().iter().map(|&k| oc.vertex_pos(k)).collect()
    }

    fn check_dexel_vs_fwn(tris: &[Tri]) -> (usize, usize, usize) {
        let oc = Octree::build_with(tris, ORG, SZ, L, &sf(), 0, SignMode::Dexel);
        let fw = classify_points_fwn(&Fwn::new(tris), &corner_pts(&oc));
        let pts = corner_pts(&oc);
        let (mut mism, mut ambiguous) = (0, 0);
        for (q, &k) in oc.vertices().iter().enumerate() {
            if oc.corner_sign(k) != fw[q] {
                mism += 1;
                let w = crate::dexel::gwn(tris, pts[q]).abs();
                if (w - 0.5).abs() < 0.1 {
                    ambiguous += 1;
                    continue;
                }
                panic!("sign mismatch at {:?}: exact |w| = {w}", pts[q]);
            }
        }
        (oc.vertices().len(), oc.fallback_vertices, mism - ambiguous)
    }

    #[test]
    fn dexel_signs_match_fwn_closed() {
        let (nv, fb, _) = check_dexel_vs_fwn(&sphere());
        println!("closed: vertices {nv} fallback {fb}");
        assert!(nv > 1000);
    }

    #[test]
    fn dexel_signs_match_fwn_holed() {
        let mut t = sphere();
        // punch a hole: drop triangles around the +x pole
        t.retain(|tr| {
            let c = [(tr[0][0] + tr[1][0] + tr[2][0]) / 3.0, (tr[0][1] + tr[1][1] + tr[2][1]) / 3.0, (tr[0][2] + tr[1][2] + tr[2][2]) / 3.0];
            !((c[0] - C[0]) > 0.35 && (c[1] - C[1]).abs() < 0.2 && (c[2] - C[2]).abs() < 0.2)
        });
        let (nv, fb, _) = check_dexel_vs_fwn(&t);
        println!("holed: vertices {nv} fallback {fb}");
        assert!(fb > 0 && nv > 1000);
    }
}
