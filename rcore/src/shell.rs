//! Opened shell: the offset surface `d = delta` with its concave-crease ridges replaced by fillets of radius `rho`.
//!
//! The new bulk is the morphological opening of `{d > delta}` with a ball of radius `rho`. With `D = delta + rho`, `U = {x fluid: d(x) >= D}`
//! and `u(x) = dist(x, U)` the shell is `{u = rho}`; the signed field `psi = rho - u` (bulk positive) equals `d - delta` wherever the
//! exact test passes and is a smooth fillet of radius `rho` at the ridges.
//! * Exact test at a fluid point `x` with `d < D`, nearest wall point `w`, `n = (x - w)/d`: `q0 = w + D n`; if `d(q0) >= D (1 - 1e-9)` then
//!   `u = D - d` exactly (every `y` in `U` has `|x - y| >= D - d` and `q0` attains it). Otherwise `x` lies in the ridge zone, `u > D - d`.
//! * Away from the ridge zone nothing changes (`psi = d - delta`, plain closest-point projection). In the ridge zone `u` comes from a sparse
//!   grid (spacing `delta / k`) on which exact nodes and nodes of `U` are frozen (`u = D - d`) and the ridge nodes are solved with a
//!   second-order fast marching method for `|grad u| = 1`; `psi` there is `min(d - delta, rho - trilinear(u))`.
//! * `rho <= 0`: no grid, everything is the plain offset (`psi = d - delta`, `t = w + delta (x - w)/d`), bit for bit.

use crate::bvh::Bvh;
use crate::cutmesh::Map;
use crate::dexel::Tri;
use crate::fwn::Fwn;
use crate::layers::nkey;
use crate::tree::par_map;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

type V3 = [f64; 3];
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Clone, Debug, Default)]
pub struct ShellStats {
    /// grid nodes (fluid, d < D + 2 hg)
    pub grid_nodes: usize,
    /// nodes whose exact test failed (solved by fast marching)
    pub ridge_nodes: usize,
    /// heap pops of the fast marching solve
    pub fmm_pops: usize,
    pub t_build: f64,
}

/// Min-heap entry (value, node index).
struct Ent(f64, u32);
impl PartialEq for Ent {
    fn eq(&self, o: &Ent) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Ent {}
impl PartialOrd for Ent {
    fn partial_cmp(&self, o: &Ent) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Ent {
    fn cmp(&self, o: &Ent) -> Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

const DIRS6: [[i64; 3]; 6] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];

pub struct ShellField<'a> {
    bvh: &'a Bvh,
    pub delta: f64,
    pub rho: f64,
    /// `delta + rho`
    pub dd: f64,
    pub hg: f64,
    org: V3,
    grid: Map<u64, f64>,
    pub stats: ShellStats,
}

impl<'a> ShellField<'a> {
    /// Plain offset shell (no grid): `psi = d - delta`.
    pub fn offset(bvh: &'a Bvh, delta: f64) -> ShellField<'a> {
        ShellField { bvh, delta, rho: 0.0, dd: delta, hg: delta, org: [0.0; 3], grid: Map::default(), stats: ShellStats::default() }
    }

    /// Opened shell of fillet radius `rho` on a grid of spacing `delta / k`; `rho <= 0` gives the plain offset (`fwn`, `tris` unused).
    pub fn new(bvh: &'a Bvh, fwn: &Fwn, tris: &[Tri], delta: f64, rho: f64, k: f64) -> ShellField<'a> {
        let mut s = ShellField::offset(bvh, delta);
        if !(rho > 0.0) {
            return s;
        }
        let t0 = Instant::now();
        let (dd, hg) = (delta + rho, delta / k.max(1e-9));
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for t in tris {
            for p in t {
                for d in 0..3 {
                    lo[d] = lo[d].min(p[d]);
                    hi[d] = hi[d].max(p[d]);
                }
            }
        }
        (s.rho, s.dd, s.hg) = (rho, dd, hg);
        s.org = [0, 1, 2].map(|d| lo[d] - dd - 4.0 * hg);
        let (marg, rad, bs) = (dd + 2.0 * hg, dd + 2.0 * hg, 8i64);
        let mut bmap: Map<u64, [i64; 3]> = Map::default();
        for t in tris {
            let (mut a, mut b) = ([0i64; 3], [0i64; 3]);
            for d in 0..3 {
                let mn = t[0][d].min(t[1][d]).min(t[2][d]) - marg;
                let mx = t[0][d].max(t[1][d]).max(t[2][d]) + marg;
                a[d] = (((mn - s.org[d]) / hg).floor() as i64).max(0) / bs;
                b[d] = (((mx - s.org[d]) / hg).ceil() as i64) / bs;
            }
            for i in a[0]..=b[0] {
                for j in a[1]..=b[1] {
                    for l in a[2]..=b[2] {
                        bmap.entry(nkey([i, j, l])).or_insert([i, j, l]);
                    }
                }
            }
        }
        let blocks: Vec<[i64; 3]> = bmap.into_values().collect();
        let cand: Vec<Vec<([i64; 3], Option<f64>)>> = par_map(&blocks, |b| {
            let mut out = Vec::new();
            for i in 0..bs {
                for j in 0..bs {
                    for l in 0..bs {
                        let n = [b[0] * bs + i, b[1] * bs + j, b[2] * bs + l];
                        let x = s.pos(n);
                        if let Some((d, w, _)) = bvh.closest_within(x, rad) {
                            if fwn.winding(x).abs() <= 0.5 {
                                out.push((n, (d >= dd || s.exact_at(x, d, w)).then_some(dd - d)));
                            }
                        }
                    }
                }
            }
            out
        });
        let (nodes, known): (Vec<[i64; 3]>, Vec<Option<f64>>) = cand.into_iter().flatten().unzip();
        let mut idx: Map<u64, u32> = Map::default();
        for (i, n) in nodes.iter().enumerate() {
            idx.insert(nkey(*n), i as u32);
        }
        let nn = nodes.len();
        let mut u: Vec<f64> = known.iter().map(|k| k.unwrap_or(f64::INFINITY)).collect();
        let mut acc: Vec<bool> = known.iter().map(|k| k.is_some()).collect();
        s.stats.grid_nodes = nn;
        s.stats.ridge_nodes = acc.iter().filter(|&&a| !a).count();
        // second-order fast marching: upwind data per axis from accepted nodes with smaller values
        let solve = |i: usize, u: &[f64], acc: &[bool]| -> f64 {
            let n = nodes[i];
            let mut terms: Vec<(f64, f64)> = Vec::with_capacity(3);
            for d in 0..3 {
                let mut best: Option<(f64, f64, f64)> = None;
                for sg in [-1i64, 1] {
                    let mut p = n;
                    p[d] += sg;
                    let Some(&j) = idx.get(&nkey(p)) else { continue };
                    if !acc[j as usize] || best.is_some_and(|b| b.0 <= u[j as usize]) {
                        continue;
                    }
                    let a1 = u[j as usize];
                    let mut p2 = p;
                    p2[d] += sg;
                    let (b, al) = match idx.get(&nkey(p2)) {
                        Some(&j2) if acc[j2 as usize] && u[j2 as usize] < a1 => ((4.0 * a1 - u[j2 as usize]) / 3.0, 2.25),
                        _ => (a1, 1.0),
                    };
                    best = Some((a1, b, al));
                }
                if let Some((_, b, al)) = best {
                    terms.push((b, al));
                }
            }
            terms.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut res = f64::INFINITY;
            for m in 1..=terms.len() {
                let (mut a, mut b, mut c) = (0.0, 0.0, -hg * hg);
                for &(bi, al) in &terms[..m] {
                    a += al;
                    b += al * bi;
                    c += al * bi * bi;
                }
                let disc = b * b - a * c;
                if disc < 0.0 {
                    break;
                }
                let um = (b + disc.sqrt()) / a;
                if um < terms[m - 1].0 {
                    break;
                }
                res = um;
                if m == terms.len() || um <= terms[m].0 {
                    break;
                }
            }
            res
        };
        let mut heap: BinaryHeap<Ent> = BinaryHeap::new();
        for i in 0..nn {
            if !acc[i] {
                let v = solve(i, &u, &acc);
                if v.is_finite() {
                    u[i] = v;
                    heap.push(Ent(v, i as u32));
                }
            }
        }
        let mut pops = 0usize;
        while let Some(Ent(v, i)) = heap.pop() {
            let i = i as usize;
            if acc[i] || v > u[i] {
                continue;
            }
            acc[i] = true;
            pops += 1;
            for d in DIRS6 {
                let p = [nodes[i][0] + d[0], nodes[i][1] + d[1], nodes[i][2] + d[2]];
                if let Some(&j) = idx.get(&nkey(p)) {
                    let j = j as usize;
                    if !acc[j] {
                        let nv = solve(j, &u, &acc);
                        if nv < u[j] {
                            u[j] = nv;
                            heap.push(Ent(nv, j as u32));
                        }
                    }
                }
            }
        }
        s.stats.fmm_pops = pops;
        // point-source correction: the nearest point of U to a ridge node is a corner of dU, where FMM (first-order at the source) overshoots by
        // ~hg/4; alternating projections onto the offset surface from the FMM estimate of that point land on the corner, and |x - y| is an
        // upper bound of u, so the minimum of both is kept
        let ridge: Vec<usize> = (0..nn).filter(|&i| known[i].is_none() && u[i].is_finite()).collect();
        let refined: Vec<f64> = par_map(&ridge, |&i| {
            let n = nodes[i];
            let at = |d: usize, sg: i64| {
                let mut p = n;
                p[d] += sg;
                idx.get(&nkey(p)).map(|&j| u[j as usize]).filter(|v| v.is_finite())
            };
            let mut g = [0.0; 3];
            for d in 0..3 {
                g[d] = match (at(d, 1), at(d, -1)) {
                    (Some(a), Some(b)) => (a - b) / (2.0 * hg),
                    (Some(a), None) => (a - u[i]) / hg,
                    (None, Some(b)) => (u[i] - b) / hg,
                    _ => 0.0,
                };
            }
            let gl = dot(g, g).sqrt();
            if !(gl > 1e-12) {
                return u[i];
            }
            let x = s.pos(n);
            let mut y = sub(x, scale(g, u[i] / gl));
            let in_u = |p: V3| bvh.closest(p).0 >= dd * (1.0 - 1e-7);
            if in_u(y) {
                // estimate overshot into U: entry point of the segment x -> y
                let (mut a, mut b) = (x, y);
                for _ in 0..40 {
                    let m = scale(add(a, b), 0.5);
                    if in_u(m) { b = m } else { a = m }
                }
                let r = sub(x, b);
                return u[i].min(dot(r, r).sqrt());
            }
            for _ in 0..40 {
                let (d, w, _) = bvh.closest(y);
                if d >= dd * (1.0 - 1e-7) {
                    let r = sub(x, y);
                    return u[i].min(dot(r, r).sqrt());
                }
                if !(d > 1e-300) {
                    break;
                }
                y = add(w, scale(sub(y, w), dd / d));
            }
            u[i]
        });
        for (&i, &v) in ridge.iter().zip(&refined) {
            u[i] = v;
        }
        for (n, v) in nodes.iter().zip(&u) {
            if v.is_finite() {
                s.grid.insert(nkey(*n), *v);
            }
        }
        s.stats.t_build = t0.elapsed().as_secs_f64();
        s
    }

    fn pos(&self, n: [i64; 3]) -> V3 {
        [0, 1, 2].map(|d| self.org[d] + n[d] as f64 * self.hg)
    }

    /// Exact test (see module doc) at a fluid point with `d < D`, wall distance `d`, nearest wall point `w`.
    fn exact_at(&self, x: V3, d: f64, w: V3) -> bool {
        if !(d > 1e-300) {
            return true;
        }
        let q = add(w, scale(sub(x, w), self.dd / d));
        let m = self.dd * (1.0 - 1e-9);
        self.bvh.closest_within(q, m).is_none_or(|h| h.0 >= m)
    }

    pub fn bvh(&self) -> &'a Bvh {
        self.bvh
    }

    /// True when `psi = d - delta` holds at `x` (no ridge influence).
    pub fn is_exact(&self, x: V3) -> bool {
        let (d, w, _) = self.bvh.closest(x);
        self.rho <= 0.0 || d >= self.dd || self.exact_at(x, d, w)
    }

    /// Trilinear `u` at `x`; None when a corner of its cell is not in the (finite) grid.
    fn interp(&self, x: V3) -> Option<f64> {
        let f = [0, 1, 2].map(|d| (x[d] - self.org[d]) / self.hg);
        let i = f.map(|v| v.floor());
        let t = [f[0] - i[0], f[1] - i[1], f[2] - i[2]];
        let mut s = 0.0;
        for c in 0..8usize {
            let o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
            let w: f64 = (0..3).map(|d| if o[d] == 1 { t[d] } else { 1.0 - t[d] }).product();
            s += w * self.grid.get(&nkey([i[0] as i64 + o[0] as i64, i[1] as i64 + o[1] as i64, i[2] as i64 + o[2] as i64]))?;
        }
        Some(s)
    }

    /// (psi, plain distance function): the flag is true when `psi` is `d - delta` (1-Lipschitz), false for the interpolated fillet.
    fn psi_dw(&self, x: V3, d: f64, w: V3) -> (f64, bool) {
        let p = d - self.delta;
        if self.rho <= 0.0 || d >= self.dd || self.exact_at(x, d, w) {
            return (p, true);
        }
        match self.interp(x) {
            Some(u) => ((self.rho - u).min(p), false),
            None => (p, true),
        }
    }

    /// Signed opened-shell field at the fluid point `x` (bulk positive).
    pub fn psi(&self, x: V3) -> f64 {
        let (d, w, _) = self.bvh.closest(x);
        self.psi_dw(x, d, w).0
    }

    /// Plain offset projection `t = w + delta (x - w)/d` of `x` (foot `w`, triangle `tri`).
    fn offset_proj(&self, x: V3) -> (V3, V3, u32) {
        let (d, w, tri) = self.bvh.closest(x);
        let t = if d > 1e-300 { add(w, scale(sub(x, w), self.delta / d)) } else { x };
        (t, w, tri)
    }

    /// Closest point of the shell for a point `x` near it: the plain offset point when it is exact, else Newton iteration on `psi`.
    /// Returns (point, nearest wall point, triangle); None when the iteration fails.
    pub fn project(&self, x: V3) -> Option<(V3, V3, u32)> {
        let r0 = self.offset_proj(x);
        if self.rho <= 0.0 {
            return Some(r0);
        }
        let (d0, w0, _) = self.bvh.closest(r0.0);
        if d0 >= self.dd || self.exact_at(r0.0, d0, w0) {
            return Some(r0);
        }
        let (tol, h) = (1e-12 * self.dd, 1e-3 * self.hg);
        let grad = |p: V3| -> V3 {
            [0, 1, 2].map(|d| {
                let (mut a, mut b) = (p, p);
                a[d] += h;
                b[d] -= h;
                (self.psi(a) - self.psi(b)) / (2.0 * h)
            })
        };
        let fin = |p: V3| {
            let (_, w, tri) = self.bvh.closest(p);
            Some((p, w, tri))
        };
        let (mut p, mut g0) = (x, [0.0; 3]);
        for it in 0..60 {
            let f = self.psi(p);
            if !f.is_finite() {
                return None;
            }
            if f.abs() < tol {
                return fin(p);
            }
            let g = grad(p);
            if it == 0 {
                g0 = g;
            }
            let g2 = dot(g, g);
            if !(g2 > 1e-24) {
                break;
            }
            let mut step = scale(g, f / g2);
            let sl = dot(step, step).sqrt();
            if sl > self.dd {
                step = scale(step, self.dd / sl);
            }
            let mut a = 1.0;
            let mut ok = false;
            for _ in 0..10 {
                let q = sub(p, scale(step, a));
                let fq = self.psi(q);
                if fq.is_finite() && fq.abs() < f.abs() {
                    p = q;
                    ok = true;
                    break;
                }
                a *= 0.5;
            }
            if !ok {
                break;
            }
        }
        // fallback: root of psi along the initial gradient direction (towards the shell from either side)
        let n = dot(g0, g0).sqrt();
        if !(n > 1e-12) {
            return None;
        }
        let dir = scale(g0, if self.psi(x) < 0.0 { 1.0 } else { -1.0 } / n);
        self.ray(x, dir, 3.0 * self.hg)
    }

    /// First intersection of the ray `v + s dir` with the shell, `0 <= s <= smax` (sphere tracing on `psi`; the interpolated fillet field
    /// is only about sqrt(3)-Lipschitz, so steps there are `psi / 1.8`, finished by a secant iteration). Returns (point, wall point, triangle).
    pub fn ray(&self, v: V3, dir: V3, smax: f64) -> Option<(V3, V3, u32)> {
        let at = |s: f64| add(v, scale(dir, s));
        let mut s = 0.0;
        for _ in 0..200 {
            let x = at(s);
            let (d, w, tri) = self.bvh.closest(x);
            let (g, ex) = self.psi_dw(x, d, w);
            if g.abs() < if ex { 1e-12 } else { 1e-12 * self.dd } {
                return Some((x, w, tri));
            }
            if !ex && g.abs() < 0.05 * self.hg {
                return self.ray_secant(&at, s, g, smax);
            }
            s += if ex { g } else { g / 1.8 };
            if !(0.0..=smax).contains(&s) {
                return None;
            }
        }
        None
    }

    fn ray_secant(&self, at: &dyn Fn(f64) -> V3, s0: f64, f0: f64, smax: f64) -> Option<(V3, V3, u32)> {
        let (mut s0, mut f0) = (s0, f0);
        let mut s1 = s0 + f0 / 1.8;
        for _ in 0..40 {
            if !(0.0..=smax).contains(&s1) {
                return None;
            }
            let x = at(s1);
            let (d, w, tri) = self.bvh.closest(x);
            let f1 = self.psi_dw(x, d, w).0;
            if f1.abs() < 1e-12 * self.dd {
                return Some((x, w, tri));
            }
            if f1 == f0 || !f1.is_finite() {
                return None;
            }
            let s2 = s1 - f1 * (s1 - s0) / (f1 - f0);
            (s0, f0, s1) = (s1, f1, s2);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hexcore::{self, Options};
    use crate::layers::{self, Columns, LayerOptions};
    use crate::octree::{Clip, Octree, SignMode, SizeField};
    use std::f64::consts::PI;

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn pt(&mut self, a: f64) -> V3 {
            [0, 1, 2].map(|_| (2.0 * self.u() - 1.0) * a)
        }
    }

    fn sphere(r: f64) -> Vec<Tri> {
        let (nu, nv) = (48usize, 24usize);
        let p = |i: usize, j: usize| {
            let (th, ph) = (PI * i as f64 / nv as f64, 2.0 * PI * (j % nu) as f64 / nu as f64);
            [r * th.sin() * ph.cos(), r * th.sin() * ph.sin(), r * th.cos()]
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

    fn bx(lo: V3, hi: V3) -> Vec<Tri> {
        let p = |i: usize, j: usize, k: usize| [if i == 1 { hi[0] } else { lo[0] }, if j == 1 { hi[1] } else { lo[1] }, if k == 1 { hi[2] } else { lo[2] }];
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

    /// two intersecting closed boxes: four concave vertical creases at (+-0.2, +-0.2), z in [-0.2, 0.2]
    fn cross() -> Vec<Tri> {
        let mut t = bx([-0.6, -0.2, -0.2], [0.6, 0.2, 0.2]);
        t.extend(bx([-0.2, -0.6, -0.2], [0.2, 0.6, 0.2]));
        t
    }

    fn seg_dist(p: V3, a: V3, b: V3) -> f64 {
        let (ab, ap) = (sub(b, a), sub(p, a));
        let t = (dot(ap, ab) / dot(ab, ab)).clamp(0.0, 1.0);
        let q = add(a, scale(ab, t));
        dot(sub(p, q), sub(p, q)).sqrt()
    }

    #[test]
    fn rho_zero_is_plain_offset() {
        let tris = sphere(0.6);
        let (bvh, fwn) = (Bvh::new(&tris), Fwn::new(&tris));
        let delta = 0.1;
        let sf = ShellField::new(&bvh, &fwn, &tris, delta, 0.0, 8.0);
        assert_eq!(sf.stats.grid_nodes, 0);
        let mut rng = Rng(7);
        let mut n = 0;
        while n < 300 {
            let x = rng.pt(0.9);
            let (d, w, tri) = bvh.closest(x);
            if fwn.winding(x).abs() > 0.5 {
                continue;
            }
            n += 1;
            assert_eq!(sf.psi(x), d - delta);
            let (t, w2, tri2) = sf.project(x).unwrap();
            let old = if d > 1e-300 { add(w, scale(sub(x, w), delta / d)) } else { x };
            assert_eq!((t, w2, tri2), (old, w, tri));
        }
    }

    #[test]
    fn concave_crease_fillet() {
        let tris = cross();
        let (bvh, fwn) = (Bvh::new(&tris), Fwn::new(&tris));
        let (delta, rho, k) = (0.1, 0.1, 8.0);
        let sf = ShellField::new(&bvh, &fwn, &tris, delta, rho, k);
        let (dd, hg) = (delta + rho, delta / k);
        println!("shell field: {:?}", sf.stats);
        assert!(sf.stats.ridge_nodes > 0 && sf.stats.ridge_nodes < sf.stats.grid_nodes);
        let creases: Vec<(V3, V3)> = [-1.0, 1.0].iter().flat_map(|&a| [-1.0, 1.0].iter().map(move |&b| ([0.2 * a, 0.2 * b, -0.2], [0.2 * a, 0.2 * b, 0.2]))).collect();
        // (a) exact away from the creases
        let mut rng = Rng(11);
        let (mut n, mut nfar) = (0, 0);
        while n < 600 {
            let x = rng.pt(1.0);
            if fwn.winding(x).abs() > 0.5 {
                continue;
            }
            n += 1;
            if creases.iter().all(|c| seg_dist(x, c.0, c.1) > 2.0 * dd) {
                nfar += 1;
                assert_eq!(sf.psi(x), bvh.closest(x).0 - delta, "psi not exact at {x:?}");
                assert!(sf.is_exact(x));
            }
        }
        assert!(nfar > 50);
        // (b) bisector of the crease at (0.2, 0.2): shell point at sqrt(2) D - rho from the crease line
        let want = 2f64.sqrt() * dd - rho;
        for s in [0.11, 0.12, 0.14, 0.17] {
            for z in [-0.05, 0.0, 0.05] {
                let (t, _, _) = sf.project([0.2 + s, 0.2 + s, z]).expect("project");
                let dc = ((t[0] - 0.2).powi(2) + (t[1] - 0.2).powi(2)).sqrt();
                assert!((dc - want).abs() < 0.1 * hg, "bisector shell point at {dc}, want {want}");
            }
        }
        // (c) projected points lie on psi = 0
        let (mut n, mut ok) = (0, 0);
        while n < 400 {
            let x = rng.pt(0.8);
            if fwn.winding(x).abs() > 0.5 || sf.psi(x).abs() > 2.0 * hg {
                continue;
            }
            n += 1;
            let (t, w, _) = sf.project(x).unwrap_or_else(|| panic!("project failed at {x:?}"));
            assert!(sf.psi(t).abs() < 1e-9, "psi(t) = {} at {x:?}", sf.psi(t));
            assert!((bvh.closest(t).1[0] - w[0]).abs() < 1e-12);
            ok += 1;
        }
        assert_eq!(ok, 400);
    }

    #[test]
    fn hexcore_layers_fillet_smoke() {
        let tris = cross();
        let (org, size, l) = (-1.0, 2.0, 5u8);
        let hs = size / (1u64 << l) as f64;
        let delta = 4.0 * hs;
        let oc = Octree::build_clipped(&tris, [org; 3], size, l, &SizeField::new(hs, 0.3, size / 8.0), 0, SignMode::Dexel, Clip::Shell { delta });
        let hc = hexcore::build(&oc, &tris, &Options { delta, fillet: 1.0, ..Options::default() });
        println!("fillet hexcore: ridge {} grid {} t_shell {:.3}", hc.stats.ridge_nodes, hc.stats.grid_nodes, hc.stats.t_shell);
        assert!(hc.rho > 0.0 && hc.stats.ridge_nodes > 0);
        let (bvh, fwn) = (Bvh::new(&tris), Fwn::new(&tris));
        let sf = ShellField::new(&bvh, &fwn, &tris, hc.delta, hc.rho, hc.fillet_k);
        assert!(!hc.shell.pts.is_empty());
        for p in &hc.shell.pts {
            assert!(sf.psi(*p).abs() < 1e-8, "shell point off psi = 0: {}", sf.psi(*p));
        }
        let (nl, r) = (6usize, 1.25);
        let dy0 = layers::fit_dy0(delta, nl, r);
        let lm = layers::build(&hc, &tris, &LayerOptions { features: false, cont: 0, cont_rings: 3, zip: false, n: nl, dy0, growth: r, snap: 0.4, repair_sj: 0.05, columns: Columns::Harmonic, smooth_sweeps: 25, hg_k: 8.0, feature_angle: 0.0, edge_nodes: false });
        println!("fillet layers: {}", lm.stats.json());
        assert_eq!((lm.stats.orient_mismatch, lm.stats.nonmanifold), (0, 0));
    }
}
