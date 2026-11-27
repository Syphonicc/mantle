//! BVH closest point / unsigned distance on a triangle soup (nearest-first traversal, box-distance pruning).

use crate::dexel::Tri;
use crate::tree::{par_map, Tree, NONE};

/// (distance, closest point, triangle id)
pub type Hit = (f64, [f64; 3], u32);

pub struct Bvh {
    tree: Tree,
    tris: Vec<Tri>,
}

type V = [f64; 3];
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn axpy(a: V, s: f64, d: V) -> V {
    [a[0] + s * d[0], a[1] + s * d[1], a[2] + s * d[2]]
}

fn seg_closest(p: V, a: V, b: V) -> V {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = if l2 > 0.0 { (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    axpy(a, t, ab)
}

/// Closest point on triangle (Ericson, Real-Time Collision Detection, ClosestPtPointTriangle).
pub fn closest_pt_triangle(p: V, t: &Tri) -> V {
    let (a, b, c) = (t[0], t[1], t[2]);
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return axpy(a, d1 / (d1 - d3), ab);
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return axpy(a, d2 / (d2 - d6), ac);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return axpy(b, (d4 - d3) / ((d4 - d3) + (d5 - d6)), sub(c, b));
    }
    let s = va + vb + vc;
    if !(s > 0.0) || !(1.0 / s).is_finite() {
        // degenerate (zero-area) triangle: best of the three edges
        let cands = [seg_closest(p, a, b), seg_closest(p, b, c), seg_closest(p, c, a)];
        let d = |x: V| dot(sub(p, x), sub(p, x));
        return *cands.iter().min_by(|x, y| d(**x).partial_cmp(&d(**y)).unwrap()).unwrap();
    }
    let (v, w) = (vb / s, vc / s);
    [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w]
}

fn box_d2(p: V, lo: V, hi: V) -> f64 {
    (0..3).map(|k| (lo[k] - p[k]).max(0.0).max(p[k] - hi[k]).powi(2)).sum()
}

impl Bvh {
    pub fn new(tris: &[Tri]) -> Bvh {
        let tree = Tree::build(tris);
        let sorted = tree.order.iter().map(|&i| tris[i as usize]).collect();
        Bvh { tree, tris: sorted }
    }

    /// nearest hit with squared distance < best.0 (best.2 == NONE: nothing yet)
    fn search(&self, q: V, best: &mut (f64, V, u32)) {
        if self.tree.nodes.is_empty() {
            return;
        }
        let mut stack = [(0u32, 0.0f64); 128];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let (ni, bd) = stack[sp];
            if bd > best.0 {
                continue;
            }
            let nd = &self.tree.nodes[ni as usize];
            if nd.left == NONE {
                for i in nd.start as usize..(nd.start + nd.count) as usize {
                    let c = closest_pt_triangle(q, &self.tris[i]);
                    let d2 = dot(sub(q, c), sub(q, c));
                    if d2 < best.0 || best.2 == NONE && d2 <= best.0 {
                        *best = (d2, c, self.tree.order[i]);
                    }
                }
            } else {
                let (l, r) = (&self.tree.nodes[nd.left as usize], &self.tree.nodes[nd.right as usize]);
                let (dl, dr) = (box_d2(q, l.lo, l.hi), box_d2(q, r.lo, r.hi));
                // push the farther child first so the nearer one is popped first
                let (far, near) = if dl <= dr { ((nd.right, dr), (nd.left, dl)) } else { ((nd.left, dl), (nd.right, dr)) };
                stack[sp] = far;
                stack[sp + 1] = near;
                sp += 2;
            }
        }
    }

    pub fn closest(&self, q: V) -> Hit {
        let mut best = (f64::INFINITY, [f64::NAN; 3], NONE);
        self.search(q, &mut best);
        (best.0.sqrt(), best.1, best.2)
    }

    /// Closest hit within `max_dist` (inclusive), else None. Prunes with `max_dist` from the start.
    pub fn closest_within(&self, q: V, max_dist: f64) -> Option<Hit> {
        let mut best = (max_dist * max_dist, [f64::NAN; 3], NONE);
        self.search(q, &mut best);
        (best.2 != NONE).then(|| (best.0.sqrt(), best.1, best.2))
    }

    /// All intersections of the segment a->b with the soup as (t in [0,1], triangle id), sorted by t.
    /// Moller-Trumbore with a 1e-12 inclusive slack on the barycentric/segment limits so hits on shared
    /// triangle edges are not lost; hits closer than 1e-12 in t are merged (a crossing on a shared edge is
    /// reported once). Triangles (nearly) parallel to the segment are skipped.
    pub fn segment_hits(&self, a: [f64; 3], b: [f64; 3]) -> Vec<(f64, u32)> {
        let mut out: Vec<(f64, u32)> = Vec::new();
        if self.tree.nodes.is_empty() {
            return out;
        }
        const EPS: f64 = 1e-12;
        let d = sub(b, a);
        let dl = dot(d, d).sqrt();
        let mut stack = [0u32; 128];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let nd = &self.tree.nodes[stack[sp] as usize];
            // slab test of the segment against the node box
            let (mut t0, mut t1) = (0.0f64, 1.0f64);
            let mut miss = false;
            for k in 0..3 {
                let (lo, hi) = (nd.lo[k] - EPS, nd.hi[k] + EPS);
                if d[k] == 0.0 {
                    if a[k] < lo || a[k] > hi {
                        miss = true;
                        break;
                    }
                } else {
                    let (u, v) = ((lo - a[k]) / d[k], (hi - a[k]) / d[k]);
                    t0 = t0.max(u.min(v));
                    t1 = t1.min(u.max(v));
                    if t0 > t1 {
                        miss = true;
                        break;
                    }
                }
            }
            if miss {
                continue;
            }
            if nd.left == NONE {
                for i in nd.start as usize..(nd.start + nd.count) as usize {
                    let tr = &self.tris[i];
                    let (e1, e2) = (sub(tr[1], tr[0]), sub(tr[2], tr[0]));
                    let pv = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
                    let det = dot(e1, pv);
                    let nn = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                    if det.abs() <= 1e-14 * dl * dot(nn, nn).sqrt() || det == 0.0 {
                        continue;
                    }
                    let inv = 1.0 / det;
                    let tv = sub(a, tr[0]);
                    let u = dot(tv, pv) * inv;
                    if u < -EPS || u > 1.0 + EPS {
                        continue;
                    }
                    let qv = [tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]];
                    let v = dot(d, qv) * inv;
                    if v < -EPS || u + v > 1.0 + EPS {
                        continue;
                    }
                    let t = dot(e2, qv) * inv;
                    if t < -EPS || t > 1.0 + EPS {
                        continue;
                    }
                    out.push((t.clamp(0.0, 1.0), self.tree.order[i]));
                }
            } else {
                stack[sp] = nd.left;
                stack[sp + 1] = nd.right;
                sp += 2;
            }
        }
        out.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap().then(x.1.cmp(&y.1)));
        out.dedup_by(|y, x| (y.0 - x.0).abs() < EPS);
        out
    }

    pub fn closest_many(&self, qs: &[V]) -> Vec<Hit> {
        par_map(qs, |&q| self.closest(q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn matches_brute_force_on_sphere() {
        let (nu, nv) = (50usize, 20usize);
        let p = |i: usize, j: usize| {
            let (th, ph) = (PI * i as f64 / nv as f64, 2.0 * PI * (j % nu) as f64 / nu as f64);
            [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()]
        };
        let mut tris: Vec<Tri> = Vec::new();
        for i in 0..nv {
            for j in 0..nu {
                if i > 0 {
                    tris.push([p(i, j), p(i + 1, j), p(i, j + 1)]);
                }
                if i < nv - 1 {
                    tris.push([p(i, j + 1), p(i + 1, j), p(i + 1, j + 1)]);
                }
            }
        }
        tris.push([[2.0; 3], [2.0; 3], [2.0; 3]]); // degenerate triangle must not break anything
        let bvh = Bvh::new(&tris);
        let mut s = 99u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let qs: Vec<V> = (0..3000).map(|_| [0, 1, 2].map(|_| 4.0 * rnd() - 2.0)).collect();
        let hits = bvh.closest_many(&qs);
        for (q, h) in qs.iter().zip(&hits) {
            let bf = tris.iter().map(|t| { let c = closest_pt_triangle(*q, t); dot(sub(*q, c), sub(*q, c)).sqrt() }).fold(f64::INFINITY, f64::min);
            assert!((h.0 - bf).abs() <= 1e-12 * bf.max(1e-300), "q={q:?} bvh={} bf={bf}", h.0);
            let c = closest_pt_triangle(*q, &tris[h.2 as usize]);
            assert_eq!(c, h.1);
            match bvh.closest_within(*q, bf * 1.0000001) {
                Some(w) => assert_eq!(w.0, h.0),
                None => panic!("closest_within missed"),
            }
            assert!(bvh.closest_within(*q, bf * 0.99).is_none() || bf == 0.0);
        }
    }
}
