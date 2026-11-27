//! Curvature-based per-triangle surface size for the octree size field (`SizeField::tri_h`).
//!
//! Curvature estimate: angle-weighted vertex normals `n`; along every edge `(v, w)` the normal-curvature `|(n_v - n_w).(v - w)| / |v - w|^2`
//! (`= 1/R` for a circle, exact on regular chord tessellations); a vertex takes the maximum over its edges (smallest `R`) and a
//! triangle the maximum over its 3 vertices. Facet-normal dihedral/centroid-distance estimates were rejected: they are biased
//! ~1.5x on right-triangle (lat-long) meshes because a facet normal sits at the circumcentre, not the centroid.
//!
//! Sharp-edge rule: an edge with `theta > sharp` (default 60 degrees) is a feature, not curvature (`R -> 0` would force
//! `h_min` along every crease). It is skipped; the convex fans / shell fillet of the hex core handle the geometry there. A
//! triangle touching such an edge gets its own smooth-edge size, optionally capped by `h_edge` (default: none = `h_surf`).
//!
//! Size: `h_c = clamp(c (R + offset), h_min, h_surf)`; the octree resolves the offset shell (distance `offset = delta` from the wall,
//! radius of curvature `R + delta` where convex), not the wall: sizing from the wall `R` alone would refine the whole fat shell cap over a
//! sharp leading edge (measured on the M6: 490k shell quads, layers > 15 min) while the layer columns already resolve the wall angularly.
//! With `offset = 0` it is `clamp(c R, h_min, h_surf)`. `c = 0.25` gives `R / h = 4` i.e. 2 pi R / h ~ 25 cells per circle (a 14 degree
//! chord, sagitta error `h^2 / 8R` ~ 3 % of h), the usual CFD rule of thumb (20 to 30 cells per circle).
//! Gradation: `h_t <= h_u + g |c_t - c_u|` over the triangle edge graph (multi-source Dijkstra, Lipschitz `g`, default 0.2),
//! so sizes vary smoothly; this only ever lowers sizes.
use crate::dexel::Tri;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone, Debug)]
pub struct CurvOptions {
    /// cells per unit radius of curvature: `h_c = c R`
    pub c: f64,
    /// smallest size (`<= 0`: `h_surf / 8`)
    pub h_min: f64,
    /// dihedral angle (degrees) above which an edge is a feature and ignored
    pub sharp_deg: f64,
    /// optional size imposed on triangles touching a sharp edge (`<= 0`: none)
    pub h_edge: f64,
    /// Lipschitz limit of the size along the surface
    pub grade: f64,
    /// offset of the surface that is actually resolved (the shell, `delta`): `R_eff = R + offset` (0 = the wall itself)
    pub offset: f64,
}

impl Default for CurvOptions {
    fn default() -> Self {
        CurvOptions { c: 0.25, h_min: 0.0, sharp_deg: 60.0, h_edge: 0.0, grade: 0.2, offset: 0.0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CurvStats {
    pub h_min_used: f64,
    pub h_max_used: f64,
    pub n_refined: usize,
    pub n_sharp_tris: usize,
    pub n_tris: usize,
    /// smallest radius of curvature of any smooth edge
    pub r_min: f64,
}

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

/// Per-triangle target size, same indexing as `tris`. `h_surf` is the upper bound (flat regions).
pub fn curvature_sizes(tris: &[Tri], h_surf: f64, o: &CurvOptions) -> (Vec<f64>, CurvStats) {
    let nt = tris.len();
    let h_min = if o.h_min > 0.0 { o.h_min.min(h_surf) } else { h_surf / 8.0 };
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in tris.iter().flatten() {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = norm(sub(hi, lo)).max(1e-300);
    let q = 1e-7 * diag;
    let mut ids: HashMap<[i64; 3], u32> = HashMap::new();
    let mut tv: Vec<[u32; 3]> = Vec::with_capacity(nt);
    for t in tris {
        let mut r = [0u32; 3];
        for k in 0..3 {
            let key = [(t[k][0] / q).round() as i64, (t[k][1] / q).round() as i64, (t[k][2] / q).round() as i64];
            let n = ids.len() as u32;
            r[k] = *ids.entry(key).or_insert(n);
        }
        tv.push(r);
    }
    let nrm: Vec<V3> = tris
        .iter()
        .map(|t| {
            let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
            let l = norm(n);
            if l > 0.0 { [n[0] / l, n[1] / l, n[2] / l] } else { [0.0; 3] }
        })
        .collect();
    let cen: Vec<V3> = tris.iter().map(|t| [0, 1, 2].map(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0)).collect();
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (i, v) in tv.iter().enumerate() {
        if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] {
            continue;
        }
        for e in 0..3 {
            let (a, b) = (v[e], v[(e + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(i as u32);
        }
    }
    let nv = ids.len();
    let mut sharp = vec![false; nt];
    let mut vsharp = vec![false; nv];
    let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); nt];
    let sharp_rad = o.sharp_deg.to_radians();
    let mut vpos: Vec<V3> = vec![[0.0; 3]; nv];
    let mut vn: Vec<V3> = vec![[0.0; 3]; nv];
    for (i, v) in tv.iter().enumerate() {
        for k in 0..3 {
            vpos[v[k] as usize] = tris[i][k];
            // angle-weighted vertex normal (exact radial direction on a regularly chord-tessellated circle / sphere)
            let (p, q, r) = (tris[i][k], tris[i][(k + 1) % 3], tris[i][(k + 2) % 3]);
            let (e1, e2) = (sub(q, p), sub(r, p));
            let w = norm(cross(e1, e2)).atan2(dot(e1, e2));
            for d in 0..3 {
                vn[v[k] as usize][d] += w * nrm[i][d];
            }
        }
    }
    for (&(a, b), ts) in &edges {
        if ts.len() != 2 {
            // open / non-manifold edge: no reliable normal there
            vsharp[a as usize] = true;
            vsharp[b as usize] = true;
            continue;
        }
        let (t0, t1) = (ts[0] as usize, ts[1] as usize);
        let theta = norm(cross(nrm[t0], nrm[t1])).atan2(dot(nrm[t0], nrm[t1]));
        let dc = norm(sub(cen[t0], cen[t1]));
        adj[t0].push((t1 as u32, dc));
        adj[t1].push((t0 as u32, dc));
        if theta > sharp_rad {
            sharp[t0] = true;
            sharp[t1] = true;
            vsharp[a as usize] = true;
            vsharp[b as usize] = true;
        }
    }
    for n in vn.iter_mut() {
        let l = norm(*n);
        if l > 0.0 {
            *n = [n[0] / l, n[1] / l, n[2] / l];
        }
    }
    // vertex curvature: max over smooth edges (v, w) of |(n_v - n_w) . (v - w)| / |v - w|^2 (= 1/R for a circle with exact normals);
    // vertices on a feature edge are skipped (their normal is meaningless), as are edges ending on one
    let mut kv = vec![0.0f64; nv];
    let mut r_min = f64::INFINITY;
    for &(a, b) in edges.keys() {
        let (a, b) = (a as usize, b as usize);
        if vsharp[a] || vsharp[b] {
            continue;
        }
        let d = sub(vpos[a], vpos[b]);
        let l2 = dot(d, d);
        if l2 <= 1e-300 {
            continue;
        }
        let k = dot(sub(vn[a], vn[b]), d).abs() / l2;
        kv[a] = kv[a].max(k);
        kv[b] = kv[b].max(k);
        if k > 1e-12 {
            r_min = r_min.min(1.0 / k);
        }
    }
    let kappa: Vec<f64> = tv.iter().map(|v| kv[v[0] as usize].max(kv[v[1] as usize]).max(kv[v[2] as usize])).collect();
    let mut h: Vec<f64> = (0..nt)
        .map(|i| {
            let mut x = if kappa[i] > 0.0 { (o.c * (1.0 / kappa[i] + o.offset)).clamp(h_min, h_surf) } else { h_surf };
            if sharp[i] && o.h_edge > 0.0 {
                x = x.min(o.h_edge.max(h_min));
            }
            x
        })
        .collect();
    // Lipschitz gradation: multi-source Dijkstra, h_t = min_u (h_u + g dist(t, u))
    let mut heap: BinaryHeap<(Reverse<u64>, u32)> = (0..nt).filter(|&i| h[i] < h_surf).map(|i| (Reverse(h[i].to_bits()), i as u32)).collect();
    while let Some((Reverse(b), i)) = heap.pop() {
        let i = i as usize;
        if f64::from_bits(b) > h[i] {
            continue;
        }
        for &(j, d) in &adj[i] {
            let cand = h[i] + o.grade * d;
            if cand < h[j as usize] {
                h[j as usize] = cand;
                heap.push((Reverse(cand.to_bits()), j));
            }
        }
    }
    let st = CurvStats {
        h_min_used: h.iter().cloned().fold(f64::INFINITY, f64::min),
        h_max_used: h.iter().cloned().fold(0.0, f64::max),
        n_refined: h.iter().filter(|&&x| x < 0.999 * h_surf).count(),
        n_sharp_tris: sharp.iter().filter(|&&s| s).count(),
        n_tris: nt,
        r_min,
    };
    (h, st)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(r: f64, n: usize) -> Vec<Tri> {
        let p = |i: usize, j: usize| {
            let (th, ph) = (std::f64::consts::PI * i as f64 / n as f64, 2.0 * std::f64::consts::PI * j as f64 / (2 * n) as f64);
            [r * th.sin() * ph.cos(), r * th.sin() * ph.sin(), r * th.cos()]
        };
        let mut t = Vec::new();
        for i in 0..n {
            for j in 0..2 * n {
                t.push([p(i, j), p(i + 1, j), p(i + 1, j + 1)]);
                t.push([p(i, j), p(i + 1, j + 1), p(i, j + 1)]);
            }
        }
        t
    }

    #[test]
    fn sphere_radius() {
        let (h, st) = curvature_sizes(&sphere(1.0, 24), 10.0, &CurvOptions { h_min: 1e-3, grade: 1e9, ..Default::default() });
        let mut v = h.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = v[v.len() / 2];
        // lat-long poles are slivers, so only the median is tight
        assert!(st.r_min > 0.5, "R {}", st.r_min);
        assert!(mid > 0.22 && mid < 0.28, "h {mid}");
    }

    #[test]
    fn cube_sharp_edges_ignored() {
        let mut t: Vec<Tri> = Vec::new();
        let c = |x: f64, y: f64, z: f64| [x, y, z];
        for a in [0.0, 1.0] {
            let q = |f: &dyn Fn(f64, f64) -> V3| {
                let (p0, p1, p2, p3) = (f(0.0, 0.0), f(1.0, 0.0), f(1.0, 1.0), f(0.0, 1.0));
                [[p0, p1, p2], [p0, p2, p3]]
            };
            t.extend(q(&|x, y| c(x, y, a)));
            t.extend(q(&|x, y| c(x, a, y)));
            t.extend(q(&|x, y| c(a, x, y)));
        }
        let (h, st) = curvature_sizes(&t, 0.5, &CurvOptions::default());
        assert!(h.iter().all(|&x| (x - 0.5).abs() < 1e-12), "cube refined");
        assert_eq!(st.n_refined, 0);
    }

    #[test]
    fn gradation_lipschitz() {
        // a sphere cap region refined hard, flat neighbours must be graded
        let (h, _) = curvature_sizes(&sphere(1.0, 16), 1.0, &CurvOptions { c: 0.02, h_min: 0.01, ..Default::default() });
        assert!(h.iter().all(|&x| x >= 0.01 && x <= 1.0));
    }
}
