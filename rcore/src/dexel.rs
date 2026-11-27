//! Tri-dexel sampling (see crate docs). Ports `proto2d/dexel3d_v1.py` (CPU) with the v3 fallback rule of
//! `proto2d/dexel3d_warp_v3.py`: a node falls back to the exact winding number if ANY ray through it is
//! inconsistent or if consistent rays disagree.

use std::thread;

pub type Tri = [[f64; 3]; 3];

/// One recorded ray crossing.
#[derive(Clone, Copy, Debug)]
pub struct Crossing {
    pub ray: u32,
    pub s: f64,
    pub sign: i8,
    pub tri: u32,
}

/// Result of sampling one ray family (rays parallel to `axis`).
pub struct Family {
    pub axis: usize,
    /// crossings sorted by (ray, s)
    pub crossings: Vec<Crossing>,
    /// net winding of each ray is zero
    pub ok: Vec<bool>,
}

/// Grid node classification.
pub struct Classified {
    /// inside flag per node, index (ix * ny + iy) * nz + iz
    pub inside: Vec<bool>,
    /// node needs the exact winding-number fallback
    pub fallback: Vec<bool>,
    pub n_crossings: usize,
}

fn nthreads() -> usize {
    thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// first index i with g[i] >= v (g ascending, uniform spacing h)
fn ceil_idx(g: &[f64], v: f64) -> i64 {
    let h = g[1] - g[0];
    let mut i = ((v - g[0]) / h).ceil() as i64;
    // guard floating error so that g[i] >= v and g[i-1] < v
    while i > 0 && (i - 1) < g.len() as i64 && g[(i - 1) as usize] >= v {
        i -= 1;
    }
    while i < g.len() as i64 && i >= 0 && g[i as usize] < v {
        i += 1;
    }
    i
}

/// last index i with g[i] <= v
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

/// All crossings of rays parallel to `axis` with the soup.
pub fn sample_family(tris: &[Tri], g: [&[f64]; 3], axis: usize) -> Family {
    let (o1, o2) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let (g1, g2) = (g[o1], g[o2]);
    let n2 = g2.len();
    let nray = g1.len() * n2;
    let nt = nthreads();
    let chunk = (tris.len() + nt - 1) / nt;
    let mut parts: Vec<Vec<Crossing>> = Vec::new();
    thread::scope(|sc| {
        let handles: Vec<_> = tris
            .chunks(chunk.max(1))
            .enumerate()
            .map(|(ci, tc)| {
                sc.spawn(move || {
                    let mut out = Vec::new();
                    for (k, t) in tc.iter().enumerate() {
                        let tid = (ci * chunk + k) as u32;
                        let (a, b, c) = (t[0], t[1], t[2]);
                        let min1 = a[o1].min(b[o1]).min(c[o1]);
                        let max1 = a[o1].max(b[o1]).max(c[o1]);
                        let min2 = a[o2].min(b[o2]).min(c[o2]);
                        let max2 = a[o2].max(b[o2]).max(c[o2]);
                        let i0 = ceil_idx(g1, min1).max(0);
                        let i1 = floor_idx(g1, max1).min(g1.len() as i64 - 1);
                        let j0 = ceil_idx(g2, min2).max(0);
                        let j1 = floor_idx(g2, max2).min(n2 as i64 - 1);
                        if i0 > i1 || j0 > j1 {
                            continue;
                        }
                        // projected normal component (sign tells entering/leaving along +axis)
                        let nrm = (b[o1] - a[o1]) * (c[o2] - a[o2]) - (b[o2] - a[o2]) * (c[o1] - a[o1]);
                        let nrm_axis = match axis {
                            // cross(B-A, C-A)[axis] with (o1, o2) cyclic or anti-cyclic
                            1 => -nrm,
                            _ => nrm,
                        };
                        let sign: i8 = if nrm_axis > 0.0 { -1 } else { 1 };
                        for i in i0..=i1 {
                            let x = g1[i as usize];
                            for j in j0..=j1 {
                                let y = g2[j as usize];
                                let ef = |p: [f64; 3], q: [f64; 3]| {
                                    (q[o1] - p[o1]) * (y - p[o2]) - (q[o2] - p[o2]) * (x - p[o1])
                                };
                                let (e0, e1, e2) = (ef(a, b), ef(b, c), ef(c, a));
                                let inside = (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);
                                let area = e0 + e1 + e2;
                                if !inside || area == 0.0 {
                                    continue;
                                }
                                let s = (e1 * a[axis] + e2 * b[axis] + e0 * c[axis]) / area;
                                out.push(Crossing { ray: (i as usize * n2 + j as usize) as u32, s, sign, tri: tid });
                            }
                        }
                    }
                    out
                })
            })
            .collect();
        for h in handles {
            parts.push(h.join().unwrap());
        }
    });
    let mut crossings: Vec<Crossing> = parts.concat();
    crossings.sort_unstable_by(|p, q| p.ray.cmp(&q.ray).then(p.s.partial_cmp(&q.s).unwrap()));
    let mut net = vec![0i32; nray];
    for c in &crossings {
        net[c.ray as usize] += c.sign as i32;
    }
    Family { axis, crossings, ok: net.iter().map(|&w| w == 0).collect() }
}

/// Per-node 1D winding along one family, written in global node order (ix, iy, iz).
fn node_winding(f: &Family, g: [&[f64]; 3], w_out: &mut [i8], ok_out: &mut [bool]) {
    let n = [g[0].len(), g[1].len(), g[2].len()];
    let (o1, o2) = match f.axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let ga = g[f.axis];
    let n2 = n[o2];
    let stride = |d: usize| match d {
        0 => n[1] * n[2],
        1 => n[2],
        _ => 1,
    };
    let (s1, s2, sa) = (stride(o1), stride(o2), stride(f.axis));
    // ray start offsets into the sorted crossing list
    let nray = n[o1] * n2;
    let mut start = vec![0usize; nray + 1];
    for c in &f.crossings {
        start[c.ray as usize + 1] += 1;
    }
    for r in 0..nray {
        start[r + 1] += start[r];
    }
    // rays are independent: split the ray range across threads; every thread writes disjoint nodes
    let nt = nthreads();
    let per = (nray + nt - 1) / nt;
    let wptr = w_out.as_mut_ptr() as usize;
    let optr = ok_out.as_mut_ptr() as usize;
    thread::scope(|sc| {
        for t in 0..nt {
            let (lo, hi) = (t * per, ((t + 1) * per).min(nray));
            let start = &start;
            sc.spawn(move || {
                // SAFETY: each ray owns a disjoint set of node indices (fixed (i, j), all k)
                let w = wptr as *mut i8;
                let ok = optr as *mut bool;
                for r in lo..hi {
                    let (i, j) = (r / n2, r % n2);
                    let base = i * s1 + j * s2;
                    let cs = &f.crossings[start[r]..start[r + 1]];
                    let rok = f.ok[r];
                    let mut acc: i32 = 0;
                    let mut p = 0usize;
                    for (k, &z) in ga.iter().enumerate() {
                        while p < cs.len() && cs[p].s < z {
                            acc += cs[p].sign as i32;
                            p += 1;
                        }
                        unsafe {
                            *w.add(base + k * sa) = acc.clamp(-100, 100) as i8;
                            *ok.add(base + k * sa) = rok;
                        }
                    }
                }
            });
        }
    });
}

/// Full tri-dexel classification (fallback nodes are flagged, not resolved).
pub fn classify(tris: &[Tri], g: [&[f64]; 3]) -> Classified {
    let nn = g[0].len() * g[1].len() * g[2].len();
    let mut w = [vec![0i8; nn], vec![0i8; nn], vec![0i8; nn]];
    let mut ok = [vec![false; nn], vec![false; nn], vec![false; nn]];
    let mut ncross = 0;
    for ax in 0..3 {
        let f = sample_family(tris, g, ax);
        ncross += f.crossings.len();
        node_winding(&f, g, &mut w[ax], &mut ok[ax]);
    }
    let mut inside = vec![false; nn];
    let mut fallback = vec![false; nn];
    for q in 0..nn {
        let ins = [w[0][q] != 0, w[1][q] != 0, w[2][q] != 0];
        let o = [ok[0][q], ok[1][q], ok[2][q]];
        inside[q] = if o[2] { ins[2] } else if o[0] { ins[0] } else { ins[1] };
        let disagree = (o[0] && o[2] && ins[0] != ins[2]) || (o[1] && o[2] && ins[1] != ins[2]) || (o[0] && o[1] && ins[0] != ins[1]);
        fallback[q] = !(o[0] && o[1] && o[2]) || disagree;
    }
    Classified { inside, fallback, n_crossings: ncross }
}

/// Exact generalized winding number (sum of signed solid angles / 4 pi), Van Oosterom-Strackee.
pub fn gwn(tris: &[Tri], p: [f64; 3]) -> f64 {
    let mut s = 0.0;
    for t in tris {
        let a = [t[0][0] - p[0], t[0][1] - p[1], t[0][2] - p[2]];
        let b = [t[1][0] - p[0], t[1][1] - p[1], t[1][2] - p[2]];
        let c = [t[2][0] - p[0], t[2][1] - p[1], t[2][2] - p[2]];
        let n = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let d = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let (la, lb, lc) = (n(a), n(b), n(c));
        let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
        let den = la * lb * lc + d(a, b) * lc + d(b, c) * la + d(c, a) * lb;
        s += 2.0 * det.atan2(den);
    }
    s / (4.0 * std::f64::consts::PI)
}

/// Resolve fallback nodes with the exact winding number (parallel brute force; tree method later).
pub fn resolve_fallback(tris: &[Tri], g: [&[f64]; 3], c: &mut Classified) -> usize {
    let n = [g[0].len(), g[1].len(), g[2].len()];
    let idx: Vec<usize> = (0..c.fallback.len()).filter(|&q| c.fallback[q]).collect();
    let nt = nthreads();
    let per = (idx.len() + nt - 1) / nt.max(1);
    let mut res = vec![false; idx.len()];
    thread::scope(|sc| {
        for (chunk_i, out) in res.chunks_mut(per.max(1)).enumerate() {
            let idx = &idx;
            sc.spawn(move || {
                for (k, o) in out.iter_mut().enumerate() {
                    let q = idx[chunk_i * per.max(1) + k];
                    let (ix, r) = (q / (n[1] * n[2]), q % (n[1] * n[2]));
                    let (iy, iz) = (r / n[2], r % n[2]);
                    *o = gwn(tris, [g[0][ix], g[1][iy], g[2][iz]]).abs() > 0.5;
                }
            });
        }
    });
    for (k, &q) in idx.iter().enumerate() {
        c.inside[q] = res[k];
    }
    idx.len()
}

/// Like `resolve_fallback` but with the fast winding number (`fwn::Fwn`, BVH built once).
pub fn resolve_fallback_fast(tris: &[Tri], g: [&[f64]; 3], c: &mut Classified) -> usize {
    let n = [g[0].len(), g[1].len(), g[2].len()];
    let idx: Vec<usize> = (0..c.fallback.len()).filter(|&q| c.fallback[q]).collect();
    let f = crate::fwn::Fwn::new(tris);
    let res = crate::tree::par_map(&idx, |&q| {
        let (ix, r) = (q / (n[1] * n[2]), q % (n[1] * n[2]));
        let (iy, iz) = (r / n[2], r % n[2]);
        f.winding([g[0][ix], g[1][iy], g[2][iz]]).abs() > 0.5
    });
    for (k, &q) in idx.iter().enumerate() {
        c.inside[q] = res[k];
    }
    idx.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> Vec<Tri> {
        // unit cube [0,1]^3, outward normals
        let v = |x: f64, y: f64, z: f64| [x, y, z];
        let q = |a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]| vec![[a, b, c], [a, c, d]];
        let mut t = Vec::new();
        t.extend(q(v(0., 0., 0.), v(0., 1., 0.), v(1., 1., 0.), v(1., 0., 0.)));
        t.extend(q(v(0., 0., 1.), v(1., 0., 1.), v(1., 1., 1.), v(0., 1., 1.)));
        t.extend(q(v(0., 0., 0.), v(1., 0., 0.), v(1., 0., 1.), v(0., 0., 1.)));
        t.extend(q(v(0., 1., 0.), v(0., 1., 1.), v(1., 1., 1.), v(1., 1., 0.)));
        t.extend(q(v(0., 0., 0.), v(0., 0., 1.), v(0., 1., 1.), v(0., 1., 0.)));
        t.extend(q(v(1., 0., 0.), v(1., 1., 0.), v(1., 1., 1.), v(1., 0., 1.)));
        t
    }

    #[test]
    fn cube_inside_matches_gwn() {
        let tris = cube();
        let g: Vec<f64> = (0..23).map(|i| -0.5 + 0.1 * i as f64 + 1e-7 * std::f64::consts::PI).collect();
        let c = classify(&tris, [&g, &g, &g]);
        assert!(c.fallback.iter().all(|&f| !f), "closed cube: no fallback expected");
        let n = g.len();
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let q = (ix * n + iy) * n + iz;
                    let exact = gwn(&tris, [g[ix], g[iy], g[iz]]).abs() > 0.5;
                    assert_eq!(c.inside[q], exact, "node {ix},{iy},{iz}");
                }
            }
        }
    }
}
