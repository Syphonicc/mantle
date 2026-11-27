//! Fast generalized winding number (Barill et al. 2018, "Fast Winding Numbers for Soups and Clouds").
//! A BVH stores, per node, the area-weighted centre `p`, the radius `r` (max vertex distance from `p`),
//! the dipole `a = sum area*n`, `B_ij = sum area*n_i*(c_t - p)_j` and the second-moment tensor
//! `C_ijk = sum area*n_i*<s_j s_k>` (s = y - p, <.> = per-triangle area average, exact per triangle).
//! With F(y) = (y-q)/|y-q|^3, d = p - q, w = 1/(4 pi) sum int n.F dA and the Taylor series of F about p:
//!   w ~ 1/(4 pi) [ a_i F_i + B_ij dF_i/dj + 1/2 C_ijk d2F_i/djdk ],  dF_i/dj = delta_ij/|d|^3 - 3 d_i d_j/|d|^5,
//!   d2F_i/djdk = 15 d_i d_j d_k/|d|^7 - 3 (delta_ij d_k + delta_ik d_j + delta_jk d_i)/|d|^5. Truncation error ~ |d|^-5.
//! Dropping the C term (error ~ d^-4, i.e. B-only 'second order') gave ~6e-2 at beta = 2 on a sphere; the third gives ~3e-3.
//! Leaves sum exact solid angles. Sign: +1 inside a closed outward-oriented surface (same as `dexel::gwn`).

use crate::dexel::Tri;
use crate::tree::{par_map, Tree, NONE};
use std::f64::consts::PI;

#[derive(Clone, Copy)]
#[derive(Default)]
struct Moment {
    p: [f64; 3],
    r: f64,
    a: [f64; 3],
    b: [[f64; 3]; 3],
    /// C_ijk = sum av_i * (int s_j s_k dA / area), s = y - p (symmetric in j,k)
    c: [[[f64; 3]; 3]; 3],
}

pub struct Fwn {
    tree: Tree,
    /// triangles in tree order
    tris: Vec<Tri>,
    mom: Vec<Moment>,
    pub beta: f64,
}

/// Signed solid angle / (2 * 4 pi) * 2 -> contribution of one triangle to w (Van Oosterom-Strackee).
fn tri_w(t: &Tri, p: [f64; 3]) -> f64 {
    let sub = |v: [f64; 3]| [v[0] - p[0], v[1] - p[1], v[2] - p[2]];
    let (a, b, c) = (sub(t[0]), sub(t[1]), sub(t[2]));
    let n = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let d = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let (la, lb, lc) = (n(a), n(b), n(c));
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    let den = la * lb * lc + d(a, b) * lc + d(b, c) * la + d(c, a) * lb;
    2.0 * det.atan2(den)
}

/// Far-field contribution (already divided by 4 pi) of a node with moments `a`, `b`, at `d = p - q`.
/// `order` 1 keeps only the dipole term, 2 adds `b`, 3 adds the second-moment tensor `c`. Derivation: w = 1/(4 pi) sum int n.F dA with F(y) = (y-q)/|y-q|^3;
/// F_i(p+s) = F_i(d) + s_j dF_i/dj + O(s^2), dF_i/dj = delta_ij/|d|^3 - 3 d_i d_j/|d|^5, and
/// int n_i s_j dA = B_ij (n constant per triangle), so the linear term is B_ij dF_i/dj.
pub(crate) fn far_field(a: &[f64; 3], b: &[[f64; 3]; 3], c: &[[[f64; 3]; 3]; 3], d: [f64; 3], order: u8) -> f64 {
    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    let inv = 1.0 / d2.sqrt();
    let (i3, i5) = (inv.powi(3), inv.powi(5));
    let mut f = (a[0] * d[0] + a[1] * d[1] + a[2] * d[2]) * i3;
    if order >= 2 {
        for i in 0..3 {
            for j in 0..3 {
                f -= b[i][j] * (3.0 * d[i] * d[j] * i5 - if i == j { i3 } else { 0.0 });
            }
        }
    }
    if order >= 3 {
        // 1/2 C_ijk d_j d_k F_i, d_j d_k F_i = 15 d_i d_j d_k/|d|^7 - 3 (delta_ij d_k + delta_ik d_j + delta_jk d_i)/|d|^5
        let i7 = i5 * inv * inv;
        let mut t = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let dl = |x: usize, y: usize| if x == y { 1.0 } else { 0.0 };
                    t += c[i][j][k] * (15.0 * d[i] * d[j] * d[k] * i7 - 3.0 * (dl(i, j) * d[k] + dl(i, k) * d[j] + dl(j, k) * d[i]) * i5);
                }
            }
        }
        f += 0.5 * t;
    }
    f / (4.0 * PI)
}

impl Fwn {
    pub fn new(tris: &[Tri]) -> Fwn {
        Self::with_beta(tris, 2.0)
    }

    pub fn with_beta(tris: &[Tri], beta: f64) -> Fwn {
        let tree = Tree::build(tris);
        let sorted: Vec<Tri> = tree.order.iter().map(|&i| tris[i as usize]).collect();
        // per triangle: area vector, centroid
        let av: Vec<[f64; 3]> = sorted
            .iter()
            .map(|t| {
                let (u, v) = ([0, 1, 2].map(|k| t[1][k] - t[0][k]), [0, 1, 2].map(|k| t[2][k] - t[0][k]));
                [0.5 * (u[1] * v[2] - u[2] * v[1]), 0.5 * (u[2] * v[0] - u[0] * v[2]), 0.5 * (u[0] * v[1] - u[1] * v[0])]
            })
            .collect();
        let ce: Vec<[f64; 3]> = sorted.iter().map(|t| [0, 1, 2].map(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0)).collect();
        let ids: Vec<usize> = (0..tree.nodes.len()).collect();
        let mom = par_map(&ids, |&ni| {
            let nd = &tree.nodes[ni];
            let rg = nd.start as usize..(nd.start + nd.count) as usize;
            let (mut aw, mut pc, mut a) = (0.0, [0.0; 3], [0.0; 3]);
            for i in rg.clone() {
                let ar = (av[i][0].powi(2) + av[i][1].powi(2) + av[i][2].powi(2)).sqrt();
                aw += ar;
                for k in 0..3 {
                    pc[k] += ar * ce[i][k];
                    a[k] += av[i][k];
                }
            }
            let p = if aw > 0.0 {
                pc.map(|x| x / aw)
            } else {
                [0, 1, 2].map(|k| 0.5 * (nd.lo[k] + nd.hi[k])) // all-degenerate node: box centre
            };
            let (mut r2, mut b, mut c) = (0.0f64, [[0.0; 3]; 3], [[[0.0; 3]; 3]; 3]);
            for i in rg {
                for v in &sorted[i] {
                    r2 = r2.max((0..3).map(|k| (v[k] - p[k]).powi(2)).sum());
                }
                for ii in 0..3 {
                    for j in 0..3 {
                        b[ii][j] += av[i][ii] * (ce[i][j] - p[j]);
                    }
                }
                // second moment about p per unit area: (c-p)(c-p)^T + 1/12 sum_k e_k e_k^T, e_k = v_k - c
                let mut sm = [[0.0; 3]; 3];
                for j in 0..3 {
                    for k in 0..3 {
                        sm[j][k] = (ce[i][j] - p[j]) * (ce[i][k] - p[k])
                            + sorted[i].iter().map(|v| (v[j] - ce[i][j]) * (v[k] - ce[i][k])).sum::<f64>() / 12.0;
                    }
                }
                for ii in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            c[ii][j][k] += av[i][ii] * sm[j][k];
                        }
                    }
                }
            }
            Moment { p, r: r2.sqrt(), a, b, c }
        });
        Fwn { tree, tris: sorted, mom, beta }
    }

    pub fn winding(&self, q: [f64; 3]) -> f64 {
        if self.tree.nodes.is_empty() {
            return 0.0;
        }
        let mut s = 0.0;
        let mut stack = [0u32; 128];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let ni = stack[sp] as usize;
            let (nd, m) = (&self.tree.nodes[ni], &self.mom[ni]);
            let d = [m.p[0] - q[0], m.p[1] - q[1], m.p[2] - q[2]];
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 > (self.beta * m.r).powi(2) {
                s += far_field(&m.a, &m.b, &m.c, d, 3);
            } else if nd.left == NONE {
                let rg = nd.start as usize..(nd.start + nd.count) as usize;
                s += self.tris[rg].iter().map(|t| tri_w(t, q)).sum::<f64>() / (4.0 * PI);
            } else {
                stack[sp] = nd.left;
                stack[sp + 1] = nd.right;
                sp += 2;
            }
        }
        s
    }

    pub fn winding_many(&self, qs: &[[f64; 3]]) -> Vec<f64> {
        par_map(qs, |&q| self.winding(q))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dexel::gwn;

    fn cube() -> Vec<Tri> {
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

    /// outward-oriented UV sphere of radius 1 centred at the origin
    fn sphere(nu: usize, nv: usize) -> Vec<Tri> {
        let p = |i: usize, j: usize| {
            let (th, ph) = (PI * i as f64 / nv as f64, 2.0 * PI * (j % nu) as f64 / nu as f64);
            [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()]
        };
        let mut t = Vec::new();
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

    fn lcg(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    }

    /// node moments of a cluster, as built in `with_beta`
    fn moments(tris: &[Tri]) -> ([f64; 3], [f64; 3], [[f64; 3]; 3], [[[f64; 3]; 3]; 3]) {
        let f = Fwn::with_beta(tris, 2.0);
        let m = f.mom[0]; // root node holds every triangle
        (m.p, m.a, m.b, m.c)
    }

    #[test]
    fn far_field_convergence_order() {
        let mut s = 777u64;
        let mut tris: Vec<Tri> = Vec::new();
        for _ in 0..20 {
            let c = [0, 1, 2].map(|_| 0.5 * (2.0 * lcg(&mut s) - 1.0));
            tris.push([0, 1, 2].map(|_| [0, 1, 2].map(|k| c[k] + 0.3 * (2.0 * lcg(&mut s) - 1.0))));
        }
        let leaf = Fwn::with_beta(&tris, 2.0);
        assert!(leaf.mom[0].r < 1.0);
        let (p, a, b, c) = moments(&tris);
        let dirs: [[f64; 3]; 5] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, -0.5, 0.8], [-0.6, 0.2, -0.4], [0.5, 0.5, 0.5]];
        let mut err = [[0.0f64; 4]; 3];
        for (di, dist) in [4.0, 8.0, 16.0, 32.0].iter().enumerate() {
            for u in &dirs {
                let un = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                let q = [0, 1, 2].map(|k| p[k] + dist * u[k] / un);
                let we = gwn(&tris, q);
                let d = [0, 1, 2].map(|k| p[k] - q[k]);
                for o in 0..3 {
                    err[o][di] = err[o][di].max((far_field(&a, &b, &c, d, o as u8 + 1) - we).abs());
                }
            }
        }
        for o in 0..3 {
            for i in 0..3 {
                let e = (err[o][i] / err[o][i + 1]).log2();
                eprintln!("order {} d={}->{}: exponent {e:.2}", o + 1, 4 << i, 8 << i);
                if o == 1 {
                    assert!(e >= 3.6, "second-order exponent {e}");
                }
            }
        }
        assert!(err[1][0] < err[0][0] * 0.5);
        assert!(err[2][3] < err[1][3]); // third order is better still
    }

    #[test]
    fn cube_matches_exact() {
        let tris = cube();
        let f = Fwn::new(&tris);
        let g: Vec<f64> = (0..23).map(|i| -0.5 + 0.1 * i as f64 + 1e-7 * PI).collect();
        let n = g.len();
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let q = [g[ix], g[iy], g[iz]];
                    assert_eq!(f.winding(q).abs() > 0.5, gwn(&tris, q).abs() > 0.5, "node {ix},{iy},{iz}");
                }
            }
        }
    }

    fn check(tris: &[Tri], surf_dist_ok: impl Fn([f64; 3]) -> bool) {
        let f = Fwn::new(tris); // default beta = 2: classification
        let f3 = Fwn::with_beta(tris, 2.0); // beta = 2 with third-order terms: |dw| < 5e-3
        let mut s = 12345u64;
        let qs: Vec<[f64; 3]> = (0..2000).map(|_| [0, 1, 2].map(|_| 3.0 * lcg(&mut s) - 1.5)).collect();
        let fast = f.winding_many(&qs);
        let fast3 = f3.winding_many(&qs);
        let (mut emax, mut n) = (0.0f64, 0);
        for ((q, &wf), &w3) in qs.iter().zip(&fast).zip(&fast3) {
            let we = gwn(tris, *q);
            if surf_dist_ok(*q) {
                emax = emax.max((wf - we).abs());
                assert!((w3 - we).abs() < 5e-3, "q={q:?} fast(beta=2)={w3} exact={we}");
                n += 1;
            }
            if (we - 0.5).abs() > 0.1 && surf_dist_ok(*q) {
                assert_eq!(wf.abs() > 0.5, we.abs() > 0.5, "class q={q:?} fast={wf} exact={we}");
            }
        }
        assert!(n > 500);
        eprintln!("max |w_fast - w_exact| (beta=2) over {n} pts = {emax:e}");
    }

    #[test]
    fn sphere_closed_and_holed() {
        let tris = sphere(52, 20); // ~1976 triangles
        assert!(tris.len() > 1900);
        let far = |q: [f64; 3]| ((q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() - 1.0).abs() > 0.05;
        assert!((gwn(&tris, [0.0; 3]) - 1.0).abs() < 1e-9, "sphere must be outward oriented");
        check(&tris, far);
        let mut holed = tris.clone();
        holed.drain(300..320); // 20 adjacent-ish triangles removed
        let hc: Vec<[f64; 3]> = tris[300..320].iter().flat_map(|t| t.iter().copied()).collect();
        // exclude points within 0.05 of the hole's vertices as well as of the surface
        let far_h = move |q: [f64; 3]| far(q) && hc.iter().all(|v| (0..3).map(|k| (q[k] - v[k]).powi(2)).sum::<f64>() > 0.05f64.powi(2));
        check(&holed, far_h);
    }
}
