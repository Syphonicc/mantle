//! OpenFOAM-equivalent mesh metrics on a `CutMesh`, computed without OpenFOAM (definitions of `primitiveMesh` /
//! `primitiveMeshTools` as used by `checkMesh`): face centres / areas (triangle fan about the point average), cell centres and
//! volumes (face-pyramid decomposition about the face-centre average), non-orthogonality, face skewness (internal and boundary
//! variants), face-pyramid orientation, non-positive volumes, cell aspect ratio, closed-cell residual, plus the corner Jacobian
//! of hexahedra. Face vertex order is outward from `Face::owner`.
use crate::cutmesh::{CutMesh, NONE};

type V3 = [f64; 3];
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn mul(a: V3, s: f64) -> V3 {
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

const VSMALL: f64 = 1e-300;
const ROOT_VSMALL: f64 = 1e-150;
/// `checkMesh` thresholds
pub const NONORTH_MAX: f64 = 70.0;
pub const SKEW_MAX: f64 = 4.0;
pub const ASPECT_MAX: f64 = 1000.0;
const PYR_MIN: f64 = -1e-15;

pub const F_NEG_VOL: u8 = 1;
pub const F_PYRAMID: u8 = 2;
pub const F_NONORTH: u8 = 4;
pub const F_SKEW: u8 = 8;
pub const F_ASPECT: u8 = 16;
pub const F_JACOBIAN: u8 = 32;
pub const FLAG_NAMES: [(u8, &str); 6] = [
    (F_NEG_VOL, "non-positive volume"),
    (F_PYRAMID, "face-pyramid orientation"),
    (F_NONORTH, "non-orthogonality > 70"),
    (F_SKEW, "skewness > 4"),
    (F_ASPECT, "aspect ratio > 1000"),
    (F_JACOBIAN, "non-positive hex corner Jacobian"),
];

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub n_cells: usize,
    pub n_faces: usize,
    pub n_internal: usize,
    pub non_orth_max: f64,
    pub non_orth_avg: f64,
    pub n_non_orth_gt70: usize,
    pub n_non_orth_gt60: usize,
    /// faces with angle >= 90 deg (checkMesh error; 70..90 is only reported as 'severely non-orthogonal', counted in n_non_orth_gt70)
    pub n_non_orth_ge90: usize,
    pub skew_max: f64,
    pub n_skew_gt4: usize,
    pub n_pyramid_err: usize,
    pub n_neg_vol: usize,
    pub vol_min: f64,
    pub vol_max: f64,
    pub vol_total: f64,
    pub aspect_max: f64,
    pub n_aspect_gt1000: usize,
    pub area_min: f64,
    pub area_max: f64,
    /// max over cells of |sum Sf| / sum |Sf| (OpenFOAM flags > 1e-9)
    pub closed_max: f64,
    pub n_unclosed: usize,
    pub n_jacobian: usize,
    /// number of cells with at least one failing check
    pub n_bad_cells: usize,
    /// cells whose only finding is 70..90 deg non-orthogonality (checkMesh accepts them): not counted as bad
    pub n_info_cells: usize,
    /// failed `checkMesh`-style checks (non-orthogonality, skewness, pyramids, volumes, aspect ratio, closed cells)
    pub failed_checks: usize,
    pub seconds: f64,
}

/// Per-cell data kept for the localisation step.
pub struct CellData {
    pub centre: Vec<V3>,
    pub vol: Vec<f64>,
    pub flags: Vec<u8>,
    pub max_skew: Vec<f32>,
    pub max_nonorth: Vec<f32>,
    pub aspect: Vec<f32>,
}

pub fn compute(m: &CutMesh) -> (Metrics, CellData) {
    let t0 = std::time::Instant::now();
    let (nc, nf) = (m.cells.len(), m.faces.len());
    let p = &m.points;
    // faces: centre and area vector (OpenFOAM makeFaceCentresAndAreas)
    let mut fc = vec![[0.0; 3]; nf];
    let mut fa = vec![[0.0; 3]; nf];
    for (i, f) in m.faces.iter().enumerate() {
        let v = &f.verts;
        let n = v.len();
        if n == 3 {
            fc[i] = mul(add(add(p[v[0] as usize], p[v[1] as usize]), p[v[2] as usize]), 1.0 / 3.0);
            fa[i] = mul(cross(sub(p[v[1] as usize], p[v[0] as usize]), sub(p[v[2] as usize], p[v[0] as usize])), 0.5);
        } else {
            let mut c0 = [0.0; 3];
            for &x in v {
                c0 = add(c0, p[x as usize]);
            }
            c0 = mul(c0, 1.0 / n as f64);
            let (mut sn, mut sa, mut sac) = ([0.0; 3], 0.0, [0.0; 3]);
            for k in 0..n {
                let (a, b) = (p[v[k] as usize], p[v[(k + 1) % n] as usize]);
                let c = add(add(a, b), c0);
                let nn = cross(sub(b, a), sub(c0, a));
                let ar = norm(nn);
                sn = add(sn, nn);
                sa += ar;
                sac = add(sac, mul(c, ar));
            }
            if sa < ROOT_VSMALL {
                fc[i] = c0;
                fa[i] = [0.0; 3];
            } else {
                fc[i] = mul(sac, 1.0 / 3.0 / sa);
                fa[i] = mul(sn, 0.5);
            }
        }
    }
    // cells: centre and volume (makeCellCentresAndVols)
    let mut cc = vec![[0.0; 3]; nc];
    let mut vol = vec![0.0; nc];
    for (c, cell) in m.cells.iter().enumerate() {
        let mut est = [0.0; 3];
        for &f in &cell.faces {
            est = add(est, fc[f as usize]);
        }
        est = mul(est, 1.0 / cell.faces.len().max(1) as f64);
        let (mut cs, mut vs) = ([0.0; 3], 0.0);
        for &f in &cell.faces {
            let fi = f as usize;
            let is_own = m.faces[fi].owner as usize == c;
            let pyr3 = if is_own { dot(fa[fi], sub(fc[fi], est)) } else { dot(fa[fi], sub(est, fc[fi])) };
            let pc = add(mul(fc[fi], 0.75), mul(est, 0.25));
            cs = add(cs, mul(pc, pyr3));
            vs += pyr3;
        }
        if vs.abs() > VSMALL {
            cc[c] = mul(cs, 1.0 / vs);
        } else {
            cc[c] = est;
        }
        vol[c] = vs / 3.0;
    }

    let mut mt = Metrics { n_cells: nc, n_faces: nf, ..Default::default() };
    let mut flags = vec![0u8; nc];
    let mut max_skew = vec![0f32; nc];
    let mut max_no = vec![0f32; nc];
    let (mut no_sum, mut no_n) = (0.0, 0usize);
    let mut pyr_bad = vec![false; nc];
    mt.area_min = f64::INFINITY;
    let mut sum_mag = vec![[0.0f64; 3]; nc];
    let mut sum_vec = vec![[0.0f64; 3]; nc];
    let mut sum_abs = vec![0.0f64; nc];
    for (i, f) in m.faces.iter().enumerate() {
        let own = f.owner as usize;
        let s = fa[i];
        let ms = norm(s);
        mt.area_min = mt.area_min.min(ms);
        mt.area_max = mt.area_max.max(ms);
        let internal = f.neigh != NONE;
        let cell_ids: [Option<usize>; 2] = [Some(own), internal.then_some(f.neigh as usize)];
        for (side, ci) in cell_ids.iter().enumerate() {
            if let Some(c) = *ci {
                let sg = if side == 0 { 1.0 } else { -1.0 };
                for k in 0..3 {
                    sum_mag[c][k] += s[k].abs();
                    sum_vec[c][k] += sg * s[k];
                }
                sum_abs[c] += ms;
            }
        }
        // face pyramids: owner sees the face as stored, neighbour as reversed (negated area)
        let po = dot(s, sub(fc[i], cc[own])) / 3.0;
        let bad = po < PYR_MIN;
        let mut bad_n = false;
        if internal {
            let nb = f.neigh as usize;
            let pn = dot(s, sub(cc[nb], fc[i])) / 3.0;
            bad_n = pn < PYR_MIN;
        }
        if bad {
            pyr_bad[own] = true;
        }
        if bad_n {
            pyr_bad[f.neigh as usize] = true;
        }
        // checkMesh counts the owner and the neighbour pyramid of a face separately
        mt.n_pyramid_err += bad as usize + bad_n as usize;
        // non-orthogonality and skewness
        if internal {
            let nb = f.neigh as usize;
            let d = sub(cc[nb], cc[own]);
            let cosv = (dot(d, s) / (norm(d) * ms).max(VSMALL)).clamp(-1.0, 1.0);
            let ang = cosv.acos().to_degrees();
            mt.non_orth_max = mt.non_orth_max.max(ang);
            no_sum += cosv;
            no_n += 1;
            mt.n_non_orth_gt60 += (ang > 60.0) as usize;
            mt.n_non_orth_ge90 += (cosv <= 1e-15) as usize;
            if ang > NONORTH_MAX {
                // checkMesh: 'severe' = between 70 and 90 degrees; >= 90 degrees are counted as errors instead
                mt.n_non_orth_gt70 += (cosv > 1e-15) as usize;
                flags[own] |= F_NONORTH;
                flags[nb] |= F_NONORTH;
            }
            for c in [own, nb] {
                max_no[c] = max_no[c].max(ang as f32);
            }
            let cpf = sub(fc[i], cc[own]);
            let sv = sub(cpf, mul(d, dot(s, cpf) / (dot(s, d) + ROOT_VSMALL)));
            let svh = mul(sv, 1.0 / (norm(sv) + ROOT_VSMALL));
            let mut fd = 0.2 * norm(d) + ROOT_VSMALL;
            for &x in &f.verts {
                fd = fd.max(dot(svh, sub(p[x as usize], fc[i])).abs());
            }
            let sk = norm(sv) / fd;
            mt.skew_max = mt.skew_max.max(sk);
            if sk > SKEW_MAX {
                mt.n_skew_gt4 += 1;
                flags[own] |= F_SKEW;
                flags[nb] |= F_SKEW;
            }
            for c in [own, nb] {
                max_skew[c] = max_skew[c].max(sk as f32);
            }
        } else {
            let cpf = sub(fc[i], cc[own]);
            let nrm = mul(s, 1.0 / (ms + VSMALL));
            let d = mul(nrm, dot(nrm, cpf));
            let sv = sub(cpf, mul(d, dot(s, cpf) / (dot(s, d) + ROOT_VSMALL)));
            let svh = mul(sv, 1.0 / (norm(sv) + ROOT_VSMALL));
            let mut fd = 0.4 * norm(d) + ROOT_VSMALL;
            for &x in &f.verts {
                fd = fd.max(dot(svh, sub(p[x as usize], fc[i])).abs());
            }
            let sk = norm(sv) / fd;
            mt.skew_max = mt.skew_max.max(sk);
            if sk > SKEW_MAX {
                mt.n_skew_gt4 += 1;
                flags[own] |= F_SKEW;
            }
            max_skew[own] = max_skew[own].max(sk as f32);
        }
    }
    mt.n_internal = m.faces.iter().filter(|f| f.neigh != NONE).count();
    // checkMesh: acos of the mean cosine over internal faces
    mt.non_orth_avg = if no_n > 0 { (no_sum / no_n as f64).clamp(-1.0, 1.0).acos().to_degrees() } else { 0.0 };
    // cell checks
    let mut aspect = vec![0f32; nc];
    mt.vol_min = f64::INFINITY;
    mt.vol_max = f64::NEG_INFINITY;
    for c in 0..nc {
        let v = vol[c];
        mt.vol_min = mt.vol_min.min(v);
        mt.vol_max = mt.vol_max.max(v);
        mt.vol_total += v;
        if v < VSMALL {
            mt.n_neg_vol += 1;
            flags[c] |= F_NEG_VOL;
        }
        if pyr_bad[c] {
            flags[c] |= F_PYRAMID;
        }
        let sm = sum_mag[c];
        let (mn, mx) = (sm[0].min(sm[1]).min(sm[2]), sm[0].max(sm[1]).max(sm[2]));
        let ar = mx / (mn + ROOT_VSMALL);
        aspect[c] = ar as f32;
        mt.aspect_max = mt.aspect_max.max(ar);
        if ar > ASPECT_MAX {
            mt.n_aspect_gt1000 += 1;
            flags[c] |= F_ASPECT;
        }
        let res = norm(sum_vec[c]) / sum_abs[c].max(VSMALL);
        mt.closed_max = mt.closed_max.max(res);
        if res > 1e-9 {
            mt.n_unclosed += 1;
        }
        if let Some(h) = m.cells[c].hex {
            let q: [V3; 8] = h.map(|i| p[i as usize]);
            if hex_min_jacobian(&q) <= 0.0 {
                flags[c] |= F_JACOBIAN;
                mt.n_jacobian += 1;
            }
        }
    }
    if nf == 0 {
        mt.area_min = 0.0;
    }
    for c in 0..nc {
        if info_only(flags[c], max_no[c] as f64) {
            mt.n_info_cells += 1;
        } else if flags[c] != 0 {
            mt.n_bad_cells += 1;
        }
    }
    mt.failed_checks = (mt.n_non_orth_ge90 > 0) as usize + (mt.n_skew_gt4 > 0) as usize + (mt.n_pyramid_err > 0) as usize + (mt.n_neg_vol > 0) as usize + (mt.n_aspect_gt1000 > 0) as usize + (mt.n_unclosed > 0) as usize;
    mt.seconds = t0.elapsed().as_secs_f64();
    (mt, CellData { centre: cc, vol, flags, max_skew, max_nonorth: max_no, aspect })
}

/// A cell whose only finding is non-orthogonality between 70 and 90 degrees (`checkMesh` reports it as 'severe' but the mesh passes).
pub fn info_only(flags: u8, max_nonorth: f64) -> bool {
    flags == F_NONORTH && max_nonorth < 90.0
}

/// Smallest scaled corner Jacobian of a hexahedron (vertex order of the OpenFOAM hex model).
fn hex_min_jacobian(q: &[V3; 8]) -> f64 {
    // corner -> (neighbour along i, j, k) so that (e_i, e_j, e_k) is right-handed for a positively oriented hex
    const N: [[usize; 3]; 8] = [[1, 3, 4], [2, 0, 5], [3, 1, 6], [0, 2, 7], [7, 5, 0], [4, 6, 1], [5, 7, 2], [6, 4, 3]];
    let mut mn = f64::INFINITY;
    for (c, n) in N.iter().enumerate() {
        let (a, b, d) = (sub(q[n[0]], q[c]), sub(q[n[1]], q[c]), sub(q[n[2]], q[c]));
        let det = dot(a, cross(b, d));
        let l = norm(a) * norm(b) * norm(d);
        mn = mn.min(if l > 0.0 { det / l } else { 0.0 });
    }
    mn
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cutmesh::{Cell, Face, Stats, INTERNAL, WALL};
    use std::collections::HashMap;

    /// Hexahedral CutMesh from points and hex node lists (OpenFOAM vertex order); faces outward from their owner.
    fn mesh(points: Vec<V3>, hexes: &[[u32; 8]]) -> CutMesh {
        const F: [[usize; 4]; 6] = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 4, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [3, 7, 6, 2]];
        let mut faces: Vec<Face> = Vec::new();
        let mut cells: Vec<Cell> = Vec::new();
        let mut idx: HashMap<[u32; 4], usize> = HashMap::new();
        for (c, h) in hexes.iter().enumerate() {
            let mut fl = Vec::new();
            for f in F {
                let v: Vec<u32> = f.iter().map(|&k| h[k]).collect();
                let mut key = [v[0], v[1], v[2], v[3]];
                key.sort_unstable();
                match idx.get(&key) {
                    Some(&i) => {
                        faces[i].neigh = c as u32;
                        faces[i].patch = INTERNAL;
                        fl.push(i as u32);
                    }
                    None => {
                        idx.insert(key, faces.len());
                        fl.push(faces.len() as u32);
                        faces.push(Face { verts: v, owner: c as u32, neigh: NONE, patch: WALL });
                    }
                }
            }
            cells.push(Cell { faces: fl, hex: Some(*h), h: 1.0, cut: false, n_orig: 1 });
        }
        CutMesh { points, faces, cells, stats: Stats::default(), patch_name: "wall", native: Default::default() }
    }

    fn two_cubes(dy: f64) -> CutMesh {
        let mut p = Vec::new();
        for x in 0..3 {
            for (y, z) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                p.push([x as f64, y as f64 + if x == 1 && y == 1 { dy } else { 0.0 }, z as f64]);
            }
        }
        // layer x: points 4x..4x+3 = (y,z) (0,0), (1,0), (1,1), (0,1); hex bottom (z=0): (y0,y1) -> use x-major ordering
        let n = |x: u32, y: u32, z: u32| 4 * x + [[0, 3], [1, 2]][y as usize][z as usize];
        let hex = |x: u32| [n(x, 0, 0), n(x + 1, 0, 0), n(x + 1, 1, 0), n(x, 1, 0), n(x, 0, 1), n(x + 1, 0, 1), n(x + 1, 1, 1), n(x, 1, 1)];
        mesh(p, &[hex(0), hex(1)])
    }

    #[test]
    fn unit_cubes_are_perfect() {
        let m = two_cubes(0.0);
        let (q, cd) = compute(&m);
        assert!((q.vol_total - 2.0).abs() < 1e-12 && q.n_neg_vol == 0 && q.n_pyramid_err == 0 && q.n_unclosed == 0);
        assert!(q.non_orth_max < 1e-6 && q.skew_max < 1e-9 && (q.aspect_max - 1.0).abs() < 1e-12 && q.n_jacobian == 0);
        assert!(cd.flags.iter().all(|&f| f == 0) && q.failed_checks == 0 && q.n_bad_cells == 0);
        assert!((cd.centre[0][0] - 0.5).abs() < 1e-12 && (cd.centre[1][0] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn warped_interface_is_skewed() {
        // the shared face moves its y=1 corners by 0.5 (cells stay convex): the face centre leaves the centre line
        let m = two_cubes(0.5);
        let (q, _) = compute(&m);
        assert!(q.skew_max > 0.1 && q.skew_max < 4.0 && q.n_neg_vol == 0, "{}", q.skew_max);
    }

    #[test]
    fn inverted_cell_is_flagged() {
        let mut m = two_cubes(0.0);
        // push the shared face's y=1 corners beyond the second cell (x = 2.5): cell 1 turns inside out
        for k in [6usize, 7] {
            m.points[k][0] = 2.5;
        }
        let (q, cd) = compute(&m);
        assert!(q.n_neg_vol + q.n_pyramid_err + q.n_jacobian > 0 && q.failed_checks > 0);
        assert!(cd.flags.iter().any(|&f| f != 0));
    }
}
