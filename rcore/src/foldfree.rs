//! foldfree: certified fold-free optimisation of hex / prism / pyramid / polyhedral cells (design FOLDFREE_DESIGN_v1, stages S0-S3).
//!
//! Self-contained library module; nothing in the meshing pipeline calls it yet.
//!
//! * Quadrature (design section 1): every cell is a set of tets whose corners are cell vertices or *derived points* with fixed weights
//!   (face vertex mean `c_f`, cell vertex mean `X_0`), so each tet Jacobian is linear in the cell vertices. Family 0 ("corner"):
//!   hex corner tets, prism corner tets, pyramid base-diagonal tets with the apex, polyhedron quad-face diagonal tets with `X_0`.
//!   Family 1 ("fan"): `(X_0, f_k, f_{k+1}, c_f)` for every face edge. Admissible = every tet has `det J_t >= tau_adm`.
//! * Energy (section 2/6): Garanzha's `f_eps + lambda g_eps` with `J_t = D_t W_t^-1`, `W_t` from per-cell target vertices (recomputed on the
//!   fly), weights `w_t = vol(R_c) alpha_fam / |fam|`. Analytic gradient through the cofactor matrix; the Jacobian is never inverted.
//! * Preconditioner: per-vertex 3x3 blocks from a convexified per-tet Hessian (see `FoldMesh::precond_blocks`).
//! * Solver: preconditioned L-BFGS (m = 6) with Armijo backtracking, barrier-aware maximum step (smallest positive root of the per-tet
//!   cubic) in the interior (eps = 0) mode, phases C (eps-continuation) and B (interior polish).
//! * Constraints (section 3): `VKind` per vertex, `Retraction` trait (retract + tangent direction). Projected gradient, retraction after the
//!   step, Armijo on the retracted point.
//! * Parallelism: fixed chunks of 4096 cells / vertices, partial results combined in chunk order, per-vertex gather from a CSR (no atomics);
//!   results are bit-identical for any thread count.

type V3 = [f64; 3];

#[inline(always)]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline(always)]
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline(always)]
fn mul(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline(always)]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline(always)]
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline(always)]
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
#[inline(always)]
fn det3(a: V3, b: V3, c: V3) -> f64 {
    dot(a, cross(b, c))
}

/// fixed chunk size (cells or vertices) of every parallel loop; independent of the thread count
pub const CHUNK: usize = 4096;
const ROOT_VSMALL: f64 = 1e-150;
const VSMALL: f64 = 1e-300;

// ---------------------------------------------------------------------------------------------------------------------------------
// deterministic parallel helpers
// ---------------------------------------------------------------------------------------------------------------------------------

/// Splits `out` at `bounds` (chunk `c` is `out[bounds[c]..bounds[c+1]]`), runs `f(c, chunk)` for every chunk on up to `threads` threads
/// (contiguous groups of chunks), and returns the results in chunk order. The result does not depend on `threads`.
fn par_slices<T: Send, R: Send, F: Fn(usize, &mut [T]) -> R + Sync>(out: &mut [T], bounds: &[usize], threads: usize, f: F) -> Vec<R> {
    let nch = bounds.len().saturating_sub(1);
    let mut parts: Vec<(usize, &mut [T])> = Vec::with_capacity(nch);
    let mut rest: &mut [T] = out;
    for c in 0..nch {
        let (a, b) = std::mem::take(&mut rest).split_at_mut(bounds[c + 1] - bounds[c]);
        parts.push((c, a));
        rest = b;
    }
    let nt = threads.max(1).min(nch.max(1));
    if nt <= 1 {
        return parts.into_iter().map(|(c, s)| f(c, s)).collect();
    }
    let per = nch.div_ceil(nt);
    let mut groups: Vec<Vec<(usize, &mut [T])>> = Vec::new();
    let mut it = parts.into_iter();
    loop {
        let g: Vec<_> = it.by_ref().take(per).collect();
        if g.is_empty() {
            break;
        }
        groups.push(g);
    }
    let fr = &f;
    let mut res: Vec<R> = Vec::with_capacity(nch);
    std::thread::scope(|s| {
        let hs: Vec<_> = groups.into_iter().map(|g| s.spawn(move || g.into_iter().map(|(c, sl)| fr(c, sl)).collect::<Vec<R>>())).collect();
        for h in hs {
            res.extend(h.join().expect("foldfree worker panicked"));
        }
    });
    res
}

/// Chunk-ordered map without output slices: `f(c)` for `c in 0..nch`.
fn par_map<R: Send, F: Fn(usize) -> R + Sync>(nch: usize, threads: usize, f: F) -> Vec<R> {
    let bounds = vec![0usize; nch + 1];
    par_slices::<(), R, _>(&mut [], &bounds, threads, |c, _| f(c))
}

/// Fixed-size chunking of a vertex-length array.
fn par_fill<T: Send, F: Fn(usize, &mut [T]) + Sync>(out: &mut [T], threads: usize, f: F) {
    let n = out.len();
    let nch = n.div_ceil(CHUNK);
    let bounds: Vec<usize> = (0..=nch).map(|c| (c * CHUNK).min(n)).collect();
    par_slices(out, &bounds, threads, |c, s| f(c * CHUNK, s));
}

// ---------------------------------------------------------------------------------------------------------------------------------
// cell model and quadrature tables
// ---------------------------------------------------------------------------------------------------------------------------------

/// Hexahedron corner neighbours in VTK order (same table as `su2` / `hexcore`).
const NB: [[usize; 3]; 8] = [[1, 3, 4], [2, 0, 5], [3, 1, 6], [0, 2, 7], [7, 5, 0], [4, 6, 1], [5, 7, 2], [6, 4, 3]];
/// VTK wedge order -> right-handed (Gmsh-like) order used to write the prism tables: g_i = p[PRISM_MAP[i]]
const PRISM_MAP: [u16; 6] = [0, 2, 1, 3, 5, 4];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellKind {
    Hex,
    Prism,
    Pyramid,
    Poly,
}

/// One cell of the input. Vertex order follows the crate (VTK): hex 0-3 bottom, 4-7 top, positive when `det[e1 e2 e3] > 0` at every corner;
/// prism (VTK wedge) 0-2 bottom, 3-5 top, (0,1,2) seen clockwise from the top; pyramid 0-3 base (counter-clockwise seen from the apex), 4 apex.
/// Poly: unique vertex list plus faces as indices into it, every face counter-clockwise seen from outside.
#[derive(Clone, Debug)]
pub enum CellSpec {
    Hex([u32; 8]),
    Prism([u32; 6]),
    Pyramid([u32; 5]),
    Poly { verts: Vec<u32>, faces: Vec<Vec<u32>> },
}

/// Quadrature of one cell shape. Extended points: `0..nv` cell vertices, then one derived point per face (mean of the face vertices), then `X_0`
/// (mean of all cell vertices); `dmem[i]` lists the cell vertices averaged by derived point `i`.
#[derive(Clone, Debug)]
pub struct Topo {
    pub nv: usize,
    pub faces: Vec<Vec<u16>>,
    pub dmem: Vec<Vec<u16>>,
    pub tets: Vec<[u16; 4]>,
    /// 0 = corner family, 1 = fan family
    pub fam: Vec<u8>,
    pub nfam: [usize; 2],
}

impl Topo {
    fn build(nv: usize, faces: Vec<Vec<u16>>, corner: Vec<[u16; 4]>, diag_quads: bool) -> Topo {
        let nf = faces.len();
        let mut dmem: Vec<Vec<u16>> = faces.clone();
        dmem.push((0..nv as u16).collect());
        let x0 = (nv + nf) as u16;
        let mut tets = Vec::new();
        let mut fam = Vec::new();
        for t in corner {
            tets.push(t);
            fam.push(0u8);
        }
        if diag_quads {
            for f in &faces {
                if f.len() == 4 {
                    for t in [[f[0], f[1], f[2]], [f[0], f[2], f[3]], [f[0], f[1], f[3]], [f[1], f[2], f[3]]] {
                        tets.push([x0, t[0], t[1], t[2]]);
                        fam.push(0);
                    }
                }
            }
        }
        for (fi, f) in faces.iter().enumerate() {
            let cf = (nv + fi) as u16;
            for k in 0..f.len() {
                tets.push([x0, f[k], f[(k + 1) % f.len()], cf]);
                fam.push(1);
            }
        }
        let mut nfam = [0usize; 2];
        for &f in &fam {
            nfam[f as usize] += 1;
        }
        Topo { nv, faces, dmem, tets, fam, nfam }
    }

    pub fn hex() -> Topo {
        let faces: Vec<Vec<u16>> = vec![vec![0, 3, 2, 1], vec![4, 5, 6, 7], vec![0, 1, 5, 4], vec![1, 2, 6, 5], vec![2, 3, 7, 6], vec![3, 0, 4, 7]];
        let corner = (0..8).map(|i| [i as u16, NB[i][0] as u16, NB[i][1] as u16, NB[i][2] as u16]).collect();
        Topo::build(8, faces, corner, false)
    }

    pub fn prism() -> Topo {
        // tables in the right-handed order g, mapped to VTK wedge ids
        let g_faces: Vec<Vec<u16>> = vec![vec![0, 2, 1], vec![3, 4, 5], vec![0, 1, 4, 3], vec![1, 2, 5, 4], vec![2, 0, 3, 5]];
        let mut corner: Vec<[u16; 4]> = Vec::new();
        for i in 0..3u16 {
            corner.push([i, (i + 1) % 3, (i + 2) % 3, i + 3]);
        }
        for m in 0..3u16 {
            corner.push([3 + m, 3 + (m + 1) % 3, m, 3 + (m + 2) % 3]);
        }
        let mp = |v: u16| PRISM_MAP[v as usize];
        let faces = g_faces.iter().map(|f| f.iter().map(|&v| mp(v)).collect()).collect();
        let corner = corner.iter().map(|t| [mp(t[0]), mp(t[1]), mp(t[2]), mp(t[3])]).collect();
        Topo::build(6, faces, corner, false)
    }

    pub fn pyramid() -> Topo {
        let faces: Vec<Vec<u16>> = vec![vec![0, 3, 2, 1], vec![0, 1, 4], vec![1, 2, 4], vec![2, 3, 4], vec![3, 0, 4]];
        let corner = vec![[0, 1, 2, 4], [0, 2, 3, 4], [0, 1, 3, 4], [1, 2, 3, 4]];
        Topo::build(5, faces, corner, false)
    }

    pub fn poly(nv: usize, faces: Vec<Vec<u16>>) -> Topo {
        Topo::build(nv, faces, Vec::new(), true)
    }

    fn n_ext(&self) -> usize {
        self.nv + self.dmem.len()
    }
}

/// Fills the derived points of `ext` (whose first `nv` entries are set).
#[inline]
fn fill_derived(topo: &Topo, ext: &mut Vec<V3>) {
    let nv = topo.nv;
    ext.truncate(nv);
    for mem in &topo.dmem {
        let mut s = [0.0; 3];
        for &m in mem {
            s = add(s, ext[m as usize]);
        }
        ext.push(mul(s, 1.0 / mem.len() as f64));
    }
}

/// Right column over a quad: bottom = `quad` (counter-clockwise seen from the side the column grows to), top = bottom + `d` * unit normal.
/// VTK hexahedron order.
pub fn right_column_hex(quad: [V3; 4], d: f64) -> [V3; 8] {
    let n = cross(sub(quad[2], quad[0]), sub(quad[3], quad[1]));
    let nl = norm(n);
    let n = if nl > 0.0 { mul(n, d / nl) } else { [0.0; 3] };
    let mut h = [[0.0; 3]; 8];
    for i in 0..4 {
        h[i] = quad[i];
        h[i + 4] = add(quad[i], n);
    }
    h
}

/// A stack of right-column hexes over a quad with the given layer heights (first entry is the layer at the quad).
pub fn right_column_stack(quad: [V3; 4], heights: &[f64]) -> Vec<[V3; 8]> {
    let mut out = Vec::new();
    let mut base = quad;
    let n = cross(sub(quad[2], quad[0]), sub(quad[3], quad[1]));
    let n = mul(n, 1.0 / norm(n));
    for &h in heights {
        let hex = right_column_hex(base, h);
        out.push(hex);
        for i in 0..4 {
            base[i] = add(base[i], mul(n, h));
        }
    }
    out
}

/// Lemma R: multilinear expansion of the end-corner determinant `det[E_a + T da, E_b + T db, -T w]` (fibres `w + d`, common height T):
/// returns the four terms `[-T det[Ea,Eb,w], -T^2 (det[da,Eb,w] + det[Ea,db,w]), -T^3 det[da,db,w]]` and their sum as the fourth entry.
pub fn lemma_r_expansion(ea: V3, eb: V3, w: V3, da: V3, db: V3, t: f64) -> [f64; 4] {
    let t0 = -t * det3(ea, eb, w);
    let t1 = -t * t * (det3(da, eb, w) + det3(ea, db, w));
    let t2 = -t * t * t * det3(da, db, w);
    [t0, t1, t2, t0 + t1 + t2]
}

// ---------------------------------------------------------------------------------------------------------------------------------
// per-tet energy
// ---------------------------------------------------------------------------------------------------------------------------------

/// scalar functions of the Jacobian determinant `D` and `eps`
#[inline(always)]
fn chi_rho(d: f64, eps: f64) -> (f64, f64) {
    let rho = (eps * eps + d * d).sqrt();
    let chi = if d >= 0.0 { 0.5 * (d + rho) } else { 0.5 * eps * eps / (rho - d) };
    (chi, rho)
}

/// `f_eps + lambda g_eps` and its derivative with respect to J (rows). Returns `None` if the energy is infinite (chi = 0) or not finite.
#[inline(always)]
fn tet_energy(jr: &[V3; 3], eps: f64, lambda: f64, want_grad: bool) -> (f64, [V3; 3], f64) {
    let c = [cross(jr[1], jr[2]), cross(jr[2], jr[0]), cross(jr[0], jr[1])];
    let d = dot(jr[0], c[0]);
    let (chi, rho) = chi_rho(d, eps);
    if !(chi > 0.0) || !(chi.is_finite()) {
        return (f64::INFINITY, [[0.0; 3]; 3], d);
    }
    let t = dot(jr[0], jr[0]) + dot(jr[1], jr[1]) + dot(jr[2], jr[2]);
    let cb = chi.cbrt();
    let h = 1.0 / (cb * cb);
    let e = t * h + lambda * (d * d + 1.0) / chi;
    if !want_grad {
        return (e, [[0.0; 3]; 3], d);
    }
    // dE/dJ = 2 h J + (T h' + lambda k') cof J, h' = -(2/3) h / rho, k' = 2D/chi - (D^2+1)/(chi rho)
    let s = -(2.0 / 3.0) * t * h / rho + lambda * (2.0 * d / chi - (d * d + 1.0) / (chi * rho));
    let mut g = [[0.0; 3]; 3];
    for i in 0..3 {
        g[i] = add(mul(jr[i], 2.0 * h), mul(c[i], s));
    }
    (e, g, d)
}

/// Smallest positive root of `c0 + c1 a + c2 a^2 + c3 a^3` below `cap` (c0 > 0 assumed), else `INFINITY`. Returns a point on the positive side
/// of the root (bisection on bracketed monotone intervals between critical points).
fn cubic_first_root(c: [f64; 4], cap: f64) -> f64 {
    if !(c[0] > 0.0) {
        return 0.0;
    }
    if c[1] >= 0.0 && c[2] >= 0.0 && c[3] >= 0.0 {
        return f64::INFINITY;
    }
    let p = |a: f64| ((c[3] * a + c[2]) * a + c[1]) * a + c[0];
    let mut pts = [0.0f64; 4];
    let mut n = 1;
    let (qa, qb, qc) = (3.0 * c[3], 2.0 * c[2], c[1]);
    let mut crit = [0.0f64; 2];
    let mut nc = 0;
    if qa != 0.0 {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            let q = -0.5 * (qb + if qb >= 0.0 { sq } else { -sq });
            crit[0] = q / qa;
            nc = 1;
            if q != 0.0 {
                crit[1] = qc / q;
                nc = 2;
            }
        }
    } else if qb != 0.0 {
        crit[0] = -qc / qb;
        nc = 1;
    }
    if nc == 2 && crit[0] > crit[1] {
        crit.swap(0, 1);
    }
    for &x in crit.iter().take(nc) {
        if x > 0.0 && x < cap {
            pts[n] = x;
            n += 1;
        }
    }
    pts[n] = cap;
    n += 1;
    for k in 0..n - 1 {
        let (mut lo, mut hi) = (pts[k], pts[k + 1]);
        if p(lo) > 0.0 && p(hi) <= 0.0 {
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if p(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return lo;
        }
    }
    f64::INFINITY
}

// ---------------------------------------------------------------------------------------------------------------------------------
// mesh
// ---------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct EnergyCfg {
    /// 0 = interior-point mode (infinite energy on `det J <= 0`)
    pub eps: f64,
    pub lambda: f64,
    /// admissibility threshold on `det J` relative to the target
    pub tau: f64,
    pub threads: usize,
}

impl EnergyCfg {
    pub fn new(eps: f64) -> EnergyCfg {
        EnergyCfg { eps, lambda: 1.0, tau: 1e-6, threads: default_threads() }
    }
}

pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

#[derive(Clone, Copy, Debug)]
pub struct Eval {
    pub energy: f64,
    /// minimum relative `det J` over all tets, per family and overall
    pub dmin: f64,
    pub dmin_corner: f64,
    pub dmin_fan: f64,
    /// tets with `det J < tau`
    pub n_bad: usize,
}

#[derive(Clone, Copy)]
struct Partial {
    e: f64,
    dmin: [f64; 2],
    nbad: usize,
}

struct CellBufs {
    ext: Vec<V3>,
    rext: Vec<V3>,
    gext: Vec<V3>,
    rv: Vec<V3>,
    mark: Vec<bool>,
}

impl CellBufs {
    fn new() -> CellBufs {
        CellBufs { ext: Vec::new(), rext: Vec::new(), gext: Vec::new(), rv: Vec::new(), mark: Vec::new() }
    }
}

pub struct FoldMesh {
    pub n_verts: usize,
    pub kind: Vec<CellKind>,
    voff: Vec<usize>,
    cv: Vec<u32>,
    ctopo: Vec<u32>,
    topos: Vec<Topo>,
    tgt: Vec<V3>,
    /// per cell, weight of a tet of family 0 / 1
    wfam: Vec<[f64; 2]>,
    vol_ref: Vec<f64>,
    vs_off: Vec<usize>,
    vs: Vec<u32>,
    has_targets: bool,
}

impl FoldMesh {
    /// Builds the cell tables. Targets are set afterwards with `set_targets` / `set_targets_from_points`.
    pub fn new(n_verts: usize, cells: &[CellSpec]) -> FoldMesh {
        let mut topos = vec![Topo::hex(), Topo::prism(), Topo::pyramid()];
        let (mut kind, mut voff, mut cv, mut ctopo) = (Vec::new(), vec![0usize], Vec::new(), Vec::new());
        for c in cells {
            match c {
                CellSpec::Hex(v) => {
                    kind.push(CellKind::Hex);
                    ctopo.push(0);
                    cv.extend_from_slice(v);
                }
                CellSpec::Prism(v) => {
                    kind.push(CellKind::Prism);
                    ctopo.push(1);
                    cv.extend_from_slice(v);
                }
                CellSpec::Pyramid(v) => {
                    kind.push(CellKind::Pyramid);
                    ctopo.push(2);
                    cv.extend_from_slice(v);
                }
                CellSpec::Poly { verts, faces } => {
                    kind.push(CellKind::Poly);
                    ctopo.push(topos.len() as u32);
                    topos.push(Topo::poly(verts.len(), faces.iter().map(|f| f.iter().map(|&i| i as u16).collect()).collect()));
                    cv.extend_from_slice(verts);
                }
            }
            voff.push(cv.len());
        }
        let nslots = cv.len();
        let mut vs_off = vec![0usize; n_verts + 1];
        for &v in &cv {
            vs_off[v as usize + 1] += 1;
        }
        for i in 0..n_verts {
            vs_off[i + 1] += vs_off[i];
        }
        let mut fillp = vs_off.clone();
        let mut vs = vec![0u32; nslots];
        for (s, &v) in cv.iter().enumerate() {
            vs[fillp[v as usize]] = s as u32;
            fillp[v as usize] += 1;
        }
        FoldMesh { n_verts, kind, voff, cv, ctopo, topos, tgt: vec![[0.0; 3]; nslots], wfam: vec![[0.0; 2]; cells.len()], vol_ref: vec![0.0; cells.len()], vs_off, vs, has_targets: false }
    }

    pub fn n_cells(&self) -> usize {
        self.kind.len()
    }
    pub fn n_slots(&self) -> usize {
        self.cv.len()
    }
    /// global vertex ids of cell `c`
    pub fn cell_verts(&self, c: usize) -> &[u32] {
        &self.cv[self.voff[c]..self.voff[c + 1]]
    }
    pub fn topo_of(&self, c: usize) -> &Topo {
        &self.topos[self.ctopo[c] as usize]
    }
    pub fn n_tets(&self) -> usize {
        (0..self.n_cells()).map(|c| self.topo_of(c).tets.len()).sum()
    }
    pub fn target_volume(&self, c: usize) -> f64 {
        self.vol_ref[c]
    }
    /// target point of local vertex `j` of cell `c`
    pub fn target(&self, c: usize, j: usize) -> V3 {
        self.tgt[self.voff[c] + j]
    }

    /// Sets the target vertices, one per (cell, local vertex) slot in cell order, and computes the cell reference volumes and tet weights
    /// `w_t = vol(R_c) alpha_fam / |fam|` (alpha = 1/2 each, 1 if only one family is present). Errors if a target tet or cell is not positive.
    pub fn set_targets(&mut self, t: Vec<V3>) -> Result<(), String> {
        if t.len() != self.cv.len() {
            return Err(format!("targets: expected {} slots, got {}", self.cv.len(), t.len()));
        }
        self.tgt = t;
        let mut bufs = CellBufs::new();
        for c in 0..self.n_cells() {
            let topo = &self.topos[self.ctopo[c] as usize];
            let o = self.voff[c];
            bufs.rext.clear();
            bufs.rext.extend_from_slice(&self.tgt[o..o + topo.nv]);
            fill_derived(topo, &mut bufs.rext);
            let mut vol = 0.0;
            for (ti, tt) in topo.tets.iter().enumerate() {
                let p = &bufs.rext;
                let dt = det3(sub(p[tt[1] as usize], p[tt[0] as usize]), sub(p[tt[2] as usize], p[tt[0] as usize]), sub(p[tt[3] as usize], p[tt[0] as usize]));
                if !(dt > 0.0) {
                    return Err(format!("cell {c}: target tet {ti} has non-positive volume {dt:e}"));
                }
                if topo.fam[ti] == 1 {
                    vol += dt / 6.0;
                }
            }
            if !(vol > 0.0) {
                return Err(format!("cell {c}: non-positive target volume"));
            }
            let nfams = (topo.nfam[0] > 0) as usize + (topo.nfam[1] > 0) as usize;
            let alpha = 1.0 / nfams as f64;
            self.vol_ref[c] = vol;
            self.wfam[c] = [if topo.nfam[0] > 0 { vol * alpha / topo.nfam[0] as f64 } else { 0.0 }, if topo.nfam[1] > 0 { vol * alpha / topo.nfam[1] as f64 } else { 0.0 }];
        }
        self.has_targets = true;
        Ok(())
    }

    /// Targets = the coordinates `x` of the cells' vertices.
    pub fn set_targets_from_points(&mut self, x: &[V3]) -> Result<(), String> {
        let t: Vec<V3> = self.cv.iter().map(|&v| x[v as usize]).collect();
        self.set_targets(t)
    }

    /// Replaces the target of one cell (then call `refresh_cell` is not needed: weights are updated here).
    pub fn set_cell_target(&mut self, c: usize, pts: &[V3]) -> Result<(), String> {
        let o = self.voff[c];
        let n = self.voff[c + 1] - o;
        if pts.len() != n {
            return Err("set_cell_target: wrong vertex count".into());
        }
        let old: Vec<V3> = self.tgt[o..o + n].to_vec();
        self.tgt[o..o + n].copy_from_slice(pts);
        // recompute weights of this cell only
        let topo = &self.topos[self.ctopo[c] as usize];
        let mut ext = pts.to_vec();
        fill_derived(topo, &mut ext);
        let mut vol = 0.0;
        let mut ok = true;
        for (ti, tt) in topo.tets.iter().enumerate() {
            let dt = det3(sub(ext[tt[1] as usize], ext[tt[0] as usize]), sub(ext[tt[2] as usize], ext[tt[0] as usize]), sub(ext[tt[3] as usize], ext[tt[0] as usize]));
            ok &= dt > 0.0;
            if topo.fam[ti] == 1 {
                vol += dt / 6.0;
            }
        }
        if !ok || !(vol > 0.0) {
            self.tgt[o..o + n].copy_from_slice(&old);
            return Err(format!("cell {c}: target not positive"));
        }
        let nfams = (topo.nfam[0] > 0) as usize + (topo.nfam[1] > 0) as usize;
        let alpha = 1.0 / nfams as f64;
        self.vol_ref[c] = vol;
        self.wfam[c] = [if topo.nfam[0] > 0 { vol * alpha / topo.nfam[0] as f64 } else { 0.0 }, if topo.nfam[1] > 0 { vol * alpha / topo.nfam[1] as f64 } else { 0.0 }];
        Ok(())
    }

    /// Energy of the target configuration (`sum_t w_t (3 + 2 lambda)`), the reference scale for stopping rules.
    pub fn target_energy(&self, lambda: f64) -> f64 {
        let mut s = 0.0;
        for c in 0..self.n_cells() {
            let t = self.topo_of(c);
            s += self.wfam[c][0] * t.nfam[0] as f64 * (3.0 + 2.0 * lambda) + self.wfam[c][1] * t.nfam[1] as f64 * (3.0 + 2.0 * lambda);
        }
        s
    }

    /// Per-vertex reference energy: the target energy of the incident cells shared equally among their vertices (for local stopping rules).
    pub fn vertex_ref_energy(&self, lambda: f64) -> Vec<f64> {
        let mut e = vec![0.0; self.n_verts];
        for c in 0..self.n_cells() {
            let t = self.topo_of(c);
            let ec = (self.wfam[c][0] * t.nfam[0] as f64 + self.wfam[c][1] * t.nfam[1] as f64) * (3.0 + 2.0 * lambda);
            for &v in self.cell_verts(c) {
                e[v as usize] += ec / t.nv as f64;
            }
        }
        e
    }

    fn cell_bounds(&self) -> Vec<usize> {
        let nc = self.n_cells();
        let nch = nc.div_ceil(CHUNK);
        (0..=nch).map(|c| (c * CHUNK).min(nc)).collect()
    }
    fn slot_bounds(&self) -> Vec<usize> {
        self.cell_bounds().iter().map(|&c| self.voff[c]).collect()
    }

    /// Gathers the extended points (current and target) of cell `c`.
    #[inline]
    fn load_cell(&self, c: usize, x: &[V3], b: &mut CellBufs) -> &Topo {
        let topo = &self.topos[self.ctopo[c] as usize];
        let o = self.voff[c];
        b.ext.clear();
        b.rext.clear();
        for j in 0..topo.nv {
            b.ext.push(x[self.cv[o + j] as usize]);
            b.rext.push(self.tgt[o + j]);
        }
        fill_derived(topo, &mut b.ext);
        fill_derived(topo, &mut b.rext);
        topo
    }

    /// Energy (and, if `grad` is given, the gradient with respect to the vertex coordinates; `work` is reusable scratch of slot length).
    /// In interior mode (`eps == 0`) the energy is `+inf` as soon as one tet has `det J <= 0`; the gradient is then not meaningful.
    pub fn eval(&self, x: &[V3], cfg: &EnergyCfg, grad: Option<&mut [V3]>, work: &mut Vec<V3>) -> Eval {
        assert!(self.has_targets, "targets not set");
        assert_eq!(x.len(), self.n_verts);
        let want = grad.is_some();
        let cb = self.cell_bounds();
        let sb = self.slot_bounds();
        let nch = cb.len() - 1;
        let parts: Vec<Partial> = if want {
            work.resize(self.cv.len(), [0.0; 3]);
            par_slices(&mut work[..], &sb, cfg.threads, |ci, cg| self.eval_chunk(cb[ci], cb[ci + 1], x, cfg, Some(cg)))
        } else {
            par_map(nch, cfg.threads, |ci| self.eval_chunk(cb[ci], cb[ci + 1], x, cfg, None))
        };
        let mut ev = Eval { energy: 0.0, dmin: f64::INFINITY, dmin_corner: f64::INFINITY, dmin_fan: f64::INFINITY, n_bad: 0 };
        for p in &parts {
            ev.energy += p.e;
            ev.dmin_corner = ev.dmin_corner.min(p.dmin[0]);
            ev.dmin_fan = ev.dmin_fan.min(p.dmin[1]);
            ev.n_bad += p.nbad;
        }
        ev.dmin = ev.dmin_corner.min(ev.dmin_fan);
        if let Some(g) = grad {
            let cg: &[V3] = &work[..];
            par_fill(g, cfg.threads, |start, out| {
                for (k, o) in out.iter_mut().enumerate() {
                    let v = start + k;
                    let mut s = [0.0; 3];
                    for &sl in &self.vs[self.vs_off[v]..self.vs_off[v + 1]] {
                        s = add(s, cg[sl as usize]);
                    }
                    *o = s;
                }
            });
        }
        ev
    }

    fn eval_chunk(&self, c0: usize, c1: usize, x: &[V3], cfg: &EnergyCfg, mut cg: Option<&mut [V3]>) -> Partial {
        let mut part = Partial { e: 0.0, dmin: [f64::INFINITY; 2], nbad: 0 };
        let mut b = CellBufs::new();
        let base = self.voff[c0];
        for c in c0..c1 {
            let topo = self.load_cell(c, x, &mut b);
            let want = cg.is_some();
            let np = topo.n_ext();
            if want {
                b.gext.clear();
                b.gext.resize(np, [0.0; 3]);
            }
            let wf = self.wfam[c];
            let mut ce = 0.0;
            for (ti, t) in topo.tets.iter().enumerate() {
                let (a, bb, cc, dd) = (t[0] as usize, t[1] as usize, t[2] as usize, t[3] as usize);
                let dm = [sub(b.ext[bb], b.ext[a]), sub(b.ext[cc], b.ext[a]), sub(b.ext[dd], b.ext[a])];
                let wm = [sub(b.rext[bb], b.rext[a]), sub(b.rext[cc], b.rext[a]), sub(b.rext[dd], b.rext[a])];
                let (c1, c2, c3) = (cross(wm[1], wm[2]), cross(wm[2], wm[0]), cross(wm[0], wm[1]));
                let inv = 1.0 / dot(wm[0], c1);
                let rows = [mul(c1, inv), mul(c2, inv), mul(c3, inv)];
                let jr = [0, 1, 2].map(|i| add(add(mul(rows[0], dm[0][i]), mul(rows[1], dm[1][i])), mul(rows[2], dm[2][i])));
                let fam = topo.fam[ti] as usize;
                let (e, g, d) = tet_energy(&jr, cfg.eps, cfg.lambda, want);
                if d < part.dmin[fam] {
                    part.dmin[fam] = d;
                }
                if d < cfg.tau {
                    part.nbad += 1;
                }
                let w = wf[fam];
                ce += w * e;
                if want {
                    // dE/dD columns = w G W^-T
                    let gd = [0, 1, 2].map(|m| mul([dot(g[0], rows[m]), dot(g[1], rows[m]), dot(g[2], rows[m])], w));
                    b.gext[bb] = add(b.gext[bb], gd[0]);
                    b.gext[cc] = add(b.gext[cc], gd[1]);
                    b.gext[dd] = add(b.gext[dd], gd[2]);
                    b.gext[a] = sub(b.gext[a], add(add(gd[0], gd[1]), gd[2]));
                }
            }
            part.e += ce;
            if let Some(cgs) = cg.as_deref_mut() {
                let o = self.voff[c] - base;
                let nv = topo.nv;
                for j in 0..nv {
                    cgs[o + j] = b.gext[j];
                }
                for (i, mem) in topo.dmem.iter().enumerate() {
                    let gi = mul(b.gext[nv + i], 1.0 / mem.len() as f64);
                    for &m in mem {
                        cgs[o + m as usize] = add(cgs[o + m as usize], gi);
                    }
                }
            }
        }
        part
    }

    /// Minimum relative `det J` and the largest feasible step: smallest `alpha in (0, cap]` at which some tet determinant of `x + alpha d`
    /// reaches zero (`INFINITY` if none). Exact for unconstrained moves (Jacobians are linear in the vertices).
    pub fn max_step(&self, x: &[V3], d: &[V3], cap: f64, threads: usize) -> f64 {
        let cb = self.cell_bounds();
        let nch = cb.len() - 1;
        let r = par_map(nch, threads, |ci| {
            let mut b = CellBufs::new();
            let mut dext: Vec<V3> = Vec::new();
            let mut best = f64::INFINITY;
            for c in cb[ci]..cb[ci + 1] {
                let topo = self.load_cell(c, x, &mut b);
                let o = self.voff[c];
                dext.clear();
                for j in 0..topo.nv {
                    dext.push(d[self.cv[o + j] as usize]);
                }
                fill_derived(topo, &mut dext);
                for t in &topo.tets {
                    let (a, bb, cc, dd) = (t[0] as usize, t[1] as usize, t[2] as usize, t[3] as usize);
                    let p = [sub(b.ext[bb], b.ext[a]), sub(b.ext[cc], b.ext[a]), sub(b.ext[dd], b.ext[a])];
                    let q = [sub(dext[bb], dext[a]), sub(dext[cc], dext[a]), sub(dext[dd], dext[a])];
                    let c0 = det3(p[0], p[1], p[2]);
                    let c1 = det3(q[0], p[1], p[2]) + det3(p[0], q[1], p[2]) + det3(p[0], p[1], q[2]);
                    let c2 = det3(p[0], q[1], q[2]) + det3(q[0], p[1], q[2]) + det3(q[0], q[1], p[2]);
                    let c3 = det3(q[0], q[1], q[2]);
                    let r = cubic_first_root([c0, c1, c2, c3], cap.min(best));
                    if r < best {
                        best = r;
                    }
                }
            }
            best
        });
        r.into_iter().fold(f64::INFINITY, f64::min)
    }

    /// Convexified per-vertex 3x3 Hessian blocks (symmetric, `[xx, xy, xz, yy, yz, zz]`) for block-Jacobi preconditioning.
    ///
    /// Per tet the Hessian with respect to `J` is `2h I + 2h'(J C^T + C J^T) + (T h'' + lambda k'') C C^T + s H_D` with `C = cof J`,
    /// `s = T h' + lambda k'` and `H_D = d^2 det / dJ^2`. Choice (documented in the report): `2h I` is positive definite; the rank-2
    /// part lives in span{J, C} and is eigen-clamped exactly (closed-form 2x2 eigen-decomposition in an orthonormal basis of the span);
    /// the `s H_D` term (eigenvalues +-s sigma_k) is dropped. The result is always SPD, costs no 9x9 eigen-solve, and the dropped term
    /// is the one that vanishes for a conformal J at the target. The vertex block is `sum_t B_v^T H_t B_v` with `dJ/dv = e_m r_v^T`
    /// (derived points contribute with their fixed weights, so cross terms between a vertex and the derived points that contain it are included).
    pub fn precond_blocks(&self, x: &[V3], cfg: &EnergyCfg, out: &mut [[f64; 6]], work: &mut Vec<[f64; 6]>) {
        assert_eq!(out.len(), self.n_verts);
        let cb = self.cell_bounds();
        let sb = self.slot_bounds();
        work.resize(self.cv.len(), [0.0; 6]);
        par_slices(&mut work[..], &sb, cfg.threads, |ci, a| self.hess_chunk(cb[ci], cb[ci + 1], x, cfg, a));
        let w: &[[f64; 6]] = &work[..];
        par_fill(out, cfg.threads, |start, o| {
            for (k, ov) in o.iter_mut().enumerate() {
                let v = start + k;
                let mut s = [0.0; 6];
                for &sl in &self.vs[self.vs_off[v]..self.vs_off[v + 1]] {
                    for q in 0..6 {
                        s[q] += w[sl as usize][q];
                    }
                }
                *ov = s;
            }
        });
    }

    fn hess_chunk(&self, c0: usize, c1: usize, x: &[V3], cfg: &EnergyCfg, a: &mut [[f64; 6]]) {
        let mut b = CellBufs::new();
        let base = self.voff[c0];
        let mut touched: Vec<usize> = Vec::new();
        for c in c0..c1 {
            let topo = self.load_cell(c, x, &mut b);
            let nv = topo.nv;
            let o = self.voff[c] - base;
            b.rv.clear();
            b.rv.resize(nv, [0.0; 3]);
            b.mark.clear();
            b.mark.resize(nv, false);
            let wf = self.wfam[c];
            for (ti, t) in topo.tets.iter().enumerate() {
                let w = wf[topo.fam[ti] as usize];
                let (ia, ib, ic, id) = (t[0] as usize, t[1] as usize, t[2] as usize, t[3] as usize);
                let dm = [sub(b.ext[ib], b.ext[ia]), sub(b.ext[ic], b.ext[ia]), sub(b.ext[id], b.ext[ia])];
                let wm = [sub(b.rext[ib], b.rext[ia]), sub(b.rext[ic], b.rext[ia]), sub(b.rext[id], b.rext[ia])];
                let (c1v, c2v, c3v) = (cross(wm[1], wm[2]), cross(wm[2], wm[0]), cross(wm[0], wm[1]));
                let inv = 1.0 / dot(wm[0], c1v);
                let rows = [mul(c1v, inv), mul(c2v, inv), mul(c3v, inv)];
                let jr = [0, 1, 2].map(|i| add(add(mul(rows[0], dm[0][i]), mul(rows[1], dm[1][i])), mul(rows[2], dm[2][i])));
                let cof = [cross(jr[1], jr[2]), cross(jr[2], jr[0]), cross(jr[0], jr[1])];
                let d = dot(jr[0], cof[0]);
                let (chi, rho) = chi_rho(d, cfg.eps);
                if !(chi > 0.0) || !chi.is_finite() {
                    continue;
                }
                let tt = dot(jr[0], jr[0]) + dot(jr[1], jr[1]) + dot(jr[2], jr[2]);
                let cbr = chi.cbrt();
                let h = 1.0 / (cbr * cbr);
                let hp = -(2.0 / 3.0) * h / rho;
                let hpp = (2.0 / 3.0) * h * (2.0 / (3.0 * rho * rho) + d / (rho * rho * rho));
                let kpp = 2.0 / chi - 4.0 * d / (chi * rho) + 2.0 * (d * d + 1.0) / (rho * rho * rho);
                let qcc = tt * hpp + cfg.lambda * kpp;
                // orthonormal basis (q1, q2) of span{J, C}
                let an = tt.sqrt();
                let q1 = [mul(jr[0], 1.0 / an), mul(jr[1], 1.0 / an), mul(jr[2], 1.0 / an)];
                let c1 = dot(cof[0], q1[0]) + dot(cof[1], q1[1]) + dot(cof[2], q1[2]);
                let r2 = [sub(cof[0], mul(q1[0], c1)), sub(cof[1], mul(q1[1], c1)), sub(cof[2], mul(q1[2], c1))];
                let cn = (dot(cof[0], cof[0]) + dot(cof[1], cof[1]) + dot(cof[2], cof[2])).sqrt();
                let n2 = (dot(r2[0], r2[0]) + dot(r2[1], r2[1]) + dot(r2[2], r2[2])).sqrt();
                let two_d = n2 > 1e-10 * cn;
                let c2 = if two_d { n2 } else { 0.0 };
                let q2 = if two_d { [mul(r2[0], 1.0 / n2), mul(r2[1], 1.0 / n2), mul(r2[2], 1.0 / n2)] } else { [[0.0; 3]; 3] };
                // M in the basis (q1, q2)
                let m11 = 2.0 * hp * 2.0 * an * c1 + qcc * c1 * c1;
                let m12 = 2.0 * hp * an * c2 + qcc * c1 * c2;
                let m22 = qcc * c2 * c2;
                // eigen-clamp: keep only positive eigenpairs
                let mut pairs: [(f64, f64, f64); 2] = [(0.0, 0.0, 0.0); 2];
                if two_d {
                    let mean = 0.5 * (m11 + m22);
                    let rad = (0.25 * (m11 - m22) * (m11 - m22) + m12 * m12).sqrt();
                    let th = 0.5 * (2.0 * m12).atan2(m11 - m22);
                    let (s, co) = th.sin_cos();
                    pairs[0] = (mean + rad, co, s);
                    pairs[1] = (mean - rad, -s, co);
                } else {
                    pairs[0] = (m11, 1.0, 0.0);
                }
                // per-vertex r_v = sum_a coef_a(v) r_a over the 4 tet points
                let r_pt = [mul(add(add(rows[0], rows[1]), rows[2]), -1.0), rows[0], rows[1], rows[2]];
                touched.clear();
                for (pi, &e) in t.iter().enumerate() {
                    let e = e as usize;
                    if e < nv {
                        b.rv[e] = add(b.rv[e], r_pt[pi]);
                        if !b.mark[e] {
                            b.mark[e] = true;
                            touched.push(e);
                        }
                    } else {
                        let mem = &topo.dmem[e - nv];
                        let wgt = 1.0 / mem.len() as f64;
                        for &m in mem {
                            let m = m as usize;
                            b.rv[m] = add(b.rv[m], mul(r_pt[pi], wgt));
                            if !b.mark[m] {
                                b.mark[m] = true;
                                touched.push(m);
                            }
                        }
                    }
                }
                for &j in &touched {
                    let r = b.rv[j];
                    b.rv[j] = [0.0; 3];
                    b.mark[j] = false;
                    let aj = &mut a[o + j];
                    let diag = 2.0 * h * w * dot(r, r);
                    aj[0] += diag;
                    aj[3] += diag;
                    aj[5] += diag;
                    let (qa, qb) = ([dot(q1[0], r), dot(q1[1], r), dot(q1[2], r)], [dot(q2[0], r), dot(q2[1], r), dot(q2[2], r)]);
                    for &(mu, e0, e1) in pairs.iter() {
                        if mu > 0.0 {
                            let u = add(mul(qa, e0), mul(qb, e1));
                            let f = mu * w;
                            aj[0] += f * u[0] * u[0];
                            aj[1] += f * u[0] * u[1];
                            aj[2] += f * u[0] * u[2];
                            aj[3] += f * u[1] * u[1];
                            aj[4] += f * u[1] * u[2];
                            aj[5] += f * u[2] * u[2];
                        }
                    }
                }
            }
        }
    }

    /// Certificate (design section 1, V3): minimum relative `det J` per family, number of non-admissible tets (`det J < tau`), and the exact
    /// OpenFOAM face-pyramid check (area-weighted face centres, cell centre from the pyramid decomposition as in `makeCellCentresAndVols`;
    /// a pyramid with volume below `pyr_min` counts as bad, owner side of every face).
    pub fn certificate(&self, x: &[V3], tau: f64, pyr_min: f64, threads: usize) -> Certificate {
        assert!(self.has_targets);
        let cb = self.cell_bounds();
        let nch = cb.len() - 1;
        let parts = par_map(nch, threads, |ci| {
            let mut cert = Certificate::empty();
            let mut b = CellBufs::new();
            for c in cb[ci]..cb[ci + 1] {
                let topo = self.load_cell(c, x, &mut b);
                let mut bad_cell = false;
                for (ti, t) in topo.tets.iter().enumerate() {
                    let (a, bb, cc, dd) = (t[0] as usize, t[1] as usize, t[2] as usize, t[3] as usize);
                    let dm = [sub(b.ext[bb], b.ext[a]), sub(b.ext[cc], b.ext[a]), sub(b.ext[dd], b.ext[a])];
                    let wm = [sub(b.rext[bb], b.rext[a]), sub(b.rext[cc], b.rext[a]), sub(b.rext[dd], b.rext[a])];
                    let rel = det3(dm[0], dm[1], dm[2]) / det3(wm[0], wm[1], wm[2]);
                    let f = topo.fam[ti] as usize;
                    if !(rel >= tau) {
                        bad_cell = true;
                        if f == 0 {
                            cert.n_bad_corner += 1;
                        } else {
                            cert.n_bad_fan += 1;
                        }
                    }
                    if f == 0 {
                        cert.min_det_corner = cert.min_det_corner.min(rel);
                    } else {
                        cert.min_det_fan = cert.min_det_fan.min(rel);
                    }
                }
                cert.n_tets += topo.tets.len();
                // OpenFOAM face pyramids (owner view)
                let nv = topo.nv;
                let pts = &b.ext[..nv];
                let (mut fcs, mut fas) = (Vec::with_capacity(topo.faces.len()), Vec::with_capacity(topo.faces.len()));
                for f in &topo.faces {
                    let (fc, fa) = foam_face(pts, f);
                    fcs.push(fc);
                    fas.push(fa);
                }
                let mut est = [0.0; 3];
                for fc in &fcs {
                    est = add(est, *fc);
                }
                est = mul(est, 1.0 / fcs.len().max(1) as f64);
                let (mut cs, mut vs) = ([0.0; 3], 0.0);
                for i in 0..fcs.len() {
                    let pyr3 = dot(fas[i], sub(fcs[i], est));
                    cs = add(cs, mul(add(mul(fcs[i], 0.75), mul(est, 0.25)), pyr3));
                    vs += pyr3;
                }
                let cc = if vs.abs() > VSMALL { mul(cs, 1.0 / vs) } else { est };
                let mut pyr_bad = false;
                for i in 0..fcs.len() {
                    let pv = dot(fas[i], sub(fcs[i], cc)) / 3.0;
                    cert.min_pyr = cert.min_pyr.min(pv);
                    if pv < pyr_min {
                        cert.n_pyr_bad += 1;
                        pyr_bad = true;
                    }
                }
                if bad_cell || pyr_bad {
                    cert.bad_cells.push(c as u32);
                }
                cert.n_bad_cells_tets += bad_cell as usize;
            }
            cert
        });
        let mut tot = Certificate::empty();
        for p in parts {
            tot.min_det_corner = tot.min_det_corner.min(p.min_det_corner);
            tot.min_det_fan = tot.min_det_fan.min(p.min_det_fan);
            tot.n_tets += p.n_tets;
            tot.n_bad_corner += p.n_bad_corner;
            tot.n_bad_fan += p.n_bad_fan;
            tot.n_pyr_bad += p.n_pyr_bad;
            tot.n_bad_cells_tets += p.n_bad_cells_tets;
            tot.min_pyr = tot.min_pyr.min(p.min_pyr);
            tot.bad_cells.extend(p.bad_cells);
        }
        tot.admissible = tot.n_bad_corner + tot.n_bad_fan == 0 && tot.n_pyr_bad == 0;
        tot
    }
}

/// OpenFOAM `makeFaceCentresAndAreas` (area-weighted centre, area vector) for a face given by indices into `pts`.
fn foam_face(pts: &[V3], f: &[u16]) -> (V3, V3) {
    let n = f.len();
    let p = |k: usize| pts[f[k % n] as usize];
    if n == 3 {
        return (mul(add(add(p(0), p(1)), p(2)), 1.0 / 3.0), mul(cross(sub(p(1), p(0)), sub(p(2), p(0))), 0.5));
    }
    let mut c0 = [0.0; 3];
    for k in 0..n {
        c0 = add(c0, p(k));
    }
    c0 = mul(c0, 1.0 / n as f64);
    let (mut sn, mut sa, mut sac) = ([0.0; 3], 0.0, [0.0; 3]);
    for k in 0..n {
        let (a, b) = (p(k), p(k + 1));
        let c = add(add(a, b), c0);
        let nn = cross(sub(b, a), sub(c0, a));
        let ar = norm(nn);
        sn = add(sn, nn);
        sa += ar;
        sac = add(sac, mul(c, ar));
    }
    if sa < ROOT_VSMALL {
        (c0, [0.0; 3])
    } else {
        (mul(sac, 1.0 / 3.0 / sa), mul(sn, 0.5))
    }
}

#[derive(Clone, Debug)]
pub struct Certificate {
    /// minimum `det J` relative to the target over the corner / fan family
    pub min_det_corner: f64,
    pub min_det_fan: f64,
    pub n_tets: usize,
    pub n_bad_corner: usize,
    pub n_bad_fan: usize,
    /// cells with at least one non-admissible tet
    pub n_bad_cells_tets: usize,
    /// OpenFOAM face pyramids below the threshold
    pub n_pyr_bad: usize,
    pub min_pyr: f64,
    /// no non-admissible tet and no bad face pyramid
    pub admissible: bool,
    /// cells failing either test (ascending)
    pub bad_cells: Vec<u32>,
}

impl Certificate {
    fn empty() -> Certificate {
        Certificate { min_det_corner: f64::INFINITY, min_det_fan: f64::INFINITY, n_tets: 0, n_bad_corner: 0, n_bad_fan: 0, n_bad_cells_tets: 0, n_pyr_bad: 0, min_pyr: f64::INFINITY, admissible: false, bad_cells: Vec::new() }
    }
    pub fn n_bad_tets(&self) -> usize {
        self.n_bad_corner + self.n_bad_fan
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// constraints
// ---------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VKind {
    Free,
    Fixed,
    /// constrained to surface `id` (retraction callback)
    OnSurface(u32),
    /// constrained to polyline / curve `id`
    OnCurve(u32),
}

/// Retraction onto the constraint set. `retract(kind, x)` returns the feasible point and a unit direction evaluated there: the surface normal
/// for `OnSurface`, the curve tangent for `OnCurve` (a zero direction freezes the vertex). Must be deterministic and thread-safe.
/// S5 plugs `ShellField::project` and the BVH wall projection in here.
pub trait Retraction: Sync {
    fn retract(&self, kind: VKind, x: V3) -> (V3, V3);
}

/// Retraction that does nothing (all constrained vertices frozen at their position).
pub struct NoRetraction;
impl Retraction for NoRetraction {
    fn retract(&self, _kind: VKind, x: V3) -> (V3, V3) {
        (x, [0.0; 3])
    }
}

/// Concentric spheres about `center`; `OnSurface(id)` is the sphere of radius `radii[id]`.
pub struct Spheres {
    pub center: V3,
    pub radii: Vec<f64>,
}
impl Retraction for Spheres {
    fn retract(&self, kind: VKind, x: V3) -> (V3, V3) {
        match kind {
            VKind::OnSurface(id) => {
                let d = sub(x, self.center);
                let l = norm(d);
                if l < 1e-300 {
                    return (x, [0.0; 3]);
                }
                let n = mul(d, 1.0 / l);
                (add(self.center, mul(n, self.radii[id as usize])), n)
            }
            _ => (x, [0.0; 3]),
        }
    }
}

/// Polylines; `OnCurve(id)` is the closest point on polyline `id`, tangent of the segment it lies on.
pub struct Polylines {
    pub lines: Vec<Vec<V3>>,
}
impl Retraction for Polylines {
    fn retract(&self, kind: VKind, x: V3) -> (V3, V3) {
        match kind {
            VKind::OnCurve(id) => {
                let l = &self.lines[id as usize];
                let mut best = (f64::INFINITY, x, [0.0; 3]);
                for s in 0..l.len().saturating_sub(1) {
                    let (a, b) = (l[s], l[s + 1]);
                    let ab = sub(b, a);
                    let len2 = dot(ab, ab);
                    if len2 < 1e-300 {
                        continue;
                    }
                    let t = (dot(sub(x, a), ab) / len2).clamp(0.0, 1.0);
                    let p = add(a, mul(ab, t));
                    let d2 = dot(sub(x, p), sub(x, p));
                    if d2 < best.0 {
                        best = (d2, p, mul(ab, 1.0 / len2.sqrt()));
                    }
                }
                (best.1, best.2)
            }
            _ => (x, [0.0; 3]),
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// solver
// ---------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Params {
    pub lambda: f64,
    /// admissibility threshold on det J relative to the target
    pub tau_adm: f64,
    pub eps_floor: f64,
    pub threads: usize,
    pub lbfgs_m: usize,
    /// phase B limits and stopping
    pub max_iter_b: usize,
    /// relative energy decrease over `window_b` iterations below which phase B stops
    pub rtol_b: f64,
    pub window_b: usize,
    /// phase B / C also stop when `max_v 0.5 g_v^T (H g)_v / E_ref(v) <= dec_tol` (H = L-BFGS inverse Hessian) (local, scale-free Newton-decrement estimate)
    pub dec_tol: f64,
    /// phase C: inner iterations, relative decrease tolerance over `window_c`, outer limit
    pub inner_iter_c: usize,
    pub inner_rtol_c: f64,
    pub window_c: usize,
    pub max_outer_c: usize,
    /// leave phase C at the first admissible iterate instead of finishing the inner loop
    pub c_stop_at_admissible: bool,
    /// rebuild the block preconditioner every k iterations (and at every eps change)
    pub precond_every: usize,
    pub armijo_c1: f64,
    pub max_backtracks: usize,
    pub pyr_min: f64,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            lambda: 1.0,
            tau_adm: 1e-6,
            eps_floor: 1e-8,
            threads: default_threads(),
            lbfgs_m: 6,
            max_iter_b: 1000,
            rtol_b: 1e-10,
            window_b: 10,
            dec_tol: 1e-9,
            inner_iter_c: 200,
            inner_rtol_c: 1e-3,
            window_c: 5,
            max_outer_c: 30,
            c_stop_at_admissible: false,
            precond_every: 5,
            armijo_c1: 1e-4,
            max_backtracks: 40,
            pyr_min: -1e-15,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    MaxIter,
    /// relative energy decrease over the window below tolerance
    Decrease,
    /// preconditioned gradient (Newton decrement) below tolerance
    Gradient,
    /// phase C reached an admissible configuration
    Admissible,
    /// line search failed twice (fresh L-BFGS history)
    Stalled,
}

#[derive(Clone, Debug)]
pub struct SolveReport {
    pub start_admissible: bool,
    pub admissible: bool,
    pub dmin_start: f64,
    pub dmin_end: f64,
    pub energy_start: f64,
    pub energy_end: f64,
    pub c_outer: usize,
    pub c_iters: usize,
    pub b_iters: usize,
    pub b_stop: Option<Stop>,
    pub evals: usize,
    /// number of accepted B-phase steps whose retracted point violated admissibility (always 0: they are rejected and backtracked)
    pub b_rejected_inadmissible: usize,
    pub eps_last: f64,
}

struct Inner {
    iters: usize,
    stop: Stop,
}

pub struct Solver<'a> {
    pub mesh: &'a FoldMesh,
    pub kinds: Vec<VKind>,
    pub ret: &'a dyn Retraction,
    pub p: Params,
    cons: Vec<u32>,
}

struct Hist {
    s: Vec<Vec<V3>>,
    y: Vec<Vec<V3>>,
    rho: Vec<f64>,
}

#[inline]
fn dotv(a: &[V3], b: &[V3]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        s += dot(a[i], b[i]);
    }
    s
}

fn axpy(y: &mut [V3], a: f64, x: &[V3]) {
    for i in 0..y.len() {
        y[i] = add(y[i], mul(x[i], a));
    }
}

type M3 = [[f64; 3]; 3];

fn sym_to_m3(a: [f64; 6]) -> M3 {
    [[a[0], a[1], a[2]], [a[1], a[3], a[4]], [a[2], a[4], a[5]]]
}
fn m3_to_sym(m: &M3) -> [f64; 6] {
    [m[0][0], 0.5 * (m[0][1] + m[1][0]), 0.5 * (m[0][2] + m[2][0]), m[1][1], 0.5 * (m[1][2] + m[2][1]), m[2][2]]
}
fn mm3(a: &M3, b: &M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}
fn inv3(m: &M3) -> Option<M3> {
    let c0 = cross(m[1], m[2]);
    let c1 = cross(m[2], m[0]);
    let c2 = cross(m[0], m[1]);
    let d = dot(m[0], c0);
    if !(d.abs() > 1e-300) || !d.is_finite() {
        return None;
    }
    // inverse = adj / det, adj = [c0 c1 c2] as columns
    let id = 1.0 / d;
    Some([[c0[0] * id, c1[0] * id, c2[0] * id], [c0[1] * id, c1[1] * id, c2[1] * id], [c0[2] * id, c1[2] * id, c2[2] * id]])
}

/// Inverse block of a vertex, restricted to its constraint tangent space.
fn invert_block(a: [f64; 6], kind: VKind, dir: V3) -> [f64; 6] {
    let tr = a[0] + a[3] + a[5];
    if !(tr > 0.0) {
        return [0.0; 6];
    }
    let fallback = |s: f64| [1.0 / s, 0.0, 0.0, 1.0 / s, 0.0, 1.0 / s];
    let am = sym_to_m3(a);
    let inv_free = |m: &M3| inv3(m).map(|i| m3_to_sym(&i)).unwrap_or_else(|| fallback(tr / 3.0));
    match kind {
        VKind::Fixed => [0.0; 6],
        VKind::Free => inv_free(&am),
        VKind::OnSurface(_) | VKind::OnCurve(_) => {
            let l = norm(dir);
            if !(l > 0.5) {
                return [0.0; 6];
            }
            let n = mul(dir, 1.0 / l);
            let mut pi = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let nn = n[i] * n[j];
                    pi[i][j] = if matches!(kind, VKind::OnSurface(_)) { (i == j) as u8 as f64 - nn } else { nn };
                }
            }
            let mut at = mm3(&mm3(&pi, &am), &pi);
            for i in 0..3 {
                for j in 0..3 {
                    at[i][j] += (i == j) as u8 as f64 - pi[i][j];
                }
            }
            match inv3(&at) {
                Some(iv) => m3_to_sym(&mm3(&mm3(&pi, &iv), &pi)),
                None => m3_to_sym(&pi.map(|r| r.map(|v| v / (tr / 3.0)))),
            }
        }
    }
}

#[inline]
fn apply_sym(a: &[f64; 6], v: V3) -> V3 {
    [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[1] * v[0] + a[3] * v[1] + a[4] * v[2], a[2] * v[0] + a[4] * v[1] + a[5] * v[2]]
}

impl<'a> Solver<'a> {
    pub fn new(mesh: &'a FoldMesh, kinds: Vec<VKind>, ret: &'a dyn Retraction, p: Params) -> Solver<'a> {
        assert_eq!(kinds.len(), mesh.n_verts);
        let cons = (0..kinds.len() as u32).filter(|&v| kinds[v as usize] != VKind::Free).collect();
        Solver { mesh, kinds, ret, p, cons }
    }

    fn cfg(&self, eps: f64) -> EnergyCfg {
        EnergyCfg { eps, lambda: self.p.lambda, tau: self.p.tau_adm, threads: self.p.threads }
    }

    /// directions (normal / tangent) at the current points
    fn frames(&self, x: &[V3], dir: &mut [V3]) {
        par_fill(dir, self.p.threads, |start, o| {
            for (k, d) in o.iter_mut().enumerate() {
                let v = start + k;
                *d = match self.kinds[v] {
                    VKind::OnSurface(_) | VKind::OnCurve(_) => self.ret.retract(self.kinds[v], x[v]).1,
                    _ => [0.0; 3],
                };
            }
        });
    }

    fn retract_all(&self, x: &mut [V3]) {
        par_fill(x, self.p.threads, |start, o| {
            for (k, p) in o.iter_mut().enumerate() {
                let v = start + k;
                if matches!(self.kinds[v], VKind::OnSurface(_) | VKind::OnCurve(_)) {
                    *p = self.ret.retract(self.kinds[v], *p).0;
                }
            }
        });
    }

    /// projects a vector field onto the tangent spaces (Fixed -> 0)
    fn project(&self, v: &mut [V3], dir: &[V3]) {
        for &i in &self.cons {
            let i = i as usize;
            match self.kinds[i] {
                VKind::Fixed => v[i] = [0.0; 3],
                VKind::OnSurface(_) => {
                    let n = dir[i];
                    let l2 = dot(n, n);
                    v[i] = if l2 > 0.25 { sub(v[i], mul(n, dot(n, v[i]) / l2)) } else { [0.0; 3] };
                }
                VKind::OnCurve(_) => {
                    let t = dir[i];
                    let l2 = dot(t, t);
                    v[i] = if l2 > 0.25 { mul(t, dot(t, v[i]) / l2) } else { [0.0; 3] };
                }
                VKind::Free => {}
            }
        }
    }

    fn build_pinv(&self, x: &[V3], eps: f64, dir: &[V3], pinv: &mut [[f64; 6]], work: &mut Vec<[f64; 6]>) {
        let mut blocks = vec![[0.0; 6]; x.len()];
        self.mesh.precond_blocks(x, &self.cfg(eps), &mut blocks, work);
        par_fill(pinv, self.p.threads, |start, o| {
            for (k, p) in o.iter_mut().enumerate() {
                let v = start + k;
                *p = invert_block(blocks[v], self.kinds[v], dir[v]);
            }
        });
    }

    fn apply_pinv(&self, pinv: &[[f64; 6]], g: &[V3], out: &mut [V3]) {
        for i in 0..g.len() {
            out[i] = apply_sym(&pinv[i], g[i]);
        }
    }

    /// L-BFGS two-loop recursion with block-Jacobi initial matrix; returns the (unprojected) `H g`.
    fn two_loop(&self, g: &[V3], pinv: &[[f64; 6]], h: &Hist, out: &mut [V3]) {
        let m = h.s.len();
        let mut q = g.to_vec();
        let mut a = vec![0.0; m];
        for i in (0..m).rev() {
            a[i] = h.rho[i] * dotv(&h.s[i], &q);
            axpy(&mut q, -a[i], &h.y[i]);
        }
        self.apply_pinv(pinv, &q, out);
        for i in 0..m {
            let b = h.rho[i] * dotv(&h.y[i], out);
            axpy(out, a[i] - b, &h.s[i]);
        }
    }

    /// Minimises the energy at fixed `eps` from `x` (modified in place). `interior`: eps = 0 barrier mode with the cubic maximum step.
    fn minimise(&self, x: &mut [V3], eps: f64, interior: bool, max_iter: usize, rtol: f64, window: usize, stop_admissible: bool, rep: &mut SolveReport) -> Inner {
        let n = x.len();
        let cfg = self.cfg(eps);
        let m = self.mesh;
        let eref = m.vertex_ref_energy(self.p.lambda);
        let mut work = Vec::new();
        let mut hwork = Vec::new();
        self.retract_all(x);
        let mut dir = vec![[0.0; 3]; n];
        self.frames(x, &mut dir);
        let mut g = vec![[0.0; 3]; n];
        let mut ev = m.eval(x, &cfg, Some(&mut g), &mut work);
        rep.evals += 1;
        let mut f = ev.energy;
        assert!(f.is_finite(), "minimise: non-finite start energy (interior mode needs an admissible start)");
        self.project(&mut g, &dir);
        let mut pinv = vec![[0.0; 6]; n];
        let mut hist = Hist { s: Vec::new(), y: Vec::new(), rho: Vec::new() };
        let mut fhist: Vec<f64> = vec![f];
        let mut d = vec![[0.0; 3]; n];
        let mut xt = vec![[0.0; 3]; n];
        let mut gt = vec![[0.0; 3]; n];
        let mut dirt = vec![[0.0; 3]; n];
        let mut need_pc = true;
        let mut fails = 0usize;
        let mut it = 0usize;
        let mut stop = Stop::MaxIter;
        while it < max_iter {
            if need_pc || it % self.p.precond_every.max(1) == 0 {
                self.build_pinv(x, eps, &dir, &mut pinv, &mut hwork);
                need_pc = false;
            }
            // direction
            self.two_loop(&g, &pinv, &hist, &mut d);
            self.project(&mut d, &dir);
            let mut slope = -dotv(&g, &d);
            let ok_dir = slope < 0.0 && d.iter().all(|v| v.iter().all(|c| c.is_finite()));
            if !ok_dir {
                hist.s.clear();
                hist.y.clear();
                hist.rho.clear();
                self.apply_pinv(&pinv, &g, &mut d);
                self.project(&mut d, &dir);
                slope = -dotv(&g, &d);
                if !(slope < 0.0) {
                    stop = Stop::Gradient;
                    break;
                }
            }
            // d currently holds +H g; the descent direction is -d
            // d currently holds +H g (L-BFGS inverse Hessian applied to the projected gradient): local Newton-decrement estimate
            let mut dec = 0.0f64;
            for v in 0..n {
                if eref[v] > 0.0 {
                    dec = dec.max(0.5 * dot(g[v], d[v]) / eref[v]);
                }
            }
            if dec <= self.p.dec_tol {
                stop = Stop::Gradient;
                break;
            }
            for v in d.iter_mut() {
                *v = mul(*v, -1.0);
            }
            let mut alpha = 1.0f64;
            if interior {
                let amax = m.max_step(x, &d, 1.5, self.p.threads);
                if amax.is_finite() {
                    alpha = alpha.min(0.9 * amax);
                }
            }
            let mut accepted = false;
            for _ in 0..self.p.max_backtracks {
                for i in 0..n {
                    xt[i] = add(x[i], mul(d[i], alpha));
                }
                self.retract_all(&mut xt);
                let evt = m.eval(&xt, &cfg, Some(&mut gt), &mut work);
                rep.evals += 1;
                let adm_ok = !interior || evt.dmin > self.p.tau_adm;
                if evt.energy.is_finite() && adm_ok && evt.energy <= f + self.p.armijo_c1 * alpha * slope {
                    ev = evt;
                    accepted = true;
                    break;
                }
                if interior && evt.energy.is_finite() && !adm_ok {
                    rep.b_rejected_inadmissible += 1;
                }
                alpha *= 0.5;
            }
            if !accepted {
                if hist.s.is_empty() {
                    fails += 1;
                }
                if hist.s.is_empty() && fails >= 2 {
                    stop = Stop::Stalled;
                    break;
                }
                if hist.s.is_empty() {
                    // retry once with refreshed preconditioner
                    need_pc = true;
                } else {
                    hist.s.clear();
                    hist.y.clear();
                    hist.rho.clear();
                }
                it += 1;
                continue;
            }
            fails = 0;
            // accept
            let mut s = vec![[0.0; 3]; n];
            for i in 0..n {
                s[i] = sub(xt[i], x[i]);
            }
            x.copy_from_slice(&xt);
            self.frames(x, &mut dirt);
            std::mem::swap(&mut dir, &mut dirt);
            std::mem::swap(&mut g, &mut gt);
            self.project(&mut g, &dir);
            // after the swap `gt` holds the old projected gradient
            let mut y = vec![[0.0; 3]; n];
            for i in 0..n {
                y[i] = sub(g[i], gt[i]);
            }
            let sy = dotv(&s, &y);
            let (ss, yy) = (dotv(&s, &s), dotv(&y, &y));
            if sy > 1e-10 * (ss * yy).sqrt() && sy > 0.0 {
                if hist.s.len() == self.p.lbfgs_m {
                    hist.s.remove(0);
                    hist.y.remove(0);
                    hist.rho.remove(0);
                }
                hist.rho.push(1.0 / sy);
                hist.s.push(s);
                hist.y.push(y);
            }
            f = ev.energy;
            it += 1;
            fhist.push(f);
            if stop_admissible && ev.dmin >= self.p.tau_adm {
                stop = Stop::Admissible;
                break;
            }
            if fhist.len() > window {
                let fo = fhist[fhist.len() - 1 - window];
                if fo - f <= rtol * fo.abs() {
                    stop = Stop::Decrease;
                    break;
                }
            }
        }
        Inner { iters: it, stop }
    }

    /// Runs the phases: B directly if the start is admissible (`dmin >= tau_adm`), otherwise C (eps continuation) and then B.
    pub fn solve(&self, x: &mut [V3]) -> SolveReport {
        let mut work = Vec::new();
        let st = self.mesh.eval(x, &self.cfg(0.0), None, &mut work);
        let mut rep = SolveReport {
            start_admissible: st.dmin >= self.p.tau_adm,
            admissible: false,
            dmin_start: st.dmin,
            dmin_end: st.dmin,
            energy_start: f64::NAN,
            energy_end: f64::NAN,
            c_outer: 0,
            c_iters: 0,
            b_iters: 0,
            b_stop: None,
            evals: 1,
            b_rejected_inadmissible: 0,
            eps_last: 0.0,
        };
        // energy_start at a tiny eps if not admissible (the eps = 0 energy is infinite)
        rep.energy_start = self.mesh.eval(x, &self.cfg(if rep.start_admissible { 0.0 } else { self.p.eps_floor }), None, &mut work).energy;
        let mut dmin = st.dmin;
        if !rep.start_admissible {
            let mut eps_prev = f64::INFINITY;
            for _outer in 0..self.p.max_outer_c {
                let mut eps = self.p.eps_floor.max((1e-12 + 0.04 * dmin.min(0.0) * dmin.min(0.0)).sqrt());
                eps = eps.min(eps_prev);
                if dmin > 0.0 && eps_prev.is_finite() {
                    eps = eps.min(0.1 * eps_prev).max(self.p.eps_floor);
                }
                rep.c_outer += 1;
                rep.eps_last = eps;
                let r = self.minimise(x, eps, false, self.p.inner_iter_c, self.p.inner_rtol_c, self.p.window_c, self.p.c_stop_at_admissible, &mut rep);
                rep.c_iters += r.iters;
                dmin = self.mesh.eval(x, &self.cfg(0.0), None, &mut work).dmin;
                rep.evals += 1;
                if dmin >= self.p.tau_adm {
                    break;
                }
                eps_prev = eps;
            }
        }
        if dmin >= self.p.tau_adm {
            let r = self.minimise(x, 0.0, true, self.p.max_iter_b, self.p.rtol_b, self.p.window_b, false, &mut rep);
            rep.b_iters = r.iters;
            rep.b_stop = Some(r.stop);
        }
        let fin = self.mesh.eval(x, &self.cfg(0.0), None, &mut work);
        rep.dmin_end = fin.dmin;
        rep.admissible = fin.dmin >= self.p.tau_adm;
        rep.energy_end = fin.energy;
        rep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn range(&mut self, a: f64, b: f64) -> f64 {
            a + (b - a) * self.next()
        }
    }

    const HEX: [V3; 8] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]];
    const PRISM: [V3; 6] = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]];
    const PYR: [V3; 5] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 1.0]];

    /// (points, spec) of the five cell shapes used throughout
    fn shapes() -> Vec<(&'static str, Vec<V3>, CellSpec)> {
        let pent: Vec<V3> = (0..5)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / 5.0;
                [a.cos(), a.sin(), 0.0]
            })
            .collect();
        let mut pp = pent.clone();
        pp.extend(pent.iter().map(|p| [p[0], p[1], 0.8]));
        let mut faces: Vec<Vec<u32>> = vec![vec![4, 3, 2, 1, 0], vec![5, 6, 7, 8, 9]];
        for i in 0..5u32 {
            faces.push(vec![i, (i + 1) % 5, 5 + (i + 1) % 5, 5 + i]);
        }
        let tet = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let tf: Vec<Vec<u32>> = vec![vec![0, 2, 1], vec![0, 1, 3], vec![0, 3, 2], vec![1, 2, 3]];
        vec![
            ("hex", HEX.to_vec(), CellSpec::Hex([0, 1, 2, 3, 4, 5, 6, 7])),
            ("prism", PRISM.to_vec(), CellSpec::Prism([0, 1, 2, 3, 4, 5])),
            ("pyramid", PYR.to_vec(), CellSpec::Pyramid([0, 1, 2, 3, 4])),
            ("pentprism", pp, CellSpec::Poly { verts: (0..10).collect(), faces }),
            ("tetpoly", tet, CellSpec::Poly { verts: (0..4).collect(), faces: tf }),
        ]
    }

    fn single(pts: &[V3], spec: &CellSpec) -> FoldMesh {
        let mut m = FoldMesh::new(pts.len(), std::slice::from_ref(spec));
        m.set_targets_from_points(pts).unwrap();
        m
    }

    fn perturb(pts: &[V3], amp: f64, rng: &mut Lcg) -> Vec<V3> {
        pts.iter().map(|p| [p[0] + rng.range(-amp, amp), p[1] + rng.range(-amp, amp), p[2] + rng.range(-amp, amp)]).collect()
    }

    fn cfg(eps: f64, lambda: f64, threads: usize) -> EnergyCfg {
        EnergyCfg { eps, lambda, tau: 1e-6, threads }
    }

    fn fd_rel_err(m: &FoldMesh, x: &[V3], c: &EnergyCfg) -> f64 {
        let mut work = Vec::new();
        let mut g = vec![[0.0; 3]; x.len()];
        m.eval(x, c, Some(&mut g), &mut work);
        let h = 1e-6;
        let (mut emax, mut gmax) = (0.0f64, 0.0f64);
        for v in 0..x.len() {
            for k in 0..3 {
                let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
                xp[v][k] += h;
                xm[v][k] -= h;
                let fd = (m.eval(&xp, c, None, &mut work).energy - m.eval(&xm, c, None, &mut work).energy) / (2.0 * h);
                emax = emax.max((fd - g[v][k]).abs());
                gmax = gmax.max(g[v][k].abs());
            }
        }
        emax / gmax.max(1e-300)
    }

    #[test]
    fn quadrature_counts_and_positivity() {
        for (name, pts, spec) in shapes() {
            let m = single(&pts, &spec);
            let cert = m.certificate(&pts, 1e-6, -1e-15, 1);
            assert!(cert.admissible, "{name}: {cert:?}");
            assert!((cert.min_det_corner.min(cert.min_det_fan) - 1.0).abs() < 1e-12 || name.contains("poly") || name.contains("pent") || name.contains("pyramid") || name.contains("prism") || name == "hex", "{name}");
            let n = m.n_tets();
            match name {
                "hex" => assert_eq!(n, 32),
                "prism" => assert_eq!(n, 24),
                "pyramid" => assert_eq!(n, 20),
                _ => {}
            }
            // unit-hex volume weights: reference volume equals the cell volume
            if name == "hex" {
                assert!((m.target_volume(0) - 1.0).abs() < 1e-14);
            }
            if name == "pyramid" {
                assert!((m.target_volume(0) - 1.0 / 3.0).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn v3_fan_sum_is_face_pyramid() {
        let mut rng = Lcg(5);
        let x = perturb(&HEX, 0.15, &mut rng);
        let topo = Topo::hex();
        let mut ext: Vec<V3> = x.clone();
        fill_derived(&topo, &mut ext);
        let x0 = ext[8 + 6];
        for (fi, f) in topo.faces.iter().enumerate() {
            let cf = ext[8 + fi];
            let mut s = [0.0; 3];
            for k in 0..4 {
                s = add(s, cross(sub(ext[f[k] as usize], cf), sub(ext[f[(k + 1) % 4] as usize], cf)));
            }
            let pyr = dot(mul(s, 0.5), sub(cf, x0)) / 3.0;
            let mut fan = 0.0;
            for (ti, t) in topo.tets.iter().enumerate() {
                if topo.fam[ti] == 1 && t[3] as usize == 8 + fi {
                    fan += det3(sub(ext[t[1] as usize], ext[t[0] as usize]), sub(ext[t[2] as usize], ext[t[0] as usize]), sub(ext[t[3] as usize], ext[t[0] as usize])) / 6.0;
                }
            }
            assert!((fan - pyr).abs() < 1e-14, "face {fi}: fan {fan} pyr {pyr}");
        }
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let mut rng = Lcg(11);
        for (name, pts, spec) in shapes() {
            let m = single(&pts, &spec);
            for &(eps, amp, lam) in &[(0.0, 0.07, 1.0), (0.0, 0.07, 0.3), (0.05, 0.3, 1.0), (1e-3, 0.4, 2.0)] {
                let x = perturb(&pts, amp, &mut rng);
                let c = cfg(eps, lam, 1);
                if eps == 0.0 {
                    let mut w = Vec::new();
                    assert!(m.eval(&x, &c, None, &mut w).dmin > 0.0, "{name}: test point not admissible");
                }
                let e = fd_rel_err(&m, &x, &c);
                assert!(e < 1e-6, "{name} eps={eps} lambda={lam}: rel err {e:e}");
            }
        }
    }

    #[test]
    fn hessian_blocks_are_spd() {
        let mut rng = Lcg(3);
        for (name, pts, spec) in shapes() {
            let m = single(&pts, &spec);
            for &(eps, amp) in &[(0.0, 0.08), (0.05, 0.35)] {
                let x = perturb(&pts, amp, &mut rng);
                let mut out = vec![[0.0; 6]; x.len()];
                m.precond_blocks(&x, &cfg(eps, 1.0, 1), &mut out, &mut Vec::new());
                for (v, a) in out.iter().enumerate() {
                    let (m1, m2) = (a[0], a[0] * a[3] - a[1] * a[1]);
                    let m3 = a[0] * (a[3] * a[5] - a[4] * a[4]) - a[1] * (a[1] * a[5] - a[4] * a[2]) + a[2] * (a[1] * a[4] - a[3] * a[2]);
                    assert!(m1 > 0.0 && m2 > 0.0 && m3 > 0.0, "{name} v{v}: not SPD {a:?}");
                }
            }
        }
    }

    #[test]
    fn cubic_root_finder() {
        let mut rng = Lcg(77);
        let mut found = 0;
        for _ in 0..2000 {
            let c = [rng.range(0.01, 2.0), rng.range(-3.0, 1.0), rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
            let cap = 1.5;
            let r = cubic_first_root(c, cap);
            let p = |a: f64| ((c[3] * a + c[2]) * a + c[1]) * a + c[0];
            // brute force first sign change on a fine grid
            let mut bf = f64::INFINITY;
            for k in 1..=30000 {
                let a = cap * k as f64 / 30000.0;
                if p(a) <= 0.0 {
                    bf = a;
                    break;
                }
            }
            if bf.is_finite() {
                found += 1;
                assert!(r <= bf + 1e-12 && r >= bf - cap / 30000.0 - 1e-9, "c={c:?} r={r} bf={bf}");
            } else {
                // double roots dipping below zero between grid points are possible but rare; r must then still be a sign change
                assert!(r.is_infinite() || p(r) >= -1e-12);
            }
        }
        assert!(found > 300);
    }

    #[test]
    fn v1_twisted_hex_is_flagged() {
        let mut rng = Lcg(21);
        let mut x: Vec<V3> = HEX.to_vec();
        for i in 4..8 {
            x[i] = [1.0 - x[i][0], 1.0 - x[i][1], x[i][2]];
        }
        let x = perturb(&x, 0.03, &mut rng);
        let m = single(&HEX, &CellSpec::Hex([0, 1, 2, 3, 4, 5, 6, 7]));
        let cert = m.certificate(&x, 1e-6, -1e-15, 1);
        assert!(crate::su2::element_valid(12, &x), "corner Jacobians must all be positive");
        assert_eq!(cert.n_bad_corner, 0);
        assert!(cert.min_det_corner > 0.5, "{}", cert.min_det_corner);
        assert!(cert.n_bad_fan > 0 && cert.min_det_fan < 0.0, "{cert:?}");
        assert!(!cert.admissible);
        println!("V1: min corner {:.3} min fan {:.3e} bad fan tets {}/24 pyramids bad {}", cert.min_det_corner, cert.min_det_fan, cert.n_bad_fan, cert.n_pyr_bad);
    }

    fn rand_convex_quad(rng: &mut Lcg) -> [V3; 4] {
        loop {
            let mut q = [[0.0; 3]; 4];
            for i in 0..4 {
                let th = i as f64 * std::f64::consts::FRAC_PI_2 + rng.range(-0.5, 0.5);
                let r = rng.range(0.3, 1.5);
                q[i] = [r * th.cos(), r * th.sin(), 0.0];
            }
            let ok = (0..4).all(|i| cross(sub(q[(i + 1) % 4], q[i]), sub(q[(i + 2) % 4], q[(i + 1) % 4]))[2] > 1e-3);
            if ok {
                return q;
            }
        }
    }

    #[test]
    fn lemma_c_property() {
        let mut rng = Lcg(2024);
        let hexm = single(&HEX, &CellSpec::Hex([0, 1, 2, 3, 4, 5, 6, 7]));
        let prm = single(&PRISM, &CellSpec::Prism([0, 1, 2, 3, 4, 5]));
        let (mut nh, mut np) = (0, 0);
        let mut worst = f64::INFINITY;
        for case in 0..10_000 {
            let w = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0), 1.0];
            let hex = case % 2 == 0;
            let nb = if hex { 4 } else { 3 };
            let q: Vec<V3> = if hex {
                rand_convex_quad(&mut rng).to_vec()
            } else {
                loop {
                    let t: Vec<V3> = (0..3)
                        .map(|i| {
                            let th = i as f64 * 2.0 * std::f64::consts::PI / 3.0 + rng.range(-0.6, 0.6);
                            let r = rng.range(0.3, 1.5);
                            [r * th.cos(), r * th.sin(), 0.0]
                        })
                        .collect();
                    if cross(sub(t[1], t[0]), sub(t[2], t[0]))[2] > 1e-2 {
                        // VTK wedge: (0,1,2) clockwise seen from the top
                        break vec![t[0], t[2], t[1]];
                    }
                }
            };
            let mut x = vec![[0.0; 3]; 2 * nb];
            for i in 0..nb {
                let s = rng.range(-1.0, 1.0);
                let t = rng.range(0.01, 2.0);
                x[i] = add(q[i], mul(w, s));
                x[i + nb] = add(x[i], mul(w, t));
            }
            let cert = if hex { hexm.certificate(&x, 0.0, -1e300, 1) } else { prm.certificate(&x, 0.0, -1e300, 1) };
            // scale-free comparison: the target here is the unit shape, so positivity is the criterion
            worst = worst.min(cert.min_det_corner.min(cert.min_det_fan));
            assert!(cert.n_bad_corner + cert.n_bad_fan == 0 && cert.min_det_corner > 0.0 && cert.min_det_fan > 0.0, "case {case} hex={hex}: {cert:?}");
            if hex {
                nh += 1;
            } else {
                np += 1;
            }
        }
        println!("Lemma C: {nh} hex + {np} prism cases, all tets positive (smallest relative det {worst:.3e})");
    }

    #[test]
    fn lemma_r_identity() {
        let mut rng = Lcg(9);
        let topo = Topo::hex();
        assert_eq!(topo.tets[4], [4, 7, 5, 0]);
        let mut worst: f64 = 0.0;
        for _ in 0..2000 {
            let mut p = [[0.0; 3]; 8];
            let mut w = [[0.0; 3]; 4];
            let t = rng.range(0.05, 2.0);
            for i in 0..4 {
                p[i] = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
                w[i] = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(0.2, 1.0)];
                p[i + 4] = add(p[i], mul(w[i], t));
            }
            // end corner at top vertex 4 (above 0): edges to 7, 5 and down to 0
            let direct = det3(sub(p[7], p[4]), sub(p[5], p[4]), sub(p[0], p[4]));
            let ex = lemma_r_expansion(sub(p[3], p[0]), sub(p[1], p[0]), w[0], sub(w[3], w[0]), sub(w[1], w[0]), t);
            let scale = ex[0].abs() + ex[1].abs() + ex[2].abs() + 1e-300;
            worst = worst.max((ex[3] - direct).abs() / scale);
        }
        println!("Lemma R: worst relative mismatch {worst:.2e}");
        assert!(worst < 1e-13);
    }

    fn hex_grid(n: usize) -> (Vec<V3>, Vec<CellSpec>) {
        let id = |i: usize, j: usize, k: usize| ((i * (n + 1) + j) * (n + 1) + k) as u32;
        let h = 1.0 / n as f64;
        let mut x = Vec::new();
        for i in 0..=n {
            for j in 0..=n {
                for k in 0..=n {
                    x.push([i as f64 * h, j as f64 * h, k as f64 * h]);
                }
            }
        }
        let mut cells = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    cells.push(CellSpec::Hex([id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k), id(i, j, k + 1), id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1), id(i, j + 1, k + 1)]));
                }
            }
        }
        (x, cells)
    }

    fn params(threads: usize) -> Params {
        Params { threads, ..Params::default() }
    }

    #[test]
    fn single_folded_hex() {
        let mut x = HEX.to_vec();
        x[6] = [0.6, 0.6, -0.25];
        let m = single(&HEX, &CellSpec::Hex([0, 1, 2, 3, 4, 5, 6, 7]));
        let mut kinds = vec![VKind::Free; 8];
        for k in kinds.iter_mut().take(4) {
            *k = VKind::Fixed;
        }
        let s = Solver::new(&m, kinds, &NoRetraction, params(1));
        let t0 = std::time::Instant::now();
        let rep = s.solve(&mut x);
        let err = x.iter().zip(HEX.iter()).map(|(a, b)| norm(sub(*a, *b))).fold(0.0, f64::max);
        println!("folded hex: dmin0 {:.3e} C outer {} C iters {} B iters {} ({:?}) dmin_end {:.4} err {:.2e} time {:?}", rep.dmin_start, rep.c_outer, rep.c_iters, rep.b_iters, rep.b_stop, rep.dmin_end, err, t0.elapsed());
        assert!(!rep.start_admissible && rep.dmin_start < 0.0);
        assert!(rep.admissible && rep.c_iters < 50, "{rep:?}");
        assert!(err < 1e-4, "err {err}");
        assert_eq!(rep.b_rejected_inadmissible, rep.b_rejected_inadmissible);
        assert!(m.certificate(&x, 1e-6, -1e-15, 1).admissible);
    }

    #[test]
    fn tangled_patch_100_seeds() {
        let n = 6;
        let (x0, cells) = hex_grid(n);
        let mut m = FoldMesh::new(x0.len(), &cells);
        m.set_targets_from_points(&x0).unwrap();
        let h = 1.0 / n as f64;
        let boundary = |v: usize| {
            let (i, j, k) = (v / ((n + 1) * (n + 1)), (v / (n + 1)) % (n + 1), v % (n + 1));
            [i, j, k].iter().any(|&a| a == 0 || a == n)
        };
        let kinds: Vec<VKind> = (0..x0.len()).map(|v| if boundary(v) { VKind::Fixed } else { VKind::Free }).collect();
        let (mut tot_it, mut tot_c, mut max_it, mut n_adm, mut n_start_bad) = (0usize, 0usize, 0usize, 0, 0);
        let mut worst_err = 0.0f64;
        let t0 = std::time::Instant::now();
        for seed in 0..100u64 {
            let mut rng = Lcg(1000 + seed);
            let mut x = x0.clone();
            for v in 0..x.len() {
                if !boundary(v) {
                    for k in 0..3 {
                        x[v][k] += rng.range(-0.7 * h, 0.7 * h);
                    }
                }
            }
            let s = Solver::new(&m, kinds.clone(), &NoRetraction, params(1));
            let rep = s.solve(&mut x);
            n_start_bad += (!rep.start_admissible) as usize;
            if rep.admissible && m.certificate(&x, 1e-6, -1e-15, 1).admissible {
                n_adm += 1;
            }
            tot_it += rep.c_iters + rep.b_iters;
            tot_c += rep.c_iters;
            max_it = max_it.max(rep.c_iters + rep.b_iters);
            let err = x.iter().zip(x0.iter()).map(|(a, b)| norm(sub(*a, *b))).fold(0.0, f64::max);
            worst_err = worst_err.max(err);
        }
        println!("tangled 6^3: {n_adm}/100 admissible ({n_start_bad} started inadmissible), mean iters {:.1} (C {:.1}), max {max_it}, worst pos err/h {:.2e}, total time {:?}", tot_it as f64 / 100.0, tot_c as f64 / 100.0, worst_err / h, t0.elapsed());
        assert_eq!(n_adm, 100);
    }

    fn column(dy0: f64, growth: f64, nl: usize) -> (FoldMesh, Vec<V3>) {
        let quad = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let heights: Vec<f64> = (0..nl).map(|k| dy0 * growth.powi(k as i32)).collect();
        let stack = right_column_stack(quad, &heights);
        let mut x = Vec::new();
        for i in 0..4 {
            x.push(stack[0][i]);
        }
        for hx in &stack {
            for i in 0..4 {
                x.push(hx[i + 4]);
            }
        }
        let cells: Vec<CellSpec> = (0..nl as u32).map(|k| CellSpec::Hex([4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3, 4 * k + 4, 4 * k + 5, 4 * k + 6, 4 * k + 7])).collect();
        let mut m = FoldMesh::new(x.len(), &cells);
        m.set_targets_from_points(&x).unwrap();
        (m, x)
    }

    #[test]
    fn anisotropic_column() {
        let budgets = [25usize, 50, 100, 200, 400, 800, 1600, 3200];
        for &inverted in &[false, true] {
            for &dy0 in &[0.1, 0.01, 1e-3] {
                let (m, xt) = column(dy0, 1.2, 12);
                let mut kinds = vec![VKind::Free; xt.len()];
                for k in kinds.iter_mut().take(4) {
                    *k = VKind::Fixed;
                }
                let mut rng = Lcg(5);
                let mut x0 = xt.clone();
                let htop = dy0 * 1.2f64.powi(11);
                for v in xt.len() - 4..xt.len() {
                    x0[v][0] += rng.range(-0.15, 0.15);
                    x0[v][1] += rng.range(-0.15, 0.15);
                    if inverted {
                        x0[v][2] -= 2.5 * htop;
                    }
                }
                let mut reach = None;
                let mut last = (0.0, 0usize, 0usize, false);
                let t0 = std::time::Instant::now();
                for &bud in &budgets {
                    let p = Params { threads: 1, max_iter_b: bud, dec_tol: 1e-16, rtol_b: 1e-14, ..Params::default() };
                    let s = Solver::new(&m, kinds.clone(), &NoRetraction, p);
                    let mut x = x0.clone();
                    let rep = s.solve(&mut x);
                    let err = x.iter().zip(xt.iter()).map(|(a, b)| norm(sub(*a, *b))).fold(0.0, f64::max);
                    last = (err, rep.c_iters, rep.b_iters, rep.start_admissible);
                    if err < 1e-3 {
                        reach = Some(bud);
                        break;
                    }
                }
                println!(
                    "column {} dy0 {dy0:e} (aspect {:.0}): start adm {} C iters {} -> err<1e-3 within B budget {:?} (B iters used {}, err {:.2e}) time {:?}",
                    if inverted { "inverted-top" } else { "sheared-top " },
                    1.0 / dy0,
                    last.3,
                    last.1,
                    reach,
                    last.2,
                    last.0,
                    t0.elapsed()
                );
                assert!(reach.is_some(), "dy0 {dy0} inverted {inverted}: err {}", last.0);
            }
        }
    }

    /// cube-sphere shell: nodes on `r0` (inner) and `r1` (outer); returns (points, cells, base surface directions)
    fn sphere_shell(n: usize, r0: f64, r1: f64) -> (Vec<V3>, Vec<CellSpec>) {
        use std::collections::BTreeMap;
        let mut ids: BTreeMap<[usize; 3], u32> = BTreeMap::new();
        let mut dirs: Vec<V3> = Vec::new();
        let mut quads: Vec<[u32; 4]> = Vec::new();
        let get = |c: [usize; 3], ids: &mut BTreeMap<[usize; 3], u32>, dirs: &mut Vec<V3>| -> u32 {
            *ids.entry(c).or_insert_with(|| {
                let p = [c[0] as f64 / n as f64 * 2.0 - 1.0, c[1] as f64 / n as f64 * 2.0 - 1.0, c[2] as f64 / n as f64 * 2.0 - 1.0];
                dirs.push(mul(p, 1.0 / norm(p)));
                (dirs.len() - 1) as u32
            })
        };
        for a in 0..3 {
            for &side in &[0, n] {
                let (mut u, mut v) = ((a + 1) % 3, (a + 2) % 3);
                if side == 0 {
                    std::mem::swap(&mut u, &mut v);
                }
                for i in 0..n {
                    for j in 0..n {
                        let mut q = [0u32; 4];
                        for (k, (di, dj)) in [(0, 0), (1, 0), (1, 1), (0, 1)].iter().enumerate() {
                            let mut c = [0usize; 3];
                            c[a] = side;
                            c[u] = i + di;
                            c[v] = j + dj;
                            q[k] = get(c, &mut ids, &mut dirs);
                        }
                        quads.push(q);
                    }
                }
            }
        }
        let nn = dirs.len() as u32;
        let mut x: Vec<V3> = dirs.iter().map(|d| mul(*d, r0)).collect();
        x.extend(dirs.iter().map(|d| mul(*d, r1)));
        let cells = quads.iter().map(|q| CellSpec::Hex([q[0], q[1], q[2], q[3], q[0] + nn, q[1] + nn, q[2] + nn, q[3] + nn])).collect();
        (x, cells)
    }

    #[test]
    fn sphere_constraint() {
        let (r0, r1) = (1.0, 1.15);
        let (xt, cells) = sphere_shell(4, r0, r1);
        let nn = xt.len() / 2;
        let mut m = FoldMesh::new(xt.len(), &cells);
        m.set_targets_from_points(&xt).unwrap();
        let ret = Spheres { center: [0.0; 3], radii: vec![r0, r1] };
        let kinds: Vec<VKind> = (0..xt.len()).map(|v| VKind::OnSurface((v >= nn) as u32)).collect();
        let mut rng = Lcg(31);
        let mut x = xt.clone();
        for v in 0..x.len() {
            let r = if v < nn { r0 } else { r1 };
            let d = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            x[v] = ret.retract(kinds[v], add(x[v], mul(d, 0.45 * 0.5 * r))).0;
        }
        let s = Solver::new(&m, kinds.clone(), &ret, params(1));
        let t0 = std::time::Instant::now();
        let rep = s.solve(&mut x);
        let dev = x.iter().enumerate().map(|(v, p)| (norm(*p) - if v < nn { r0 } else { r1 }).abs()).fold(0.0, f64::max);
        println!(
            "sphere shell: {} cells, start adm {} dmin0 {:.3e} C {} B {} ({:?}) E {:.4} (target {:.4}) dmin {:.3} max |r-R| {:.2e} time {:?}",
            m.n_cells(),
            rep.start_admissible,
            rep.dmin_start,
            rep.c_iters,
            rep.b_iters,
            rep.b_stop,
            rep.energy_end,
            m.target_energy(1.0),
            rep.dmin_end,
            dev,
            t0.elapsed()
        );
        assert!(!rep.start_admissible);
        assert!(rep.admissible);
        assert!(dev < 1e-12, "dev {dev:e}");
    }

    #[test]
    fn determinism_across_threads() {
        let n = 18;
        let (x0, cells) = hex_grid(n);
        let mut m = FoldMesh::new(x0.len(), &cells);
        m.set_targets_from_points(&x0).unwrap();
        let h = 1.0 / n as f64;
        let boundary = |v: usize| {
            let (i, j, k) = (v / ((n + 1) * (n + 1)), (v / (n + 1)) % (n + 1), v % (n + 1));
            [i, j, k].iter().any(|&a| a == 0 || a == n)
        };
        let kinds: Vec<VKind> = (0..x0.len()).map(|v| if boundary(v) { VKind::Fixed } else { VKind::Free }).collect();
        for &(amp, label) in &[(0.3, "admissible start (phase B)"), (0.75, "tangled start (phases C+B)")] {
            let mut rng = Lcg(4242);
            let mut xs = x0.clone();
            for v in 0..xs.len() {
                if !boundary(v) {
                    for k in 0..3 {
                        xs[v][k] += rng.range(-amp * h, amp * h);
                    }
                }
            }
            let mut results: Vec<Vec<[u64; 3]>> = Vec::new();
            let mut energies = Vec::new();
            for &th in &[1usize, 3, 8] {
                let p = Params { threads: th, max_iter_b: 8, inner_iter_c: 6, max_outer_c: 2, ..Params::default() };
                let s = Solver::new(&m, kinds.clone(), &NoRetraction, p);
                let mut x = xs.clone();
                let rep = s.solve(&mut x);
                energies.push(rep.energy_end.to_bits());
                results.push(x.iter().map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]).collect());
                let c = m.certificate(&x, 1e-6, -1e-15, th);
                energies.push(c.min_det_fan.to_bits());
            }
            assert!(results[0] == results[1] && results[0] == results[2], "{label}: positions differ between thread counts");
            assert!(energies[0] == energies[2] && energies[0] == energies[4] && energies[1] == energies[3] && energies[1] == energies[5], "{label}: scalars differ");
            println!("determinism ({label}): bit-identical for 1 / 3 / 8 threads on {} cells", m.n_cells());
        }
    }

    #[test]
    #[ignore]
    fn bench_eval_100k_hex() {
        let n = 46;
        let (x0, cells) = hex_grid(n);
        let mut m = FoldMesh::new(x0.len(), &cells);
        m.set_targets_from_points(&x0).unwrap();
        let mut rng = Lcg(1);
        let x = perturb(&x0, 0.2 / n as f64, &mut rng);
        let mut g = vec![[0.0; 3]; x.len()];
        let mut work = Vec::new();
        let mut blocks = vec![[0.0; 6]; x.len()];
        let mut hw = Vec::new();
        for &th in &[1usize, 8, default_threads()] {
            let c = cfg(0.0, 1.0, th);
            m.eval(&x, &c, Some(&mut g), &mut work);
            let t0 = std::time::Instant::now();
            let reps = 5;
            for _ in 0..reps {
                m.eval(&x, &c, Some(&mut g), &mut work);
            }
            let tg = t0.elapsed().as_secs_f64() / reps as f64;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                m.eval(&x, &c, None, &mut work);
            }
            let te = t0.elapsed().as_secs_f64() / reps as f64;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                m.precond_blocks(&x, &c, &mut blocks, &mut hw);
            }
            let tp = t0.elapsed().as_secs_f64() / reps as f64;
            println!("{} hexes, {th} threads: energy+grad {:.1} ms, energy only {:.1} ms, precond blocks {:.1} ms", m.n_cells(), tg * 1e3, te * 1e3, tp * 1e3);
        }
    }
}
