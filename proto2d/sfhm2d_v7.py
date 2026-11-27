#!/usr/bin/env python3
"""
SFHM-2D v7: shell-first hybrid mesher, 2D prototype.

v7 changes vs v6: native SU2 writer (--su2: tri/quad elements, polygons split by max-min-angle
fans, markers wall/farfield).

v6 changes vs v5: bulk size field h = hmin + grade*d (d = wall distance beyond the shell),
optional wake band from sharp trailing edges (--wake-len/--wake-dir/--wake-h/--wake-growth).
v5's quadtree refined only ~1 cell around the shell and 2:1 balance doubled h every 1-2 cells,
which gave a mesh-independent spurious pressure drag. --grade 0 reproduces v5.

v5 changes vs v4: graded extra BL columns next to convex-corner fans (--fan-grade,
1 disables = v4) so the column spacing steps down geometrically to the fan cells.

v4 changes vs v3: ingest defeaturing (clean_soup): opposite coincident segments cancel
(zero-width spikes, folds), open ends closer than --bridge*hwall are bridged (holes),
non-separating segments are dropped; every repair is reported. --no-clean reproduces v3.

v3 changes vs v2: y+-aware layers. The first cell is held at the spec dy0 everywhere;
where the shell is thinner than the nominal stack, the layer count N_i drops per column
(|N_a - N_b| <= 1 between neighbours, the extra layer closes with a top triangle) and the
growth ratio is re-solved per column. --layers fixedN reproduces v2.

v2 changes vs v1: shell from the eikonal equation |grad T| = 1/delta(x) (option B,
research doc section 11) instead of the warped level set; safety net after the
bulk is clipped (one shell loop per body, no wall crossing), on failure delta*0.7
on the offending wall segments and rebuild. --shell warped reproduces v1.

Pipeline (see ../RESEARCH_volume_meshing_v1.md section 4):
  1. polyline soup -> exact distance + nearest-point label (KD-tree on dense samples),
     sign from generalized winding number (robust to holes / overlaps)
  2. boundary-layer spec: delta (shell thickness), N layers, growth r
     variable thickness: delta_eff(s) = min(delta(s), 0.4*gap_half(s)), then a
     Lipschitz lower envelope along the wall so the offset cannot fold
  3. shell Gamma = { psi = 0 },  psi = delta(x) * (T(x) - 1), T from second-order fast
     marching on the local grid (v1: psi = SDF - delta_eff(label))
  4. bulk = quadtree (2:1 balanced) clipped to psi>=0 by marching-squares style
     polygon clipping; vertex snapping removes slivers; polygons stay polygons
     (OpenFOAM is polyhedral), optional midpoint subdivision not needed in 2D
  5. BL fill: Laplace phi in the shell (0 wall, 1 shell), trace -grad(phi) from
     every bulk boundary vertex down to the wall, place N layer nodes with
     geometric growth, snap wall nodes to exact wall + feature vertices
  6. write OpenFOAM polyMesh (extruded 1 cell), report metrics
"""
import numpy as np, time, os, json, argparse, math, sys
from scipy.spatial import cKDTree
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ----------------------------------------------------------------------------
# geometry library
# ----------------------------------------------------------------------------
def naca4(t=0.12, n=160, chord=1.0, closed_te=True):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))            # cosine spacing
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3
                  - (0.1036 if closed_te else 0.1015) * x**4)
    up = np.c_[x, yt][::-1]                  # TE -> LE on upper
    lo = np.c_[x, -yt][1:]                   # LE -> TE on lower
    pts = np.vstack([up, lo]) * chord
    # orientation: upper TE->LE then lower LE->TE is clockwise; make CCW
    return pts[::-1].copy()

def rot(pts, ang_deg, about=(0, 0)):
    a = np.deg2rad(ang_deg); c, s = np.cos(a), np.sin(a)
    p = pts - np.asarray(about)
    return np.c_[c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]] + np.asarray(about)

def square(side=1.0):
    h = side / 2
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]])

def _dedupe(polys, tol=1e-12):
    out = []
    for pts, closed in polys:
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) > tol: keep.append(i)
        if closed and np.linalg.norm(pts[keep[-1]] - pts[keep[0]]) <= tol: keep.pop()
        out.append((pts[keep], closed))
    return out

def geometry(name):
    return _dedupe(_geometry(name))

def _geometry(name):
    """returns list of (pts, closed) polylines"""
    if name == "airfoil":
        return [(naca4(), True)]
    if name.startswith("tmr"):                              # NASA TMR NACA 0012, sharp TE; "tmr<n>" = n points per side
        n = int(name[3:] or 200)
        x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n + 1)))
        yt = 0.594689181 * (0.298222773 * np.sqrt(x) - 0.127125232 * x - 0.357907906 * x**2 + 0.291984971 * x**3 - 0.105174606 * x**4)
        up = np.c_[x[::-1], yt[::-1]]; lo = np.c_[x[1:-1], -yt[1:-1]]
        return [(np.vstack([up, lo]), True)]
    if name == "square":
        return [(square(), True)]
    if name == "twoelement":
        main = naca4(t=0.12, n=160)
        flap = rot(naca4(t=0.10, n=120) * 0.35, -25.0) + np.array([0.98, -0.045])
        return [(main, True), (flap, True)]
    if name == "dirty":
        a = naca4(n=160)
        i0, i1 = 110, 114                                   # hole on the upper surface
        a = np.vstack([a[i1:], a[:i0 + 1]])                 # open polyline
        j = 40                                              # backtracking fold on the lower surface
        fold = np.vstack([a[:j + 1], a[j - 3:j][::-1], a[j - 2:j + 1], a[j + 1:]])
        k = 200                                             # zero-area sliver spike
        spike = np.vstack([fold[:k + 1], fold[k:k + 1] + np.array([0.0, -0.004]), fold[k:]])
        return [(spike, False)]
    raise ValueError(name)

# ----------------------------------------------------------------------------
# ingest defeaturing (v4): exact, reported repairs of the polyline soup
# ----------------------------------------------------------------------------
def _seg_winding(A, B, P):
    da = A[None] - P[:, None]; db = B[None] - P[:, None]
    cr = da[..., 0] * db[..., 1] - da[..., 1] * db[..., 0]
    return np.arctan2(cr, (da * db).sum(-1)).sum(1) / (2 * np.pi)

def _chain(segs):
    """directed segments (u, v) -> list of (vertex ids, closed); open chains first, then cycles"""
    out_ = {}; indeg = {}
    for u, v in segs:
        out_.setdefault(u, []).append(v); indeg[v] = indeg.get(v, 0) + 1
    used = set(); chains = []
    def walk(u):
        ids = [u]
        while out_.get(u):
            v = out_[u].pop()
            used.add((u, v)); ids.append(v); u = v
            if v == ids[0]: break
        return ids
    for u in list(out_):
        if indeg.get(u, 0) == 0:
            while out_.get(u): chains.append((walk(u), False))
    for u in list(out_):
        while out_.get(u):
            ids = walk(u)
            closed = len(ids) > 2 and ids[-1] == ids[0]
            chains.append((ids[:-1] if closed else ids, closed))
    return chains

def clean_soup(polys, bridge_tol, merge_tol=1e-9):
    """1. merge coincident vertices, drop zero-length segments
       2. cancel opposite coincident segment pairs (zero-width spikes, back-tracking folds) and duplicates
       3. bridge open ends closer than bridge_tol (holes, cracks)
       4. drop segments that do not separate inside from outside (winding equal on both sides)
       returns (polylines, report list of strings)"""
    rep = []
    allp = np.vstack([p for p, _ in polys]); q = merge_tol * np.ptp(allp, 0).max()
    vid = {}; V = []
    def gid(p):
        k = tuple(np.round(p / q).astype(np.int64))
        if k not in vid: vid[k] = len(V); V.append(np.asarray(p, float))
        return vid[k]
    segs = []; nzero = 0
    for pts, closed in polys:
        ids = [gid(p) for p in pts]; n = len(ids)
        for i in range(n if closed else n - 1):
            u, v = ids[i], ids[(i + 1) % n]
            if u == v: nzero += 1
            else: segs.append((u, v))
    if nzero: rep.append(f'{nzero} zero-length segments removed')
    net = {}
    for u, v in segs:
        k = (min(u, v), max(u, v)); net[k] = net.get(k, 0) + (1 if u < v else -1)
    ncancel = sum((sum(1 for u, v in segs if (min(u, v), max(u, v)) == k) - abs(c)) // 2 for k, c in net.items()) if len(segs) < 20000 else None
    kept = []
    ndup = 0
    for (u, v), c in net.items():
        if c == 0: continue
        if abs(c) > 1: ndup += abs(c) - 1
        kept.append((u, v) if c > 0 else (v, u))
    V = np.array(V)
    if ncancel: rep.append(f'{ncancel} opposite coincident segment pairs cancelled (zero-width spikes / folds)')
    if ndup: rep.append(f'{ndup} duplicate segments merged')
    used_v = {w for e in kept for w in e}
    if len(used_v) < len(V): rep.append(f'{len(V) - len(used_v)} vertices left isolated (spike tips / fold points)')
    chains = _chain(kept)
    # bridge open ends, greedy by gap width
    while True:
        opens = [i for i, (ids, cl) in enumerate(chains) if not cl]
        best = None
        for i in opens:
            e = V[chains[i][0][-1]]
            for j in opens:
                d = np.linalg.norm(V[chains[j][0][0]] - e)
                if d < bridge_tol and (best is None or d < best[0]): best = (d, i, j)
        if best is None: break
        d, i, j = best
        if i == j:
            chains[i] = (chains[i][0], True); rep.append(f'hole of width {d:.4g} closed by a bridging segment')
        else:
            chains[i] = (chains[i][0] + chains[j][0], False); chains.pop(j)
            rep.append(f'crack of width {d:.4g} between two open curves bridged')
    # non-separating segments
    A = []; B = []; owner = []
    for ci, (ids, cl) in enumerate(chains):
        n = len(ids)
        for t in range(n if cl else n - 1): A.append(V[ids[t]]); B.append(V[ids[(t + 1) % n]]); owner.append((ci, t))
    A = np.array(A); B = np.array(B)
    m = 0.5 * (A + B); t_ = B - A; L = np.linalg.norm(t_, axis=1); nrm = np.c_[-t_[:, 1], t_[:, 0]] / L[:, None]
    eps = 1e-3 * L[:, None]
    w1 = np.abs(_seg_winding(A, B, m + eps * nrm)) > 0.5; w2 = np.abs(_seg_winding(A, B, m - eps * nrm)) > 0.5
    dang = w1 == w2
    if dang.any():
        rep.append(f'{dang.sum()} non-separating segments removed (dangling / internal walls)')
        keep = [(tuple(A[k]), tuple(B[k])) for k in np.flatnonzero(~dang)]
        vid.clear(); V2 = []
        def gid2(p):
            k = tuple(np.round(np.array(p) / q).astype(np.int64))
            if k not in vid: vid[k] = len(V2); V2.append(np.array(p))
            return vid[k]
        chains = _chain([(gid2(a), gid2(b)) for a, b in keep]); V = np.array(V2)
    nopen = sum(1 for _, cl in chains if not cl)
    if nopen: rep.append(f'{nopen} curves remain open (gap > bridge tolerance {bridge_tol:.4g})')
    out = [(V[ids], cl) for ids, cl in chains if len(ids) >= 2]
    return out, rep

# ----------------------------------------------------------------------------
# implicit geometry: distance, label, winding
# ----------------------------------------------------------------------------
class Implicit:
    def __init__(self, polylines, hs):
        self.segs = []     # (a, b, curve id)
        samples, lab = [], []
        self.curve_pts = polylines
        for cid, (pts, closed) in enumerate(polylines):
            P = pts if not closed else np.vstack([pts, pts[:1]])
            for k in range(len(P) - 1):
                self.segs.append((P[k], P[k + 1], cid))
        self.A = np.array([s[0] for s in self.segs]); self.B = np.array([s[1] for s in self.segs])
        self.cid = np.array([s[2] for s in self.segs])
        # dense samples on segments for KD-tree (exact distance computed to the owning segment)
        for k, (a, b, c) in enumerate(self.segs):
            L = np.linalg.norm(b - a); m = max(2, int(np.ceil(L / hs)) + 1)
            t = np.linspace(0, 1, m)
            samples.append(a[None] + t[:, None] * (b - a)[None])
            lab.append(np.full(m, k))
        self.S = np.vstack(samples); self.Sseg = np.concatenate(lab)
        self.tree = cKDTree(self.S)
        # feature vertices: turning angle > 30 deg; type decided later (needs winding): convex solid corner or other
        feats = []; bis = []
        for pts, closed in polylines:
            n = len(pts)
            rng = range(n) if closed else range(1, n - 1)
            for i in rng:
                p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % n]
                d0 = (p1 - p0) / np.linalg.norm(p1 - p0); d1 = (p2 - p1) / np.linalg.norm(p2 - p1)
                ang = np.degrees(np.arccos(np.clip(np.dot(d0, d1), -1, 1)))
                if ang > 30:
                    feats.append(p1); b = d1 - d0; bis.append(b / (np.linalg.norm(b) + 1e-300))
            if not closed:
                feats += [pts[0], pts[-1]]; bis += [np.zeros(2), np.zeros(2)]
        self.features = np.array(feats) if feats else np.zeros((0, 2))
        self.feat_convex = np.zeros(len(feats), bool)
        if feats:
            test = self.features + 1e-5 * np.array(bis)
            self.feat_convex = (np.abs(self.winding(test)) > 0.5) & (np.linalg.norm(bis, axis=1) > 0)

    def seg_dist(self, P, k):
        a = self.A[k]; b = self.B[k]; ab = b - a
        t = np.clip(np.einsum('ij,ij->i', P - a, ab) / (np.einsum('ij,ij->i', ab, ab) + 1e-300), 0, 1)
        q = a + t[:, None] * ab
        return np.linalg.norm(P - q, axis=1), q

    def nearest(self, P):
        """exact-ish distance: nearest sample's segment plus the two neighbouring segments"""
        P = np.atleast_2d(P)
        _, idx = self.tree.query(P, k=1)
        k0 = self.Sseg[idx]
        best_d = np.full(len(P), np.inf); best_q = np.zeros_like(P); best_k = k0.copy()
        for off in (-1, 0, 1):
            k = np.clip(k0 + off, 0, len(self.segs) - 1)
            d, q = self.seg_dist(P, k)
            m = d < best_d
            best_d[m] = d[m]; best_q[m] = q[m]; best_k[m] = k[m]
        return best_d, best_q, best_k

    def winding(self, P, chunk=20000):
        P = np.atleast_2d(P); w = np.zeros(len(P))
        for s in range(0, len(P), chunk):
            p = P[s:s + chunk]
            da = self.A[None] - p[:, None]; db = self.B[None] - p[:, None]
            cr = da[..., 0] * db[..., 1] - da[..., 1] * db[..., 0]
            dt = (da * db).sum(-1)
            w[s:s + chunk] = np.arctan2(cr, dt).sum(1) / (2 * np.pi)
        return w

    def sdf(self, P):
        d, q, k = self.nearest(P)
        inside = np.abs(self.winding(P)) > 0.5
        return np.where(inside, -d, d), q, k

    def sdf_grid(self, X, Y, hg, stride=4):
        """same as sdf; winding exact within 3*stride*hg of the surface, elsewhere bilinear from an exact
        coarse grid (winding is harmonic, smooth away from the wall). v1 assigned one winding value per far
        component, which leaks through holes wider than 4*hg (dirty case: body interior marked fluid)."""
        G = np.c_[X.ravel(), Y.ravel()]
        d, q, k = self.nearest(G); d = d.reshape(X.shape)
        ny, nx = X.shape
        ic = np.r_[np.arange(0, ny, stride), ny - 1]; jc = np.r_[np.arange(0, nx, stride), nx - 1]
        ic = np.unique(ic); jc = np.unique(jc)
        Wc = self.winding(np.c_[X[np.ix_(ic, jc)].ravel(), Y[np.ix_(ic, jc)].ravel()]).reshape(len(ic), len(jc))
        from scipy.interpolate import RegularGridInterpolator
        w = RegularGridInterpolator((ic.astype(float), jc.astype(float)), Wc)(
            np.c_[np.repeat(np.arange(ny), nx), np.tile(np.arange(nx), ny)].astype(float)).reshape(X.shape)
        near = d <= 3 * stride * hg
        w[near] = self.winding(G[near.ravel()])
        inside = np.abs(w) > 0.5
        return np.where(inside, -d, d), q.reshape(X.shape + (2,)), k.reshape(X.shape), inside

# ----------------------------------------------------------------------------
# variable thickness: gap detection on a local grid + Lipschitz envelope
# ----------------------------------------------------------------------------
def thickness_field(imp, delta0, grid, sdf_g, q_g, k_g, gap_ratio=0.4, lip=0.7):
    """per-segment delta_eff. grid: (X, Y) meshgrid arrays."""
    nseg = len(imp.segs)
    dcand = np.full(nseg, delta0)
    X, Y = grid; h = X[0, 1] - X[0, 0]
    fluid = sdf_g > 0
    # medial pairs: 4-neighbour pairs of fluid nodes whose nearest wall points are far apart
    def pairs(sl1, sl2):
        f = fluid[sl1] & fluid[sl2]
        qa = q_g[sl1][f]; qb = q_g[sl2][f]
        da = sdf_g[sl1][f]; db = sdf_g[sl2][f]
        sep = np.linalg.norm(qa - qb, axis=1)
        dm = np.minimum(da, db)
        gap = (sep > 1.7 * dm) & (sep > 3 * h)
        ka = k_g[sl1][f][gap]; kb = k_g[sl2][f][gap]; g = dm[gap]
        np.minimum.at(dcand, ka, gap_ratio * g)
        np.minimum.at(dcand, kb, gap_ratio * g)
    pairs((slice(None), slice(0, -1)), (slice(None), slice(1, None)))
    pairs((slice(0, -1), slice(None)), (slice(1, None), slice(None)))
    return lipschitz_envelope(imp, dcand, lip)

def lipschitz_envelope(imp, d, lip=0.7):
    """Lipschitz lower envelope along each curve (closed or open) in arc length"""
    deff = d.copy()
    for cid in np.unique(imp.cid):
        ks = np.where(imp.cid == cid)[0]
        L = np.linalg.norm(imp.B[ks] - imp.A[ks], axis=1)
        v = deff[ks]
        closed = imp.curve_pts[cid][1]
        for _ in range(2 if closed else 1):
            for i in range(1, len(ks)):
                v[i] = min(v[i], v[i - 1] + lip * L[i - 1])
            for i in range(len(ks) - 2, -1, -1):
                v[i] = min(v[i], v[i + 1] + lip * L[i])
            if closed:
                v[0] = min(v[0], v[-1] + lip * L[-1]); v[-1] = min(v[-1], v[0] + lip * L[-1])
        deff[ks] = v
    return deff

# ----------------------------------------------------------------------------
# eikonal shell (option B): |grad T| = 1/delta(x), T = 0 on the wall, shell = {T = 1}
# ----------------------------------------------------------------------------
def eikonal(sdf_g, dl_g, hg, tmax):
    """second-order fast marching on the local grid, fluid side only.
    T = sdf/delta (exact) on fluid nodes within 1.5 hg of the wall; marching stops once T > tmax.
    returns T, inf where not reached (solid, beyond tmax)"""
    import heapq
    ny, nx = sdf_g.shape
    solid = sdf_g <= 0
    init = (~solid) & (sdf_g < 1.5 * hg)
    T = np.full(ny * nx, np.inf); T[init.ravel()] = (sdf_g / dl_g)[init]
    T = T.tolist()
    K = (solid | init).ravel().tolist()         # solid is never reached: far nodes have only fluid neighbours
    W2 = ((hg / dl_g) ** 2).ravel().tolist()     # (h * slowness)^2
    INF = math.inf

    def nbrs(n):
        i, j = divmod(n, nx)
        if j > 0: yield n - 1
        if j < nx - 1: yield n + 1
        if i > 0: yield n - nx
        if i < ny - 1: yield n + nx

    def upd(n):
        i, j = divmod(n, nx)
        terms = []                              # per axis: (t1, t2) upwind values, t2 = second node or inf
        for d, lo, hi, c in ((1, j > 0, j < nx - 1, j), (nx, i > 0, i < ny - 1, i)):
            m = nx if d == 1 else ny
            best = INF; best2 = INF
            for s in (-1, 1):
                if (s < 0 and not lo) or (s > 0 and not hi): continue
                n1 = n + s * d
                if K[n1] and T[n1] < best:
                    best = T[n1]; best2 = INF
                    if 0 <= c + 2 * s < m:
                        n2 = n + 2 * s * d
                        if K[n2] and T[n2] <= best: best2 = T[n2]
            if best < INF: terms.append((best, best2))
        for order in (2, 1):
            a = b = c = 0.0
            for t1, t2 in terms:
                if order == 2 and t2 < INF: al, tau = 2.25, (4 * t1 - t2) / 3
                else: al, tau = 1.0, t1
                a += al; b -= 2 * al * tau; c += al * tau * tau
            c -= W2[n]
            disc = b * b - 4 * a * c
            if disc >= 0:
                t = (-b + math.sqrt(disc)) / (2 * a)
                if t >= max(t1 for t1, _ in terms): return t
        return min(t1 for t1, _ in terms) + math.sqrt(W2[n])

    heap = []
    for n0 in np.flatnonzero(init.ravel()).tolist():
        for nb in nbrs(n0):
            if not K[nb]:
                t = upd(nb)
                if t < T[nb]: T[nb] = t; heapq.heappush(heap, (t, nb))
    while heap:
        t, n = heapq.heappop(heap)
        if K[n] or t > T[n]: continue
        K[n] = True
        if t > tmax: break
        for nb in nbrs(n):
            if not K[nb]:
                t2 = upd(nb)
                if t2 < T[nb]: T[nb] = t2; heapq.heappush(heap, (t2, nb))
    T = np.array(T).reshape(ny, nx); Km = np.array(K).reshape(ny, nx)
    T[~Km | solid] = np.inf
    return T

def poly_winding(L, pts):
    """winding number of closed polygon L around pts"""
    A = L; B = np.roll(L, -1, 0)
    da = A[None] - pts[:, None]; db = B[None] - pts[:, None]
    cr = da[..., 0] * db[..., 1] - da[..., 1] * db[..., 0]
    return np.arctan2(cr, (da * db).sum(-1)).sum(1) / (2 * np.pi)

def seg_cross(A, B, C, D, chunk=2000):
    """bool per segment AB: properly crosses any segment CD"""
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    hit = np.zeros(len(A), bool)
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk, None]; b = B[s:s + chunk, None]; c = C[None]; d = D[None]
        o1 = orient(a, b, c); o2 = orient(a, b, d); o3 = orient(c, d, a); o4 = orient(c, d, b)
        hit[s:s + chunk] = ((o1 * o2 < 0) & (o3 * o4 < 0)).any(1)
    return hit

def shell_check(imp, loops, P, psi_g, solid, sdf_g, q_g, k_g, hg, body_lab, body_rep, hres):
    """safety net: exactly one shell loop per body, no shell vertex in the solid, no shell edge crossing the wall.
    returns (reasons, wall segment ids whose delta should be reduced)"""
    why = []; kbad = set()
    nbody = len(body_rep); enc = np.zeros(nbody, int); enc_by = [[] for _ in range(nbody)]
    for li, loop in enumerate(loops):
        L = P[loop]
        s, q, k = imp.sdf(L)
        ins = s <= 0
        if ins.any(): why.append(f'{ins.sum()} shell vertices in solid'); kbad.update(k[ins].tolist())
        hit = seg_cross(L, np.roll(L, -1, 0), imp.A, imp.B)
        if hit.any():
            why.append(f'{hit.sum()} shell edges cross the wall')
            kbad.update(imp.nearest(0.5 * (L + np.roll(L, -1, 0))[hit])[2].tolist())
        e = np.abs(poly_winding(L, body_rep)) > 0.5 if nbody else np.zeros(0, bool)
        enc += e
        for b in np.flatnonzero(e): enc_by[b].append(li)
        if nbody and e.sum() == 0: why.append(f'shell loop {li} encloses no body'); kbad.update(k.tolist())
        if e.sum() > 1:
            why.append(f'shell loop {li} encloses {e.sum()} bodies')
            # collision zone: nodes in the shell or within hres of it (channel the bulk cannot resolve) whose
            # 4-neighbour has its nearest wall point on another curve, far away
            inshell = (~solid) & (psi_g < hres)
            for s1, s2 in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                           ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
                f = inshell[s1] & inshell[s2]
                ka = k_g[s1][f]; kb = k_g[s2][f]
                sep = np.linalg.norm(q_g[s1][f] - q_g[s2][f], axis=1); dm = np.minimum(sdf_g[s1][f], sdf_g[s2][f])
                m = (imp.cid[ka] != imp.cid[kb]) & (sep > 1.7 * dm) & (sep > 3 * hg)
                kbad.update(ka[m].tolist()); kbad.update(kb[m].tolist())
    for b in range(nbody):
        if enc[b] == 0:
            why.append(f'body {b} has no shell loop'); kbad.update(k_g[body_lab == b + 1].tolist())
        elif enc[b] > 1:
            why.append(f'body {b} inside {enc[b]} shell loops')
            for li in enc_by[b][1:]: kbad.update(imp.nearest(P[loops[li]])[2].tolist())
    return why, kbad

# ----------------------------------------------------------------------------
# quadtree
# ----------------------------------------------------------------------------
class Quadtree:
    def __init__(self, x0, y0, size, lmax):
        self.x0, self.y0, self.size, self.lmax = x0, y0, size, lmax
        self.leaves = {(0, 0, 0)}
        self.occ = [set() for _ in range(lmax + 1)]; self.occ[0].add((0, 0))
    def h(self, l): return self.size / 2 ** l
    def box(self, key):
        l, i, j = key; h = self.h(l)
        return self.x0 + i * h, self.y0 + j * h, h
    def split(self, key):
        l, i, j = key
        self.leaves.discard(key)
        for di in (0, 1):
            for dj in (0, 1):
                c = (l + 1, 2 * i + di, 2 * j + dj)
                self.leaves.add(c); self.occ[l + 1].add((c[1], c[2]))
    def balance(self):
        changed = True
        while changed:
            changed = False
            for key in list(self.leaves):
                l, i, j = key
                if l + 2 > self.lmax or key not in self.leaves: continue
                o = self.occ[l + 2]
                need = False
                for k in range(4):
                    if ((4 * i - 1, 4 * j + k) in o or (4 * i + 4, 4 * j + k) in o or
                        (4 * i + k, 4 * j - 1) in o or (4 * i + k, 4 * j + 4) in o):
                        need = True; break
                if need:
                    self.split(key); changed = True
    def finer_neighbor(self, key, side):
        l, i, j = key
        if l + 1 > self.lmax: return False
        if side == 'S': return (l + 1, 2 * i, 2 * j - 1) in self.leaves
        if side == 'E': return (l + 1, 2 * i + 2, 2 * j) in self.leaves
        if side == 'N': return (l + 1, 2 * i, 2 * j + 2) in self.leaves
        if side == 'W': return (l + 1, 2 * i - 1, 2 * j) in self.leaves
    def polygon_ikeys(self, key):
        """CCW integer-coordinate vertices (units of h(lmax)) including hanging midpoints"""
        l, i, j = key; u = 2 ** (self.lmax - l); hu = u // 2
        c = [(i * u, j * u), ((i + 1) * u, j * u), ((i + 1) * u, (j + 1) * u), (i * u, (j + 1) * u)]
        out = []
        for s, side in enumerate('SENW'):
            out.append(c[s])
            if self.finer_neighbor(key, side):
                a, b = c[s], c[(s + 1) % 4]
                out.append(((a[0] + b[0]) // 2, (a[1] + b[1]) // 2))
        return out

# ----------------------------------------------------------------------------
# main mesher
# ----------------------------------------------------------------------------
def build(args):
    T = {}; t0 = time.time()
    polys = geometry(args.geom)
    ingest_report = []
    if args.clean:
        polys, ingest_report = clean_soup(polys, args.bridge * args.hwall)
        for line in ingest_report: print('ingest:', line, file=sys.stderr)
    allp = np.vstack([p for p, _ in polys])
    bb0, bb1 = allp.min(0), allp.max(0)
    h_wall = args.hwall                        # target tangential spacing at the shell
    delta = args.delta; N = args.nlayers; r = args.growth
    imp = Implicit(polys, hs=h_wall / 4)
    T['ingest'] = time.time() - t0

    # ---- local grid for gap detection and Laplace ----
    t1 = time.time()
    hg = delta / args.gridres
    m = 4 * delta
    gx = np.arange(bb0[0] - m, bb1[0] + m + hg, hg); gy = np.arange(bb0[1] - m, bb1[1] + m + hg, hg)
    X, Y = np.meshgrid(gx, gy)                 # shape (ny, nx)
    G = np.c_[X.ravel(), Y.ravel()]
    sdf_g, q_g, k_g, solid = imp.sdf_grid(X, Y, hg)
    deff = thickness_field(imp, delta, (X, Y), sdf_g, q_g, k_g) if args.variable else np.full(len(imp.segs), delta)
    T['grid+thickness'] = time.time() - t1

    wake_src = imp.features[imp.feat_convex] if len(imp.features) else np.zeros((0, 2))
    if args.wake_len > 0 and args.wake_src_xmin is not None: wake_src = wake_src[wake_src[:, 0] >= args.wake_src_xmin]
    # ---- shell field (eikonal or warped level set) + safety net ----
    from scipy import ndimage
    ny, nx = X.shape
    # bodies = groups of curves whose solid regions touch on the grid (grid components alone split thin TEs)
    comp, ncomp = ndimage.label(solid)
    grp = list(range(len(imp.curve_pts)))
    def find(a):
        while grp[a] != a: a = grp[a]
        return a
    for ic in range(1, ncomp + 1):
        cs = np.unique(imp.cid[k_g[comp == ic]])
        for c2 in cs[1:]: grp[find(c2)] = find(cs[0])
    has_solid = np.zeros(len(grp), bool); has_solid[np.unique(imp.cid[k_g[solid]])] = True
    roots = sorted({find(c) for c in range(len(grp)) if has_solid[c]})
    node_body = np.array([find(c) for c in range(len(grp))])[imp.cid[k_g]]
    body_lab = np.zeros(X.shape, int); body_rep = []
    for ib, root in enumerate(roots):
        mb = solid & (node_body == root); body_lab[mb] = ib + 1
        body_rep.append(G[np.argmin(np.where(mb.ravel(), sdf_g.ravel(), np.inf))])
    body_rep = np.array(body_rep).reshape(-1, 2)
    retries = []; nscaled = np.zeros(len(imp.segs), bool); T['eikonal'] = 0.0
    for attempt in range(args.maxretry + 1):
        te = time.time()
        dl_g = deff[k_g]
        psi_g = sdf_g - dl_g
        if args.shell == 'eikonal':
            Tg = eikonal(sdf_g, dl_g, hg, args.tmax)
            reach = np.isfinite(Tg) & ~solid
            psi_g[reach] = dl_g[reach] * (Tg[reach] - 1)
            far_ = ~solid & ~reach
            psi_g[far_] = np.maximum(psi_g[far_], (args.tmax - 1) * dl_g[far_])
        T['eikonal'] += time.time() - te

        def psi(P):
            s, q, k = imp.sdf(P)
            p = s - deff[k]
            if args.shell == 'eikonal':
                fx = (P[:, 0] - gx[0]) / hg; fy = (P[:, 1] - gy[0]) / hg
                m = (fx >= 0) & (fy >= 0) & (fx < nx - 1) & (fy < ny - 1)
                j = np.floor(fx[m]).astype(int); i = np.floor(fy[m]).astype(int); a = fx[m] - j; b = fy[m] - i
                p[m] = ((1 - a) * (1 - b) * psi_g[i, j] + a * (1 - b) * psi_g[i, j + 1]
                        + (1 - a) * b * psi_g[i + 1, j] + a * b * psi_g[i + 1, j + 1])
            return p, s

        # ---- quadtree ----
        t2 = time.time()
        D = args.domain; qt = Quadtree(-D / 2, -D / 2 + 0.5 * (bb0[1] + bb1[1]), D, args.lmax)
        hmin = qt.h(args.lmax)
        # base refinement
        for _ in range(args.lmin): 
            for key in list(qt.leaves): qt.split(key)
        for l in range(args.lmin, args.lmax):
            keys = [k for k in qt.leaves if k[0] == l]
            if not keys: continue
            C = np.array([[qt.box(k)[0] + qt.box(k)[2] / 2, qt.box(k)[1] + qt.box(k)[2] / 2] for k in keys])
            p, s = psi(C)
            hl = qt.h(l)
            # refine cells whose centre is within ~1.2 cell sizes of Gamma, or inside the band/solid near Gamma
            ref = np.abs(p) < 1.2 * hl * np.sqrt(2)
            # also keep solid interior coarse: cells deep inside solid (s < -hl*2) are never used; skip refining them
            ref &= ~(s < -2 * hl * np.sqrt(2))
            # v6 size field: h(x) = hmin + grade * (wall distance beyond the shell) [+ wake band]; refine while
            # the cell is larger than h at its nearest point (centre distance minus half diagonal)
            if args.grade > 0:
                dmin = np.maximum(s - delta - hl / np.sqrt(2), 0)
                htgt = hmin + args.grade * dmin
                if args.wake_len > 0 and len(wake_src):
                    wd = np.array([np.cos(np.radians(args.wake_dir)), np.sin(np.radians(args.wake_dir))])
                    for src in wake_src:
                        rel = C - src; xa = rel @ wd; xp = np.abs(rel @ np.array([-wd[1], wd[0]]))
                        on = (xa > -hl) & (xa < args.wake_len)
                        hw = args.wake_h + args.wake_growth * np.maximum(xa, 0) + args.grade * np.maximum(xp - hl / np.sqrt(2), 0)
                        htgt = np.where(on, np.minimum(htgt, hw), htgt)
                ref |= (hl > htgt) & (s > -hl)
            for k, f in zip(keys, ref):
                if f: qt.split(k)
        qt.balance()
        T['quadtree'] = time.time() - t2

        # ---- vertices, psi at vertices, snapping ----
        t3 = time.time()
        cell_keys = [k for k in qt.leaves]
        polys_i = [qt.polygon_ikeys(k) for k in cell_keys]
        vid = {}; V = []
        for pg in polys_i:
            for ik in pg:
                if ik not in vid: vid[ik] = len(V); V.append(ik)
        Vi = np.array(V); Vx = qt.x0 + Vi[:, 0] * hmin; Vy = qt.y0 + Vi[:, 1] * hmin
        P = np.c_[Vx, Vy]
        pv, sv = psi(P)
        # local size at each vertex = smallest incident cell size
        vsize = np.full(len(V), np.inf)
        for key, pg in zip(cell_keys, polys_i):
            h = qt.h(key[0])
            for ik in pg: vsize[vid[ik]] = min(vsize[vid[ik]], h)
        snap = (np.abs(pv) < args.snap * vsize) & (vsize <= 2.01 * hmin)
        if snap.any():
            # move along -grad(psi) (numerical) to psi=0 (one Newton step, then exact: nearest point + delta along normal)
            Ps = P[snap]
            eps = 1e-6
            gx_ = (psi(Ps + [eps, 0])[0] - psi(Ps - [eps, 0])[0]) / (2 * eps)
            gy_ = (psi(Ps + [0, eps])[0] - psi(Ps - [0, eps])[0]) / (2 * eps)
            g2 = gx_**2 + gy_**2 + 1e-300
            Ps = Ps - (pv[snap] / g2)[:, None] * np.c_[gx_, gy_]
            P[snap] = Ps; pv[snap] = 0.0
        # cell polygons as vertex ids
        cells_bulk = []
        cut_cache = {}
        def cut(u, v):
            key = (min(u, v), max(u, v))
            if key not in cut_cache:
                a, b = P[u], P[v]; pa, pb = pv[u], pv[v]
                t = pa / (pa - pb)
                cut_cache[key] = len(P_list); P_list.append(a + t * (b - a)); pv_list.append(0.0)
            return cut_cache[key]
        P_list = list(P); pv_list = list(pv)
        ncut4 = 0
        for key, pg in zip(cell_keys, polys_i):
            ids = [vid[ik] for ik in pg]
            vals = [pv[i] for i in ids]
            if all(v > 0 for v in vals): cells_bulk.append(ids); continue
            if all(v <= 0 for v in vals): 
                if all(v < 0 for v in vals) or True: continue
            # clip
            out = []; entries = []
            n = len(ids)
            for k in range(n):
                u, v = ids[k], ids[(k + 1) % n]; pa, pb = pv[u], pv[v]
                if pa >= 0: out.append(u)
                if (pa > 0 and pb < 0) or (pa < 0 and pb > 0):
                    c = cut(u, v); 
                    if pa < 0: entries.append(len(out))
                    out.append(c)
            if len(entries) >= 2:
                x0, y0, h = qt.box(key); pc, _ = psi(np.array([[x0 + h / 2, y0 + h / 2]]))
                ncut4 += 1
                if pc[0] < 0:
                    # split into components at entry cuts
                    comps = []
                    for a_, b_ in zip(entries, entries[1:] + [entries[0] + len(out)]):
                        comp = [out[(t) % len(out)] for t in range(a_, b_)]
                        comps.append(comp)
                    for comp in comps:
                        if len(comp) >= 3: cells_bulk.append(comp)
                    continue
            # remove consecutive duplicates
            o2 = [out[0]]
            for t in out[1:]:
                if t != o2[-1]: o2.append(t)
            if o2[0] == o2[-1]: o2.pop()
            if len(o2) >= 3: cells_bulk.append(o2)
        P = np.array(P_list); pv = np.array(pv_list)
        T['bulk'] = time.time() - t3

        # ---- boundary edges, shell loops ----
        t4 = time.time()
        edge_cells = {}
        for ci, c in enumerate(cells_bulk):
            n = len(c)
            for k in range(n):
                u, v = c[k], c[(k + 1) % n]
                edge_cells.setdefault((min(u, v), max(u, v)), []).append((ci, u, v))
        nxt = {}
        box_edges = []
        xmin, xmax = qt.x0, qt.x0 + D; ymin, ymax = qt.y0, qt.y0 + D
        for e, lst in edge_cells.items():
            if len(lst) == 1:
                ci, u, v = lst[0]
                if pv[u] == 0 and pv[v] == 0 and not (abs(P[u, 0] - xmin) < 1e-12 or abs(P[u, 0] - xmax) < 1e-12 or abs(P[u, 1] - ymin) < 1e-12 or abs(P[u, 1] - ymax) < 1e-12):
                    nxt[u] = v
                else:
                    box_edges.append((ci, u, v))
            elif len(lst) > 2:
                raise RuntimeError("non-manifold bulk edge")
        loops = []; seen = set()
        for s in list(nxt):
            if s in seen: continue
            loop = [s]; seen.add(s); c = nxt[s]
            while c != s and c not in seen:
                loop.append(c); seen.add(c); c = nxt.get(c, s)
            if len(loop) >= 3: loops.append(loop)

        why, kbad = shell_check(imp, loops, P, psi_g, solid, sdf_g, q_g, k_g, hg, body_lab, body_rep, 2 * hmin)
        if not why: break
        kbad = np.array(sorted(kbad), int)
        retries.append(dict(attempt=attempt, reasons=why, n_segments=int(len(kbad))))
        print(f'shell check failed (attempt {attempt}): {"; ".join(why)} -> delta*0.7 on {len(kbad)} segments', file=sys.stderr)
        if attempt == args.maxretry or len(kbad) == 0:
            print('shell check: giving up, continuing with last shell', file=sys.stderr); break
        deff[kbad] *= 0.7; nscaled[kbad] = True
        deff = lipschitz_envelope(imp, deff, 0.7)
    # shell placement accuracy: true wall distance of shell vertices relative to delta_eff
    if loops:
        Ls = P[np.concatenate(loops)]; sL, _, kL = imp.sdf(Ls); ratio = sL / deff[kL]
        shell_ratio = dict(mean=float(ratio.mean()), min=float(ratio.min()), max=float(ratio.max()))
    else:
        shell_ratio = None


    # ---- v5: graded columns on the shell edges next to convex-corner fans ----
    # the first fan cell is ~dy0 wide, the neighbouring BL column one quadtree cell (~hmin) away: the face
    # between them has interpolation weight ~0.02. Extra shell vertices on that edge at fractions g^-m
    # (densest at the fan) give a geometric transition of the column spacing; the owning bulk polygon
    # takes them as collinear vertices.
    no_cap = set(); n_graded = 0; frac_of = {}       # graded vertex -> (fraction from the fan side, fan vertex, regular vertex)
    if args.fan_grade > 1 and len(imp.features) and loops:
        dy0g = args.dy0 if args.dy0 else delta * (r - 1) / (r**N - 1) if abs(r - 1) > 1e-9 else delta / N
        ft0 = cKDTree(imp.features); hs0 = h_wall / 4
        P_list = list(P)
        for li, loop in enumerate(loops):
            capm = np.zeros(len(loop), bool)
            d_, q_, k_ = imp.nearest(P[loop])
            df_, fi_ = ft0.query(q_)
            for a_ in range(len(loop)):
                fi = fi_[a_]
                capm[a_] = (imp.feat_convex[fi] and df_[a_] < 1.5 * hs0 and
                            np.linalg.norm(P[loop[a_]] - imp.features[fi]) < 1.3 * deff[k_[a_]] + 2 * hs0)
            new = []
            n = len(loop)
            for a_ in range(n):
                b_ = (a_ + 1) % n; u, v = loop[a_], loop[b_]
                new.append(u)
                if capm[a_] == capm[b_]: continue
                he = np.linalg.norm(P[v] - P[u]); fr = []; t = 1.0 / args.fan_grade
                while t * he > 1.5 * dy0g: fr.append(t); t /= args.fan_grade
                if not fr: continue
                fan_v, reg_v = (u, v) if capm[a_] else (v, u)
                xs = sorted(fr) if capm[a_] else sorted(fr, reverse=True)      # in loop order u -> v
                ids = []
                # wall side graded (fraction x), shell side evenly spaced (rank-matched, so both stay monotone)
                xsh = {x: (j + 1) / (len(fr) + 1) for j, x in enumerate(sorted(fr))}
                for x in xs:
                    ids.append(len(P_list)); P_list.append(P[fan_v] + xsh[x] * (P[reg_v] - P[fan_v])); no_cap.add(ids[-1])
                    frac_of[ids[-1]] = (x, fan_v, reg_v)
                # owning bulk cell: directed edge u -> v
                ci = [c_ for c_, uu, vv in edge_cells[(min(u, v), max(u, v))] if uu == u and vv == v][0]
                pg = cells_bulk[ci]; j = pg.index(u)
                cells_bulk[ci] = pg[:j + 1] + ids + pg[j + 1:]
                new += ids; n_graded += len(ids)
            loops[li] = new
        P = np.array(P_list)

    # ---- Laplace in the shell on the local grid ----
    shell = (~solid) & (psi_g < 0)
    phi = np.where(solid, 0.0, 1.0)
    idx = -np.ones(X.shape, int); unk = np.where(shell.ravel())[0]; idx.ravel()[unk] = np.arange(len(unk))
    I, J = np.unravel_index(unk, X.shape)
    rows = [np.arange(len(unk))]; cols = [np.arange(len(unk))]; vals = [np.full(len(unk), 4.0)]; rhs = np.zeros(len(unk))
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ii = I + di; jj = J + dj
        inb = (ii >= 0) & (jj >= 0) & (ii < ny) & (jj < nx)
        ii_c = np.clip(ii, 0, ny - 1); jj_c = np.clip(jj, 0, nx - 1)
        nb_shell = inb & shell[ii_c, jj_c]
        rows.append(np.arange(len(unk))[nb_shell]); cols.append(idx[ii_c, jj_c][nb_shell]); vals.append(np.full(nb_shell.sum(), -1.0))
        dir_ = inb & ~nb_shell
        rhs[dir_] += phi[ii_c, jj_c][dir_]
        rhs[~inb] += 1.0
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(unk), len(unk)))
    sol = spla.spsolve(A, rhs)
    phi.ravel()[unk] = sol
    T['laplace'] = time.time() - t4

    # ---- streamlines ----
    t5 = time.time()
    dy0s = args.dy0 if args.dy0 else delta * (r - 1) / (r**N - 1) if abs(r - 1) > 1e-9 else delta / N
    gphi_y, gphi_x = np.gradient(phi, hg)
    solid_f = solid.astype(float)
    def interp(F, p):
        fx = (p[0] - gx[0]) / hg; fy = (p[1] - gy[0]) / hg
        i = int(np.clip(np.floor(fy), 0, ny - 2)); j = int(np.clip(np.floor(fx), 0, nx - 2))
        a = fx - j; b = fy - i
        return ((1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i, j + 1] + (1 - a) * b * F[i + 1, j] + a * b * F[i + 1, j + 1])
    bl_cells = []; wall_pts = []; dy0 = []; ncol_all = []; ntrans = 0; bad = 0; nfeat_snap = 0; nonconvex = 0; ncap = 0
    feat_tree = cKDTree(imp.features) if len(imp.features) else None
    hs = h_wall / 4
    for loop in loops:
        cols_nodes = []      # per seed: list of node ids from wall (0) to shell (N)
        wall_xy = []
        cap = {}     # seed index in loop -> feature id
        for a_, s in enumerate(loop):
            d, q, k = imp.nearest(P[s][None])
            if feat_tree is not None:
                df, fi = feat_tree.query(q[0])
                if s not in no_cap and imp.feat_convex[fi] and df < 1.5 * hs and np.linalg.norm(P[s] - imp.features[fi]) < 1.3 * deff[k[0]] + 2 * hs:
                    cap[a_] = fi
        for a_, s in enumerate(loop):
            if a_ in cap:
                F = imp.features[cap[a_]]
                wall_xy.append(F.copy()); cols_nodes.append(np.array([P[s].copy(), F.copy()])); continue
            p = P[s].copy(); path = [p.copy()]
            # first step toward the wall along -grad(psi) direction = -(p - q)/|p-q| ... use nearest point
            d, q, k = imp.nearest(p[None]); dirw = (q[0] - p); dirw /= (np.linalg.norm(dirw) + 1e-300)
            p = p + dirw * min(0.5 * hg, 0.25 * d[0]); path.append(p.copy())
            for it in range(20000):
                g = np.array([interp(gphi_x, p), interp(gphi_y, p)]); gn = np.linalg.norm(g)
                if gn < 1e-12: 
                    d, q, k = imp.nearest(p[None]); g = (p - q[0]); gn = np.linalg.norm(g) + 1e-300
                step = -g / gn * (0.5 * hg)
                pn = p + step
                # exact winding near the wall: the bilinear solid mask lets a step enter the body by up to ~hg/2,
                # which folds the first layers once dy0 << hg (wall-resolved meshes)
                if (interp(solid_f, pn) > 0.5 or imp.nearest(pn[None])[0][0] < 2 * hg) and abs(imp.winding(pn[None])[0]) > 0.5:
                    # bisect for the wn=0.5 crossing
                    a, b = p, pn
                    for _ in range(12):
                        mid = 0.5 * (a + b)
                        if abs(imp.winding(mid[None])[0]) > 0.5: b = mid
                        else: a = mid
                    p = 0.5 * (a + b); path.append(p.copy()); break
                p = pn; path.append(p.copy())
                d, q, k = imp.nearest(p[None])
                if d[0] < 0.25 * hg:
                    p = q[0]; path.append(p.copy()); break
            # exact projection if a real wall is near
            d, q, k = imp.nearest(p[None])
            if d[0] < 1.5 * hs: p = q[0]; path[-1] = p.copy()
            wall_xy.append(p.copy()); cols_nodes.append(np.array(path))
        wall_xy = np.array(wall_xy)
        # graded columns (v5): the harmonic field fans out at convex corners, so shell spacing does not carry
        # to the wall; place the wall point at the same fraction between fan and regular wall points instead
        pos = {v_: i_ for i_, v_ in enumerate(loop)}
        for a_, s in enumerate(loop):
            if s in frac_of:
                x, fv, rv = frac_of[s]
                wt = wall_xy[pos[fv]] + x * (wall_xy[pos[rv]] - wall_xy[pos[fv]])
                qw = imp.nearest(wt[None])[1][0]
                wall_xy[a_] = qw; cols_nodes[a_] = np.array([P[s].copy(), qw.copy()])
        # feature snapping: each feature vertex grabs its nearest wall point
        if feat_tree is not None:
            wt = cKDTree(wall_xy)
            for fi, F in enumerate(imp.features):
                if imp.feat_convex[fi] and fi in cap.values(): continue
                d, i = wt.query(F)
                if d < 2.5 * h_wall:
                    wall_xy[i] = F; nfeat_snap += 1
        # per-column layer count (v3): first cell on spec dy0s, N_i from the streamline length, |N_a - N_b| <= 1
        nseed = len(loop)
        Ls = np.array([max(np.linalg.norm(np.diff(np.vstack([w, path[::-1][1:]]), axis=0), axis=1).sum(), 1e-12)
                       for path, w in zip(cols_nodes, wall_xy)])
        if args.layers == 'yplus':
            nstar = np.log1p(Ls * (r - 1) / dy0s) / np.log(r) if abs(r - 1) > 1e-9 else Ls / dy0s
            Ncol = np.clip(np.round(nstar).astype(int), 1, N)
            for _ in range(nseed):                     # Lipschitz-1 lower envelope around the loop
                Nn = np.minimum(Ncol, np.minimum(np.roll(Ncol, 1), np.roll(Ncol, -1)) + 1)
                if (Nn == Ncol).all(): break
                Ncol = Nn
        else:
            Ncol = np.full(nseed, N)
        ncol_all.append(Ncol)
        # place nodes along each streamline by geometric growth
        node_ids = []; feat_pid = {}
        for s, path, w, Ni, L in zip(loop, cols_nodes, wall_xy, Ncol, Ls):
            path = path[::-1].copy(); path[0] = w          # wall -> shell
            seg = np.linalg.norm(np.diff(path, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]
            if args.layers == 'yplus':
                # growth ratio r_i such that d1 * sum_{k<N_i} r_i^k = L (monotone in r_i, bisection);
                # second pass rescales d1 so the first node's true wall distance (not arc length) is dy0s
                d1 = dy0s
                for npass in range(2):
                    lo, hi = 0.2, 4.0
                    for _ in range(60):
                        rm = 0.5 * (lo + hi)
                        if d1 * np.sum(rm ** np.arange(Ni)) < L: lo = rm
                        else: hi = rm
                    tk = np.r_[0, np.cumsum(d1 * rm ** np.arange(Ni))]; tk *= L / tk[-1]
                    if npass == 0 and Ni > 1:
                        p1 = np.array([np.interp(tk[1], cum, path[:, 0]), np.interp(tk[1], cum, path[:, 1])])
                        hn = imp.nearest(p1[None])[0][0]
                        if hn > 0: d1 = min(dy0s * tk[1] / hn, 2 * dy0s, 0.9 * L / Ni)
            else:
                dy = L * (r - 1) / (r**Ni - 1) if abs(r - 1) > 1e-9 else L / Ni
                tk = np.array([dy * (r**k - 1) / (r - 1) if abs(r - 1) > 1e-9 else dy * k for k in range(Ni + 1)])
                tk[-1] = L
            px = np.interp(tk, cum, path[:, 0]); py = np.interp(tk, cum, path[:, 1])
            ids = []
            a_ = len(node_ids)
            for k in range(Ni):
                if k == 0 and a_ in cap:
                    fi = cap[a_]
                    if fi not in feat_pid: feat_pid[fi] = len(P_list); P_list.append(imp.features[fi].copy())
                    ids.append(feat_pid[fi]); continue
                ids.append(len(P_list)); P_list.append(np.array([px[k], py[k]]))
            ids.append(s)
            node_ids.append(ids); dy0.append(tk[1])
        wall_pts.append(wall_xy); ncap += len(cap)
        # feature-local smoothing: interior layer nodes within 3*delta of a feature vertex
        if feat_tree is not None and args.smooth > 0:
            near = np.array([feat_tree.query(w)[0] < 3 * delta for w in wall_xy])
            for it in range(args.smooth):
                for a in range(nseed):
                    if not near[a]: continue
                    am, ap_ = (a - 1) % nseed, (a + 1) % nseed
                    for k in range(1, min(Ncol[a], Ncol[am], Ncol[ap_])):
                        i = node_ids[a][k]
                        P_list[i] = 0.5 * P_list[i] + 0.25 * (P_list[node_ids[am][k]] + P_list[node_ids[ap_][k]])
        # cells: between consecutive seeds; where N differs by one, the extra layer closes with a top triangle
        for a in range(nseed):
            b = (a + 1) % nseed
            ia, ib = node_ids[a], node_ids[b]; na, nb = Ncol[a], Ncol[b]
            cand = [[ia[k], ia[k + 1], ib[k + 1], ib[k]] for k in range(min(na, nb))]
            if na == nb + 1: cand.append([ia[nb], ia[nb + 1], ib[nb]]); ntrans += 1
            elif nb == na + 1: cand.append([ia[na], ib[na + 1], ib[na]]); ntrans += 1
            for quad in cand:
                pts = np.array([P_list[q] for q in quad])
                area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
                if area < 0: quad = quad[::-1]; area = -area
                # dedupe
                q2 = [quad[0]]
                for t in quad[1:]:
                    if t != q2[-1]: q2.append(t)
                if q2[0] == q2[-1]: q2.pop()
                if len(q2) < 3 or area < 1e-18: bad += 1; continue
                if len(q2) == 4:
                    pp = np.array([P_list[q] for q in q2]); e = np.roll(pp, -1, 0) - pp
                    cr = e[:, 0] * np.roll(e, -1, 0)[:, 1] - e[:, 1] * np.roll(e, -1, 0)[:, 0]
                    if cr.min() <= 0: nonconvex += 1
                bl_cells.append(q2)
    P = np.array(P_list)
    T['layers'] = time.time() - t5
    T['total'] = time.time() - t0

    # ---- assemble and write ----
    cells = cells_bulk + bl_cells
    used = np.zeros(len(P), bool)
    for c in cells: used[c] = True
    remap = -np.ones(len(P), int); remap[used] = np.arange(used.sum())
    P = P[used]; cells = [[int(remap[q]) for q in c] for c in cells]
    wall_edge_set = set()
    for loop_i, loop in enumerate(loops):
        pass
    info = dict(geom=args.geom, n_bulk=len(cells_bulk), n_bl=len(bl_cells), n_cells=len(cells),
                n_seeds=sum(len(l) for l in loops), n_loops=len(loops), bad_bl_cells=bad, nonconvex_bl_quads=nonconvex, n_cap_seeds=ncap,
                dy0_min=float(np.min(dy0)) if dy0 else None, dy0_max=float(np.max(dy0)) if dy0 else None,
                dy0_mean=float(np.mean(dy0)) if dy0 else None, feature_snaps=nfeat_snap,
                deff_min=float(deff.min()), deff_max=float(deff.max()), n_ambiguous_cells=ncut4,
                laplace_unknowns=int(len(unk)), timings=T,
                shell=args.shell, shell_retries=retries, n_segments_thinned=int(nscaled.sum()),
                ingest_report=ingest_report, n_graded_fan_columns=n_graded, shell_dist_over_delta=shell_ratio, layers=args.layers, dy0_spec=dy0s, n_transition_tris=ntrans,
                ncol_min=int(min(c.min() for c in ncol_all)) if ncol_all else None,
                frac_cols_reduced=float(np.mean(np.concatenate(ncol_all) < N)) if ncol_all else None)
    out = args.out
    os.makedirs(out, exist_ok=True)
    write_polymesh(out, P, cells, (xmin, xmax, ymin, ymax), args.dz)
    write_system(out)
    if args.su2: info['su2'] = write_su2(os.path.join(out, 'mesh.su2'), P, cells, (xmin, xmax, ymin, ymax))
    with open(os.path.join(out, 'sfhm_info.json'), 'w') as f: json.dump(info, f, indent=1)
    if args.plot:
        plot(out, P, cells[:len(cells_bulk)], cells[len(cells_bulk):], polys, loops, (bb0, bb1), delta)
    return info

# ----------------------------------------------------------------------------
def write_polymesh(case, P, cells, box, dz):
    xmin, xmax, ymin, ymax = box
    npt = len(P)
    edge_cells = {}
    for ci, c in enumerate(cells):
        n = len(c)
        for k in range(n):
            u, v = c[k], c[(k + 1) % n]
            edge_cells.setdefault((min(u, v), max(u, v)), []).append((ci, u, v))
    internal = []; bnd = {'wall': [], 'inlet': [], 'outlet': [], 'top': [], 'bottom': []}
    for e, lst in edge_cells.items():
        if len(lst) == 2:
            (c1, u1, v1), (c2, u2, v2) = lst
            if c1 < c2: internal.append((c1, c2, u1, v1))
            else: internal.append((c2, c1, u2, v2))
        elif len(lst) == 1:
            ci, u, v = lst[0]
            mx = 0.5 * (P[u] + P[v]); tol = 1e-9
            if abs(mx[0] - xmin) < tol: bnd['inlet'].append((ci, u, v))
            elif abs(mx[0] - xmax) < tol: bnd['outlet'].append((ci, u, v))
            elif abs(mx[1] - ymax) < tol: bnd['top'].append((ci, u, v))
            elif abs(mx[1] - ymin) < tol: bnd['bottom'].append((ci, u, v))
            else: bnd['wall'].append((ci, u, v))
        else:
            raise RuntimeError(f"edge with {len(lst)} cells")
    internal.sort(key=lambda t: (t[0], t[1]))
    faces = []; owner = []; neigh = []
    for c1, c2, u, v in internal:
        faces.append([u, v, v + npt, u + npt]); owner.append(c1); neigh.append(c2)
    patches = []
    for name in ['wall', 'inlet', 'outlet', 'top', 'bottom']:
        start = len(faces)
        for ci, u, v in bnd[name]:
            faces.append([u, v, v + npt, u + npt]); owner.append(ci)
        patches.append((name, 'wall' if name == 'wall' else 'patch', start, len(faces) - start))
    start = len(faces)
    for ci, c in enumerate(cells):
        faces.append(list(reversed(c))); owner.append(ci)
    for ci, c in enumerate(cells):
        faces.append([q + npt for q in c]); owner.append(ci)
    patches.append(('frontAndBack', 'empty', start, len(faces) - start))
    d = os.path.join(case, 'constant', 'polyMesh'); os.makedirs(d, exist_ok=True)
    def hdr(cls, obj, note=None):
        s = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n" % cls
        if note: s += '    note "%s";\n' % note
        s += "    location \"constant/polyMesh\";\n    object %s;\n}\n" % obj
        return s
    with open(os.path.join(d, 'points'), 'w') as f:
        f.write(hdr('vectorField', 'points')); f.write(f"{2*npt}\n(\n")
        for z in (0.0, dz):
            for p in P: f.write(f"({p[0]:.12g} {p[1]:.12g} {z})\n")
        f.write(")\n")
    with open(os.path.join(d, 'faces'), 'w') as f:
        f.write(hdr('faceList', 'faces')); f.write(f"{len(faces)}\n(\n")
        for fc in faces: f.write(f"{len(fc)}(" + " ".join(map(str, fc)) + ")\n")
        f.write(")\n")
    note = f"nPoints: {2*npt} nCells: {len(cells)} nFaces: {len(faces)} nInternalFaces: {len(internal)}"
    with open(os.path.join(d, 'owner'), 'w') as f:
        f.write(hdr('labelList', 'owner', note)); f.write(f"{len(owner)}\n(\n" + "\n".join(map(str, owner)) + "\n)\n")
    with open(os.path.join(d, 'neighbour'), 'w') as f:
        f.write(hdr('labelList', 'neighbour', note)); f.write(f"{len(neigh)}\n(\n" + "\n".join(map(str, neigh)) + "\n)\n")
    with open(os.path.join(d, 'boundary'), 'w') as f:
        f.write(hdr('polyBoundaryMesh', 'boundary')); f.write(f"{len(patches)}\n(\n")
        for name, typ, st, n in patches:
            f.write(f"    {name}\n    {{\n        type {typ};\n        nFaces {n};\n        startFace {st};\n    }}\n")
        f.write(")\n")

def _split_polygon(P, c):
    """convex polygon (possibly with collinear hanging vertices) -> triangles/quads for element-based solvers.
    tri/convex quad kept; otherwise a fan from the apex that maximises the minimum triangle angle"""
    n = len(c)
    pts = P[c]
    def angles(tri):
        a, b, d = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        out = []
        for p, q, r in ((a, b, d), (b, d, a), (d, a, b)):
            u = q - p; v = r - p
            out.append(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300), -1, 1)))
        return out
    if n == 3: return [list(c)]
    if n == 4:
        e = np.roll(pts, -1, 0) - pts; cr = e[:, 0] * np.roll(e, -1, 0)[:, 1] - e[:, 1] * np.roll(e, -1, 0)[:, 0]
        if (cr > 1e-14 * np.abs(cr).max()).all(): return [list(c)]
    best = None
    for a in range(n):
        tris = [(a, (a + k) % n, (a + k + 1) % n) for k in range(1, n - 1)]
        q = min(min(angles(t)) for t in tris)
        if best is None or q > best[0]: best = (q, tris)
    return [[c[i] for i in t] for t in best[1]]

def write_su2(path, P, cells, box, wall_name='wall', far_name='farfield'):
    """native SU2 2D mesh: triangles (5) and quads (9); polygons split by _split_polygon; markers wall + farfield"""
    xmin, xmax, ymin, ymax = box
    elems = []
    for c in cells:
        for e in _split_polygon(P, c):
            pts = P[e]; area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
            elems.append(e if area > 0 else e[::-1])
    cnt = {}; dirn = {}
    for c in cells:
        pc = P[c]; ccw = np.sum(pc[:, 0] * np.roll(pc[:, 1], -1) - np.roll(pc[:, 0], -1) * pc[:, 1]) > 0
        n = len(c)
        for k in range(n):
            u, v = c[k], c[(k + 1) % n]; key = (min(u, v), max(u, v)); cnt[key] = cnt.get(key, 0) + 1
            dirn[key] = (u, v) if ccw else (v, u)          # boundary edge with its cell on the left
    wall, far = [], []
    for key, m in cnt.items():
        if m != 1: continue
        u, v = dirn[key]
        mx = 0.5 * (P[u] + P[v]); tol = 1e-9
        onbox = abs(mx[0] - xmin) < tol or abs(mx[0] - xmax) < tol or abs(mx[1] - ymin) < tol or abs(mx[1] - ymax) < tol
        (far if onbox else wall).append((u, v))
    with open(path, 'w') as f:
        f.write(f"NDIME= 2\nNELEM= {len(elems)}\n")
        for i, e in enumerate(elems):
            f.write(f"{5 if len(e) == 3 else 9} " + " ".join(map(str, e)) + f" {i}\n")
        f.write(f"NPOIN= {len(P)}\n")
        for i, p in enumerate(P): f.write(f"{p[0]:.15g} {p[1]:.15g} {i}\n")
        f.write("NMARK= 2\n")
        for name, lst in ((wall_name, wall), (far_name, far)):
            f.write(f"MARKER_TAG= {name}\nMARKER_ELEMS= {len(lst)}\n")
            for u, v in lst: f.write(f"3 {u} {v}\n")
    ntri = sum(1 for e in elems if len(e) == 3)
    return dict(n_elem=len(elems), n_tri=ntri, n_quad=len(elems) - ntri, n_wall_edges=len(wall), n_far_edges=len(far))

def write_system(case):
    d = os.path.join(case, 'system'); os.makedirs(d, exist_ok=True)
    H = lambda cls, obj: "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n" % (cls, obj)
    open(os.path.join(d, 'controlDict'), 'w').write(H('dictionary', 'controlDict') + """
application simpleFoam;
startFrom latestTime; startTime 0; stopAt endTime; endTime 500; deltaT 1;
writeControl timeStep; writeInterval 500; purgeWrite 1; writeFormat ascii; writePrecision 8; timeFormat general; runTimeModifiable true;
functions
{
    yPlus1 { type yPlus; libs (fieldFunctionObjects); writeControl writeTime; }
    forces1 { type forceCoeffs; libs (forces); patches (wall); rho rhoInf; rhoInf 1; CofR (0.25 0 0); liftDir (0 1 0); dragDir (1 0 0); pitchAxis (0 0 1); magUInf 1; lRef 1; Aref 0.1; writeControl timeStep; writeInterval 1; log false; }
}
""")
    open(os.path.join(d, 'fvSchemes'), 'w').write(H('dictionary', 'fvSchemes') + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; limited cellLimited Gauss linear 1; grad(U) $limited; grad(k) $limited; grad(omega) $limited; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind limited; turbulence bounded Gauss limitedLinear 1; div(phi,k) $turbulence; div(phi,omega) $turbulence; div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default limited corrected 0.5; }
wallDist { method meshWave; }
""")
    open(os.path.join(d, 'fvSolution'), 'w').write(H('dictionary', 'fvSolution') + """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.01; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
}
SIMPLE { nNonOrthogonalCorrectors 1; consistent yes; residualControl { p 1e-5; U 1e-5; "(k|omega)" 1e-5; } }
relaxationFactors { equations { U 0.9; ".*" 0.7; } }
""")

def plot(out, P, bulk, bl, polys, loops, bb, delta):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    bb0, bb1 = bb
    for name, xl, yl in [('full', (bb0[0] - 0.3, bb1[0] + 0.3), (bb0[1] - 0.3, bb1[1] + 0.3)),
                         ('zoom', (bb0[0] - 3 * delta, bb0[0] + 10 * delta), (-6 * delta, 6 * delta)),
                         ('zoomTE', (bb1[0] - 10 * delta, bb1[0] + 3 * delta), (-6 * delta, 6 * delta))]:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.add_collection(PolyCollection([P[c] for c in bulk], facecolors='none', edgecolors='0.4', linewidths=0.3))
        ax.add_collection(PolyCollection([P[c] for c in bl], facecolors='none', edgecolors='tab:red', linewidths=0.3))
        for pts, closed in polys:
            q = np.vstack([pts, pts[:1]]) if closed else pts
            ax.plot(q[:, 0], q[:, 1], 'k-', lw=0.8)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal'); ax.set_title(f'{out} {name}')
        fig.savefig(os.path.join(out, f'mesh_{name}.png'), dpi=160); plt.close(fig)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--geom', default='airfoil'); ap.add_argument('--out', default='runs/v7/sfhm_airfoil')
    ap.add_argument('--delta', type=float, default=0.02); ap.add_argument('--nlayers', type=int, default=8)
    ap.add_argument('--growth', type=float, default=1.25); ap.add_argument('--hwall', type=float, default=0.01)
    ap.add_argument('--domain', type=float, default=16.0); ap.add_argument('--lmin', type=int, default=4)
    ap.add_argument('--lmax', type=int, default=11); ap.add_argument('--gridres', type=float, default=16)
    ap.add_argument('--snap', type=float, default=0.25); ap.add_argument('--dz', type=float, default=0.1)
    ap.add_argument('--variable', action='store_true'); ap.add_argument('--smooth', type=int, default=0); ap.add_argument('--plot', action='store_true')
    ap.add_argument('--shell', choices=['eikonal', 'warped'], default='eikonal')
    ap.add_argument('--tmax', type=float, default=1.5); ap.add_argument('--maxretry', type=int, default=8)
    ap.add_argument('--layers', choices=['yplus', 'fixedN'], default='yplus'); ap.add_argument('--dy0', type=float, default=None)
    ap.add_argument('--no-clean', dest='clean', action='store_false'); ap.add_argument('--bridge', type=float, default=5.0)
    ap.add_argument('--fan-grade', dest='fan_grade', type=float, default=2.5)
    ap.add_argument('--grade', type=float, default=0.15, help='bulk size gradation dh/dd (0 = v5 behaviour)')
    ap.add_argument('--wake-len', dest='wake_len', type=float, default=0.0); ap.add_argument('--wake-dir', dest='wake_dir', type=float, default=0.0)
    ap.add_argument('--wake-h', dest='wake_h', type=float, default=None); ap.add_argument('--wake-growth', dest='wake_growth', type=float, default=0.02)
    ap.add_argument('--wake-src-xmin', dest='wake_src_xmin', type=float, default=None)
    ap.add_argument('--su2', action='store_true', help='also write native SU2 mesh (mesh.su2)')
    args = ap.parse_args()
    if args.wake_h is None: args.wake_h = 2 * args.hwall
    info = build(args)
    print(json.dumps(info, indent=1))
