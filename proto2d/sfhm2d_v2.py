#!/usr/bin/env python3
"""
SFHM-2D v2: shell-first hybrid mesher, 2D prototype.

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
    gphi_y, gphi_x = np.gradient(phi, hg)
    solid_f = solid.astype(float)
    def interp(F, p):
        fx = (p[0] - gx[0]) / hg; fy = (p[1] - gy[0]) / hg
        i = int(np.clip(np.floor(fy), 0, ny - 2)); j = int(np.clip(np.floor(fx), 0, nx - 2))
        a = fx - j; b = fy - i
        return ((1 - a) * (1 - b) * F[i, j] + a * (1 - b) * F[i, j + 1] + (1 - a) * b * F[i + 1, j] + a * b * F[i + 1, j + 1])
    bl_cells = []; wall_pts = []; dy0 = []; bad = 0; nfeat_snap = 0; nonconvex = 0; ncap = 0
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
                if imp.feat_convex[fi] and df < 1.5 * hs and np.linalg.norm(P[s] - imp.features[fi]) < 1.3 * deff[k[0]] + 2 * hs:
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
                if interp(solid_f, pn) > 0.5 and abs(imp.winding(pn[None])[0]) > 0.5:
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
        # feature snapping: each feature vertex grabs its nearest wall point
        if feat_tree is not None:
            wt = cKDTree(wall_xy)
            for fi, F in enumerate(imp.features):
                if imp.feat_convex[fi] and fi in cap.values(): continue
                d, i = wt.query(F)
                if d < 2.5 * h_wall:
                    wall_xy[i] = F; nfeat_snap += 1
        # place nodes along each streamline by geometric growth
        node_ids = []; feat_pid = {}
        for s, path, w in zip(loop, cols_nodes, wall_xy):
            path = path[::-1].copy(); path[0] = w          # wall -> shell
            seg = np.linalg.norm(np.diff(path, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]; L = cum[-1]
            if L <= 0: L = 1e-12
            dy = L * (r - 1) / (r**N - 1) if abs(r - 1) > 1e-9 else L / N
            tk = np.array([dy * (r**k - 1) / (r - 1) if abs(r - 1) > 1e-9 else dy * k for k in range(N + 1)])
            tk[-1] = L
            px = np.interp(tk, cum, path[:, 0]); py = np.interp(tk, cum, path[:, 1])
            ids = []
            a_ = len(node_ids)
            for k in range(N):
                if k == 0 and a_ in cap:
                    fi = cap[a_]
                    if fi not in feat_pid: feat_pid[fi] = len(P_list); P_list.append(imp.features[fi].copy())
                    ids.append(feat_pid[fi]); continue
                ids.append(len(P_list)); P_list.append(np.array([px[k], py[k]]))
            ids.append(s)
            node_ids.append(ids); dy0.append(dy)
        wall_pts.append(wall_xy); ncap += len(cap)
        # feature-local smoothing: interior layer nodes within 3*delta of a feature vertex
        if feat_tree is not None and args.smooth > 0:
            nseed = len(loop)
            near = np.array([feat_tree.query(w)[0] < 3 * delta for w in wall_xy])
            for it in range(args.smooth):
                for a in range(nseed):
                    if not near[a]: continue
                    am, ap_ = (a - 1) % nseed, (a + 1) % nseed
                    for k in range(1, N):
                        i = node_ids[a][k]
                        P_list[i] = 0.5 * P_list[i] + 0.25 * (P_list[node_ids[am][k]] + P_list[node_ids[ap_][k]])
        # cells: between consecutive seeds (loop order has body on the right -> go reverse for CCW)
        nseed = len(loop)
        for a in range(nseed):
            b = (a + 1) % nseed
            for k in range(N):
                quad = [node_ids[a][k], node_ids[a][k + 1], node_ids[b][k + 1], node_ids[b][k]]
                # orientation check
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
                shell_dist_over_delta=shell_ratio)
    out = args.out
    os.makedirs(out, exist_ok=True)
    write_polymesh(out, P, cells, (xmin, xmax, ymin, ymax), args.dz)
    write_system(out)
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
    ap.add_argument('--geom', default='airfoil'); ap.add_argument('--out', default='runs/v2/sfhm_airfoil')
    ap.add_argument('--delta', type=float, default=0.02); ap.add_argument('--nlayers', type=int, default=8)
    ap.add_argument('--growth', type=float, default=1.25); ap.add_argument('--hwall', type=float, default=0.01)
    ap.add_argument('--domain', type=float, default=16.0); ap.add_argument('--lmin', type=int, default=4)
    ap.add_argument('--lmax', type=int, default=11); ap.add_argument('--gridres', type=float, default=16)
    ap.add_argument('--snap', type=float, default=0.25); ap.add_argument('--dz', type=float, default=0.1)
    ap.add_argument('--variable', action='store_true'); ap.add_argument('--smooth', type=int, default=0); ap.add_argument('--plot', action='store_true')
    ap.add_argument('--shell', choices=['eikonal', 'warped'], default='eikonal')
    ap.add_argument('--tmax', type=float, default=1.5); ap.add_argument('--maxretry', type=int, default=8)
    args = ap.parse_args()
    info = build(args)
    print(json.dumps(info, indent=1))
