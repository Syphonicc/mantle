#!/usr/bin/env python3
"""Tri-dexel test in 3D (CPU, numpy): rays along x, y, z through every grid line, ALL crossings with a
triangle soup recorded (exact point, sign of the normal component, triangle id -> normal). Inside/outside
of every grid node from the signed crossing count along its rays (1D winding). Rays whose net winding
is non-zero (holes, open soups) are flagged; nodes with no consistent ray, or where consistent rays disagree,
fall back to the exact generalized winding number (solid angles, Van Oosterom-Strackee).

Test soup: UV sphere overlapping a torus (two closed soups, intersecting -> internal faces), with
--hole triangles deleted from the sphere. Accuracy is checked against the exact GWN on a random node sample.

usage: dexel3d_v1.py [--n 128 256] [--hole 6] [--sample 20000]
"""
import argparse, time, json, numpy as np

def uv_sphere(c, r, nu=96, nv=48):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False); v = np.linspace(0, np.pi, nv + 1)
    V = [np.array([c[0], c[1], c[2] + r])]
    for j in range(1, nv):
        for i in range(nu):
            V.append(c + r * np.array([np.sin(v[j]) * np.cos(u[i]), np.sin(v[j]) * np.sin(u[i]), np.cos(v[j])]))
    V.append(np.array([c[0], c[1], c[2] - r])); V = np.array(V); S = len(V) - 1
    idx = lambda j, i: 1 + (j - 1) * nu + (i % nu)
    F = [[0, idx(1, i), idx(1, i + 1)] for i in range(nu)]
    for j in range(1, nv - 1):
        for i in range(nu):
            F += [[idx(j, i), idx(j + 1, i), idx(j + 1, i + 1)], [idx(j, i), idx(j + 1, i + 1), idx(j, i + 1)]]
    F += [[S, idx(nv - 1, i + 1), idx(nv - 1, i)] for i in range(nu)]
    return V, np.array(F)

def torus(c, R, r, nu=128, nv=48):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False); v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    V = np.array([c + np.array([(R + r * np.cos(b)) * np.cos(a), (R + r * np.cos(b)) * np.sin(a), r * np.sin(b)]) for a in u for b in v])
    idx = lambda i, j: (i % nu) * nv + (j % nv)
    F = []
    for i in range(nu):
        for j in range(nv):
            F += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    return V, np.array(F)

def soup(hole):
    Vs, Fs = uv_sphere(np.array([0.0, 0.0, 0.0]), 0.6)
    Vt, Ft = torus(np.array([0.35, 0.1, 0.05]), 0.55, 0.18)
    if hole: Fs = np.delete(Fs, np.arange(len(Fs) // 3, len(Fs) // 3 + hole), axis=0)   # hole in the sphere
    T = np.concatenate([Vs[Fs], Vt[Ft]])          # (ntri, 3, 3) soup, no shared indexing needed
    return T

def gwn(T, P, chunk=2000):
    """exact generalized winding number (sum of signed solid angles / 4 pi)"""
    w = np.zeros(len(P))
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk][:, None, None, :]
        a, b, c = (T[None, :, 0] - p[:, :, 0]), (T[None, :, 1] - p[:, :, 0]), (T[None, :, 2] - p[:, :, 0])
        la, lb, lc = [np.linalg.norm(x, axis=-1) for x in (a, b, c)]
        det = np.einsum('ijk,ijk->ij', a, np.cross(b, c))
        den = la * lb * lc + np.einsum('ijk,ijk->ij', a, b) * lc + np.einsum('ijk,ijk->ij', b, c) * la + np.einsum('ijk,ijk->ij', c, a) * lb
        w[s:s + chunk] = 2 * np.arctan2(det, den).sum(1) / (4 * np.pi)
    return w

def ray_axis(T, g, ax):
    """crossings of all rays parallel to axis 'ax' (through grid lines of the other two axes) with the soup.
    returns flat ray id, coordinate along the ray, sign"""
    o1, o2 = [k for k in range(3) if k != ax]
    g1, g2, ga = g[o1], g[o2], g[ax]; h1, h2 = g1[1] - g1[0], g2[1] - g2[0]
    P1, P2 = T[:, :, o1], T[:, :, o2]
    i0 = np.ceil((P1.min(1) - g1[0]) / h1).astype(np.int64); i1 = np.floor((P1.max(1) - g1[0]) / h1).astype(np.int64)
    j0 = np.ceil((P2.min(1) - g2[0]) / h2).astype(np.int64); j1 = np.floor((P2.max(1) - g2[0]) / h2).astype(np.int64)
    i0 = np.clip(i0, 0, len(g1) - 1); i1 = np.clip(i1, -1, len(g1) - 1); j0 = np.clip(j0, 0, len(g2) - 1); j1 = np.clip(j1, -1, len(g2) - 1)
    ni = np.maximum(i1 - i0 + 1, 0); nj = np.maximum(j1 - j0 + 1, 0); cnt = ni * nj
    tri = np.repeat(np.arange(len(T)), cnt)
    k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ii = np.repeat(i0, cnt) + k // np.repeat(np.maximum(nj, 1), cnt); jj = np.repeat(j0, cnt) + k % np.repeat(np.maximum(nj, 1), cnt)
    x, y = g1[ii], g2[jj]
    A, B, C = T[tri, 0], T[tri, 1], T[tri, 2]
    # 2D edge functions in the (o1, o2) plane; top-left style tie rule via a fixed tiny grid offset (see caller)
    def ef(P, Q): return (Q[:, o1] - P[:, o1]) * (y - P[:, o2]) - (Q[:, o2] - P[:, o2]) * (x - P[:, o1])
    e0, e1, e2 = ef(A, B), ef(B, C), ef(C, A)
    inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
    area = e0 + e1 + e2
    inside &= area != 0
    tri, ii, jj, e0, e1, e2, area = tri[inside], ii[inside], jj[inside], e0[inside], e1[inside], e2[inside], area[inside]
    A, B, C = T[tri, 0], T[tri, 1], T[tri, 2]
    s = (e1 * A[:, ax] + e2 * B[:, ax] + e0 * C[:, ax]) / area            # barycentric interpolation along the ray
    nrm = np.cross(B - A, C - A)[:, ax]
    sign = np.where(nrm > 0, -1, 1)                                         # crossing a face whose normal points +ax = leaving
    ray = ii * len(g2) + jj
    return ray, s, sign, len(g1) * len(g2)

def winding_ray(ray, s, sign, nray, ga):
    o = np.lexsort((s, ray)); ray, s, sign = ray[o], s[o], sign[o]
    span = ga[-1] - ga[0] + 1.0
    key = s - ga[0] + ray * span
    csum = np.r_[0, np.cumsum(sign)]
    start = np.searchsorted(ray, np.arange(nray))
    q = (ga[None, :] - ga[0]) + (np.arange(nray) * span)[:, None]
    W = csum[np.searchsorted(key, q.ravel()).reshape(q.shape)] - csum[start][:, None]
    net = np.bincount(ray, weights=sign, minlength=nray)
    return W, net

def tridexel(T, g):
    n = [len(x) for x in g]; res = []; ncross = 0
    for ax in range(3):
        ray, s, sign, nray = ray_axis(T, g, ax); ncross += len(s)
        W, net = winding_ray(ray, s, sign, nray, g[ax])
        o1, o2 = [k for k in range(3) if k != ax]
        W = W.reshape(n[o1], n[o2], n[ax]); ok = (np.abs(net) < 0.5).reshape(n[o1], n[o2])
        perm = np.argsort([o1, o2, ax])                                     # back to (x, y, z) order
        res.append((np.abs(np.transpose(W, perm)) > 0.5, np.transpose(np.broadcast_to(ok[:, :, None], W.shape), perm)))
    (ix, okx), (iy, oky), (iz, okz) = res
    inside = np.where(okz, iz, np.where(okx, ix, iy))
    cons = [(i, o) for i, o in res]
    nok = okx.astype(int) + oky + okz
    disagree = (okx & okz & (ix != iz)) | (oky & okz & (iy != iz)) | (okx & oky & (ix != iy))
    fallback = (nok == 0) | disagree
    return inside, fallback, ncross

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--n', nargs='+', type=int, default=[128, 256])
    ap.add_argument('--hole', type=int, default=6); ap.add_argument('--sample', type=int, default=20000)
    a = ap.parse_args()
    T = soup(a.hole)
    rng = np.random.default_rng(0)
    for n in a.n:
        off = 1e-7 * np.pi                                                   # avoid exact ray-vertex/edge ties
        g = [np.linspace(-1.0, 1.2, n) + off * (k + 1) for k in range(3)]
        t = time.time(); inside, fb, ncross = tridexel(T, g); t_d = time.time() - t
        G = np.stack(np.meshgrid(*g, indexing='ij'), -1)
        t = time.time(); nfb = int(fb.sum())
        if nfb: inside[fb] = np.abs(gwn(T, G[fb])) > 0.5
        t_fb = time.time() - t
        idx = rng.choice(G.size // 3, size=min(a.sample, G.size // 3), replace=False)
        Ps = G.reshape(-1, 3)[idx]
        t = time.time(); ref = np.abs(gwn(T, Ps)) > 0.5; t_ref = time.time() - t
        mism = int((inside.reshape(-1)[idx] != ref).sum())
        per_node = t_ref / len(Ps)
        print(json.dumps(dict(grid=f'{n}^3', nodes=int(G.size // 3), tris=len(T), hole_tris=a.hole, crossings=ncross,
                              t_tridexel=round(t_d, 3), fallback_nodes=nfb, t_fallback=round(t_fb, 3),
                              sample=len(Ps), mismatches_in_sample=mism,
                              gwn_brute_s_per_node=per_node, gwn_brute_all_nodes_est_s=round(per_node * G.size / 3, 1))))
