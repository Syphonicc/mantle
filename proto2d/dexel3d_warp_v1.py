#!/usr/bin/env python3
"""GPU tri-dexel with NVIDIA Warp vs Warp's fast generalized winding number (Barill-style dipole tree).

tri-dexel: one thread per ray (x, y and z families); the thread marches the ray through ALL crossings
(BVH ray queries, restarting just past each hit), accumulates the signed crossing count and writes the
1D winding of every grid node it passes. Net winding != 0 at the end flags an inconsistent ray (holes).
Node classification: z-ray if consistent, else x, else y; nodes without a consistent ray or with
disagreeing consistent rays fall back to the fast winding number.

baseline: wp.mesh_query_point_sign_winding_number on every node (the strongest sign query in Warp).

usage: dexel3d_warp_v1.py [--n 128 256 384] [--hole 200] [--device cuda:0]
"""
import argparse, time, json, numpy as np, warp as wp, importlib.util
spec = importlib.util.spec_from_file_location('d', __file__.replace('dexel3d_warp_v1.py', 'dexel3d_v1.py'))
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)

@wp.kernel
def dexel_rays(mesh: wp.uint64, ax: int, g0: wp.array(dtype=float), g1: wp.array(dtype=float), ga: wp.array(dtype=float),
               n0: int, n1: int, na: int, W: wp.array(dtype=wp.int8), ok: wp.array(dtype=wp.int8)):
    tid = wp.tid()
    i = tid // n1
    j = tid - i * n1
    a = g0[i]
    b = g1[j]
    # ray direction = +axis 'ax'; the two other coordinates are (a, b) in increasing axis order
    d = wp.vec3(0.0, 0.0, 0.0)
    p = wp.vec3(0.0, 0.0, 0.0)
    lo = ga[0] - 1.0
    if ax == 0:
        d = wp.vec3(1.0, 0.0, 0.0)
        p = wp.vec3(lo, a, b)
    elif ax == 1:
        d = wp.vec3(0.0, 1.0, 0.0)
        p = wp.vec3(a, lo, b)
    else:
        d = wp.vec3(0.0, 0.0, 1.0)
        p = wp.vec3(a, b, lo)
    span = ga[na - 1] - lo + 1.0
    w = int(0)
    k = int(0)
    s = lo
    eps = float(1.0e-9)
    for it in range(4096):
        q = wp.mesh_query_ray(mesh, p + d * (s - lo), d, span)
        hit_s = float(1.0e30)
        if q.result:
            hit_s = s + q.t
        # write nodes strictly before the hit
        while k < na and ga[k] < hit_s:
            W[tid * na + k] = wp.int8(wp.clamp(w, -100, 100))
            k += 1
        if not q.result:
            break
        nd = wp.dot(q.normal, d)
        if nd < 0.0:
            w += 1
        else:
            w -= 1
        s = hit_s + eps
    ok[tid] = wp.int8(1) if w == 0 else wp.int8(0)

@wp.kernel
def gwn_sign(mesh: wp.uint64, P: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.int8)):
    tid = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, P[tid], 1.0e6, 2.0, 0.5)
    out[tid] = wp.int8(1) if (q.result and q.sign < 0.0) else wp.int8(0)

def tridexel_gpu(mesh, g, dev):
    n = [len(x) for x in g]; res = []
    gs = [wp.array(x.astype(np.float32), dtype=float, device=dev) for x in g]
    for ax in range(3):
        o0, o1 = [k for k in range(3) if k != ax]
        W = wp.zeros(n[o0] * n[o1] * n[ax], dtype=wp.int8, device=dev); ok = wp.zeros(n[o0] * n[o1], dtype=wp.int8, device=dev)
        wp.launch(dexel_rays, dim=n[o0] * n[o1], inputs=[mesh.id, ax, gs[o0], gs[o1], gs[ax], n[o0], n[o1], n[ax], W, ok], device=dev)
        res.append((W, ok, (o0, o1, ax)))
    wp.synchronize_device(dev)
    return res

def combine(res, n):
    out = []
    for W, ok, (o0, o1, ax) in res:
        Wn = W.numpy().reshape(n[o0], n[o1], n[ax]); okn = ok.numpy().reshape(n[o0], n[o1]).astype(bool)
        perm = np.argsort([o0, o1, ax])
        out.append((np.transpose(np.abs(Wn) > 0, perm), np.transpose(np.broadcast_to(okn[:, :, None], Wn.shape), perm)))
    (ix, okx), (iy, oky), (iz, okz) = out
    inside = np.where(okz, iz, np.where(okx, ix, iy))
    disagree = (okx & okz & (ix != iz)) | (oky & okz & (iy != iz)) | (okx & oky & (ix != iy))
    fallback = ~(okx | oky | okz) | disagree
    return inside, fallback

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--n', nargs='+', type=int, default=[128, 256, 384])
    ap.add_argument('--hole', type=int, default=200); ap.add_argument('--device', default='cuda:0'); ap.add_argument('--check', type=int, default=3000)
    a = ap.parse_args()
    wp.init(); dev = a.device
    T = D.soup(a.hole)
    pts = wp.array(T.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=dev)
    idx = wp.array(np.arange(len(T) * 3, dtype=np.int32), dtype=int, device=dev)
    t = time.time(); mesh = wp.Mesh(points=pts, indices=idx, support_winding_number=True); wp.synchronize_device(dev); t_bvh = time.time() - t
    rng = np.random.default_rng(1)
    for n in a.n:
        off = 1e-4 * np.pi
        g = [np.linspace(-1.0, 1.2, n) + off * (k + 1) * (2.2 / n) for k in range(3)]
        tridexel_gpu(mesh, g, dev)                                          # warm-up (kernel compile)
        t = time.time(); res = tridexel_gpu(mesh, g, dev); t_gpu = time.time() - t
        t = time.time(); inside, fb = combine(res, [n, n, n]); t_comb = time.time() - t
        G = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
        # fallback with the fast winding number (GPU)
        t = time.time(); fbi = np.flatnonzero(fb.ravel())
        if len(fbi):
            Pf = wp.array(G[fbi], dtype=wp.vec3, device=dev); of = wp.zeros(len(fbi), dtype=wp.int8, device=dev)
            wp.launch(gwn_sign, dim=len(fbi), inputs=[mesh.id, Pf, of], device=dev); wp.synchronize_device(dev)
            inside.reshape(-1)[fbi] = of.numpy().astype(bool)
        t_fb = time.time() - t
        # baseline: fast winding number on ALL nodes (GPU)
        Pa = wp.array(G, dtype=wp.vec3, device=dev); oa = wp.zeros(len(G), dtype=wp.int8, device=dev)
        wp.launch(gwn_sign, dim=min(len(G), 1000), inputs=[mesh.id, Pa, oa], device=dev); wp.synchronize_device(dev)   # warm-up
        t = time.time(); wp.launch(gwn_sign, dim=len(G), inputs=[mesh.id, Pa, oa], device=dev); wp.synchronize_device(dev); t_gwn = time.time() - t
        ins_gwn = oa.numpy().astype(bool)
        # near-surface check against the exact (brute) GWN, excluding ambiguous |w| in (0.4, 0.6)
        insr = inside.reshape(n, n, n); nb = np.zeros_like(insr)
        for axx in range(3):
            dd = np.diff(insr.astype(np.int8), axis=axx) != 0; s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
            s1[axx] = slice(0, -1); s2[axx] = slice(1, None); nb[tuple(s1)] |= dd; nb[tuple(s2)] |= dd
        cand = np.flatnonzero(nb.ravel()); pick = rng.choice(cand, size=min(a.check, len(cand)), replace=False)
        w = D.gwn(T, G[pick].astype(float)); clear = np.abs(np.abs(w) - 0.5) > 0.1; ref = np.abs(w) > 0.5
        mis_dex = int(((inside.reshape(-1)[pick] != ref) & clear).sum()); mis_fast = int(((ins_gwn[pick] != ref) & clear).sum())
        print(json.dumps(dict(grid=f'{n}^3', nodes=int(len(G)), tris=len(T), hole=a.hole, t_bvh=round(t_bvh, 3),
                              t_tridexel_gpu=round(t_gpu, 3), t_combine_cpu=round(t_comb, 3), fallback_nodes=int(len(fbi)), t_fallback_gpu=round(t_fb, 3),
                              t_fastGWN_all_nodes_gpu=round(t_gwn, 3), agree_dexel_vs_fastGWN=float(np.mean(inside.reshape(-1) == ins_gwn)),
                              near_check=int(len(pick)), near_clear=int(clear.sum()), mism_dexel_clear=mis_dex, mism_fastGWN_clear=mis_fast)))
