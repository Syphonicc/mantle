#!/usr/bin/env python3
"""v2: Rust fast-GWN fallback (fwn_bench) vs Warp GPU v3, near-surface exact-GWN check, and BVH unsigned distance
(dist_bench) vs numpy brute force.

usage: compare_rcore_v2.py [--n 128 256] [--hole 200] [--npts 20000] [--nbrute 2000]
"""
import argparse, json, os, subprocess, sys, time, numpy as np, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, file)); m = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [file]; spec.loader.exec_module(m); sys.argv = argv; return m
D = load('d', 'dexel3d_v1.py'); W = load('w', 'dexel3d_warp_v3.py')
import warp as wp
BIN = os.path.join(here, '..', 'rcore', 'target', 'release', 'fwn_bench')
TMP = '/tmp/claude-1000/-home-syphonic-Auto-Mesh/032fe367-fa31-4d4b-a988-c5cdbfdbd638/scratchpad/rcore'

DBIN = os.path.join(here, '..', 'rcore', 'target', 'release', 'dist_bench')

def np_dist(T, P):
    """exact unsigned distance from each point of P (m,3) to the soup T (nt,3,3), Ericson closest point, vectorised over triangles"""
    A, B, C = T[:, 0], T[:, 1], T[:, 2]
    out = np.empty(len(P))
    for k, p in enumerate(P):
        ab, ac, ap = B - A, C - A, p - A
        d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1); bp = p - B; d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
        cp = p - C; d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
        vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
        with np.errstate(all='ignore'):
            den = va + vb + vc; v = vb / den; w = vc / den
            q = A + ab * v[:, None] + ac * w[:, None]
            m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0); q[m] = (B + (C - B) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None])[m]
            m = (vb <= 0) & (d2 >= 0) & (d6 <= 0); q[m] = (A + ac * (d2 / (d2 - d6))[:, None])[m]
            m = (d6 >= 0) & (d5 <= d6); q[m] = C[m]
            m = (vc <= 0) & (d1 >= 0) & (d3 <= 0); q[m] = (A + ab * (d1 / (d1 - d3))[:, None])[m]
            m = (d3 >= 0) & (d4 <= d3); q[m] = B[m]
            m = (d1 <= 0) & (d2 <= 0); q[m] = A[m]
            q[~np.isfinite(q).all(1)] = A[~np.isfinite(q).all(1)]  # degenerate triangles: nearest vertex fallback is conservative only for A
        out[k] = np.sqrt(((q - p) ** 2).sum(1).min())
    return out

ap = argparse.ArgumentParser(); ap.add_argument('--n', nargs='+', type=int, default=[128, 256])
ap.add_argument('--hole', type=int, default=200); ap.add_argument('--npts', type=int, default=20000); ap.add_argument('--nbrute', type=int, default=2000)
a = ap.parse_args()
os.makedirs(TMP, exist_ok=True)
T = D.soup(a.hole)
with open(f'{TMP}/soup.bin', 'wb') as f:
    f.write(np.uint64(len(T)).tobytes()); f.write(T.astype('<f8').tobytes())
wp.init(); dev = 'cuda:0'
mesh = wp.Mesh(points=wp.array(T.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=dev),
               indices=wp.array(np.arange(len(T) * 3, dtype=np.int32), dtype=int, device=dev), support_winding_number=True)
rng = np.random.default_rng(3)
for n in a.n:
    g = [np.linspace(-1.0, 1.2, n) + 1e-7 * np.pi * (k + 1) for k in range(3)]
    with open(f'{TMP}/grid.bin', 'wb') as f:
        f.write(np.array([n, n, n], dtype='<u8').tobytes())
        for x in g: f.write(x.astype('<f8').tobytes())
    rs = json.loads(subprocess.run([BIN, f'{TMP}/soup.bin', f'{TMP}/grid.bin', f'{TMP}/out'], capture_output=True, text=True, check=True).stdout)
    ins_r = np.fromfile(f'{TMP}/out.inside.u8', dtype=np.uint8).astype(bool)
    fb_r = np.fromfile(f'{TMP}/out.fallback.u8', dtype=np.uint8).astype(bool)
    W.tridexel_gpu(mesh, g, dev)
    t = time.time(); res = W.tridexel_gpu(mesh, g, dev); t_gpu = time.time() - t
    ins_w, fb_w = W.combine(res, [n, n, n])
    G = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)
    fbi = np.flatnonzero(fb_w.ravel())
    if len(fbi):
        Pf = wp.array(G[fbi].astype(np.float32), dtype=wp.vec3, device=dev); of = wp.zeros(len(fbi), dtype=wp.int8, device=dev)
        wp.launch(W.gwn_sign, dim=len(fbi), inputs=[mesh.id, Pf, of], device=dev); wp.synchronize_device(dev)
        ins_w.reshape(-1)[fbi] = of.numpy().astype(bool)
    ins_w = ins_w.reshape(-1)
    insr = ins_r.reshape(n, n, n); nb = np.zeros_like(insr)
    for ax in range(3):
        dd = np.diff(insr.astype(np.int8), axis=ax) != 0; s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1); s2[ax] = slice(1, None); nb[tuple(s1)] |= dd; nb[tuple(s2)] |= dd
    cand = np.flatnonzero(nb.ravel()); pick = rng.choice(cand, size=min(2000, len(cand)), replace=False)
    wv = D.gwn(T, G[pick]); clear = np.abs(np.abs(wv) - 0.5) > 0.1
    # distance bench: random points in the grid box
    lo, hi = np.array([x[0] for x in g]), np.array([x[-1] for x in g])
    P = lo + rng.random((a.npts, 3)) * (hi - lo)
    with open(f'{TMP}/pts.bin', 'wb') as f:
        f.write(np.uint64(len(P)).tobytes()); f.write(P.astype('<f8').tobytes())
    ds = json.loads(subprocess.run([DBIN, f'{TMP}/soup.bin', f'{TMP}/pts.bin', f'{TMP}/dist.bin'], capture_output=True, text=True, check=True).stdout)
    dr = np.fromfile(f'{TMP}/dist.bin', dtype='<f8')
    sub = rng.choice(len(P), size=min(a.nbrute, len(P)), replace=False)
    t = time.time(); db = np_dist(T, P[sub]); t_bf = time.time() - t
    out = dict(grid=f'{n}^3', nodes=n ** 3, tris=int(len(T)), rust_threads=rs['threads'], t_build_fwn=rs['t_build_fwn'], t_rust_classify=rs['t_classify'],
               t_warp_gpu_rays=round(t_gpu, 4), rust_fallback=rs['fallback_nodes'], warp_fallback=int(len(fbi)), same_fallback_set=bool(np.array_equal(fb_r, fb_w.reshape(-1))),
               t_rust_fallback_fast=rs['t_fallback_fast'], agree_rust_vs_warp_all=float(np.mean(ins_r == ins_w)),
               agree_rust_vs_warp_nonfallback=float(np.mean(ins_r[~fb_r] == ins_w[~fb_r])),
               near_surface_clear_mismatch_rust=int(((ins_r[pick] != (np.abs(wv) > 0.5)) & clear).sum()), near_surface_checked=int(clear.sum()),
               dist_npts=ds['npts'], t_dist_build=ds['t_build'], t_dist_query=ds['t_query'], t_numpy_brute_dist=round(t_bf, 3), dist_brute_checked=int(len(sub)),
               dist_max_abs_err=float(np.abs(dr[sub] - db).max()), dist_max_rel_err=float((np.abs(dr[sub] - db) / np.maximum(db, 1e-300)).max()))
    print(json.dumps(out))
