#!/usr/bin/env python3
"""Benchmark + cross-check of the Rust tri-dexel core (rcore dexel_bench) against the numpy CPU version
(dexel3d_v1) and the Warp GPU version (dexel3d_warp_v3) on the same dirty soup and the same grid.

usage: compare_rcore_v1.py [--n 128 256] [--hole 200] [--resolve-max 2000000]
--resolve-max: let Rust resolve fallback nodes with exact (brute) GWN only when nodes <= this.
"""
import argparse, json, os, subprocess, sys, time, numpy as np, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, file)); m = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [file]; spec.loader.exec_module(m); sys.argv = argv; return m
D = load('d', 'dexel3d_v1.py'); W = load('w', 'dexel3d_warp_v3.py')
import warp as wp
BIN = os.path.join(here, '..', 'rcore', 'target', 'release', 'dexel_bench')
TMP = '/tmp/claude-1000/-home-syphonic-Auto-Mesh/032fe367-fa31-4d4b-a988-c5cdbfdbd638/scratchpad/rcore'

ap = argparse.ArgumentParser(); ap.add_argument('--n', nargs='+', type=int, default=[128, 256])
ap.add_argument('--hole', type=int, default=200); ap.add_argument('--resolve-max', type=int, default=2_200_000)
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
    resolve = n ** 3 <= a.resolve_max
    cmd = [BIN, f'{TMP}/soup.bin', f'{TMP}/grid.bin', f'{TMP}/out'] + (['--resolve'] if resolve else [])
    rs = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    ins_r = np.fromfile(f'{TMP}/out.inside.u8', dtype=np.uint8).astype(bool)
    fb_r = np.fromfile(f'{TMP}/out.fallback.u8', dtype=np.uint8).astype(bool)
    # numpy single-thread CPU (classification only)
    t = time.time(); ins_np, fb_np, _ = D.tridexel(T, g); t_np = time.time() - t
    # Warp GPU v3 + fast GWN fallback
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
    same_fb = bool(np.array_equal(fb_r, fb_w.reshape(-1)))
    nonfb = ~fb_r
    agree_nonfb = float(np.mean(ins_r[nonfb] == ins_w[nonfb]))
    out = dict(grid=f'{n}^3', nodes=n ** 3, rust_threads=rs['threads'], t_rust_classify=rs['t_classify'], t_numpy_classify=round(t_np, 3),
               t_warp_gpu_rays=round(t_gpu, 4), rust_fallback=rs['fallback_nodes'], warp_fallback=int(len(fbi)), same_fallback_set=same_fb,
               agree_rust_vs_warp_nonfallback=agree_nonfb)
    if resolve:
        out['t_rust_fallback_exact'] = rs['t_fallback']
        out['agree_rust_vs_warp_all'] = float(np.mean(ins_r == ins_w))
        insr = ins_r.reshape(n, n, n); nb = np.zeros_like(insr)
        for ax in range(3):
            dd = np.diff(insr.astype(np.int8), axis=ax) != 0; s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
            s1[ax] = slice(0, -1); s2[ax] = slice(1, None); nb[tuple(s1)] |= dd; nb[tuple(s2)] |= dd
        cand = np.flatnonzero(nb.ravel()); pick = rng.choice(cand, size=min(2000, len(cand)), replace=False)
        wv = D.gwn(T, G[pick]); clear = np.abs(np.abs(wv) - 0.5) > 0.1
        out['near_surface_clear_mismatch_rust'] = int(((ins_r[pick] != (np.abs(wv) > 0.5)) & clear).sum())
        out['near_surface_checked'] = int(clear.sum())
    print(json.dumps(out))
