#!/usr/bin/env python3
"""Run rcore octree_bench on the dirty soup (dexel3d_v1.soup) and cross-check independently in numpy:
2:1 balance (face/edge/corner) from leaves.bin, cut points vs brute-force distance on a subset, slice plot.

usage: compare_octree_v1.py [--L 8 10] [--hole 200] [--grade 0.2] [--sample 2000]
Domain [-1.5,1.7]^3, h_surf = size / 2^L. One JSON line per setting; PNG + binaries in the run directory.
"""
import argparse, json, os, subprocess, sys, numpy as np, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, file)); m = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [file]; spec.loader.exec_module(m); sys.argv = argv; return m
D = load('d', 'dexel3d_v1.py')
BIN = os.path.join(here, '..', 'rcore', 'target', 'release', 'octree_bench')
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'octree')
ORG, SIZE = -1.5, 3.2

ap = argparse.ArgumentParser(); ap.add_argument('--L', nargs='+', type=int, default=[8, 10])
ap.add_argument('--hole', type=int, default=200); ap.add_argument('--grade', type=float, default=0.2)
ap.add_argument('--sample', type=int, default=2000)
a = ap.parse_args()
os.makedirs(TMP, exist_ok=True)
T = D.soup(a.hole)
with open(f'{TMP}/SOUP.bin', 'wb') as f:
    f.write(np.uint64(len(T)).tobytes()); f.write(T.astype('<f8').tobytes())

def pt_tri_d2(P, T):
    """squared distance of points P (m,3) to triangles T (n,3,3), min over triangles (Ericson, vectorised over triangles)"""
    best = np.full(len(P), np.inf)
    A, B, C = T[:, 0], T[:, 1], T[:, 2]; AB, AC = B - A, C - A
    for s in range(len(P)):
        p = P[s]; ap_ = p - A
        d1 = (AB * ap_).sum(1); d2 = (AC * ap_).sum(1)
        bp = p - B; d3 = (AB * bp).sum(1); d4 = (AC * bp).sum(1)
        cp = p - C; d5 = (AB * cp).sum(1); d6 = (AC * cp).sum(1)
        vc = d1 * d4 - d3 * d2; vb = d5 * d2 - d1 * d6; va = d3 * d6 - d5 * d4
        den = va + vb + vc; den = np.where(np.abs(den) < 1e-300, 1e-300, den)
        v = vb / den; w = vc / den
        Q = A + AB * v[:, None] + AC * w[:, None]            # interior
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0); Q[m] = (A + AB * (d1 / np.where(d1 - d3 == 0, 1, d1 - d3))[:, None])[m]
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0); Q[m] = (A + AC * (d2 / np.where(d2 - d6 == 0, 1, d2 - d6))[:, None])[m]
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        t = (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1, (d4 - d3) + (d5 - d6)); Q[m] = (B + (C - B) * t[:, None])[m]
        m = (d6 >= 0) & (d5 <= d6); Q[m] = C[m]
        m = (d3 >= 0) & (d4 <= d3); Q[m] = B[m]
        m = (d1 <= 0) & (d2 <= 0); Q[m] = A[m]
        best[s] = ((Q - p) ** 2).sum(1).min()
    return best

for L in a.L:
    pre = f'{TMP}/L{L}'
    cmd = [BIN, f'{TMP}/SOUP.bin', f'origin={ORG},{ORG},{ORG}', f'size={SIZE}', f'L={L}', f'h_surf={SIZE / 2 ** L}',
           f'grade={a.grade}', f'h_max={SIZE / 8}', pre]
    rs = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    raw = open(f'{pre}.leaves.bin', 'rb').read(); n = int(np.frombuffer(raw[:8], '<u8')[0])
    rec = np.frombuffer(raw[8:8 + 13 * n], dtype=np.dtype([('l', 'u1'), ('i', '<u4'), ('j', '<u4'), ('k', '<u4')]))
    lev = rec['l'].astype(np.int64); ijk = np.stack([rec['i'], rec['j'], rec['k']], 1).astype(np.int64)
    # --- independent 2:1 check: dict (level, i, j, k) -> exists; for each leaf and each of 26 neighbour cells at its level,
    # a leaf ancestor 2+ levels coarser is a violation (vectorised per level, dict replaced by python set of packed ints)
    def pk(l, c): return (l.astype(np.uint64) << np.uint64(60)) + (c[:, 0].astype(np.uint64)) + (c[:, 1].astype(np.uint64) << np.uint64(20)) + (c[:, 2].astype(np.uint64) << np.uint64(40))
    keys = np.sort(pk(lev, ijk)); viol = 0
    offs = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]
    for l in np.unique(lev):
        if l < 2: continue
        c = ijk[lev == l]
        for o in offs:
            nb = c + np.array(o); ok = ((nb >= 0) & (nb < (1 << int(l)))).all(1); nb = nb[ok]
            bad = np.zeros(len(nb), bool)
            for al in range(0, int(l) - 1):                    # ancestors at levels 0 .. l-2
                q = pk(np.full(len(nb), al, np.int64), nb >> (int(l) - al))
                idx = np.searchsorted(keys, q); idx[idx >= len(keys)] = len(keys) - 1
                bad |= keys[idx] == q
            viol += int(bad.sum())
    # --- cut points vs brute force on a subset
    rawc = open(f'{pre}.cuts.bin', 'rb').read(); nc = int(np.frombuffer(rawc[:8], '<u8')[0])
    cuts = np.frombuffer(rawc[8:8 + 48 * nc], '<f8').reshape(-1, 6) if nc else np.zeros((0, 6))
    rng = np.random.default_rng(1); sub = cuts[rng.choice(nc, min(a.sample, nc), replace=False)] if nc else cuts
    d = np.sqrt(pt_tri_d2(sub[:, :3], T)) if len(sub) else np.zeros(0)
    nl = np.linalg.norm(sub[:, 3:], axis=1) if len(sub) else np.zeros(0)
    out = dict(L=L, tris=rs['tris'], leaves=rs['leaves'], cut_edges=rs['cut_edges'], multi_hit_edges=rs['multi_hit_edges'],
               unresolved_edges=rs['unresolved_edges'], rust_balance_violations=rs['balance_violations'], py_balance_violations=viol,
               cut_max_dist_to_soup=float(d.max()) if len(d) else None, cut_normal_unit_err=float(np.abs(nl - 1).max()) if len(nl) else None,
               t_build=rs['t_build'], t_balance=rs['t_balance'], t_signs=rs['t_signs'], t_cuts=rs['t_cuts'], t_total=rs['t_total'], threads=rs['threads'])
    print(json.dumps(out), flush=True)
    # --- slice plot z = 0.05
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection; from matplotlib.patches import Rectangle
    z = 0.05; h = SIZE / (1 << lev); lo = ORG + ijk * h[:, None]
    m = (lo[:, 2] <= z) & (z < lo[:, 2] + h)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(PatchCollection([Rectangle((x, y), s, s) for x, y, s in zip(lo[m, 0], lo[m, 1], h[m])], fc='none', ec='0.4', lw=0.3))
    cm = np.abs(cuts[:, 2] - z) < SIZE / 2 ** L if nc else []
    if nc: ax.plot(cuts[cm, 0], cuts[cm, 1], 'r.', ms=1.5)
    ax.set_xlim(ORG, ORG + SIZE); ax.set_ylim(ORG, ORG + SIZE); ax.set_aspect('equal'); ax.set_title(f'octree L={L}, z={z} slice, {int(m.sum())} leaves')
    fig.savefig(f'{pre}.slice.png', dpi=130); plt.close(fig)
