#!/usr/bin/env python3
"""Dexel test (2D analogue of tri-dexel): cast rays along every grid row (x-rays) and column (y-rays),
record ALL crossings with the segment soup (exact position, sign, normal), and derive inside/outside of
every grid node by signed crossing counts along each ray (1D winding number).

Dirty soups (holes, folds, spikes) make some rays inconsistent (net winding at the far end != 0).
Those rays are detected for free; their nodes fall back to the exact generalized winding number.

Compared against the current method (exact generalized winding number, sfhm2d_v7.Implicit.sdf_grid)
for accuracy and wall-clock, plus edge-crossing (Hermite) accuracy vs linear interpolation of the SDF.

usage: dexel2d_v1.py [--geoms airfoil twoelement dirty tmr656] [--res 16 64]
"""
import argparse, time, json, numpy as np, importlib.util, sys
spec = importlib.util.spec_from_file_location('m', __file__.replace('dexel2d_v1.py', 'sfhm2d_v7.py'))
M = importlib.util.module_from_spec(spec); sys.argv, _argv = [sys.argv[0]], sys.argv; spec.loader.exec_module(M); sys.argv = _argv

def ray_crossings(A, B, lines, axis):
    """all crossings of segments AB with lines {p[axis] = c}; half-open rule avoids double counts at vertices.
    returns line index, coordinate along the ray, sign (+1 if segment goes up in 'axis'), segment id"""
    o = 1 - axis                                    # ray direction coordinate
    a, b = A[:, axis], B[:, axis]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    c0, h = lines[0], lines[1] - lines[0]
    j0 = np.ceil((lo - c0) / h).astype(np.int64); j1 = np.ceil((hi - c0) / h).astype(np.int64) - 1   # lo <= c < hi
    j0 = np.clip(j0, 0, len(lines)); j1 = np.clip(j1, -1, len(lines) - 1)
    cnt = np.maximum(j1 - j0 + 1, 0)
    seg = np.repeat(np.arange(len(A)), cnt)
    j = np.repeat(j0, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    c = lines[j]
    t = (c - a[seg]) / (b[seg] - a[seg])
    s = A[seg, o] + t * (B[seg, o] - A[seg, o])
    sign = np.where(b[seg] > a[seg], 1, -1)
    return j, s, sign, seg

def winding_along(j, s, sign, nlines, nodes):
    """1D winding at every (line j, node coordinate) from crossings: cumulative sign of crossings with s < node.
    also returns per-line net winding (consistency: 0 for a closed watertight soup)"""
    o = np.lexsort((s, j)); j, s, sign = j[o], s[o], sign[o]
    span = nodes[-1] - nodes[0] + 1.0
    key = s - nodes[0] + j * span                    # rows laid end to end on one axis
    csum = np.r_[0, np.cumsum(sign)]
    row_start = np.searchsorted(j, np.arange(nlines))
    q = (nodes[None, :] - nodes[0]) + (np.arange(nlines) * span)[:, None]
    idx = np.searchsorted(key, q.ravel()).reshape(q.shape)
    W = csum[idx] - csum[row_start][:, None]
    net = np.bincount(j, weights=sign, minlength=nlines)
    return W, net

def dexel_inside(imp, gx, gy):
    A, B = imp.A, imp.B
    jx, sx, gx_sign, segx = ray_crossings(A, B, gy, axis=1)     # x-rays: rows y = gy[j]
    Wx, netx = winding_along(jx, sx, gx_sign, len(gy), gx)       # (ny, nx)
    jy, sy, gy_sign, segy = ray_crossings(A, B, gx, axis=0)      # y-rays: columns x = gx[i]
    Wy, nety = winding_along(jy, sy, gy_sign, len(gx), gy)       # (nx, ny)
    Wy = Wy.T
    okx = np.abs(netx) < 0.5; oky = np.abs(nety) < 0.5
    inx = np.abs(Wx) > 0.5; iny = np.abs(Wy) > 0.5
    good = okx[:, None] & oky[None, :]
    inside = np.where(okx[:, None], inx, iny)
    agree = (inx == iny) | ~good
    fallback = ~(okx[:, None] | oky[None, :]) | ~agree          # both rays inconsistent, or consistent rays disagree
    return inside, fallback, dict(n_cross=int(len(sx) + len(sy)), bad_xrays=int((~okx).sum()), bad_yrays=int((~oky).sum())), (jx, sx, segx)

def run(name, res, delta=0.02):
    polys = M.geometry(name)
    imp = M.Implicit(polys, hs=0.01 / 4)
    allp = np.vstack([p for p, _ in polys]); bb0, bb1 = allp.min(0), allp.max(0)
    hg = delta / res; m = 4 * delta
    gx = np.arange(bb0[0] - m, bb1[0] + m + hg, hg); gy = np.arange(bb0[1] - m, bb1[1] + m + hg, hg)
    X, Y = np.meshgrid(gx, gy)
    out = dict(geom=name, grid=f'{len(gx)}x{len(gy)}', nodes=int(X.size), segs=len(imp.A))
    t = time.time(); inside_d, fb, st, (jx, sx, segx) = dexel_inside(imp, gx, gy); t_d = time.time() - t
    nfb = int(fb.sum())
    t = time.time()
    if nfb: inside_d[fb] = np.abs(imp.winding(np.c_[X[fb], Y[fb]])) > 0.5
    t_fb = time.time() - t
    out.update(st, t_dexel=t_d, n_fallback=nfb, t_fallback=t_fb)
    t = time.time(); _, _, _, inside_ref_grid = imp.sdf_grid(X, Y, hg); t_cur = time.time() - t
    out['t_current_sdf_grid'] = t_cur
    # exact reference: brute-force generalized winding on all nodes (affordable at these sizes)
    t = time.time(); inside_exact = (np.abs(imp.winding(np.c_[X.ravel(), Y.ravel()])) > 0.5).reshape(X.shape); t_ex = time.time() - t
    out.update(t_exact_all=t_ex, mismatch_dexel=int((inside_d != inside_exact).sum()), mismatch_current=int((inside_ref_grid != inside_exact).sum()))
    # Hermite data: exact x-ray crossings vs linear interpolation of the (exact) signed distance along the row
    sdf = np.where(inside_exact, -1, 1) * imp.nearest(np.c_[X.ravel(), Y.ravel()])[0].reshape(X.shape)
    i = np.clip(np.searchsorted(gx, sx) - 1, 0, len(gx) - 2); jj = jx
    f0, f1 = sdf[jj, i], sdf[jj, i + 1]
    ok = f0 * f1 < 0
    s_lin = gx[i[ok]] + f0[ok] / (f0[ok] - f1[ok]) * hg
    err = np.abs(s_lin - sx[ok])
    out.update(hermite_edges=int(ok.sum()), lin_cut_err_max_over_h=float(err.max() / hg) if ok.any() else None,
               lin_cut_err_mean_over_h=float(err.mean() / hg) if ok.any() else None)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--geoms', nargs='+', default=['airfoil', 'twoelement', 'dirty', 'tmr656'])
    ap.add_argument('--res', nargs='+', type=float, default=[16, 64])
    a = ap.parse_args()
    for g in a.geoms:
        for r in a.res:
            o = run(g, r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}))
