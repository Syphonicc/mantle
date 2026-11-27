#!/usr/bin/env python3
"""Cp at ONERA M6 span stations from SU2 surface_flow.csv files (conservative variables, FREESTREAM_VEL_EQ_ONE).

usage: m6_cp_v1.py OUT.png LABEL=CASE_DIR [LABEL=CASE_DIR ...] [--mach 0.8395] [--tol 0.008]
Stations: eta = y / b = 0.20, 0.44, 0.65, 0.80, 0.90, 0.95 with semi-span b = 1.1963 (|y| is used, so full-span meshes work).
Cp = 2 (p - p_inf), p = (gamma - 1)(rhoE - |m|^2 / (2 rho)), p_inf = 1 / (gamma M^2). x is normalised by the local chord of the
surface points in each slab |y - y_s| < tol. Prints, per case and station, the suction peak and the mean |dCp| to the first case.
"""
import argparse, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('cases', nargs='+')
ap.add_argument('--exp', default=None, help='experimental file (x/c, -Cp blocks), e.g. proto2d/data/m6_exp_schmitt_charpin.txt')
ap.add_argument('--mach', type=float, default=0.8395); ap.add_argument('--tol', type=float, default=0.008)
a = ap.parse_args()
g, b = 1.4, 1.1963
pinf = 1.0 / (g * a.mach ** 2)
ETA = [0.20, 0.44, 0.65, 0.80, 0.90, 0.95]


def load(d):
    with open(f'{d}/surface_flow.csv') as f:
        hdr = [h.strip().strip('"') for h in f.readline().split(',')]
    D = np.loadtxt(f'{d}/surface_flow.csv', delimiter=',', skiprows=1)
    c = {h: D[:, i] for i, h in enumerate(hdr)}
    m2 = c['Momentum_x'] ** 2 + c['Momentum_y'] ** 2 + c['Momentum_z'] ** 2
    p = (g - 1) * (c['Energy'] - 0.5 * m2 / c['Density'])
    return c['x'], np.abs(c['y']), c['z'], 2 * (p - pinf)


data = [(lab, load(d)) for lab, d in (s.split('=', 1) for s in a.cases)]
exp = {}
if a.exp:
    key = None
    for line in open(a.exp):
        if line.startswith('# eta='):
            e, side = line[6:].split(); key = (round(float(e), 2), side); exp[key] = []
        elif line.strip() and not line.startswith('#') and key:
            x, mcp = map(float, line.split()[:2]); exp[key].append((x, -mcp))
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
for ax, eta in zip(axs.flat, ETA):
    ys = eta * b
    ref_curve = None
    for k, (lab, (x, y, z, cp)) in enumerate(data):
        m = np.abs(y - ys) < a.tol
        if m.sum() < 10:
            continue
        xs, zs, cs = x[m], z[m], cp[m]
        xle, xte = xs.min(), xs.max()
        xc = (xs - xle) / (xte - xle)
        # upper / lower split by z relative to the chord line through LE and TE points
        zle, zte = zs[np.argmin(xs)], zs[np.argmax(xs)]
        up = zs >= zle + (zte - zle) * xc
        for sel, ls in ((up, '-'), (~up, '--')):
            o = np.argsort(xc[sel])
            ax.plot(xc[sel][o], cs[sel][o], ls, color=f'C{k}', lw=1.2, label=lab if ls == '-' else None)
        # deviation to the first case on a common grid
        grid = np.linspace(0.02, 0.98, 97)
        cur = []
        for sel in (up, ~up):
            o = np.argsort(xc[sel]); cur.append(np.interp(grid, xc[sel][o], cs[sel][o]))
        cur = np.array(cur)
        if ref_curve is None:
            ref_curve = cur
            print(f'eta {eta:.2f} {lab:>12}: min Cp {cs.min():+.3f}')
        else:
            print(f'eta {eta:.2f} {lab:>12}: min Cp {cs.min():+.3f}  mean |dCp| to {data[0][0]} {np.abs(cur - ref_curve).mean():.3f}')
    for side, mk in (('upper', 'o'), ('lower', 's')):
        pts = np.array(exp.get((round(eta, 2), side), []))
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], mk, mfc='none', mec='k', ms=4, label='experiment' if side == 'upper' else None)
    ax.invert_yaxis(); ax.set_title(f'η = {eta:.2f}'); ax.set_xlabel('x/c'); ax.set_ylabel('Cp'); ax.grid(alpha=0.3)
axs.flat[0].legend()
fig.suptitle(f'ONERA M6, Euler, M {a.mach}, α 3.06° (solid: upper, dashed: lower)')
fig.tight_layout(); fig.savefig(a.out, dpi=110)
print('wrote', a.out)
