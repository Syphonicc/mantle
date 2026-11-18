#!/usr/bin/env python3
"""Cp and Cf along an aerofoil from OpenFOAM 'surfaces' raw samples (p, wallShearStress), several cases overlaid.
usage: plot_cpcf_v1.py OUT.png LABEL=CASE [LABEL=CASE ...]   (U = 1, rho = 1, chord 1, one z-plane used)"""
import sys, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out = sys.argv[1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for spec in sys.argv[2:]:
    lab, case = spec.split('=', 1)
    fp = sorted(glob.glob(f'{case}/postProcessing/wallsurf/*/p_wall.raw'))[-1]
    fw = sorted(glob.glob(f'{case}/postProcessing/wallsurf/*/wallShearStress_wall.raw'))[-1]
    p = np.loadtxt(fp, comments='#'); w = np.loadtxt(fw, comments='#')
    z0 = p[:, 2].min(); m = np.abs(p[:, 2] - z0) < 1e-9 * max(1, abs(z0)) + 1e-12
    if m.sum() < 10: m = np.ones(len(p), bool)
    x, y, cp = p[m, 0], p[m, 1], p[m, 3] / 0.5
    cf = np.linalg.norm(w[m, 3:6], axis=1) / 0.5
    for side, sm, ls in (('upper', y >= 0, '-'), ('lower', y < 0, '--')):
        o = np.argsort(x[sm])
        a1.plot(x[sm][o], cp[sm][o], ls, lw=1, label=f'{lab} {side}')
        a2.plot(x[sm][o], cf[sm][o], ls, lw=1, label=f'{lab} {side}')
a1.invert_yaxis(); a1.set_xlabel('x/c'); a1.set_ylabel('Cp'); a1.grid(alpha=.3); a1.legend(fontsize=7)
a2.set_xlabel('x/c'); a2.set_ylabel('Cf'); a2.set_ylim(0, 0.02); a2.grid(alpha=.3); a2.legend(fontsize=7)
fig.suptitle('NACA 0012 (TMR), Re 6e6, alpha 10 deg, SA, SFHM v6 mesh family, 800c domain')
fig.tight_layout(); fig.savefig(out, dpi=150); print('wrote', out)
