#!/usr/bin/env python3
"""Print last iteration, density residual, CL, CD from SU2 history.csv files. usage: su2_hist_v1.py DIR [DIR ...]"""
import sys, csv
for d in sys.argv[1:]:
    try:
        rows = list(csv.reader(open(f'{d}/history.csv')))
    except OSError:
        print(d, 'no history'); continue
    h = [c.strip().strip('"') for c in rows[0]]; r = rows[-1]
    g = lambda k: float(r[h.index(k)]) if k in h else float('nan')
    print(f"{d}: it {int(g('Inner_Iter'))} rms[Rho] {g('rms[Rho]'):.2f} CL {g('CL'):.5f} CD {g('CD'):.6f}")
