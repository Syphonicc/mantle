#!/usr/bin/env python3
"""SU2 config from the official SU2 TestCases rans/naca0012/turb_NACA0012_sa.cfg (TMR NACA 0012: M 0.15,
Re 6e6, alpha 10, SA-neg), only adapted to our meshes: markers wall/farfield, mesh.su2, no restart.
v2 vs v1: official settings (CFL 1000 fixed, no multigrid, linear solver 1e-10, MUSCL turbulence).
usage: su2_case_v2.py CASE_DIR [--iters 6000] [--aoa 10]"""
import argparse, os, re
ap = argparse.ArgumentParser(); ap.add_argument('case'); ap.add_argument('--iters', type=int, default=6000); ap.add_argument('--aoa', type=float, default=10.0)
a = ap.parse_args()
ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs/su2/ref/turb_NACA0012_sa.cfg')
s = open(ref).read()
s = s.replace('airfoil', 'wall')
for k, v in {'RESTART_SOL': 'NO', 'MESH_FILENAME': 'mesh.su2', 'ITER': str(a.iters), 'AOA': str(a.aoa),
             'OUTPUT_WRT_FREQ': '100000'}.items():
    s = re.sub(rf'^{k}\s*=.*$', f'{k}= {v}', s, flags=re.M)
s += '\nHISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )\nOUTPUT_FILES= ( RESTART, SURFACE_CSV )\n'
open(os.path.join(a.case, 'turb.cfg'), 'w').write(s)
print('wrote', os.path.join(a.case, 'turb.cfg'))
