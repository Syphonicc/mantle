#!/usr/bin/env python3
"""ONERA M6 Euler validation: run SU2 on a mesh with the official test-case settings and report CL / CD / residual.

usage: m6_validate_v1.py NAME MESH.su2 [--wall WING] [--far FARFIELD] [--sym SYMMETRY|none] [--iter 2000] [--threads 8]
Case dir: proto2d/runs/m6/NAME/ (config, history.csv, log). Settings are $SU2_TESTCASES/euler/oneram6/inv_ONERAM6.cfg (default ~/SU2/TestCases)
(M 0.8395, AoA 3.06, JST, multigrid) with markers replaced; REF_AREA = 0 (SU2 projected area), so a full-span mesh and the
half-span reference give comparable coefficients. Prints one JSON line.
"""
import argparse, json, os, re, subprocess, time

ap = argparse.ArgumentParser()
ap.add_argument('name'); ap.add_argument('mesh')
ap.add_argument('--wall', default='wall'); ap.add_argument('--far', default='farfield'); ap.add_argument('--sym', default='none')
ap.add_argument('--iter', type=int, default=2000); ap.add_argument('--threads', type=int, default=8)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
case = os.path.join(here, 'runs', 'm6', a.name)
os.makedirs(case, exist_ok=True)
tc = os.path.expanduser(os.environ.get('SU2_TESTCASES', '~/SU2/TestCases'))   # SU2 source tree TestCases (config files)
cfg = open(os.path.join(tc, 'euler/oneram6/inv_ONERAM6.cfg')).read()
cfg = re.sub(r'(?m)^MARKER_EULER=.*$', f'MARKER_EULER= ( {a.wall} )', cfg)
cfg = re.sub(r'(?m)^MARKER_FAR=.*$', f'MARKER_FAR= ( {a.far} )', cfg)
cfg = re.sub(r'(?m)^MARKER_SYM=.*$', '' if a.sym == 'none' else f'MARKER_SYM= ( {a.sym} )', cfg)
for k in ('MARKER_PLOTTING', 'MARKER_MONITORING', 'MARKER_DESIGNING', 'DV_MARKER'):
    cfg = re.sub(rf'(?m)^{k}\s*=.*$', f'{k}= ( {a.wall} )', cfg)
cfg = re.sub(r'(?m)^ITER=.*$', f'ITER= {a.iter}', cfg)
cfg = re.sub(r'(?m)^CONV_RESIDUAL_MINVAL=.*$', 'CONV_RESIDUAL_MINVAL= -10', cfg)
cfg = re.sub(r'(?m)^MESH_FILENAME=.*$', f'MESH_FILENAME= {os.path.abspath(a.mesh)}', cfg)
cfg = re.sub(r'(?m)^OUTPUT_FILES\s*=.*$', 'OUTPUT_FILES= ( RESTART_ASCII, SURFACE_CSV )', cfg)
cfg = re.sub(r'(?m)^SCREEN_OUTPUT\s*=.*$', 'SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, LIFT, DRAG )', cfg)
cfg = re.sub(r'(?m)^HISTORY_OUTPUT\s*=.*$', '', cfg) + '\nHISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )\n'
open(os.path.join(case, 'm6.cfg'), 'w').write(cfg)
t = time.time()
r = subprocess.run([os.environ.get('SU2_CFD', 'SU2_CFD'), '-t', str(a.threads), 'm6.cfg'], cwd=case, capture_output=True, text=True)
open(os.path.join(case, 'log'), 'w').write(r.stdout + r.stderr)
hist = [l for l in open(os.path.join(case, 'history.csv')) if l.strip()] if os.path.exists(os.path.join(case, 'history.csv')) else []
res = dict(name=a.name, rc=r.returncode, t=round(time.time() - t, 1))
if len(hist) > 1:
    hdr = [h.strip().strip('"') for h in hist[0].split(',')]
    last = [float(x) for x in hist[-1].split(',')]
    d = dict(zip(hdr, last))
    res.update(iters=int(d.get('Inner_Iter', len(hist) - 1)), rms_rho=d.get('rms[Rho]'), CL=d.get('CL'), CD=d.get('CD'))
res['warnings'] = [l.strip() for l in (r.stdout + r.stderr).splitlines() if re.search(r're-orientation|distort|ERROR|Error', l)][:5]
print(json.dumps(res))
