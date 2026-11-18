#!/usr/bin/env python3
"""Gmsh baseline (SU2-ecosystem mesher) for the TMR NACA 0012, same spec as sfhm2d_v7:
wall spacing hwall, first layer dy0, growth r, layer stack thickness delta (BoundaryLayer field, quads,
fan at the sharp TE), bulk size h = hwall + grade * wall distance (MathEval on a Distance field),
farfield circle of radius R. Writes CASE/mesh.su2 (markers wall, farfield) and CASE/gmsh_info.json.

usage: python3 gmsh_compare_v1.py --out DIR --npts 656
       --hwall 0.00305 --dy0 4e-6 --growth 1.14 [--delta 0.02] [--grade 0.15] [--radius 400]
"""
import argparse, json, os, sys, time, glob, numpy as np
try:
    import gmsh
except ImportError:                       # user-site gmsh built for another Python: append (do not shadow numpy)
    sys.path += glob.glob(os.path.expanduser('~/.local/lib/python3*/site-packages')); import gmsh
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--npts', type=int, default=328)
ap.add_argument('--hwall', type=float, default=0.0061); ap.add_argument('--dy0', type=float, default=8e-6)
ap.add_argument('--growth', type=float, default=1.3); ap.add_argument('--delta', type=float, default=0.02)
ap.add_argument('--grade', type=float, default=0.15); ap.add_argument('--radius', type=float, default=400.0)
ap.add_argument('--algo', type=int, default=6)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
n = a.npts
x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n + 1)))
yt = 0.594689181 * (0.298222773 * np.sqrt(x) - 0.127125232 * x - 0.357907906 * x**2 + 0.291984971 * x**3 - 0.105174606 * x**4)
gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0); gmsh.model.add('tmr')
geo = gmsh.model.geo
te = geo.addPoint(1.0, 0.0, 0.0); le = geo.addPoint(0.0, 0.0, 0.0)
up = [te] + [geo.addPoint(x[i], yt[i], 0.0) for i in range(n - 1, 0, -1)] + [le]
lo = [le] + [geo.addPoint(x[i], -yt[i], 0.0) for i in range(1, n)] + [te]
cu = geo.addSpline(up); cl = geo.addSpline(lo)
c0 = geo.addPoint(0.5, 0, 0)
f = [geo.addPoint(0.5 + a.radius * np.cos(t), a.radius * np.sin(t), 0) for t in np.linspace(0, 2 * np.pi, 5)[:-1]]
arcs = [geo.addCircleArc(f[i], c0, f[(i + 1) % 4]) for i in range(4)]
outer = geo.addCurveLoop(arcs); inner = geo.addCurveLoop([cu, cl])
surf = geo.addPlaneSurface([outer, inner])
geo.synchronize()
gmsh.model.addPhysicalGroup(1, [cu, cl], name='wall')
gmsh.model.addPhysicalGroup(1, arcs, name='farfield')
gmsh.model.addPhysicalGroup(2, [surf], name='fluid')
fm = gmsh.model.mesh.field
fd = fm.add('Distance'); fm.setNumbers(fd, 'CurvesList', [cu, cl]); fm.setNumber(fd, 'Sampling', 4 * n)
fs = fm.add('MathEval'); fm.setString(fs, 'F', f'{a.hwall} + {a.grade} * F{fd}')
fm.setAsBackgroundMesh(fs)
bl = fm.add('BoundaryLayer')
fm.setNumbers(bl, 'CurvesList', [cu, cl]); fm.setNumber(bl, 'Size', a.dy0); fm.setNumber(bl, 'Ratio', a.growth)
fm.setNumber(bl, 'Thickness', a.delta); fm.setNumber(bl, 'Quads', 1); fm.setNumbers(bl, 'FanPointsList', [te])
fm.setNumber(bl, 'SizeFar', a.hwall)
fm.setAsBoundaryLayer(bl)
gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0); gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0); gmsh.option.setNumber('Mesh.Algorithm', a.algo)
status = 'ok'
try:
    gmsh.model.mesh.generate(2)
except Exception as e:
    status = f'generate failed: {e}'
tmesh = time.time() - t0
info = dict(status=status, t_mesh=tmesh)
if status == 'ok':
    types, tags, _ = gmsh.model.mesh.getElements(2)
    for ty, tg in zip(types, tags):
        info[gmsh.model.mesh.getElementProperties(ty)[0]] = int(len(tg))
    gmsh.option.setNumber('Mesh.SaveAll', 0)
    gmsh.write(os.path.join(a.out, 'mesh.su2'))
gmsh.finalize()
info.update(vars(a))
json.dump(info, open(os.path.join(a.out, 'gmsh_info.json'), 'w'), indent=1)
print(json.dumps(info))
