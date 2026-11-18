#!/usr/bin/env python3
"""Wall-resolved steady RANS (kOmegaSST) on an existing 2D mesh, NASA TMR NACA 0012 conditions by default.
Same setup for every mesher. v3 vs v2: --model sa (Spalart-Allmaras, nuTilda wall 0,
freestream nuTilda = 3 nu as in TMR). v2 vs v1: low-Re wall treatment (y+ ~ 1), TMR freestream turbulence
(Tu 0.052 %, nut/nu 0.009), endTime / processor count arguments, residual-converged stop.

usage: flow_case_v3.py CASE [--aoa 10] [--re 6e6] [--iters 8000] [--model sst|sa]
"""
import os, sys, re, argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('case'); ap.add_argument('--aoa', type=float, default=10.0); ap.add_argument('--re', type=float, default=6e6)
ap.add_argument('--iters', type=int, default=8000); ap.add_argument('--tu', type=float, default=0.00052); ap.add_argument('--nutr', type=float, default=0.009)
ap.add_argument('--model', choices=['sst', 'sa'], default='sst')
a = ap.parse_args()
case = a.case; U = 1.0; nu = U * 1.0 / a.re
k = 1.5 * (U * a.tu) ** 2; nut = a.nutr * nu; omega = k / nut
ux, uy = U * np.cos(np.radians(a.aoa)), U * np.sin(np.radians(a.aoa))
H = lambda cls, obj: "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n" % (cls, obj)
b = open(f"{case}/constant/polyMesh/boundary").read()
b = re.sub(r'(\n\s*(front|back|frontAndBack)\s*\{[^}]*?type\s+)\w+;', r'\1empty;', b)
open(f"{case}/constant/polyMesh/boundary", 'w').write(b)
empties = [n for n in ('front', 'back', 'frontAndBack') if re.search(r'\n\s*%s\s*\{' % n, b)]
os.makedirs(f"{case}/0", exist_ok=True)
def field(name, cls, dim, internal, wall, far):
    s = H(cls, name) + f"dimensions {dim};\ninternalField uniform {internal};\nboundaryField\n{{\n    wall {{ {wall} }}\n"
    s += f'    "(inlet|outlet|top|bottom)" {{ {far} }}\n'
    for e in empties: s += f"    {e} {{ type empty; }}\n"
    s += "}\n"; open(f"{case}/0/{name}", 'w').write(s)
field('U', 'volVectorField', '[0 1 -1 0 0 0 0]', f"({ux} {uy} 0)", "type noSlip;", f"type freestreamVelocity; freestreamValue uniform ({ux} {uy} 0);")
field('p', 'volScalarField', '[0 2 -2 0 0 0 0]', "0", "type zeroGradient;", "type freestreamPressure; freestreamValue uniform 0;")
field('k', 'volScalarField', '[0 2 -2 0 0 0 0]', f"{k}", "type fixedValue; value uniform 1e-14;", f"type freestream; freestreamValue uniform {k};")
field('omega', 'volScalarField', '[0 0 -1 0 0 0 0]', f"{omega}", f"type omegaWallFunction; value uniform {omega};", f"type freestream; freestreamValue uniform {omega};")
field('nut', 'volScalarField', '[0 2 -1 0 0 0 0]', f"{nut}", "type nutLowReWallFunction; value uniform 0;", "type calculated; value uniform 0;")
open(f"{case}/constant/transportProperties", 'w').write(H('dictionary', 'transportProperties') + f"transportModel Newtonian;\nnu {nu};\n")
if a.model == 'sa':
    nt = 3 * nu; nutsa = nt * (3**3 / (3**3 + 7.1**3))
    field('nuTilda', 'volScalarField', '[0 2 -1 0 0 0 0]', f"{nt}", "type fixedValue; value uniform 0;", f"type freestream; freestreamValue uniform {nt};")
    field('nut', 'volScalarField', '[0 2 -1 0 0 0 0]', f"{nutsa}", "type fixedValue; value uniform 0;", "type calculated; value uniform 0;")
    for fn in ('k', 'omega'): os.remove(f"{case}/0/{fn}")
rasm = 'SpalartAllmaras' if a.model == 'sa' else 'kOmegaSST'
open(f"{case}/constant/turbulenceProperties", 'w').write(H('dictionary', 'turbulenceProperties') + f"simulationType RAS;\nRAS {{ RASModel {rasm}; turbulence on; printCoeffs off; }}\n")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sfhm2d_v5 import write_system
write_system(case)
cd = open(f"{case}/system/controlDict").read()
cd = re.sub(r'liftDir \([^)]*\)', f'liftDir ({-np.sin(np.radians(a.aoa))} {np.cos(np.radians(a.aoa))} 0)', cd)
cd = re.sub(r'dragDir \([^)]*\)', f'dragDir ({np.cos(np.radians(a.aoa))} {np.sin(np.radians(a.aoa))} 0)', cd)
cd = cd.replace('endTime 500', f'endTime {a.iters}').replace('writeInterval 500', f'writeInterval {a.iters}')
cd = cd.replace('startFrom latestTime', 'startFrom startTime')
open(f"{case}/system/controlDict", 'w').write(cd)
fs = open(f"{case}/system/fvSolution").read()
fs = fs.replace('"(U|k|omega)"', '"(U|k|omega|nuTilda)"')
sc = open(f"{case}/system/fvSchemes").read()
sc = sc.replace('div(phi,omega) $turbulence;', 'div(phi,omega) $turbulence; div(phi,nuTilda) $turbulence;')
open(f"{case}/system/fvSchemes", 'w').write(sc)
fs = fs.replace('residualControl { p 1e-5; U 1e-5; "(k|omega)" 1e-5; }', 'residualControl { p 1e-7; U 1e-8; "(k|omega)" 1e-8; }')
fs = fs.replace('relaxationFactors { equations { U 0.9; ".*" 0.7; } }', 'relaxationFactors { equations { U 0.9; ".*" 0.6; } }')
open(f"{case}/system/fvSolution", 'w').write(fs)
open(f"{case}/system/decomposeParDict", 'w').write(H('dictionary', 'decomposeParDict') + "numberOfSubdomains 8; method scotch;\n")
print("ok", case, "nu", nu, "k", k, "omega", omega, "empties", empties)
