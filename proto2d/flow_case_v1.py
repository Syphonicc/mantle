#!/usr/bin/env python3
"""Populate 0/ and constant/ for a steady RANS (kOmegaSST) run on an existing 2D mesh; same setup for both meshers."""
import os, sys, re, numpy as np
case = sys.argv[1]; aoa = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0; Re = float(sys.argv[3]) if len(sys.argv) > 3 else 1e6
U = 1.0; nu = U * 1.0 / Re; I = 0.01; k = 1.5 * (U * I) ** 2; omega = k ** 0.5 / (0.09 ** 0.25 * 0.1)
ux, uy = U * np.cos(np.radians(aoa)), U * np.sin(np.radians(aoa))
H = lambda cls, obj: "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n" % (cls, obj)
# make front/back empty if present (snappy+extrudeMesh output)
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
field('k', 'volScalarField', '[0 2 -2 0 0 0 0]', f"{k}", f"type kqRWallFunction; value uniform {k};", f"type freestream; freestreamValue uniform {k};")
field('omega', 'volScalarField', '[0 0 -1 0 0 0 0]', f"{omega}", "type omegaWallFunction; value uniform 1;", f"type freestream; freestreamValue uniform {omega};")
field('nut', 'volScalarField', '[0 2 -1 0 0 0 0]', "0", "type nutkWallFunction; value uniform 0;", "type calculated; value uniform 0;")
os.makedirs(f"{case}/constant", exist_ok=True)
open(f"{case}/constant/transportProperties", 'w').write(H('dictionary', 'transportProperties') + f"transportModel Newtonian;\nnu {nu};\n")
open(f"{case}/constant/turbulenceProperties", 'w').write(H('dictionary', 'turbulenceProperties') + "simulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; printCoeffs off; }\n")
# system from the sfhm writer (controlDict with forceCoeffs/yPlus), copy if missing
if not os.path.exists(f"{case}/system/fvSolution") or 'GAMG' not in open(f"{case}/system/fvSolution").read():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sfhm2d_v1 import write_system; write_system(case)
cd = open(f"{case}/system/controlDict").read()
cd = re.sub(r'liftDir \([^)]*\)', f'liftDir ({-np.sin(np.radians(aoa))} {np.cos(np.radians(aoa))} 0)', cd)
cd = re.sub(r'dragDir \([^)]*\)', f'dragDir ({np.cos(np.radians(aoa))} {np.sin(np.radians(aoa))} 0)', cd)
cd = cd.replace('endTime 500', 'endTime 1500').replace('writeInterval 500', 'writeInterval 1500')
open(f"{case}/system/controlDict", 'w').write(cd)
print("ok", case, "nu", nu, "empties", empties)
