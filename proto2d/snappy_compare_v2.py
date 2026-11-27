#!/usr/bin/env python3
"""
snappyHexMesh baseline for the same 2D geometries, same wall spacing, same layer spec.
v2 vs v1: geometry from sfhm2d_v5 (adds the TMR NACA 0012), explicit --dy0 (first layer),
--lev (surface refinement levels above the base mesh; lmin = lmax - lev keeps the slab z-refinement bounded),
--relaxq (determinant / tet-quality / twist checks off, as commonly done for y+ ~ 1 layers).
Workflow: STL (polyline extruded in z) -> blockMesh slab -> snappyHexMesh (castellate, snap, layers)
          -> extrudeMesh from the front patch to a 1-cell 2D mesh -> checkMesh.
Note: snappy refines in z as well (no 2D mode), so its wall-clock is for a 3D slab and is
reported but not directly comparable.
"""
import numpy as np, os, sys, subprocess, time, json, argparse, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfhm2d_v5 import geometry

FOAM = "source /usr/lib/openfoam/openfoam2512/etc/bashrc >/dev/null 2>&1; "

def write_stl(polys, path, z0=-1.0, z1=2.0):
    with open(path, 'w') as f:
        f.write("solid wall\n")
        for pts, closed in polys:
            P = np.vstack([pts, pts[:1]]) if closed else pts
            for k in range(len(P) - 1):
                a, b = P[k], P[k + 1]
                for tri in (((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1)),
                            ((a[0], a[1], z0), (b[0], b[1], z1), (a[0], a[1], z1))):
                    f.write("facet normal 0 0 0\n outer loop\n")
                    for v in tri: f.write(f"  vertex {v[0]:.10g} {v[1]:.10g} {v[2]:.10g}\n")
                    f.write(" endloop\nendfacet\n")
        f.write("endsolid wall\n")

H = lambda cls, obj: "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n" % (cls, obj)

def setup(case, polys, D, lmin, lmax, nlayers, growth, delta, yc, first=None, relaxq=False):
    os.makedirs(f"{case}/system", exist_ok=True); os.makedirs(f"{case}/constant/triSurface", exist_ok=True)
    write_stl(polys, f"{case}/constant/triSurface/wall.stl")
    nb = 2 ** lmin; h0 = D / nb
    lev = lmax - lmin
    open(f"{case}/system/blockMeshDict", 'w').write(H('dictionary', 'blockMeshDict') + f"""
scale 1;
vertices
(
 ({-D/2} {yc-D/2} 0) ({D/2} {yc-D/2} 0) ({D/2} {yc+D/2} 0) ({-D/2} {yc+D/2} 0)
 ({-D/2} {yc-D/2} {h0}) ({D/2} {yc-D/2} {h0}) ({D/2} {yc+D/2} {h0}) ({-D/2} {yc+D/2} {h0})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nb} {nb} 1) simpleGrading (1 1 1) );
edges ();
boundary
(
 inlet  {{ type patch; faces ((0 4 7 3)); }}
 outlet {{ type patch; faces ((1 2 6 5)); }}
 top    {{ type patch; faces ((3 7 6 2)); }}
 bottom {{ type patch; faces ((0 1 5 4)); }}
 front  {{ type patch; faces ((0 3 2 1)); }}
 back   {{ type patch; faces ((4 5 6 7)); }}
);
""")
    # layers: snappy's final-layer thickness relative to the local cell size; total BL = delta
    hmin = D / 2 ** lmax
    if first is None: first = delta * (growth - 1) / (growth ** nlayers - 1)
    open(f"{case}/system/snappyHexMeshDict", 'w').write(H('dictionary', 'snappyHexMeshDict') + f"""
castellatedMesh true; snap true; addLayers true;
geometry {{ wall.stl {{ type triSurfaceMesh; name wall; }} }}
castellatedMeshControls
{{
    maxLocalCells 4000000; maxGlobalCells 8000000; minRefinementCells 0; maxLoadUnbalance 0.1; nCellsBetweenLevels 2;
    features ( {{ file "wall.eMesh"; level {lev}; }} );
    refinementSurfaces {{ wall {{ level ({lev} {lev}); patchInfo {{ type wall; }} }} }}
    resolveFeatureAngle 30;
    refinementRegions {{ wall {{ mode distance; levels (({3*delta} {lev})); }} }}
    locationInMesh ({-D/2+0.3*D} {yc + 0.37*D} {h0*0.5});
    allowFreeStandingZoneFaces true;
}}
snapControls {{ nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5; nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true; multiRegionFeatureSnap false; }}
addLayersControls
{{
    relativeSizes false; layers {{ wall {{ nSurfaceLayers {nlayers}; }} }}
    expansionRatio {growth}; firstLayerThickness {first}; minThickness {first*0.1};
    nGrow 0; featureAngle 180; slipFeatureAngle 30; nRelaxIter 5; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50; nRelaxedIter 20;
}}
meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol {'-1e30' if relaxq else '1e-13'}; minTetQuality {'-1e30' if relaxq else '1e-15'};
    minArea -1; minTwist {'-1' if relaxq else '0.02'}; minDeterminant {'-1' if relaxq else '0.001'}; minFaceWeight {'0.01' if relaxq else '0.02'}; minVolRatio {'0.001' if relaxq else '0.01'}; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75;
    relaxed {{ maxNonOrtho 75; }}
}}
mergeTolerance 1e-6;
""")
    open(f"{case}/system/surfaceFeatureExtractDict", 'w').write(H('dictionary', 'surfaceFeatureExtractDict') + """
wall.stl { extractionMethod extractFromSurface; includedAngle 150; subsetFeatures { nonManifoldEdges no; openEdges yes; } writeObj no; }
""")
    open(f"{case}/system/extrudeMeshDict", 'w').write(H('dictionary', 'extrudeMeshDict') + """
constructFrom patch; sourceCase "."; sourcePatches (front); exposedPatchName back; flipNormals false;
extrudeModel linearNormal; nLayers 1; expansionRatio 1.0; linearNormalCoeffs { thickness 0.1; } mergeFaces false; mergeTol 0;
""")
    open(f"{case}/system/controlDict", 'w').write(H('dictionary', 'controlDict') + """
application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 8; timeFormat general;
""")
    open(f"{case}/system/fvSchemes", 'w').write(H('dictionary', 'fvSchemes') + "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} interpolationSchemes{default linear;} snGradSchemes{default corrected;}\n")
    open(f"{case}/system/fvSolution", 'w').write(H('dictionary', 'fvSolution') + "solvers{} \n")
    open(f"{case}/system/decomposeParDict", 'w').write(H('dictionary', 'decomposeParDict') + "numberOfSubdomains 8; method scotch;\n")

def run(case, np_=8):
    log = {}
    def sh(cmd, name):
        t = time.time()
        r = subprocess.run(["bash", "-c", FOAM + f"cd {case} && {cmd} > log.{name} 2>&1"])
        log[name] = time.time() - t
        return r.returncode
    sh("surfaceFeatureExtract", "surfaceFeatureExtract")
    sh("blockMesh", "blockMesh")
    sh(f"decomposePar -force && mpirun -np {np_} snappyHexMesh -parallel -overwrite && reconstructParMesh -constant -mergeTol 1e-6", "snappyHexMesh")
    sh("extrudeMesh", "extrudeMesh")
    sh("checkMesh -allGeometry -allTopology", "checkMesh")
    return log

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--geom', default='airfoil'); ap.add_argument('--out', default='runs/v5/snappy_airfoil')
    ap.add_argument('--delta', type=float, default=0.02); ap.add_argument('--nlayers', type=int, default=8)
    ap.add_argument('--growth', type=float, default=1.25); ap.add_argument('--domain', type=float, default=16.0)
    ap.add_argument('--lmin', type=int, default=4); ap.add_argument('--lmax', type=int, default=11); ap.add_argument('--np', type=int, default=8)
    ap.add_argument('--dy0', type=float, default=None); ap.add_argument('--lev', type=int, default=None)
    ap.add_argument('--relaxq', action='store_true', help='high-aspect layer quality settings (y+~1 practice)')
    a = ap.parse_args()
    polys = geometry(a.geom)
    if a.lev is not None: a.lmin = a.lmax - a.lev
    allp = np.vstack([p for p, _ in polys]); yc = 0.5 * (allp[:, 1].min() + allp[:, 1].max())
    setup(a.out, polys, a.domain, a.lmin, a.lmax, a.nlayers, a.growth, a.delta, yc, a.dy0, a.relaxq)
    t = run(a.out, a.np)
    print(json.dumps(t, indent=1))
