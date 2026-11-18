#!/usr/bin/env python3
"""rcore hex-core + pillow harness v1 (compare_shell_v1.py untouched). Cases: (i) UV sphere r=0.6, (ii) sphere+torus soup (dexel3d_v1.soup(0)),
(iii) closed cube soup, side 1 (sharp edges). For each, at L (default 7, domain [-1.5,1.7]^3, h_surf = size/2^L, grade 0.25, delta = 4 h_surf) run
hexcore_bench, then checkMesh -allGeometry -allTopology (failed checks, counts, max non-orth, max skewness), SU2_CFD ITER=2 (Euler on `shell`;
only re-orientation / distortion lines kept) and report the histogram of pillow scaled Jacobians (from the bench: bins <0, 0-0.2, 0.2-0.5, 0.5-0.8, >0.8).
All outputs persist under proto2d/runs/hexcore/. One JSON line per case.
usage: compare_hexcore_v1.py [--L 7] [--delta-h 4] [--shapes sphere soup cube] [--gamma 0.3] [--theta-max 60] [--smooth-iters 5] [--extra key=val ...]
"""
import argparse, json, os, re, subprocess, sys, time, numpy as np, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, file)); m = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [file]; spec.loader.exec_module(m); sys.argv = argv; return m
D = load('d', 'dexel3d_v1.py')
BIN = os.path.join(here, '..', 'rcore', 'target', 'release', 'hexcore_bench')
SU2 = '/home/syphonic/SU2_install/bin/SU2_CFD'
TMP = os.path.join(here, 'runs', 'hexcore')
BASHRC = '/usr/lib/openfoam/openfoam2512/etc/bashrc'
ORG, SIZE = -1.5, 3.2
CFG = """SOLVER= EULER
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
MACH_NUMBER= 0.3
AOA= 0.0
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 288.15
REF_LENGTH= 1.2
REF_AREA= 1.1309733552923256
MARKER_EULER= ( {patch} )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( {patch} )
MARKER_MONITORING= ( {patch} )
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.5, 5.0, 100.0 )
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
TIME_DISCRE_FLOW= EULER_IMPLICIT
ITER= {iter}
CONV_FIELD= RMS_DENSITY
CONV_RESIDUAL_MINVAL= {minval}
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, DRAG, LIFT )
HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )
OUTPUT_FILES= ( RESTART )
"""

def cube_soup(lo=-0.5, hi=0.5):
    """closed cube, 12 triangles, outward (counter-clockwise seen from outside) orientation"""
    P = lambda i, j, k: [hi if i else lo, hi if j else lo, hi if k else lo]
    quads = [  # (ccw seen from outside)
        [P(0,0,0), P(0,1,0), P(0,1,1), P(0,0,1)],  # -x
        [P(1,0,0), P(1,0,1), P(1,1,1), P(1,1,0)],  # +x
        [P(0,0,0), P(0,0,1), P(1,0,1), P(1,0,0)],  # -y
        [P(0,1,0), P(1,1,0), P(1,1,1), P(0,1,1)],  # +y
        [P(0,0,0), P(1,0,0), P(1,1,0), P(0,1,0)],  # -z
        [P(0,0,1), P(0,1,1), P(1,1,1), P(1,0,1)],  # +z
    ]
    T = []
    for q in quads: T += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    return np.array(T, dtype=float)

def write_soup(name):
    if name == 'sphere':
        V, F = D.uv_sphere(np.array([0.0, 0.0, 0.0]), 0.6); T = V[F]
    elif name == 'cube':
        T = cube_soup()
    else:
        T = D.soup(0)
    p = f'{TMP}/{name}.bin'
    with open(p, 'wb') as f:
        f.write(np.uint64(len(T)).tobytes()); f.write(np.ascontiguousarray(T, dtype='<f8').tobytes())
    return p

def foam_list(path):
    t = open(path).read(); t = t[t.index('}') + 1:]; t = t[t.index('(') + 1:t.rindex(')')]
    return np.array(t.split(), dtype=np.int64)

def run_checkmesh(case, patch):
    sets = f'{case}/constant/polyMesh/sets'
    subprocess.run(['rm', '-rf', sets])
    out = subprocess.run(['bash', '-c', f'source {BASHRC} >/dev/null 2>&1; checkMesh -allGeometry -allTopology -writeSets raw -case {case}'],
                         capture_output=True, text=True).stdout
    g = lambda pat: (re.search(pat, out) or [None, None])[1]
    failed = [re.sub(r'\s+', ' ', l.strip(' *')) for l in out.split('\n') if l.startswith(' ***')]
    pm = f'{case}/constant/polyMesh'
    own = foam_list(f'{pm}/owner'); nei = foam_list(f'{pm}/neighbour'); ncell = int(g(r'\n\s+cells:\s+(\d+)'))
    b = open(f'{pm}/boundary').read()
    pat = {n: (int(s), int(s) + int(c)) for n, c, s in re.findall(r'(\w+)\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+)', b)}
    cut = np.zeros(ncell, bool); cut[own[pat[patch][0]:pat[patch][1]]] = True
    res = {}
    if os.path.isdir(sets):
        for nm in sorted(os.listdir(sets)):
            if nm == 'upperTriangularFace': continue
            ids = foam_list(f'{sets}/{nm}')
            if nm in ('concaveCells', 'underdeterminedCells'):
                c = int(cut[ids].sum())
            else:
                ic = cut[own[ids]].copy(); inn = ids < len(nei)
                ic[inn] |= cut[nei[ids[inn]]]; c = int(ic.sum())
            res[nm] = {'total': int(len(ids)), 'clipped': c, 'other': int(len(ids)) - c}
    mx = lambda pat: (lambda v: float(v.rstrip('.,')) if v else None)(g(pat))
    return dict(cells=ncell, n_clipped_cells=int(cut.sum()), failed_checks=failed, mesh_ok='Mesh OK.' in out, sets=res,
                max_nonorth=mx(r'Mesh non-orthogonality Max: ([0-9.eE+-]+)'), max_skew=mx(r'Max skewness = ([0-9.eE+-]+)'))


def foam_check(case):
    out = subprocess.run(['bash', '-c', f'source {BASHRC} >/dev/null 2>&1; checkMesh -allGeometry -allTopology -case {case}'],
                         capture_output=True, text=True).stdout
    open(f'{case}/checkMesh.log', 'w').write(out)
    g = lambda pat: (re.search(pat, out) or [None, None])[1]
    failed = [re.sub(r'\s+', ' ', l.strip(' *')) for l in out.split('\n') if l.startswith(' ***')]
    mx = lambda pat: (lambda v: float(v.rstrip('.,')) if v else None)(g(pat))
    counts = {k: int(g(rf'\n\s+{k}:\s+(\d+)')) for k in ('points', 'faces', 'cells', 'hexahedra', 'polyhedra') if g(rf'\n\s+{k}:\s+(\d+)')}
    return dict(counts=counts, failed_checks=failed, mesh_ok='Mesh OK.' in out,
                max_nonorth=mx(r'Mesh non-orthogonality Max: ([0-9.eE+-]+)'), max_skew=mx(r'Max skewness = ([0-9.eE+-]+)'))

def su2_run(case, it, patch):
    cfg = f'{case}/it.cfg'
    open(cfg, 'w').write(CFG.format(iter=it, minval=-8, patch=patch))
    t = time.time()
    r = subprocess.run([SU2, '-t', '8', cfg], cwd=case, capture_output=True, text=True, timeout=3000)
    log = r.stdout + r.stderr
    open(f'{case}/it.log', 'w').write(log)
    lines = [re.sub(r'\s+', ' ', l.strip('> ')) for l in log.split('\n') if re.search(r're-orientation|distorted', l)]
    return lines, time.time() - t, r.returncode

ap = argparse.ArgumentParser(); ap.add_argument('--L', nargs='+', type=int, default=[7])
ap.add_argument('--grade', type=float, default=0.25); ap.add_argument('--delta-h', type=float, default=4.0)
ap.add_argument('--gamma', type=float, default=0.3); ap.add_argument('--theta-max', type=float, default=60.0)
ap.add_argument('--smooth-iters', type=int, default=5); ap.add_argument('--tag', default='')
ap.add_argument('--extra', nargs='*', default=[])
ap.add_argument('--shapes', nargs='+', default=['sphere', 'soup', 'cube'])
ap.add_argument('--no-su2', action='store_true')
a = ap.parse_args()
os.makedirs(TMP, exist_ok=True)

for shape in a.shapes:
    soup = write_soup(shape)
    for L in a.L:
        case = f'{TMP}/{shape}_L{L}{a.tag}'
        hs = SIZE / 2 ** L
        t = time.time()
        r = subprocess.run([BIN, soup, f'origin={ORG},{ORG},{ORG}', f'size={SIZE}', f'L={L}', f'h_surf={hs}', f'grade={a.grade}', f'delta={a.delta_h * hs}',
                            f'gamma={a.gamma}', f'theta_max={a.theta_max}', f'smooth_iters={a.smooth_iters}'] + a.extra + [case], capture_output=True, text=True)
        if r.returncode: print(json.dumps(dict(shape=shape, L=L, error=r.stderr[-500:])), flush=True); continue
        bench = json.loads(r.stdout.strip().split('\n')[-1]); t_bench = time.time() - t
        open(f'{case}/bench.json', 'w').write(json.dumps(bench))
        keys = ['leaves', 'core_initial', 'core_cells', 'removed_margin', 'removed_nonmanifold', 'removed_hanging', 'removed_split', 'removed_normal', 'removed_jacobian',
                'rounds', 'core_boundary_quads', 'pillow_hexes', 'core_split_faces', 'nonmanifold_edges_final', 'nonmanifold_verts_final', 'normal_violations_final',
                'sj_min_before', 'sj_min_after', 'sj_mean', 'n_sj_below_0p2', 'n_sj_below_0', 'sj_hist_before', 'sj_hist_after', 'max_proj_dist_over_h',
                'cells', 'hex_cells', 'poly_cells', 'orphan_faces', 'min_vol_ratio', 'volume', 'su2', 't_octree', 't_hexcore', 't_total']
        t = time.time(); cm = foam_check(case); t_cm = time.time() - t
        if a.no_su2: su_lines, t_su, rc = [], 0.0, None
        else: su_lines, t_su, rc = su2_run(case, 2, 'shell')
        print(json.dumps(dict(shape=shape, L=L, bench={k: bench[k] for k in keys}, sj_hist_bins='<0,0-0.2,0.2-0.5,0.5-0.8,>0.8', checkMesh=cm, su2_load_lines=su_lines, su2_rc=rc,
                              t_bench=round(t_bench, 2), t_checkmesh=round(t_cm, 2), t_su2_2iter=round(t_su, 2))), flush=True)
