#!/usr/bin/env python3
"""rcore cut-cell mesh harness v2 (v1 untouched). For (i) a single closed UV sphere (r=0.6, origin) and (ii) the sphere+torus
soup (dexel3d_v1.soup(0)), at each L, with and without cut-cell agglomeration (merge_tau=0 / --tau):
  * cutmesh_bench (JSON),
  * OpenFOAM checkMesh -allGeometry -allTopology -writeSets raw; every flagged set (constant/polyMesh/sets/*) is classified
    into CUT cells (owners of wall faces) and UNCUT cells (faces: owner or neighbour cut),
  * SU2_CFD for 1 iteration: its re-orientation / distortion lines,
  * sphere at --euler-L (default 6): SU2 Euler run, M 0.3, up to 2000 iterations, CONV_RESIDUAL_MINVAL=-8 -> final rms density,
    iterations, CD (reference area = pi r^2).
One JSON line per case. Cases are written to a temporary run directory.
usage: compare_cutmesh_v2.py [--L 6 7] [--tau 0.1] [--euler-L 6] [--no-euler] [--shapes sphere soup]
"""
import argparse, json, os, re, subprocess, sys, time, numpy as np, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, file)); m = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [file]; spec.loader.exec_module(m); sys.argv = argv; return m
D = load('d', 'dexel3d_v1.py')
BIN = os.path.join(here, '..', 'rcore', 'target', 'release', 'cutmesh_bench')
SU2 = os.environ.get('SU2_CFD', 'SU2_CFD')  # SU2 executable: $SU2_CFD, else found on PATH
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'cutmesh2')
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
MARKER_EULER= ( wall )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( wall )
MARKER_MONITORING= ( wall )
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
ap = argparse.ArgumentParser(); ap.add_argument('--L', nargs='+', type=int, default=[6, 7])
ap.add_argument('--tau', type=float, default=0.1); ap.add_argument('--grade', type=float, default=0.25); ap.add_argument('--eps', type=float, default=0.05)
ap.add_argument('--euler-L', type=int, default=6); ap.add_argument('--no-euler', action='store_true')
ap.add_argument('--shapes', nargs='+', default=['sphere', 'soup']); ap.add_argument('--variants', nargs='+', default=['nomerge', 'merge'])
a = ap.parse_args()
os.makedirs(TMP, exist_ok=True)

def write_soup(name):
    if name == 'sphere':
        V, F = D.uv_sphere(np.array([0.0, 0.0, 0.0]), 0.6); T = V[F]
    else:
        T = D.soup(0)
    p = f'{TMP}/{name}.bin'
    with open(p, 'wb') as f:
        f.write(np.uint64(len(T)).tobytes()); f.write(np.ascontiguousarray(T, dtype='<f8').tobytes())
    return p

def foam_list(path):
    """ints of an OpenFOAM ascii label list / set file (after the FoamFile header, inside the first parenthesis)"""
    t = open(path).read(); t = t[t.index('}') + 1:]; t = t[t.index('(') + 1:t.rindex(')')]
    return np.array(t.split(), dtype=np.int64)

def run_checkmesh(case):
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
    cut = np.zeros(ncell, bool); cut[own[pat['wall'][0]:pat['wall'][1]]] = True
    res = {}
    if os.path.isdir(sets):
        for nm in sorted(os.listdir(sets)):
            if nm == 'upperTriangularFace': continue  # equal-neighbour faces of cell pairs sharing 2+ faces; checkMesh reports 'ordering OK'
            ids = foam_list(f'{sets}/{nm}')
            if nm in ('concaveCells', 'underdeterminedCells'):
                c = int(cut[ids].sum())
            else:
                ic = cut[own[ids]].copy(); inn = ids < len(nei)
                ic[inn] |= cut[nei[ids[inn]]]; c = int(ic.sum())
            res[nm] = {'total': int(len(ids)), 'cut': c, 'uncut': int(len(ids)) - c}
    mx = lambda pat: (lambda v: float(v.rstrip('.,')) if v else None)(g(pat))
    return dict(cells=ncell, n_cut_cells=int(cut.sum()), failed_checks=failed, mesh_ok='Mesh OK.' in out, sets=res,
                total_volume=mx(r'Total volume = ([0-9.eE+-]+)'), max_nonorth=mx(r'Mesh non-orthogonality Max: ([0-9.eE+-]+)'),
                max_skew=mx(r'Max skewness = ([0-9.eE+-]+)'))

def su2_run(case, it, minval=-8, name='run'):
    cfg = f'{case}/{name}.cfg'
    open(cfg, 'w').write(CFG.format(iter=it, minval=minval))
    t = time.time()
    r = subprocess.run([SU2, '-t', '8', cfg], cwd=case, capture_output=True, text=True, timeout=3000)
    log = r.stdout + r.stderr
    open(f'{case}/{name}.log', 'w').write(log)
    lines = [re.sub(r'\s+', ' ', l.strip('> ')) for l in log.split('\n') if re.search(r're-orientation|distorted', l)]
    return log, lines, time.time() - t

def euler_result(case):
    H = np.genfromtxt(f'{case}/history.csv', delimiter=',', names=True, dtype=None, encoding=None)
    names = list(H.dtype.names); low = [n.strip('"').lower() for n in names]
    rho = names[[i for i, n in enumerate(low) if n.startswith('rms') and 'rho' in n][0]]
    cd = names[[i for i, n in enumerate(low) if n in ('cd', 'd') or n.endswith('cd')][0]]
    return dict(iterations=int(len(H)), final_rms_rho=float(H[rho][-1]), CD=float(H[cd][-1]), CD_min_abs_last50=float(np.abs(H[cd][-50:]).min()))

for shape in a.shapes:
    soup = write_soup(shape)
    for L in a.L:
        for var in a.variants:
            case = f'{TMP}/{shape}_L{L}_{var}'
            t = time.time()
            r = subprocess.run([BIN, soup, f'origin={ORG},{ORG},{ORG}', f'size={SIZE}', f'L={L}', f'h_surf={SIZE / 2 ** L}', f'grade={a.grade}',
                                f'eps={a.eps}', f'merge_tau={a.tau if var == "merge" else 0}', case], capture_output=True, text=True)
            if r.returncode: print(json.dumps(dict(shape=shape, L=L, variant=var, error=r.stderr[-500:])), flush=True); continue
            bench = json.loads(r.stdout.strip().split('\n')[-1]); t_bench = time.time() - t
            keys = ['fluid_cells', 'cut_cells', 'hex_cells', 'poly_cells', 'orphan_faces', 'unclosed_cells', 'volume', 'cells_merged', 'merge_groups', 'merge_passes',
                    'unmergeable_bad_cells', 'bad_cells_before', 'bad_cells_after', 'min_vol_ratio_before', 'min_vol_ratio_after', 'quality_before',
                    'quality_after', 'su2', 'su2_elements', 't_merge', 't_total']
            t = time.time(); cm = run_checkmesh(case); t_cm = time.time() - t
            _, su_lines, t_su1 = su2_run(case, 1, name='it1')
            out = dict(shape=shape, L=L, variant=var, bench={k: bench[k] for k in keys}, checkMesh=cm, su2_load_lines=su_lines,
                       t_bench=round(t_bench, 2), t_checkmesh=round(t_cm, 2), t_su2_1iter=round(t_su1, 2))
            if shape == 'sphere' and L == a.euler_L and not a.no_euler:
                log, _, t_e = su2_run(case, 2000, -8, name='euler')
                try: out['euler'] = dict(euler_result(case), wall_s=round(t_e, 1))
                except Exception as e: out['euler'] = dict(error=str(e), tail=log[-400:])
            print(json.dumps(out), flush=True)
