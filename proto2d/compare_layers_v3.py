#!/usr/bin/env python3
"""rcore boundary-layer harness v2: columns=nearest vs columns=harmonic (hexcore_bench layers=N columns=...). Cases: sphere R 0.6, sphere+torus soup
(dexel3d_v1.soup(0)), cube side 1, cross (union of two overlapping closed boxes [-.6,.6]x[-.2,.2]x[-.2,.2] and [-.2,.2]x[-.6,.6]x[-.2,.2]: concave edges
and corners). Domain [-1.5,1.7]^3, L (default 7), h_surf = size/2^L, grade 0.25, delta = 4 h_surf, layers=12, growth 1.25, dy0 = stack-fit value
delta (r-1)/(r^N-1) (reported). Per (case, columns): hexcore_bench JSON (layer stats incl. laplace / fallback / first-cell distances for ALL columns);
default checkMesh (Mesh OK / failed checks); checkMesh -allGeometry -allTopology (old sets deleted; failures attributed to layer / pillow / core cells
by cell id: core cells first, then pillow hexes, then layer cells); SU2_CFD ITER=2 Euler load check (re-orientation / distortion / warning lines, rc).
Outputs under proto2d/runs/layers_v2/. One JSON line per (case, columns).
usage: compare_layers_v2.py [--cases sphere:7 soup:7 cube:7 cross:7] [--columns nearest harmonic] [--layers 12] [--growth 1.25] [--snap S] [--hg-k K] [--tag T] [--no-su2] [--no-cm] [--extra k=v ...]
"""
import argparse, json, os, re, subprocess, sys, time, numpy as np, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, file)); m = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [file]; spec.loader.exec_module(m); sys.argv = argv; return m
D = load('d', 'dexel3d_v1.py')
BIN = os.path.join(here, '..', 'rcore', 'target', 'release', 'hexcore_bench')
SU2 = os.environ.get('SU2_CFD', 'SU2_CFD')  # SU2 executable: $SU2_CFD, else found on PATH
TMP = os.path.join(here, 'runs', 'layers_v2')
BASHRC = '/usr/lib/openfoam/openfoam2512/etc/bashrc'
ORG, SIZE = -1.5, 3.2
EULER = """SOLVER= EULER
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
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
TIME_DISCRE_FLOW= EULER_IMPLICIT
ITER= 2
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, DRAG, LIFT )
HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )
OUTPUT_FILES= ( RESTART )
"""
RANS = """SOLVER= RANS
KIND_TURB_MODEL= SA
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
MACH_NUMBER= 0.2
AOA= 0.0
FREESTREAM_TEMPERATURE= 288.15
REYNOLDS_NUMBER= 1.0E5
REYNOLDS_LENGTH= 1.2
REF_LENGTH= 1.2
REF_AREA= 1.1309733552923256
REF_DIMENSIONALIZATION= FREESTREAM_PRESS_EQ_ONE
MARKER_HEATFLUX= ( wall, 0.0 )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( wall )
MARKER_MONITORING= ( wall )
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 5.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.5, 1.0, 100.0 )
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
TIME_DISCRE_FLOW= EULER_IMPLICIT
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO
TIME_DISCRE_TURB= EULER_IMPLICIT
ITER= 300
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, RMS_NU_TILDE, DRAG )
HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )
OUTPUT_FILES= ( RESTART, SURFACE_CSV )
"""

def cube_soup(lo=-0.5, hi=0.5): return cube_soup_box([lo] * 3, [hi] * 3)

def cube_soup_box(lo, hi):
    """closed box, 12 triangles, outward (counter-clockwise seen from outside) orientation"""
    P = lambda i, j, k: [hi[0] if i else lo[0], hi[1] if j else lo[1], hi[2] if k else lo[2]]
    quads = [
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
    elif name == 'cross':
        T = np.concatenate([cube_soup_box([-.6, -.2, -.2], [.6, .2, .2]), cube_soup_box([-.2, -.6, -.2], [.2, .6, .2])])
    else:
        T = D.soup(0)
    p = f'{TMP}/{name}.bin'
    with open(p, 'wb') as f:
        f.write(np.uint64(len(T)).tobytes()); f.write(np.ascontiguousarray(T, dtype='<f8').tobytes())
    return p

def foam_list(path):
    t = open(path).read(); t = t[t.index('}') + 1:]; t = t[t.index('(') + 1:t.rindex(')')]
    return np.array(t.split(), dtype=np.int64)

def run_checkmesh_default(case):
    out = subprocess.run(['bash', '-c', f'source {BASHRC} >/dev/null 2>&1; checkMesh -case {case}'], capture_output=True, text=True).stdout
    open(f'{case}/checkMesh_default.log', 'w').write(out)
    failed = [re.sub(r'\s+', ' ', l.strip(' *')) for l in out.split('\n') if l.startswith(' ***')]
    return dict(mesh_ok='Mesh OK.' in out, failed_checks=failed)

def foam_class(path):
    m = re.search(r'class\s+(\w+);', open(path).read(2000))
    return m.group(1) if m else None

def run_checkmesh(case, n_core, n_pillow):
    # v3: set class (cellSet / faceSet / pointSet) is read from each set's header instead of guessed from the name; cells are attributed to
    # core / pillow / layer by id (layer cells are appended after the core and pillow cells, n_core + n_pillow = first layer cell id)
    sets = f'{case}/constant/polyMesh/sets'
    subprocess.run(['rm', '-rf', sets])
    out = subprocess.run(['bash', '-c', f'source {BASHRC} >/dev/null 2>&1; checkMesh -allGeometry -allTopology -writeSets raw -case {case}'],
                         capture_output=True, text=True).stdout
    open(f'{case}/checkMesh.log', 'w').write(out)
    g = lambda pat: (re.search(pat, out) or [None, None])[1]
    failed = [re.sub(r'\s+', ' ', l.strip(' *')) for l in out.split('\n') if l.startswith(' ***')]
    pm = f'{case}/constant/polyMesh'
    own = foam_list(f'{pm}/owner'); nei = foam_list(f'{pm}/neighbour'); ncell = int(g(r'\n\s+cells:\s+(\d+)'))
    cls = np.where(np.arange(ncell) < n_core, 0, np.where(np.arange(ncell) < n_core + n_pillow, 1, 2))
    names = ['core', 'pillow', 'layer']
    res = {}
    if os.path.isdir(sets):
        for nm in sorted(os.listdir(sets)):
            if nm == 'upperTriangularFace': continue
            kind = foam_class(f'{sets}/{nm}')
            ids = foam_list(f'{sets}/{nm}')
            if kind == 'cellSet':
                c = cls[ids]
            elif kind == 'faceSet':
                inn = ids < len(nei); c = cls[own[ids]].copy(); c[inn] = np.maximum(c[inn], cls[nei[ids[inn]]])  # worst class touching the face
            else:
                res[nm] = {'total': int(len(ids)), 'class': kind}; continue
            res[nm] = {'total': int(len(ids)), 'class': kind, **{names[k]: int((c == k).sum()) for k in range(3)}}
    mx = lambda pat: (lambda v: float(v.rstrip('.,')) if v else None)(g(pat))
    return dict(cells=ncell, failed_checks=failed, mesh_ok='Mesh OK.' in out, sets=res,
                max_nonorth=mx(r'Mesh non-orthogonality Max: ([0-9.eE+-]+)'), max_skew=mx(r'Max skewness = ([0-9.eE+-]+)'),
                types={k: int(v) for k, v in re.findall(r'\n\s+(\w+):\s+(\d+)\s*\n', out) if k in ('hexahedra', 'prisms', 'pyramids', 'polyhedra', 'tetrahedra', 'tet wedges')})

def su2_run(case, cfgtext, name, timeout=3000):
    cfg = f'{case}/{name}.cfg'
    open(cfg, 'w').write(cfgtext)
    t = time.time()
    r = subprocess.run([SU2, '-t', '8', cfg], cwd=case, capture_output=True, text=True, timeout=timeout)
    log = r.stdout + r.stderr
    open(f'{case}/{name}.log', 'w').write(log)
    lines = [re.sub(r'\s+', ' ', l.strip('> ')) for l in log.split('\n') if re.search(r're-orientation|distorted|[Ww]arning|WARNING|negative|Negative', l)]
    return lines, time.time() - t, r.returncode, log

def rans_report(case):
    out = {}
    h = f'{case}/history.csv'
    if os.path.exists(h):
        rows = [l.strip().replace('"', '').split(',') for l in open(h) if l.strip()]
        hd = [c.strip() for c in rows[0]]
        ir = next((i for i, c in enumerate(hd) if c.lower().startswith('rms[rho]')), None)
        it = next((i for i, c in enumerate(hd) if 'nu' in c.lower() and c.lower().startswith('rms')), None)
        if ir is not None and len(rows) > 1:
            v = [float(r[ir]) for r in rows[1:]]
            out.update(n_iter=len(v), rms_rho_first=v[0], rms_rho_min=min(v), rms_rho_last=v[-1], rms_rho_at_50=v[min(49, len(v) - 1)], rms_rho_at_150=v[min(149, len(v) - 1)],
                       finite=bool(np.all(np.isfinite(v))))
        if it is not None and len(rows) > 1:
            out['rms_nu_last'] = float(rows[-1][it])
    # y+: SURFACE_CSV carries conservative variables only, so continue 1 iteration from the restart and write the surface .vtk (has Y_Plus)
    try:
        import shutil
        shutil.copy(f'{case}/restart.dat', f'{case}/solution.dat')
        cfg = open(f'{case}/rans.cfg').read().replace('RESTART_SOL= NO', 'RESTART_SOL= YES')
        cfg = re.sub(r'^ITER= .*$', 'ITER= 1', cfg, flags=re.M); cfg = re.sub(r'^OUTPUT_FILES.*$', 'OUTPUT_FILES= ( SURFACE_PARAVIEW_ASCII )', cfg, flags=re.M)
        open(f'{case}/yplus.cfg', 'w').write(cfg + 'VOLUME_OUTPUT= ( COORDINATES, SOLUTION, PRIMITIVE )\n')
        subprocess.run([SU2, '-t', '8', 'yplus.cfg'], cwd=case, capture_output=True, text=True, timeout=1800)
        t = open(f'{case}/surface.vtk', errors='ignore').read().split('\n')
        n = int(next(x for x in t if x.startswith('POINT_DATA')).split()[1]); i = next(k for k, l in enumerate(t) if l.startswith('SCALARS Y_Plus'))
        vals = []; j = i + 2
        while len(vals) < n and j < len(t): vals += [float(x) for x in t[j].split()]; j += 1
        y = np.array(vals[:n]); out.update(yplus_min=float(y.min()), yplus_mean=float(y.mean()), yplus_median=float(np.median(y)), yplus_max=float(y.max()), yplus_n=int(n))
    except Exception as e:
        out['yplus'] = f'unavailable: {e}'
    return out

ap = argparse.ArgumentParser(); ap.add_argument('--cases', nargs='+', default=['sphere:7', 'soup:7', 'cube:7', 'cross:7'])
ap.add_argument('--columns', nargs='+', default=['nearest', 'harmonic']); ap.add_argument('--hg-k', type=float, default=8.0)
ap.add_argument('--grade', type=float, default=0.25); ap.add_argument('--delta-h', type=float, default=4.0)
ap.add_argument('--layers', type=int, default=12); ap.add_argument('--growth', type=float, default=1.25)
ap.add_argument('--snap', type=float, default=0.4)
ap.add_argument('--tag', default=''); ap.add_argument('--extra', nargs='*', default=[]); ap.add_argument('--no-su2', action='store_true'); ap.add_argument('--no-cm', action='store_true')
a = ap.parse_args()
os.makedirs(TMP, exist_ok=True)

for cs in a.cases:
    shape, L = cs.split(':'); L = int(L)
    soup = write_soup(shape)
    hs = SIZE / 2 ** L; delta = a.delta_h * hs
    dy0 = delta * (a.growth - 1) / (a.growth ** a.layers - 1)
    for col in a.columns:
        case = f'{TMP}/{shape}_L{L}_{col}{a.tag}'
        t = time.time()
        r = subprocess.run([BIN, soup, f'origin={ORG},{ORG},{ORG}', f'size={SIZE}', f'L={L}', f'h_surf={hs}', f'grade={a.grade}', f'delta={delta}',
                            f'layers={a.layers}', f'dy0={dy0}', f'growth={a.growth}', f'snap={a.snap}', f'columns={col}', f'hg_k={a.hg_k}'] + a.extra + [case], capture_output=True, text=True)
        if r.returncode: print(json.dumps(dict(shape=shape, L=L, columns=col, error=r.stderr[-500:])), flush=True); continue
        bench = json.loads(r.stdout.strip().split('\n')[-1]); t_bench = time.time() - t
        open(f'{case}/bench.json', 'w').write(json.dumps(bench))
        ly = bench['layers']
        cmd = cma = None; tot = None; t_cm = 0.0
        if not a.no_cm:
            t = time.time(); cmd = run_checkmesh_default(case); cma = run_checkmesh(case, bench['core_cells'], bench['cells'] - bench['core_cells'] - ly['shell_quads'] * ly['layers']); t_cm = time.time() - t
            tot = {k: sum(v.get(k, 0) for v in cma['sets'].values() if v.get('class') != 'pointSet') for k in ('core', 'pillow', 'layer')}
        if a.no_su2: su_lines, t_su, rc = [], 0.0, None
        else: su_lines, t_su, rc, _ = su2_run(case, EULER, 'it')
        keys = ['leaves', 'core_cells', 'pillow_hexes', 'sj_min_after', 'n_sj_below_0p2', 'cells', 'hex_cells', 'poly_cells', 'orphan_faces', 'nonpositive_cells', 'volume', 'su2',
                'patches', 't_octree', 't_hexcore', 't_layers_total', 't_total']
        print(json.dumps(dict(shape=shape, L=L, columns=col, delta=delta, dy0=dy0, layers=a.layers, growth=a.growth, snap=a.snap, hg_k=a.hg_k, bench={k: bench[k] for k in keys if k in bench}, layer=ly,
                              checkMesh_default=cmd, checkMesh_all=cma, checkMesh_defects_by_class=tot, su2_warning_lines=su_lines[:8], su2_n_warnings=len(su_lines), su2_rc=rc,
                              t_bench=round(t_bench, 2), t_checkmesh=round(t_cm, 2), t_su2_2iter=round(t_su, 2))), flush=True)
