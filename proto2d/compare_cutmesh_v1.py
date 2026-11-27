#!/usr/bin/env python3
"""Run rcore cutmesh_bench on the CLOSED test soup (dexel3d_v1.soup(0): sphere + overlapping torus) and verify the
fluid cut-cell mesh independently:
  * OpenFOAM `checkMesh -allGeometry -allTopology` (cells, faces, failed checks, total volume),
  * the SU2 file parsed in numpy: every element volume > 0, every internal tri/quad face shared by exactly 2 elements,
    boundary faces by exactly 1 (and the boundary elements are exactly the once-used faces), total volume vs the
    polyhedral volume reported by the mesher (same fan definition) and vs the OpenFOAM total volume.
The OpenFOAM total volume uses a different definition for non-planar (cut) faces, so it differs from the closed-surface
volume at ~1e-7; both relative differences are printed. One JSON line per L.

usage: compare_cutmesh_v1.py [--L 6 7] [--grade 0.25] [--eps 0.05] [--keep]
Domain [-1.5,1.7]^3, h_surf = size / 2^L. Cases are written to a temporary run directory.
"""
import argparse, json, os, re, subprocess, sys, time, numpy as np, importlib.util
here = os.path.dirname(os.path.abspath(__file__))
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, file)); m = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [file]; spec.loader.exec_module(m); sys.argv = argv; return m
D = load('d', 'dexel3d_v1.py')
BIN = os.path.join(here, '..', 'rcore', 'target', 'release', 'cutmesh_bench')
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'cutmesh')
BASHRC = '/usr/lib/openfoam/openfoam2512/etc/bashrc'
ORG, SIZE = -1.5, 3.2

ap = argparse.ArgumentParser(); ap.add_argument('--L', nargs='+', type=int, default=[6, 7])
ap.add_argument('--grade', type=float, default=0.25); ap.add_argument('--eps', type=float, default=0.05)
a = ap.parse_args()
os.makedirs(TMP, exist_ok=True)
T = D.soup(0)
with open(f'{TMP}/SOUP.bin', 'wb') as f:
    f.write(np.uint64(len(T)).tobytes()); f.write(T.astype('<f8').tobytes())

def tet_vol(A, B, C, Dd): return np.einsum('ij,ij->i', np.cross(B - A, C - A), Dd - A) / 6.0

def parse_su2(path):
    txt = open(path).read().split('\n')
    i = 0
    assert txt[0].startswith('NDIME= 3'); ne = int(txt[1].split('=')[1]); lines = txt[2:2 + ne]
    assert txt[2 + ne].startswith('NPOIN='); npn = int(txt[2 + ne].split('=')[1])
    P = np.fromstring(' '.join(txt[3 + ne:3 + ne + npn]), sep=' ').reshape(-1, 4)[:, :3]
    j = 3 + ne + npn; assert txt[j].startswith('NMARK='); nm = int(txt[j].split('=')[1]); j += 1
    marks = {}
    for _ in range(nm):
        tag = txt[j].split('=')[1].strip(); n = int(txt[j + 1].split('=')[1]); j += 2
        marks[tag] = txt[j:j + n]; j += n
    el = {}
    for ty, nn in ((10, 4), (12, 8), (14, 5)):
        g = [l for l in lines if l.startswith(f'{ty} ')]
        el[ty] = np.fromstring(' '.join(g), sep=' ', dtype=np.int64).reshape(-1, nn + 2)[:, 1:1 + nn] if g else np.zeros((0, nn), np.int64)
    assert sum(len(v) for v in el.values()) == ne, 'unexpected element type in file'
    return P, el, marks

def fan_vol(P, polys_idx, apex):
    """sum over outward polygons (as arrays of vertex ids, all same length n) of fan tets about the face mean point"""
    tot = np.zeros(len(apex))
    for p in polys_idx:
        pts = P[p]; fc = pts.mean(1); n = p.shape[1]
        for t in range(n):
            tot += tet_vol(pts[:, t], fc, pts[:, (t + 1) % n], apex)
    return tot

def check_su2(path):
    P, el, marks = parse_su2(path)
    res = {'n_points': len(P)}
    vols = []; tri_f = []; quad_f = []; nonpos = 0
    t = el[10]
    if len(t):
        v = tet_vol(P[t[:, 0]], P[t[:, 1]], P[t[:, 2]], P[t[:, 3]]); vols.append(v); nonpos += int((v <= 0).sum())
        for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)): tri_f.append(t[:, f])
    q = el[14]
    if len(q):
        c = P[q[:, :4]].mean(1); ap = P[q[:, 4]]
        parts = np.stack([tet_vol(P[q[:, i]], P[q[:, (i + 1) % 4]], c, ap) for i in range(4)])
        vols.append(parts.sum(0)); nonpos += int((parts.min(0) <= 0).sum())
        quad_f.append(q[:, :4])
        for f in ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)): tri_f.append(q[:, f])
    h = el[12]
    if len(h):
        ctr = P[h].mean(1)
        faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        v = fan_vol(P, [h[:, f] for f in faces], ctr); vols.append(v); nonpos += int((v <= 0).sum())
        nb = [[1, 3, 4], [2, 0, 5], [3, 1, 6], [0, 2, 7], [7, 5, 0], [4, 6, 1], [5, 7, 2], [6, 4, 3]]
        for i in range(8):
            e = [P[h[:, k]] - P[h[:, i]] for k in nb[i]]
            nonpos += int((np.einsum('ij,ij->i', np.cross(e[0], e[1]), e[2]) <= 0).sum())
        for f in faces: quad_f.append(h[:, f])
    allv = np.concatenate(vols)
    res.update(n_elems=int(len(allv)), min_elem_vol=float(allv.min()), nonpositive=nonpos, total_volume=float(allv.sum()))
    # face sharing
    def counts(rows):
        if not len(rows): return np.zeros((0, rows.shape[1] if rows.ndim == 2 else 3), np.int64), np.zeros(0, np.int64)
        r = np.sort(np.concatenate(rows) if isinstance(rows, list) else rows, axis=1)
        o = np.lexsort(r.T[::-1]); r = r[o]
        new = np.r_[True, (r[1:] != r[:-1]).any(1)]
        idx = np.flatnonzero(new); cnt = np.diff(np.r_[idx, len(r)])
        return r[idx], cnt
    def bcast(lines):
        tri = [l.split()[1:4] for l in lines if l.startswith('5 ')]; qd = [l.split()[1:5] for l in lines if l.startswith('9 ')]
        return np.array(tri, np.int64).reshape(-1, 3), np.array(qd, np.int64).reshape(-1, 4)
    bt = np.concatenate([bcast(l)[0] for l in marks.values()]); bq = np.concatenate([bcast(l)[1] for l in marks.values()])
    bad = 0; internal = 0; boundary = 0
    for rows, bnd in ((tri_f, bt), (quad_f, bq)):
        u, c = counts(rows) if len(rows) else (np.zeros((0, bnd.shape[1]), np.int64), np.zeros(0, np.int64))
        bad += int((c > 2).sum()); internal += int((c == 2).sum())
        single = u[c == 1]; boundary += len(single)
        # boundary elements must be exactly the once-used faces, each used once
        ub, cb = counts(bnd) if len(bnd) else (np.zeros((0, bnd.shape[1]), np.int64), np.zeros(0, np.int64))
        bad += int((cb != 1).sum())
        both, cc = counts(np.concatenate([single, ub])) if len(single) + len(ub) else (None, np.zeros(0))
        bad += int((cc != 2).sum())
    res.update(internal_faces=internal, boundary_faces=boundary, face_sharing_violations=bad,
               wall_elems=len(marks.get('wall', [])), farfield_elems=len(marks.get('farfield', [])))
    return res

def run_checkmesh(case):
    out = subprocess.run(['bash', '-c', f'source {BASHRC} >/dev/null 2>&1; checkMesh -allGeometry -allTopology -case {case}'],
                         capture_output=True, text=True).stdout
    g = lambda pat: (re.search(pat, out) or [None, None])[1]
    failed = [re.sub(r'\s+', ' ', l.strip(' *')) for l in out.split('\n') if l.startswith(' ***')]
    return dict(cells=int(g(r'\n\s+cells:\s+(\d+)')), faces=int(g(r'\n\s+faces:\s+(\d+)')), internal_faces=int(g(r'internal faces:\s+(\d+)')),
                n_failed=int((g(r'Failed (\d+) mesh checks') or 0)), failed_checks=failed,
                total_volume=float(g(r'Total volume = ([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)')), max_nonorth=float(g(r'Max: ([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)')),
                mesh_ok='Mesh OK.' in out)

for L in a.L:
    case = f'{TMP}/L{L}'
    t = time.time()
    r = subprocess.run([BIN, f'{TMP}/SOUP.bin', f'origin={ORG},{ORG},{ORG}', f'size={SIZE}', f'L={L}', f'h_surf={SIZE / 2 ** L}',
                        f'grade={a.grade}', f'eps={a.eps}', case], capture_output=True, text=True)
    if r.returncode: print(json.dumps(dict(L=L, error=r.stderr[-500:]))); continue
    bench = json.loads(r.stdout.strip().split('\n')[-1]); t_bench = time.time() - t
    t = time.time(); cm = run_checkmesh(case); t_cm = time.time() - t
    t = time.time(); su = check_su2(f'{case}/mesh.su2'); t_su = time.time() - t
    rel = lambda x, y: abs(x - y) / abs(y)
    keys = ['leaves', 'fluid_cells', 'hex_cells', 'poly_cells', 'cut_cells', 'multi_component_cells', 'unresolved_edges', 'snapped_points',
            'min_vol_ratio', 'orphan_faces', 'n_points', 'n_faces', 'su2', 'nonpositive_su2_elements', 'volume', 'volume_foam_def', 't_total']
    su.update(rel_vol_vs_mesher=rel(su['total_volume'], bench['volume']), rel_vol_vs_checkmesh=rel(su['total_volume'], cm['total_volume']),
              ok_vs_mesher=bool(rel(su['total_volume'], bench['volume']) < 1e-9), ok_vs_checkmesh=bool(rel(su['total_volume'], cm['total_volume']) < 1e-9))
    print(json.dumps(dict(L=L, bench={k: bench[k] for k in keys}, checkMesh=cm, su2_check=su,
                          rel_foam_vs_foamdef=rel(cm['total_volume'], bench['volume_foam_def']),
                          t_bench=round(t_bench, 2), t_checkmesh=round(t_cm, 2), t_su2check=round(t_su, 2))), flush=True)
