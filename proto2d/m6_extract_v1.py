#!/usr/bin/env python3
"""Extract the ONERA M6 wing surface from the SU2 test-case mesh and write a closed full-span STL.

The SU2 test case is a half wing on the symmetry plane (span along +y). The WING marker is mirrored about y = 0, so the
full wing is a closed surface (mantle has no symmetry-plane boundary yet). Quads are split into two triangles.
usage: m6_extract_v1.py [SU2_MESH] [OUT.stl]   (defaults: $SU2_TESTCASE_MESHES/euler/oneram6/mesh_ONERAM6_inv_ffd.su2 [~/TestCases, the su2code/TestCases repo], examples/oneram6_full.stl)
Prints triangle count, bounding box and the open-edge count of the result (0 = watertight).
"""
import os, sys, struct, numpy as np
from collections import Counter

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser(os.environ.get('SU2_TESTCASE_MESHES', '~/TestCases')), 'euler/oneram6/mesh_ONERAM6_inv_ffd.su2')
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples', 'oneram6_full.stl')

pts, tris, marker, mode, left = [], [], None, None, 0
with open(src) as f:
    for line in f:
        s = line.split('%')[0].strip()
        if not s:
            continue
        if mode == 'poin' and left > 0:
            v = s.split(); pts.append([float(v[0]), float(v[1]), float(v[2])]); left -= 1; continue
        if mode == 'melem' and left > 0:
            v = list(map(int, s.split()))
            if marker == 'WING':
                if v[0] == 5:
                    tris.append(v[1:4])
                elif v[0] == 9:
                    tris += [[v[1], v[2], v[3]], [v[1], v[3], v[4]]]
            left -= 1; continue
        if s.startswith('NPOIN='):
            mode, left = 'poin', int(s.split('=')[1].split()[0])
        elif s.startswith('NELEM='):
            mode, left = 'skip', 0
        elif s.startswith('MARKER_TAG='):
            marker = s.split('=')[1].strip()
        elif s.startswith('MARKER_ELEMS='):
            mode, left = 'melem', int(s.split('=')[1])
P = np.array(pts); T = np.array(tris)
half = P[T]                                   # (n, 3, 3)
mir = half.copy(); mir[:, :, 1] *= -1; mir = mir[:, ::-1]   # mirror and flip winding
full = np.concatenate([half, mir])
# orient outward: signed volume must be positive
vol = np.einsum('ij,ij->i', full[:, 0], np.cross(full[:, 1], full[:, 2])).sum() / 6
if vol < 0:
    full = full[:, ::-1]
# watertightness: count undirected edges after welding
key = np.round(full / 1e-9).astype(np.int64)
ids = {}
vid = np.array([[ids.setdefault(tuple(p), len(ids)) for p in t] for t in key])
ec = Counter(tuple(sorted((a, b))) for t in vid for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
open_edges = sum(1 for c in ec.values() if c == 1)
with open(out, 'wb') as f:
    f.write(b'mantle: ONERA M6 full span (mirrored from SU2 test case)'.ljust(80, b' '))
    f.write(struct.pack('<I', len(full)))
    for t in full:
        n = np.cross(t[1] - t[0], t[2] - t[0]); n = n / (np.linalg.norm(n) + 1e-300)
        f.write(struct.pack('<12fH', *n, *t[0], *t[1], *t[2], 0))
print(f'triangles {len(full)}  bbox {full.reshape(-1, 3).min(0).round(4)} .. {full.reshape(-1, 3).max(0).round(4)}  '
      f'volume {abs(vol):.5f}  open_edges {open_edges}  -> {out}')
