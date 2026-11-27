#!/usr/bin/env python3
"""Mesh metrics for a 2D native SU2 mesh (any mesher), so SFHM and Gmsh are measured identically.
usage: su2_metrics_v1.py MESH.su2 DY0 [--wall wall]
prints JSON: element counts, first-cell wall-normal height stats vs DY0 (fraction within +-20 %),
min interior angle, max aspect ratio, non-positive-area elements."""
import sys, json, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('mesh'); ap.add_argument('dy0', type=float); ap.add_argument('--wall', default='wall')
a = ap.parse_args()
L = open(a.mesh).read().split('\n'); i = 0; elems = []; P = None; markers = {}
while i < len(L):
    t = L[i].strip()
    if t.startswith('NELEM='):
        n = int(t.split('=')[1]); elems = [list(map(int, L[i + 1 + k].split()))[1:-1] if len(L[i + 1 + k].split()) > 1 else [] for k in range(n)]
        # SU2 lines: type v0..vk [index]; index optional
        raw = [list(map(int, L[i + 1 + k].split())) for k in range(n)]
        elems = [r[1:1 + (3 if r[0] == 5 else 4)] for r in raw]; i += n + 1; continue
    if t.startswith('NPOIN='):
        n = int(t.split('=')[1].split()[0]); P = np.array([list(map(float, L[i + 1 + k].split()[:2])) for k in range(n)]); i += n + 1; continue
    if t.startswith('MARKER_TAG='):
        name = t.split('=')[1].strip(); m = int(L[i + 1].split('=')[1])
        markers[name] = [tuple(map(int, L[i + 2 + k].split()[1:3])) for k in range(m)]; i += m + 2; continue
    i += 1
area = []; minang = []; aspect = []
edge_elem = {}
for ei, e in enumerate(elems):
    Q = P[e]; ar = 0.5 * np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]); area.append(ar)
    E = np.roll(Q, -1, 0) - Q; Ln = np.linalg.norm(E, axis=1)
    cosv = [-(E[k - 1] @ E[k]) / (Ln[k - 1] * Ln[k] + 1e-300) for k in range(len(e))]
    minang.append(np.degrees(np.arccos(np.clip(cosv, -1, 1))).min()); aspect.append(Ln.max() / max(Ln.min(), 1e-300))
    for k in range(len(e)):
        u, v = e[k], e[(k + 1) % len(e)]; edge_elem[(min(u, v), max(u, v))] = ei
area = np.array(area)
dy = []
for u, v in markers.get(a.wall, []):
    ei = edge_elem[(min(u, v), max(u, v))]; Ledge = np.linalg.norm(P[u] - P[v])
    dy.append((2 if len(elems[ei]) == 3 else 1) * abs(area[ei]) / Ledge)
dy = np.array(dy)
out = dict(mesh=a.mesh, n_elem=len(elems), n_tri=sum(len(e) == 3 for e in elems), n_quad=sum(len(e) == 4 for e in elems),
           n_nonpositive=int((area <= 0).sum()), min_angle_deg=float(np.min(minang)), max_aspect=float(np.max(aspect)),
           wall_edges=len(dy), dy_min=float(dy.min()), dy_median=float(np.median(dy)), dy_max=float(dy.max()),
           frac_within_20pct=float(np.mean(np.abs(dy / a.dy0 - 1) < 0.2)))
print(json.dumps(out))
