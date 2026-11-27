"""Minimal ascii polyMesh reader for 1-cell-thick 2D meshes + metrics + window plot."""
import numpy as np, re, os, sys, json
def _body(p):
    s = open(p).read(); i = s.index('(', s.index('}')); return s[i + 1:s.rindex(')')]
def read(case):
    d = os.path.join(case, 'constant', 'polyMesh')
    pts = np.array([list(map(float, m)) for m in re.findall(r'\(([-\d.eE+]+) ([-\d.eE+]+) ([-\d.eE+]+)\)', _body(d + '/points'))])
    faces = [np.array(list(map(int, m.split()))) for m in re.findall(r'\d+\(([^)]*)\)', _body(d + '/faces'))]
    owner = np.array(list(map(int, _body(d + '/owner').split())))
    neigh = np.array(list(map(int, _body(d + '/neighbour').split())))
    b = open(d + '/boundary').read()
    patches = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in re.finditer(r'(\w+)\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);', b)}
    return pts, faces, owner, neigh, patches
def cell_centres(pts, faces, owner, neigh, ncell):
    # area-weighted face-centre average (good enough for 2D prisms)
    C = np.zeros((ncell, 3)); W = np.zeros(ncell)
    for fi, f in enumerate(faces):
        P = pts[f]; c = P.mean(0)
        a = 0.5 * np.linalg.norm(sum(np.cross(P[i] - c, P[(i + 1) % len(P)] - c) for i in range(len(P))))
        C[owner[fi]] += a * c; W[owner[fi]] += a
        if fi < len(neigh): C[neigh[fi]] += a * c; W[neigh[fi]] += a
    return C / W[:, None]
def metrics(case, wall='wall', dy_target=None):
    pts, faces, owner, neigh, patches = read(case)
    ncell = owner.max() + 1
    C = cell_centres(pts, faces, owner, neigh, ncell)
    st, n = patches[wall][1], patches[wall][0]
    dy = []
    for fi in range(st, st + n):
        f = faces[fi]; P = pts[f]; c = P.mean(0)
        # face normal (2D side face): use polygon normal
        nrm = sum(np.cross(P[i] - c, P[(i + 1) % len(P)] - c) for i in range(len(P))); nrm /= np.linalg.norm(nrm)
        dy.append(2 * abs(np.dot(C[owner[fi]] - c, nrm)))       # distance centre->face, x2 = cell height
    dy = np.array(dy)
    out = dict(case=case, n_cells=int(ncell), n_wall_faces=int(n), dy_min=float(dy.min()), dy_median=float(np.median(dy)), dy_max=float(dy.max()))
    if dy_target: out['frac_within_20pct'] = float(np.mean(np.abs(dy / dy_target - 1) < 0.2))
    return out, dy
def plot_window(case, cx, cy, r, out, title=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    pts, faces, owner, neigh, patches = read(case)
    key = 'frontAndBack' if 'frontAndBack' in patches else ('front' if 'front' in patches else 'originalPatch')
    st, n = patches[key][1], patches[key][0]
    if key == 'frontAndBack': n = n // 2
    segs = []
    for fi in range(st, st + n):
        f = faces[fi]; P = pts[f][:, :2]
        if np.any((abs(P[:, 0] - cx) < r) & (abs(P[:, 1] - cy) < r)):
            segs += [[P[i], P[(i + 1) % len(P)]] for i in range(len(P))]
    fig, ax = plt.subplots(figsize=(8, 8)); ax.add_collection(LineCollection(segs, colors='0.2', linewidths=0.4))
    if 'wall' in patches:
        ws, wn = patches['wall'][1], patches['wall'][0]
        wsegs = [[pts[faces[fi]][0, :2], pts[faces[fi]][1, :2]] for fi in range(ws, ws + wn)]
        ax.add_collection(LineCollection(wsegs, colors='tab:red', linewidths=1.0))
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r); ax.set_aspect('equal'); ax.set_title(title or case)
    fig.savefig(out, dpi=150); plt.close(fig)
if __name__ == '__main__':
    if sys.argv[1] == 'metrics':
        m, _ = metrics(sys.argv[2], dy_target=float(sys.argv[3]) if len(sys.argv) > 3 else None); print(json.dumps(m))
    elif sys.argv[1] == 'plot':
        plot_window(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6])
