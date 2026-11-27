"""Write small test geometries (binary STL, numpy only): sphere.stl (R 0.5, UV sphere) and box.stl (1 x 0.6 x 0.4)."""
import os
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))


def write_stl(path, tris):
    tris = np.asarray(tris, dtype="<f4")  # (n, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tris
    with open(path, "wb") as f:
        f.write(b"mantle example".ljust(80, b"\0"))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(rec.tobytes())


def sphere(r=0.5, nu=36, nv=28):
    """Outward-oriented UV sphere: nu*(2*nv-2) = 2016 triangles."""
    th, ph = np.linspace(0, np.pi, nv + 1), np.linspace(0, 2 * np.pi, nu, endpoint=False)
    P = lambda i, j: r * np.array([np.sin(th[i]) * np.cos(ph[j % nu]), np.sin(th[i]) * np.sin(ph[j % nu]), np.cos(th[i])])
    t = []
    for i in range(nv):
        for j in range(nu):
            if i > 0:
                t.append([P(i, j), P(i + 1, j), P(i, j + 1)])
            if i < nv - 1:
                t.append([P(i, j + 1), P(i + 1, j), P(i + 1, j + 1)])
    return t


def box(size=(1.0, 0.6, 0.4)):
    """Outward-oriented box, 12 triangles."""
    h = np.array(size) / 2
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * h
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]  # corner idx = 4*(x>0)+2*(y>0)+(z>0)
    t = []
    for a, b, d, e in q:
        t += [[c[a], c[b], c[d]], [c[a], c[d], c[e]]]
    return t


def fix_orientation(tris):
    """Flip triangles so the signed volume is positive (outward normals)."""
    t = np.asarray(tris, dtype=float)
    vol = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
    return t if vol > 0 else t[:, [0, 2, 1]]


if __name__ == "__main__":
    for name, t in (("sphere", sphere()), ("box", box())):
        t = fix_orientation(t)
        write_stl(os.path.join(here, name + ".stl"), t)
        print(name, len(t), "triangles")
