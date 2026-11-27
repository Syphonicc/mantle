//! Feature detection on a triangle soup: the feature graph of the WETTED wall (the boundary of the union of the solids; the surface the BVH
//! closest point returns for fluid points).
//!
//! * Sharp edges: mesh edges shared by two triangles whose normals deviate by more than `angle_deg`. They are classified convex / concave
//!   and clipped against the other parts of the soup (breakpoints at the exact edge / triangle intersections); a piece is kept when both of
//!   its triangles lie on the boundary of the union there (boundary test: generalized winding number about 0 just outside the surface and
//!   about 1 or more just inside).
//! * Intersection curves: triangle-triangle intersection segments of triangles of different parts (connected components of the soup, joined by
//!   shared vertices). They are the concave creases of the union and are kept where the part of each triangle that is not inside the other part
//!   is on the boundary of the union (this drops flat seams, e.g. coplanar faces of overlapping boxes).
//! * Corners: nodes of the welded graph where the incident pieces are not exactly two of one kind (3 or more curves meet, a curve ends, convex
//!   meets concave). They need not be soup vertices (mixed vertices of overlapping parts are intersection points). Curve ends lying inside other
//!   pieces split them first (T-junctions).
//!
//! Orientation of every part is fixed by its signed volume, so soups with inward-facing parts are handled. Degenerate triangles, open
//! boundaries and non-manifold edges are ignored (no feature there), nothing panics on dirty input.
//!
//! Queries: `nearest_curve` (point on any feature curve, grid accelerated) and `corners_near`.

use crate::bvh::Bvh;
use crate::cutmesh::Map;
use crate::dexel::Tri;
use crate::fwn::Fwn;
use std::time::Instant;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn mul(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: V3) -> V3 {
    mul(a, 1.0 / norm(a).max(1e-300))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Convex,
    Concave,
    Mixed,
}

#[derive(Clone, Debug)]
pub struct Curve {
    pub pts: Vec<V3>,
    pub kind: Kind,
    /// corner ids at the two ends (`None`: closed loop)
    pub ends: Option<[usize; 2]>,
    /// smallest deviation angle between the face normals (degrees) along the curve (180 for intersection curves)
    pub angle_deg: f64,
}

#[derive(Clone, Debug)]
pub struct Corner {
    pub p: V3,
    pub curves: Vec<usize>,
    pub kind: Kind,
}

#[derive(Clone, Debug)]
pub struct FeatureOptions {
    /// minimum deviation of the face normals (degrees) for a sharp edge
    pub angle_deg: f64,
}

impl Default for FeatureOptions {
    fn default() -> Self {
        FeatureOptions { angle_deg: 30.0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FeatureStats {
    pub sharp_pieces: usize,
    pub isect_pieces: usize,
    pub curves: usize,
    pub corners: usize,
    pub convex_corners: usize,
    pub concave_corners: usize,
    pub mixed_corners: usize,
    pub t_splits: usize,
    pub seconds: f64,
}

impl FeatureStats {
    pub fn json(&self) -> String {
        format!(
            "{{\"sharp_pieces\": {}, \"isect_pieces\": {}, \"curves\": {}, \"corners\": {}, \"convex_corners\": {}, \"concave_corners\": {}, \"mixed_corners\": {}, \"t_splits\": {}, \"seconds\": {:.3}}}",
            self.sharp_pieces, self.isect_pieces, self.curves, self.corners, self.convex_corners, self.concave_corners, self.mixed_corners, self.t_splits, self.seconds
        )
    }
}

pub struct CurveHit {
    pub p: V3,
    pub curve: usize,
    pub dist: f64,
    /// segment of the curve polyline and parameter (0..1) of the hit on it
    pub seg: usize,
    pub u: f64,
}

pub struct Features {
    pub curves: Vec<Curve>,
    pub corners: Vec<Corner>,
    pub stats: FeatureStats,
    cell: f64,
    org: V3,
    grid: Map<(i32, i32, i32), Vec<(u32, u32)>>,
}

#[derive(Clone, Copy)]
struct Piece {
    a: V3,
    b: V3,
    ang: f64,
    convex: bool,
}

fn closest_on_seg(p: V3, a: V3, b: V3) -> (V3, f64) {
    let ab = sub(b, a);
    let t = (dot(sub(p, a), ab) / dot(ab, ab).max(1e-300)).clamp(0.0, 1.0);
    (add(a, mul(ab, t)), t)
}

/// Segment of the intersection of two triangles (planes with unit normals `n1`, `n2`); None when coplanar, disjoint or touching in a point.
fn tri_tri_segment(t1: &Tri, n1: V3, t2: &Tri, n2: V3, eps: f64) -> Option<(V3, V3)> {
    let d = cross(n1, n2);
    let dl = norm(d);
    if dl < 1e-6 {
        return None;
    }
    let d = mul(d, 1.0 / dl);
    let seg = |t: &Tri, n_other: V3, o: V3| -> Option<(f64, f64, V3)> {
        let ds: [f64; 3] = [0, 1, 2].map(|i| dot(n_other, sub(t[i], o)));
        let s: [i32; 3] = ds.map(|x| if x > eps { 1 } else if x < -eps { -1 } else { 0 });
        if (s[0] > 0 && s[1] > 0 && s[2] > 0) || (s[0] < 0 && s[1] < 0 && s[2] < 0) || (s[0] == 0 && s[1] == 0 && s[2] == 0) {
            return None;
        }
        let mut pts: Vec<V3> = Vec::new();
        for i in 0..3 {
            if s[i] == 0 {
                pts.push(t[i]);
            }
            let j = (i + 1) % 3;
            if s[i] * s[j] == -1 {
                let f = ds[i] / (ds[i] - ds[j]);
                pts.push(add(t[i], mul(sub(t[j], t[i]), f)));
            }
        }
        if pts.len() < 2 {
            return None;
        }
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in &pts {
            let u = dot(*p, d);
            lo = lo.min(u);
            hi = hi.max(u);
        }
        Some((lo, hi, pts[0]))
    };
    let (l1, h1, p1) = seg(t1, n2, t2[0])?;
    let (l2, h2, _) = seg(t2, n1, t1[0])?;
    let (lo, hi) = (l1.max(l2), h1.min(h2));
    if hi - lo <= 4.0 * eps {
        return None;
    }
    // a point of the line: p1 projected onto the plane of t2 along the line direction is p1 itself (p1 lies on both planes)
    let base = sub(p1, mul(d, dot(p1, d)));
    Some((add(base, mul(d, lo)), add(base, mul(d, hi))))
}

impl Features {
    pub fn empty() -> Features {
        Features { curves: Vec::new(), corners: Vec::new(), stats: FeatureStats::default(), cell: 1.0, org: [0.0; 3], grid: Map::default() }
    }

    pub fn detect(tris: &[Tri], opt: &FeatureOptions) -> Features {
        let t0 = Instant::now();
        let mut f = Features::empty();
        // drop degenerate / non-finite triangles
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for t in tris {
            for v in t {
                for d in 0..3 {
                    lo[d] = lo[d].min(v[d]);
                    hi[d] = hi[d].max(v[d]);
                }
            }
        }
        if tris.is_empty() || !(0..3).all(|d| lo[d].is_finite() && hi[d].is_finite()) {
            return f;
        }
        let size = norm(sub(hi, lo)).max(1e-300);
        let q = 1e-9 * size;
        let key = |p: V3| -> (i64, i64, i64) { ((p[0] / q).round() as i64, (p[1] / q).round() as i64, (p[2] / q).round() as i64) };
        let mut vid: Map<(i64, i64, i64), u32> = Map::default();
        let mut tv: Vec<[u32; 3]> = Vec::new();
        let mut tr: Vec<Tri> = Vec::new();
        for t in tris {
            let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
            if !(norm(n) > 1e-14 * size * size) || !n.iter().all(|x| x.is_finite()) {
                continue;
            }
            let ids = t.map(|p| {
                let k = key(p);
                let nid = vid.len() as u32;
                *vid.entry(k).or_insert(nid)
            });
            if ids[0] == ids[1] || ids[1] == ids[2] || ids[0] == ids[2] {
                continue;
            }
            tv.push(ids);
            tr.push(*t);
        }
        if tr.is_empty() {
            return f;
        }
        // parts: connected components over shared vertices
        let nvert = vid.len();
        let mut par: Vec<u32> = (0..nvert as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        for ids in &tv {
            for k in 1..3 {
                let (a, b) = (find(&mut par, ids[0]), find(&mut par, ids[k]));
                if a != b {
                    par[a.max(b) as usize] = a.min(b);
                }
            }
        }
        let comp: Vec<u32> = tv.iter().map(|ids| find(&mut par, ids[0])).collect();
        // orientation by signed volume per part
        let mut vol: Map<u32, f64> = Map::default();
        for (i, t) in tr.iter().enumerate() {
            *vol.entry(comp[i]).or_insert(0.0) += dot(t[0], cross(t[1], t[2])) / 6.0;
        }
        for (i, t) in tr.iter_mut().enumerate() {
            if vol[&comp[i]] < 0.0 {
                t.swap(1, 2);
                tv[i].swap(1, 2);
            }
        }
        let nrm: Vec<V3> = tr.iter().map(|t| unit(cross(sub(t[1], t[0]), sub(t[2], t[0])))).collect();
        let bvh = Bvh::new(&tr);
        let fwn = Fwn::new(&tr);
        let eps = 1e-4 * size;
        let epn = 1e-7 * size;
        let on_boundary = |p: V3, n: V3| -> bool { fwn.winding(add(p, mul(n, epn))).abs() < 0.25 && fwn.winding(sub(p, mul(n, epn))).abs() > 0.75 };
        let mut pieces: Vec<Piece> = Vec::new();
        let mut st = FeatureStats::default();

        // ---- sharp edges
        let mut emap: Map<(u32, u32), Vec<u32>> = Map::default();
        for (i, ids) in tv.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (ids[k], ids[(k + 1) % 3]);
                emap.entry((a.min(b), a.max(b))).or_default().push(i as u32);
            }
        }
        let cos_thr = (opt.angle_deg.to_radians()).cos();
        let mut edges: Vec<((u32, u32), Vec<u32>)> = emap.into_iter().filter(|(_, v)| v.len() == 2).collect();
        edges.sort_unstable_by_key(|e| e.0);
        let mut vpos: Vec<V3> = vec![[0.0; 3]; nvert];
        for (i, ids) in tv.iter().enumerate() {
            for k in 0..3 {
                vpos[ids[k] as usize] = tr[i][k];
            }
        }
        for ((a, b), ts) in &edges {
            let (t1, t2) = (ts[0] as usize, ts[1] as usize);
            if dot(nrm[t1], nrm[t2]) > cos_thr {
                continue;
            }
            let (pa, pb) = (vpos[*a as usize], vpos[*b as usize]);
            let third = |t: usize| -> V3 {
                let k = (0..3).find(|&k| tv[t][k] != *a && tv[t][k] != *b).unwrap();
                tr[t][k]
            };
            let (c1, c2) = (third(t1), third(t2));
            let convex = dot(nrm[t1], sub(c2, pa)) < 0.0;
            // breakpoints from the other parts
            let mut ts_br: Vec<f64> = vec![0.0, 1.0];
            for (tt, id) in bvh.segment_hits(pa, pb) {
                if comp[id as usize] != comp[t1] && tt > 1e-9 && tt < 1.0 - 1e-9 {
                    ts_br.push(tt);
                }
            }
            ts_br.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let e = sub(pb, pa);
            let el = norm(e);
            for w in ts_br.windows(2) {
                if (w[1] - w[0]) * el < 1e-9 * size {
                    continue;
                }
                let m = add(pa, mul(e, 0.5 * (w[0] + w[1])));
                let eh = mul(e, 1.0 / el);
                let inward = |c: V3| unit(sub(sub(c, m), mul(eh, dot(sub(c, m), eh))));
                let ok1 = on_boundary(add(m, mul(inward(c1), eps)), nrm[t1]);
                let ok2 = ok1 && on_boundary(add(m, mul(inward(c2), eps)), nrm[t2]);
                if ok1 && ok2 {
                    pieces.push(Piece { a: add(pa, mul(e, w[0])), b: add(pa, mul(e, w[1])), ang: dot(nrm[t1], nrm[t2]).clamp(-1.0, 1.0).acos().to_degrees(), convex });
                    st.sharp_pieces += 1;
                }
            }
        }

        // ---- intersection curves of different parts (sweep along the longest axis)
        let ax = (0..3).max_by(|&i, &j| (hi[i] - lo[i]).partial_cmp(&(hi[j] - lo[j])).unwrap()).unwrap();
        let bb: Vec<(V3, V3)> = tr.iter().map(|t| ([0, 1, 2].map(|d| t[0][d].min(t[1][d]).min(t[2][d])), [0, 1, 2].map(|d| t[0][d].max(t[1][d]).max(t[2][d])))).collect();
        let mut ord: Vec<u32> = (0..tr.len() as u32).collect();
        ord.sort_by(|&i, &j| bb[i as usize].0[ax].partial_cmp(&bb[j as usize].0[ax]).unwrap());
        let teps = 1e-12 * size;
        let mut short: Vec<Piece> = Vec::new();
        for ii in 0..ord.len() {
            let i = ord[ii] as usize;
            for &j in &ord[ii + 1..] {
                let j = j as usize;
                if bb[j].0[ax] > bb[i].1[ax] + teps {
                    break;
                }
                if comp[i] == comp[j] {
                    continue;
                }
                if (0..3).any(|d| bb[j].0[d] > bb[i].1[d] + teps || bb[i].0[d] > bb[j].1[d] + teps) {
                    continue;
                }
                let Some((p, r)) = tri_tri_segment(&tr[i], nrm[i], &tr[j], nrm[j], teps) else { continue };
                let m = mul(add(p, r), 0.5);
                let (na, nb) = (nrm[i], nrm[j]);
                let ua = unit(sub(nb, mul(na, dot(nb, na))));
                let ub = unit(sub(na, mul(nb, dot(na, nb))));
                if on_boundary(add(m, mul(ua, eps)), na) && on_boundary(add(m, mul(ub, eps)), nb) {
                    pieces.push(Piece { a: p, b: r, ang: 180.0, convex: false });
                    st.isect_pieces += 1;
                } else if norm(sub(r, p)) < 40.0 * eps {
                    // short segments at triangle corners: the offset sample can leave the triangles; decided below by their neighbours
                    short.push(Piece { a: p, b: r, ang: 180.0, convex: false });
                }
            }
        }

        // short rejected segments that continue an accepted intersection curve belong to it
        {
            let wt = 1e-7 * size;
            let mut changed = true;
            while changed && !short.is_empty() {
                changed = false;
                let mut keep: Vec<Piece> = Vec::new();
                for sp in short.drain(..) {
                    let touches = pieces.iter().filter(|q| !q.convex).any(|q| [q.a, q.b].iter().any(|e| norm(sub(*e, sp.a)) < wt || norm(sub(*e, sp.b)) < wt));
                    if touches {
                        pieces.push(sp);
                        st.isect_pieces += 1;
                        changed = true;
                    } else {
                        keep.push(sp);
                    }
                }
                short = keep;
            }
        }
        // ---- weld endpoints into nodes
        let wtol = 1e-7 * size;
        let mut nodes: Vec<(V3, u32)> = Vec::new();
        let mut cellmap: Map<(i64, i64, i64), Vec<u32>> = Map::default();
        let ck = |p: V3| -> (i64, i64, i64) { ((p[0] / (2.0 * wtol)).floor() as i64, (p[1] / (2.0 * wtol)).floor() as i64, (p[2] / (2.0 * wtol)).floor() as i64) };
        let mut weld = |p: V3, nodes: &mut Vec<(V3, u32)>| -> u32 {
            let c = ck(p);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(l) = cellmap.get(&(c.0 + dx, c.1 + dy, c.2 + dz)) {
                            for &n in l {
                                if norm(sub(nodes[n as usize].0, p)) < wtol {
                                    let (q0, cnt) = nodes[n as usize];
                                    let c1 = cnt as f64 + 1.0;
                                    nodes[n as usize] = ([0, 1, 2].map(|d| q0[d] + (p[d] - q0[d]) / c1), cnt + 1);
                                    return n;
                                }
                            }
                        }
                    }
                }
            }
            nodes.push((p, 1));
            cellmap.entry(c).or_default().push(nodes.len() as u32 - 1);
            nodes.len() as u32 - 1
        };
        // pieces as (node a, node b, convex)
        let mut pn: Vec<(u32, u32, bool)> = Vec::new();
        let mut pang: Vec<f64> = Vec::new();
        for p in &pieces {
            let (a, b) = (weld(p.a, &mut nodes), weld(p.b, &mut nodes));
            if a != b {
                pn.push((a, b, p.convex));
                pang.push(p.ang);
            }
        }
        // T-junctions: free ends lying inside other pieces split them
        let ttol = 1e-6 * size;
        let mut deg: Vec<u32> = vec![0; nodes.len()];
        for &(a, b, _) in &pn {
            deg[a as usize] += 1;
            deg[b as usize] += 1;
        }
        for nidx in 0..nodes.len() {
            if deg[nidx] != 1 {
                continue;
            }
            let p = nodes[nidx].0;
            let mut hit: Option<usize> = None;
            for (k, &(a, b, _)) in pn.iter().enumerate() {
                if a as usize == nidx || b as usize == nidx {
                    continue;
                }
                let (c, t) = closest_on_seg(p, nodes[a as usize].0, nodes[b as usize].0);
                let l = norm(sub(nodes[b as usize].0, nodes[a as usize].0));
                if norm(sub(c, p)) < ttol && t * l > wtol && (1.0 - t) * l > wtol {
                    hit = Some(k);
                    break;
                }
            }
            if let Some(k) = hit {
                let (a, b, cv) = pn[k];
                pn[k] = (a, nidx as u32, cv);
                pn.push((nidx as u32, b, cv));
                pang.push(pang[k]);
                st.t_splits += 1;
            }
        }
        // ---- graph
        let nn = nodes.len();
        let mut inc: Vec<Vec<u32>> = vec![Vec::new(); nn];
        for (k, &(a, b, _)) in pn.iter().enumerate() {
            inc[a as usize].push(k as u32);
            inc[b as usize].push(k as u32);
        }
        let is_corner: Vec<bool> = (0..nn)
            .map(|v| {
                let d = inc[v].len();
                d >= 3 || d == 1 || (d == 2 && pn[inc[v][0] as usize].2 != pn[inc[v][1] as usize].2)
            })
            .collect();
        let mut cid: Vec<usize> = vec![usize::MAX; nn];
        let mut corners: Vec<Corner> = Vec::new();
        for v in 0..nn {
            if is_corner[v] {
                cid[v] = corners.len();
                let nc = inc[v].iter().filter(|&&k| pn[k as usize].2).count();
                let kind = if nc == inc[v].len() {
                    Kind::Convex
                } else if nc == 0 {
                    Kind::Concave
                } else {
                    Kind::Mixed
                };
                corners.push(Corner { p: nodes[v].0, curves: Vec::new(), kind });
            }
        }
        let mut used: Vec<bool> = vec![false; pn.len()];
        let mut curves: Vec<Curve> = Vec::new();
        let walk = |start: usize, first: usize, used: &mut Vec<bool>| -> (Vec<V3>, usize, bool, f64) {
            let mut pts = vec![nodes[start].0];
            let mut amin = f64::INFINITY;
            let (mut cur, mut k) = (start, first);
            loop {
                used[k] = true;
                amin = amin.min(pang[k]);
                let (a, b, cv) = pn[k];
                let nxt = if a as usize == cur { b as usize } else { a as usize };
                pts.push(nodes[nxt].0);
                if is_corner[nxt] || nxt == start {
                    return (pts, nxt, cv, amin);
                }
                cur = nxt;
                let nk = inc[nxt].iter().map(|&x| x as usize).find(|&x| !used[x]);
                match nk {
                    Some(x) => k = x,
                    None => return (pts, nxt, cv, amin),
                }
            }
        };
        for v in 0..nn {
            if !is_corner[v] {
                continue;
            }
            for &k in &inc[v] {
                if used[k as usize] {
                    continue;
                }
                let (pts, end, cv, amin) = walk(v, k as usize, &mut used);
                let id = curves.len();
                let (e0, e1) = (cid[v], cid[end]);
                if e0 != usize::MAX {
                    corners[e0].curves.push(id);
                }
                if e1 != usize::MAX && e1 != e0 {
                    corners[e1].curves.push(id);
                } else if e1 == e0 && e0 != usize::MAX {
                    corners[e0].curves.push(id);
                }
                curves.push(Curve { pts, kind: if cv { Kind::Convex } else { Kind::Concave }, ends: if e0 != usize::MAX && e1 != usize::MAX { Some([e0, e1]) } else { None }, angle_deg: amin });
            }
        }
        // closed loops without corners
        for v in 0..nn {
            if is_corner[v] {
                continue;
            }
            if let Some(&k) = inc[v].iter().find(|&&k| !used[k as usize]) {
                let (pts, _, cv, amin) = walk(v, k as usize, &mut used);
                curves.push(Curve { pts, kind: if cv { Kind::Convex } else { Kind::Concave }, ends: None, angle_deg: amin });
            }
        }
        st.curves = curves.len();
        st.corners = corners.len();
        st.convex_corners = corners.iter().filter(|c| c.kind == Kind::Convex).count();
        st.concave_corners = corners.iter().filter(|c| c.kind == Kind::Concave).count();
        st.mixed_corners = corners.iter().filter(|c| c.kind == Kind::Mixed).count();
        // ---- query grid over curve segments
        f.cell = (size / 48.0).max(1e-300);
        f.org = lo;
        let cellof = |p: V3, c: f64, o: V3| -> (i32, i32, i32) { (((p[0] - o[0]) / c).floor() as i32, ((p[1] - o[1]) / c).floor() as i32, ((p[2] - o[2]) / c).floor() as i32) };
        for (ci, c) in curves.iter().enumerate() {
            for s in 0..c.pts.len() - 1 {
                let (a, b) = (cellof(c.pts[s], f.cell, f.org), cellof(c.pts[s + 1], f.cell, f.org));
                for x in a.0.min(b.0)..=a.0.max(b.0) {
                    for y in a.1.min(b.1)..=a.1.max(b.1) {
                        for z in a.2.min(b.2)..=a.2.max(b.2) {
                            f.grid.entry((x, y, z)).or_default().push((ci as u32, s as u32));
                        }
                    }
                }
            }
        }
        f.curves = curves;
        f.corners = corners;
        st.seconds = t0.elapsed().as_secs_f64();
        f.stats = st;
        f
    }

    /// Nearest point on any feature curve within `max_dist` of `p`.
    pub fn nearest_curve(&self, p: V3, max_dist: f64) -> Option<CurveHit> {
        if self.curves.is_empty() {
            return None;
        }
        let c0 = ((p[0] - self.org[0]) / self.cell).floor() as i32;
        let c1 = ((p[1] - self.org[1]) / self.cell).floor() as i32;
        let c2 = ((p[2] - self.org[2]) / self.cell).floor() as i32;
        let rmax = (max_dist / self.cell).ceil() as i32 + 1;
        let mut best: Option<CurveHit> = None;
        for r in 0..=rmax {
            if let Some(b) = &best {
                if b.dist <= (r - 1).max(0) as f64 * self.cell {
                    break;
                }
            }
            for x in c0 - r..=c0 + r {
                for y in c1 - r..=c1 + r {
                    for z in c2 - r..=c2 + r {
                        if (x - c0).abs().max((y - c1).abs()).max((z - c2).abs()) != r {
                            continue;
                        }
                        if let Some(l) = self.grid.get(&(x, y, z)) {
                            for &(ci, s) in l {
                                let c = &self.curves[ci as usize];
                                let (q, u) = closest_on_seg(p, c.pts[s as usize], c.pts[s as usize + 1]);
                                let d = norm(sub(q, p));
                                if d <= max_dist && best.as_ref().is_none_or(|b| d < b.dist || (d == b.dist && (ci as usize, s as usize) < (b.curve, b.seg))) {
                                    best = Some(CurveHit { p: q, curve: ci as usize, dist: d, seg: s as usize, u });
                                }
                            }
                        }
                    }
                }
            }
        }
        best
    }

    /// Corners within `r` of `p` as (corner id, distance), nearest first.
    pub fn corners_near(&self, p: V3, r: f64) -> Vec<(usize, f64)> {
        let mut v: Vec<(usize, f64)> = self.corners.iter().enumerate().map(|(i, c)| (i, norm(sub(c.p, p)))).filter(|x| x.1 <= r).collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v
    }

    /// Total length of the curves of the given kind.
    pub fn length(&self, kind: Kind) -> f64 {
        self.curves.iter().filter(|c| c.kind == kind).map(|c| c.pts.windows(2).map(|w| norm(sub(w[1], w[0]))).sum::<f64>()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn bx(lo: V3, hi: V3) -> Vec<Tri> {
        let p = |i: usize, j: usize, k: usize| [if i == 1 { hi[0] } else { lo[0] }, if j == 1 { hi[1] } else { lo[1] }, if k == 1 { hi[2] } else { lo[2] }];
        let quads = [
            [p(0, 0, 0), p(0, 1, 0), p(0, 1, 1), p(0, 0, 1)],
            [p(1, 0, 0), p(1, 0, 1), p(1, 1, 1), p(1, 1, 0)],
            [p(0, 0, 0), p(0, 0, 1), p(1, 0, 1), p(1, 0, 0)],
            [p(0, 1, 0), p(1, 1, 0), p(1, 1, 1), p(0, 1, 1)],
            [p(0, 0, 0), p(1, 0, 0), p(1, 1, 0), p(0, 1, 0)],
            [p(0, 0, 1), p(0, 1, 1), p(1, 1, 1), p(1, 0, 1)],
        ];
        quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).collect()
    }

    fn sphere(c: V3, r: f64, nu: usize, nv: usize) -> Vec<Tri> {
        let p = |i: usize, j: usize| {
            let (th, ph) = (PI * i as f64 / nv as f64, 2.0 * PI * (j % nu) as f64 / nu as f64);
            [c[0] + r * th.sin() * ph.cos(), c[1] + r * th.sin() * ph.sin(), c[2] + r * th.cos()]
        };
        let mut t: Vec<Tri> = Vec::new();
        for i in 0..nv {
            for j in 0..nu {
                if i > 0 {
                    t.push([p(i, j), p(i + 1, j), p(i, j + 1)]);
                }
                if i < nv - 1 {
                    t.push([p(i, j + 1), p(i + 1, j), p(i + 1, j + 1)]);
                }
            }
        }
        t
    }

    fn torus(c: V3, big: f64, small: f64, nu: usize, nv: usize) -> Vec<Tri> {
        let p = |i: usize, j: usize| {
            let (a, b) = (2.0 * PI * (i % nu) as f64 / nu as f64, 2.0 * PI * (j % nv) as f64 / nv as f64);
            [c[0] + (big + small * b.cos()) * a.cos(), c[1] + (big + small * b.cos()) * a.sin(), c[2] + small * b.sin()]
        };
        let mut t: Vec<Tri> = Vec::new();
        for i in 0..nu {
            for j in 0..nv {
                t.push([p(i, j), p(i + 1, j), p(i + 1, j + 1)]);
                t.push([p(i, j), p(i + 1, j + 1), p(i, j + 1)]);
            }
        }
        t
    }

    #[test]
    fn single_box() {
        let f = Features::detect(&bx([0.0; 3], [1.0, 2.0, 3.0]), &FeatureOptions::default());
        assert_eq!((f.curves.len(), f.corners.len()), (12, 8), "{}", f.stats.json());
        assert!(f.corners.iter().all(|c| c.kind == Kind::Convex && c.curves.len() == 3));
        assert!(f.curves.iter().all(|c| c.kind == Kind::Convex));
        assert!((f.length(Kind::Convex) - 4.0 * (1.0 + 2.0 + 3.0)).abs() < 1e-9);
    }

    #[test]
    fn smooth_sphere_has_no_features() {
        let f = Features::detect(&sphere([0.0; 3], 1.0, 48, 24), &FeatureOptions::default());
        assert_eq!((f.curves.len(), f.corners.len()), (0, 0));
    }

    #[test]
    fn overlapping_box_cross() {
        let mut t = bx([-0.6, -0.2, -0.2], [0.6, 0.2, 0.2]);
        t.extend(bx([-0.2, -0.6, -0.2], [0.2, 0.6, 0.2]));
        let f = Features::detect(&t, &FeatureOptions::default());
        let concave: Vec<&Curve> = f.curves.iter().filter(|c| c.kind == Kind::Concave).collect();
        assert_eq!(concave.len(), 4, "{}", f.stats.json());
        for c in &concave {
            let (a, b) = (c.pts[0], *c.pts.last().unwrap());
            assert!((a[0].abs() - 0.2).abs() < 1e-9 && (a[1].abs() - 0.2).abs() < 1e-9 && (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
            assert!(((a[2] - b[2]).abs() - 0.4).abs() < 1e-9, "crease length");
        }
        let mixed: Vec<&Corner> = f.corners.iter().filter(|c| c.kind == Kind::Mixed).collect();
        assert_eq!(mixed.len(), 8, "{}", f.stats.json());
        for c in &mixed {
            assert!(c.p.iter().all(|x| (x.abs() - 0.2).abs() < 1e-9), "mixed corner at {:?}", c.p);
            assert_eq!(c.curves.len(), 3);
        }
        assert_eq!(f.stats.convex_corners, 16);
        assert_eq!(f.stats.concave_corners, 0);
        // box 1: 4 x-edges of 1.2 minus the 0.4 inside box 2, 8 short edges; box 2 the same
        assert!((f.length(Kind::Convex) - 12.8).abs() < 1e-9, "convex length {}", f.length(Kind::Convex));
        // nearest curve / corner queries
        let h = f.nearest_curve([-0.3, -0.2, -0.25], 0.5).unwrap();
        assert!((h.dist - 0.05).abs() < 1e-9 && (h.p[2] + 0.2).abs() < 1e-9);
        let cn = f.corners_near([-0.21, -0.19, -0.2], 0.05);
        assert_eq!(cn.len(), 1);
    }

    #[test]
    fn sphere_torus_intersection_curve() {
        let mut t = sphere([0.0; 3], 0.6, 64, 32);
        t.extend(torus([0.35, 0.1, 0.05], 0.55, 0.18, 96, 32));
        let f = Features::detect(&t, &FeatureOptions::default());
        assert_eq!(f.corners.len(), 0, "{}", f.stats.json());
        // the torus tube enters and leaves the sphere: two closed concave loops
        assert_eq!(f.curves.len(), 2, "{}", f.stats.json());
        for c in &f.curves {
            assert!(c.kind == Kind::Concave && c.ends.is_none());
            let (a, b) = (c.pts[0], *c.pts.last().unwrap());
            assert!(norm(sub(a, b)) < 1e-6, "closed loop");
            // every point of the curve lies on both surfaces: |x| = 0.6 (sphere)
            assert!(c.pts.iter().all(|p| (norm(*p) - 0.6).abs() < 2e-3));
        }
    }

    #[test]
    fn dirty_soup_does_not_panic() {
        let mut t = bx([0.0; 3], [1.0; 3]);
        t.push([[0.0; 3], [0.0; 3], [1.0, 0.0, 0.0]]); // degenerate
        t.push([[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        t.truncate(t.len() - 5); // open box
        let _ = Features::detect(&t, &FeatureOptions::default());
        let _ = Features::detect(&[], &FeatureOptions::default());
    }
}
