//! Graded, conforming shell quad meshes: the core boundary as a polygon surface P, a surface T-join that restores parity, and
//! face templates that turn P into an all-quad mesh Q (design: `docs/GRADED_SHELL_DESIGN_v1.md`).
//!
//! * Points are keys on the *doubled* lattice (`key2`): the lattice of the finest octree level with every coordinate doubled, so
//!   midpoints of lattice edges and face centres are exact integers. Needs `max_level <= 19` (21 bits per coordinate).
//! * A boundary face (`BFace`) is a lattice quad of one core leaf; its side `t` (corner `t` to `t + 1`) is *hanging* when the side
//!   midpoint is a vertex of a finer core leaf. Its polygon in P has the four corners plus the midpoint of every hanging side.
//! * A face with an odd number of hanging sides cannot be quad-meshed (4F = 2E_int + n). `Surface::tjoin` selects *full* edges
//!   (same-level, unsplit on both faces) to split at their midpoint so that every face gets an even split mask (hanging sides plus
//!   join edges). Half edges are never split again.
//! * `Surface::expand` replaces every face by one of four templates according to its (even) mask: T0 (the face), Tc (two adjacent
//!   sides, 3 quads about the face centre), Ts (two opposite sides, 2 quads), T4 (all sides, 4 quads).

use crate::octree::{vertex_coords, vertex_key};

pub const NONE: u32 = u32::MAX;

/// Doubled-lattice key of the lattice point `c`.
pub fn key2(c: [u32; 3]) -> u64 {
    vertex_key(2 * c[0], 2 * c[1], 2 * c[2])
}

/// Midpoint of two doubled-lattice keys (exact when both have even coordinates, i.e. are lattice points).
pub fn mid_key(a: u64, b: u64) -> u64 {
    let (p, q) = (vertex_coords(a), vertex_coords(b));
    vertex_key((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2)
}

/// Boundary face of the core: owner leaf, corner keys (doubled lattice) in right-hand order about the outward normal, owner edge
/// `h`, and the mask of hanging sides (bit `t` = the midpoint of side `t` is a vertex of a finer core leaf).
#[derive(Clone, Copy, Debug)]
pub struct BFace {
    pub leaf: u32,
    pub key: [u64; 4],
    pub h: f64,
    pub hang: u8,
}

impl BFace {
    /// Polygon of the face in P: corners with the hanging midpoints inserted (at most 8 vertices).
    pub fn polygon(&self) -> ([u64; 8], usize) {
        let (mut p, mut n) = ([0u64; 8], 0);
        for t in 0..4 {
            p[n] = self.key[t];
            n += 1;
            if self.hang >> t & 1 == 1 {
                p[n] = mid_key(self.key[t], self.key[(t + 1) % 4]);
                n += 1;
            }
        }
        (p, n)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EKind {
    /// full side of two faces (same level, unsplit): joinable
    FF,
    /// half edge: half side of the coarse face, full side of the fine face
    HF,
    /// half edge: half side of both faces (both see the hanging midpoint)
    HH,
    /// used by one face only (shell touching the domain boundary)
    Open,
    /// used by more than two faces
    NonManifold,
}

/// An edge of P: a segment between two consecutive polygon vertices.
#[derive(Clone, Copy, Debug)]
pub struct PEdge {
    pub a: u64,
    pub b: u64,
    pub n: u32,
    /// first two faces using the edge and whether each uses it as half of a hanging side
    pub f: [u32; 2],
    pub half: [bool; 2],
}

impl PEdge {
    pub fn kind(&self) -> EKind {
        match self.n {
            1 => EKind::Open,
            2 => match (self.half[0], self.half[1]) {
                (false, false) => EKind::FF,
                (true, true) => EKind::HH,
                _ => EKind::HF,
            },
            _ => EKind::NonManifold,
        }
    }
}

pub struct Surface {
    pub faces: Vec<BFace>,
    /// sorted by (a, b), a < b
    pub edges: Vec<PEdge>,
    /// per face and side: the P-edge of the side, or the two half edges (second is NONE for a full side)
    pub fside: Vec<[[u32; 2]; 4]>,
    /// edges used by more than two faces: (edge, faces)
    pub nm: Vec<(u32, Vec<u32>)>,
}

#[derive(Default, Debug)]
pub struct ManCheck {
    /// edges with one face
    pub open_edges: usize,
    /// faces of every edge with more than two faces
    pub nm_edges: Vec<Vec<u32>>,
    /// vertices whose faces form several fans
    pub nm_verts: usize,
    /// faces of all fans but the largest, over all non-manifold vertices (with repeats)
    pub minor_faces: Vec<u32>,
}

#[derive(Default, Debug)]
pub struct TJoin {
    /// final split mask per face (hanging sides plus join edges)
    pub mask: Vec<u8>,
    /// per P-edge: in the join J
    pub is_j: Vec<bool>,
    /// faces with an odd number of hanging sides before the join
    pub odd_faces: usize,
    pub n_j: usize,
    /// longest path (in edges) used by the BFS pairing
    pub max_path: usize,
    /// faces whose mask is still odd (an odd component without an open edge: the assumptions of Theorem 1 failed)
    pub unmatched: Vec<u32>,
}

/// Q mesh: all-quad refinement of P.
#[derive(Default, Debug)]
pub struct QMesh {
    /// sorted unique doubled-lattice keys
    pub verts: Vec<u64>,
    pub quads: Vec<[u32; 4]>,
    /// parent face of every quad; quads of face `f` are `fq[f]..fq[f + 1]`
    pub parent: Vec<u32>,
    pub fq: Vec<u32>,
    /// faces per template T0, Tc, Ts, T4
    pub tpl_count: [usize; 4],
}

/// Template ids: corners 0..4, side midpoints 4 + t, face centre 8. Corner order is counter-clockwise about the outward normal.
pub const TPL: [&[[u8; 4]]; 4] = [
    &[[0, 1, 2, 3]],
    // Tc, sides 0 and 1 split (common corner 1)
    &[[4, 1, 5, 8], [0, 4, 8, 3], [8, 5, 2, 3]],
    // Ts, sides 0 and 2 split
    &[[0, 4, 6, 3], [4, 1, 2, 6]],
    // T4
    &[[0, 4, 8, 7], [4, 1, 5, 8], [8, 5, 2, 6], [7, 8, 6, 3]],
];

/// Mask (bit t = side t split) -> (template, rotation); odd masks have no template.
pub const MASK_TPL: [Option<(u8, u8)>; 16] = [
    Some((0, 0)), // 0000
    None,
    None,
    Some((1, 0)), // sides 0,1
    None,
    Some((2, 0)), // sides 0,2
    Some((1, 1)), // sides 1,2
    None,
    None,
    Some((1, 3)), // sides 0,3
    Some((2, 1)), // sides 1,3
    None,
    Some((1, 2)), // sides 2,3
    None,
    None,
    Some((3, 0)), // all
];

/// Local template point `i` of rotation `rot` (corner i -> i + rot, side midpoint 4 + t -> 4 + t + rot).
fn rot_id(i: u8, rot: u8) -> usize {
    match i {
        0..=3 => ((i + rot) % 4) as usize,
        4..=7 => 4 + ((i - 4 + rot) % 4) as usize,
        _ => 8,
    }
}

impl Surface {
    pub fn build(faces: &[BFace]) -> Surface {
        let nf = faces.len();
        let mut uses: Vec<(u64, u64, u32, u8, u8, bool)> = Vec::with_capacity(nf * 5);
        for (fi, f) in faces.iter().enumerate() {
            for t in 0..4 {
                let (a, b) = (f.key[t], f.key[(t + 1) % 4]);
                if f.hang >> t & 1 == 1 {
                    let m = mid_key(a, b);
                    uses.push((a.min(m), a.max(m), fi as u32, t as u8, 0, true));
                    uses.push((m.min(b), m.max(b), fi as u32, t as u8, 1, true));
                } else {
                    uses.push((a.min(b), a.max(b), fi as u32, t as u8, 0, false));
                }
            }
        }
        uses.sort_unstable();
        let mut edges: Vec<PEdge> = Vec::with_capacity(uses.len() / 2 + 1);
        let mut fside = vec![[[NONE; 2]; 4]; nf];
        let mut nm: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut i = 0;
        while i < uses.len() {
            let mut j = i;
            while j < uses.len() && uses[j].0 == uses[i].0 && uses[j].1 == uses[i].1 {
                j += 1;
            }
            let e = edges.len() as u32;
            let mut pe = PEdge { a: uses[i].0, b: uses[i].1, n: (j - i) as u32, f: [NONE; 2], half: [false; 2] };
            for (s, u) in uses[i..j].iter().enumerate() {
                if s < 2 {
                    pe.f[s] = u.2;
                    pe.half[s] = u.5;
                }
                fside[u.2 as usize][u.3 as usize][u.4 as usize] = e;
            }
            if j - i > 2 {
                nm.push((e, uses[i..j].iter().map(|u| u.2).collect()));
            }
            edges.push(pe);
            i = j;
        }
        Surface { faces: faces.to_vec(), edges, fside, nm }
    }

    /// Open / non-manifold edges and non-manifold vertices (faces around a vertex, hanging vertices included, that do not form a
    /// single fan connected through shared edges).
    pub fn manifold_check(&self) -> ManCheck {
        let mut r = ManCheck::default();
        for e in &self.edges {
            if e.n == 1 {
                r.open_edges += 1;
            }
        }
        for (_, fs) in &self.nm {
            r.nm_edges.push(fs.clone());
        }
        let mut vv: Vec<(u64, u32, u64, u64)> = Vec::with_capacity(self.faces.len() * 4);
        for (fi, f) in self.faces.iter().enumerate() {
            let (p, n) = f.polygon();
            for t in 0..n {
                vv.push((p[t], fi as u32, p[(t + n - 1) % n], p[(t + 1) % n]));
            }
        }
        vv.sort_unstable();
        let mut i = 0;
        while i < vv.len() {
            let mut j = i;
            while j < vv.len() && vv[j].0 == vv[i].0 {
                j += 1;
            }
            let g = &vv[i..j];
            let n = g.len();
            let mut par: Vec<usize> = (0..n).collect();
            fn find(p: &mut Vec<usize>, mut x: usize) -> usize {
                while p[x] != x {
                    p[x] = p[p[x]];
                    x = p[x];
                }
                x
            }
            for x in 0..n {
                for y in x + 1..n {
                    if g[x].2 == g[y].2 || g[x].2 == g[y].3 || g[x].3 == g[y].2 || g[x].3 == g[y].3 {
                        let (rx, ry) = (find(&mut par, x), find(&mut par, y));
                        par[rx] = ry;
                    }
                }
            }
            let roots: Vec<usize> = (0..n).map(|x| find(&mut par, x)).collect();
            let mut sizes: Vec<(usize, usize)> = Vec::new();
            for &rt in &roots {
                match sizes.iter_mut().find(|s| s.0 == rt) {
                    Some(s) => s.1 += 1,
                    None => sizes.push((rt, 1)),
                }
            }
            if sizes.len() > 1 {
                r.nm_verts += 1;
                let big = sizes.iter().max_by_key(|(k, c)| (*c, std::cmp::Reverse(*k))).unwrap().0;
                for x in 0..n {
                    if roots[x] != big {
                        r.minor_faces.push(g[x].1);
                    }
                }
            }
            i = j;
        }
        r
    }

    /// Union-find components of the full-edge graph (faces joined by FF edges), as a root id per face.
    pub fn ff_components(&self) -> Vec<u32> {
        let mut p: Vec<u32> = (0..self.faces.len() as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        for e in &self.edges {
            if e.kind() == EKind::FF {
                let (a, b) = (find(&mut p, e.f[0]), find(&mut p, e.f[1]));
                if a != b {
                    p[a.max(b) as usize] = a.min(b);
                }
            }
        }
        (0..p.len() as u32).map(|x| find(&mut p, x)).collect()
    }

    /// T-join over the full-edge graph (faces are nodes, an FF edge joins its two faces, an open full edge joins its face to a
    /// virtual sink of free parity): greedy pairing of adjacent odd faces (preferring edges between faces with `pref > 0`, ties by
    /// edge order), then BFS paths with XOR for the rest. Deterministic: faces and edges are visited in key order.
    pub fn tjoin(&self, pref: &[u8]) -> TJoin {
        let nf = self.faces.len();
        let sink = nf as u32;
        let mut parity: Vec<u8> = self.faces.iter().map(|f| f.hang.count_ones() as u8 & 1).collect();
        let odd_faces = parity.iter().filter(|&&p| p == 1).count();
        // adjacency (other node, edge), in edge order
        let mut deg = vec![0u32; nf + 1];
        let ends = |e: &PEdge| -> Option<(u32, u32)> {
            match e.kind() {
                EKind::FF => Some((e.f[0], e.f[1])),
                EKind::Open if !e.half[0] => Some((e.f[0], sink)),
                _ => None,
            }
        };
        for e in &self.edges {
            if let Some((a, b)) = ends(e) {
                deg[a as usize] += 1;
                if b != sink {
                    deg[b as usize] += 1;
                }
            }
        }
        let mut start = vec![0usize; nf + 1];
        for i in 0..nf {
            start[i + 1] = start[i] + deg[i] as usize;
        }
        let mut fill = start.clone();
        let mut adj: Vec<(u32, u32)> = vec![(0, 0); start[nf]];
        for (ei, e) in self.edges.iter().enumerate() {
            if let Some((a, b)) = ends(e) {
                adj[fill[a as usize]] = (b, ei as u32);
                fill[a as usize] += 1;
                if b != sink {
                    adj[fill[b as usize]] = (a, ei as u32);
                    fill[b as usize] += 1;
                }
            }
        }
        let mut is_j = vec![false; self.edges.len()];
        // greedy pairing: among the edges to a still-odd neighbour prefer the one giving the better templates (an even mask of two opposite
        // sides or all four, planar SJ 1, beats two adjacent sides, 0.707), then edges between faces whose owners are already polyhedral,
        // then the first edge in key order
        let mut cur: Vec<u8> = self.faces.iter().map(|f| f.hang).collect();
        let side_of = |f: usize, e: u32| (0..4).find(|&t| self.fside[f][t][0] == e).unwrap_or(0);
        let qual = |m: u8| -> u8 {
            if m.count_ones() % 2 == 1 {
                1
            } else if m == 0b0101 || m == 0b1010 || m == 0b1111 {
                2
            } else {
                1
            }
        };
        for f in 0..nf {
            if parity[f] == 0 {
                continue;
            }
            let mut best: Option<((u8, u8), u32, u32)> = None; // ((template quality, preference), edge, other)
            for &(o, e) in &adj[start[f]..start[f + 1]] {
                if o == sink || parity[o as usize] == 0 || is_j[e as usize] {
                    continue;
                }
                let q = qual(cur[f] | 1 << side_of(f, e)) + qual(cur[o as usize] | 1 << side_of(o as usize, e));
                let w = (q, pref.get(f).copied().unwrap_or(0) + pref.get(o as usize).copied().unwrap_or(0));
                if best.is_none_or(|b| w > b.0) {
                    best = Some((w, e, o));
                }
            }
            if let Some((_, e, o)) = best {
                is_j[e as usize] = true;
                cur[f] |= 1 << side_of(f, e);
                cur[o as usize] |= 1 << side_of(o as usize, e);
                parity[f] = 0;
                parity[o as usize] = 0;
            }
        }
        // BFS paths with XOR
        let mut max_path = 0usize;
        let (mut seen, mut pe, mut pn) = (vec![0u32; nf], vec![NONE; nf], vec![NONE; nf]);
        let (mut stamp, mut queue) = (0u32, Vec::<u32>::new());
        for f in 0..nf {
            if parity[f] == 0 {
                continue;
            }
            stamp += 1;
            queue.clear();
            queue.push(f as u32);
            seen[f] = stamp;
            let mut head = 0;
            // (target node or sink, node the last edge leaves from, last edge for the sink)
            let mut found: Option<(u32, u32, u32)> = None;
            'bfs: while head < queue.len() {
                let u = queue[head] as usize;
                head += 1;
                for &(o, e) in &adj[start[u]..start[u + 1]] {
                    if o == sink {
                        found = Some((sink, u as u32, e));
                        break 'bfs;
                    }
                    if seen[o as usize] == stamp {
                        continue;
                    }
                    seen[o as usize] = stamp;
                    pe[o as usize] = e;
                    pn[o as usize] = u as u32;
                    if parity[o as usize] == 1 {
                        found = Some((o, o, NONE));
                        break 'bfs;
                    }
                    queue.push(o);
                }
            }
            let Some((target, from, last)) = found else { continue };
            let mut len = 0;
            if target == sink {
                is_j[last as usize] ^= true;
                len += 1;
            } else {
                parity[target as usize] ^= 1;
            }
            let mut x = from as usize;
            while x != f {
                is_j[pe[x] as usize] ^= true;
                x = pn[x] as usize;
                len += 1;
            }
            parity[f] ^= 1;
            max_path = max_path.max(len);
        }
        let mut mask = vec![0u8; nf];
        for (fi, f) in self.faces.iter().enumerate() {
            for t in 0..4 {
                if f.hang >> t & 1 == 1 || is_j[self.fside[fi][t][0] as usize] {
                    mask[fi] |= 1 << t;
                }
            }
        }
        let unmatched: Vec<u32> = (0..nf as u32).filter(|&f| mask[f as usize].count_ones() & 1 == 1).collect();
        let n_j = is_j.iter().filter(|&&j| j).count();
        TJoin { mask, is_j, odd_faces, n_j, max_path, unmatched }
    }

    /// Replace every face by its template (the masks must be even). Quads keep the orientation of the face.
    pub fn expand(&self, mask: &[u8]) -> QMesh {
        let nf = self.faces.len();
        let mut keys: Vec<[u64; 4]> = Vec::with_capacity(nf * 2);
        let mut q = QMesh { fq: Vec::with_capacity(nf + 1), ..Default::default() };
        for (fi, f) in self.faces.iter().enumerate() {
            q.fq.push(keys.len() as u32);
            let (tpl, rot) = MASK_TPL[mask[fi] as usize & 15].expect("odd split mask");
            q.tpl_count[tpl as usize] += 1;
            let mut pt = [0u64; 9];
            pt[..4].copy_from_slice(&f.key);
            for t in 0..4 {
                pt[4 + t] = mid_key(f.key[t], f.key[(t + 1) % 4]);
            }
            pt[8] = mid_key(f.key[0], f.key[2]);
            for tq in TPL[tpl as usize] {
                keys.push(std::array::from_fn(|i| pt[rot_id(tq[i], rot)]));
                q.parent.push(fi as u32);
            }
        }
        q.fq.push(keys.len() as u32);
        q.verts = keys.iter().flatten().copied().collect();
        q.verts.sort_unstable();
        q.verts.dedup();
        q.quads = keys.iter().map(|k| k.map(|x| q.verts.binary_search(&x).unwrap() as u32)).collect();
        q
    }

    /// Q mesh without any split (T0 everywhere).
    pub fn plain(faces: &[BFace]) -> QMesh {
        let nf = faces.len();
        let mut q = QMesh { fq: (0..=nf as u32).collect(), parent: (0..nf as u32).collect(), tpl_count: [nf, 0, 0, 0], ..Default::default() };
        q.verts = faces.iter().flat_map(|f| f.key).collect();
        q.verts.sort_unstable();
        q.verts.dedup();
        q.quads = faces.iter().map(|f| f.key.map(|x| q.verts.binary_search(&x).unwrap() as u32)).collect();
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p2(k: u64) -> [f64; 2] {
        let c = vertex_coords(k);
        [c[0] as f64, c[1] as f64]
    }

    #[test]
    fn templates_tile_square() {
        // unit square on the doubled lattice scaled by 8 (z = 0); counter-clockwise corners
        let c = [[0u32, 0], [8, 0], [8, 8], [0, 8]];
        let key = |x: u32, y: u32| vertex_key(x, y, 0);
        let f = |hang: u8| BFace { leaf: 0, key: [key(c[0][0], c[0][1]), key(c[1][0], c[1][1]), key(c[2][0], c[2][1]), key(c[3][0], c[3][1])], h: 1.0, hang };
        let mut n_even = 0;
        for mask in 0..16u8 {
            assert_eq!(MASK_TPL[mask as usize].is_some(), mask.count_ones() % 2 == 0, "mask table parity {mask:04b}");
            if mask.count_ones() % 2 == 1 {
                continue;
            }
            n_even += 1;
            let face = f(0);
            let s = Surface::build(&[face]);
            let q = s.expand(&[mask]);
            let expect_tpl = match mask.count_ones() {
                0 => 0,
                4 => 3,
                _ if mask == 0b0101 || mask == 0b1010 => 2,
                _ => 1,
            };
            assert_eq!(q.tpl_count[expect_tpl], 1, "template of mask {mask:04b}");
            let (mut area, mut sjmin) = (0.0, f64::INFINITY);
            let mut dir: std::collections::HashMap<(u64, u64), i32> = Default::default();
            for qd in &q.quads {
                let p: Vec<[f64; 2]> = qd.iter().map(|&v| p2(q.verts[v as usize])).collect();
                let a = 0.5 * (0..4).map(|i| p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1]).sum::<f64>();
                assert!(a > 0.0, "non-positive quad area, mask {mask:04b}");
                area += a;
                for i in 0..4 {
                    let (cur, nx, pv) = (p[i], p[(i + 1) % 4], p[(i + 3) % 4]);
                    let (e1, e2) = ([nx[0] - cur[0], nx[1] - cur[1]], [pv[0] - cur[0], pv[1] - cur[1]]);
                    sjmin = sjmin.min((e1[0] * e2[1] - e1[1] * e2[0]) / (e1[0].hypot(e1[1]) * e2[0].hypot(e2[1])));
                    let (u, v) = (q.verts[qd[i] as usize], q.verts[qd[(i + 1) % 4] as usize]);
                    *dir.entry((u.min(v), u.max(v))).or_insert(0) += if u < v { 1 } else { -1 };
                }
            }
            assert!((area - 64.0).abs() < 1e-12, "area {area} mask {mask:04b}");
            assert!(sjmin >= 0.70, "planar SJ {sjmin} mask {mask:04b}");
            // boundary vertices: corners plus the midpoints of the masked sides, all on the square boundary
            let on_bd = |p: [f64; 2]| p[0] == 0.0 || p[0] == 8.0 || p[1] == 0.0 || p[1] == 8.0;
            let nb = q.verts.iter().filter(|&&k| on_bd(p2(k))).count();
            assert_eq!(nb, 4 + mask.count_ones() as usize, "boundary vertices of mask {mask:04b}");
            // every edge: interior ones are used twice with opposite orientation (sum 0, both uses), boundary ones once
            let mut uses: std::collections::HashMap<(u64, u64), usize> = Default::default();
            for qd in &q.quads {
                for i in 0..4 {
                    let (u, v) = (q.verts[qd[i] as usize], q.verts[qd[(i + 1) % 4] as usize]);
                    *uses.entry((u.min(v), u.max(v))).or_insert(0) += 1;
                }
            }
            for (&(u, v), &n) in &uses {
                let bd = on_bd(p2(u)) && on_bd(p2(v)) && (p2(u)[0] == p2(v)[0] && (p2(u)[0] == 0.0 || p2(u)[0] == 8.0) || p2(u)[1] == p2(v)[1] && (p2(u)[1] == 0.0 || p2(u)[1] == 8.0));
                if bd {
                    assert_eq!(n, 1, "boundary edge used {n} times");
                    assert_eq!(dir[&(u, v)].abs(), 1);
                } else {
                    assert_eq!(n, 2, "interior edge used {n} times");
                    assert_eq!(dir[&(u, v)], 0, "interior edge not used with opposite orientation");
                }
            }
        }
        assert_eq!(n_even, 8);
        // rotations are consistent with the mask: the template boundary carries a vertex exactly on the masked sides
        for mask in [0b0011u8, 0b0110, 0b1100, 0b1001, 0b0101, 0b1010, 0b1111, 0] {
            let s = Surface::build(&[f(0)]);
            let q = s.expand(&[mask]);
            for t in 0..4 {
                let m = mid_key(vertex_key(c[t][0], c[t][1], 0), vertex_key(c[(t + 1) % 4][0], c[(t + 1) % 4][1], 0));
                assert_eq!(q.verts.binary_search(&m).is_ok(), mask >> t & 1 == 1, "mid of side {t}, mask {mask:04b}");
            }
        }
    }

    #[test]
    fn tjoin_two_faces() {
        // two coplanar faces of level L sharing an FF edge; the left one has a hanging side towards a finer neighbour (odd)
        let k = |x: u32, y: u32| vertex_key(x, y, 0);
        let l = BFace { leaf: 0, key: [k(0, 0), k(8, 0), k(8, 8), k(0, 8)], h: 1.0, hang: 0b0001 };
        let r = BFace { leaf: 1, key: [k(8, 0), k(16, 0), k(16, 8), k(8, 8)], h: 1.0, hang: 0 };
        let s = Surface::build(&[l, r]);
        let tj = s.tjoin(&[0, 0]);
        // no sink partner for the left face except the open edges of the right one: pairs through the open edge of face 0 or 1
        assert_eq!(tj.odd_faces, 1);
        assert!(tj.unmatched.is_empty());
        assert!(tj.mask.iter().all(|m| m.count_ones() % 2 == 0));
        let q = s.expand(&tj.mask);
        assert!(q.quads.len() >= 2);
    }
}
