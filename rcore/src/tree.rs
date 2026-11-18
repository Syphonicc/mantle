//! Shared binary BVH over a triangle soup: median split on the longest axis of the centroid bounds,
//! leaves of <= `LEAF` triangles. Every node (leaf or inner) owns the contiguous range
//! `order[start..start + count]` of the triangle permutation, so per-node aggregates are cheap to build.

use crate::dexel::Tri;
use std::thread;

pub const LEAF: usize = 8;
pub const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub lo: [f64; 3],
    pub hi: [f64; 3],
    /// child node indices (NONE for a leaf)
    pub left: u32,
    pub right: u32,
    pub start: u32,
    pub count: u32,
}

pub struct Tree {
    pub nodes: Vec<Node>,
    /// permutation of triangle ids; node ranges index into this
    pub order: Vec<u32>,
}

fn bounds(t: &Tri) -> ([f64; 3], [f64; 3]) {
    let (mut lo, mut hi) = (t[0], t[0]);
    for v in &t[1..] {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    (lo, hi)
}

impl Tree {
    pub fn build(tris: &[Tri]) -> Tree {
        let cen: Vec<[f64; 3]> = tris.iter().map(|t| [0, 1, 2].map(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0)).collect();
        let mut order: Vec<u32> = (0..tris.len() as u32).collect();
        let mut nodes = Vec::with_capacity(2 * tris.len() / LEAF + 2);
        if !tris.is_empty() {
            Self::rec(tris, &cen, &mut order, 0, tris.len(), &mut nodes);
        }
        Tree { nodes, order }
    }

    fn rec(tris: &[Tri], cen: &[[f64; 3]], order: &mut [u32], s: usize, e: usize, nodes: &mut Vec<Node>) -> u32 {
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        let (mut clo, mut chi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for &i in &order[s..e] {
            let (l, h) = bounds(&tris[i as usize]);
            for k in 0..3 {
                lo[k] = lo[k].min(l[k]);
                hi[k] = hi[k].max(h[k]);
                clo[k] = clo[k].min(cen[i as usize][k]);
                chi[k] = chi[k].max(cen[i as usize][k]);
            }
        }
        let id = nodes.len() as u32;
        nodes.push(Node { lo, hi, left: NONE, right: NONE, start: s as u32, count: (e - s) as u32 });
        if e - s <= LEAF {
            return id;
        }
        let ax = (0..3).max_by(|&a, &b| (chi[a] - clo[a]).partial_cmp(&(chi[b] - clo[b])).unwrap()).unwrap();
        let m = (s + e) / 2;
        order[s..e].select_nth_unstable_by(m - s, |&a, &b| cen[a as usize][ax].partial_cmp(&cen[b as usize][ax]).unwrap());
        let l = Self::rec(tris, cen, order, s, m, nodes);
        let r = Self::rec(tris, cen, order, m, e, nodes);
        nodes[id as usize].left = l;
        nodes[id as usize].right = r;
        id
    }
}

/// Map `f` over `items` on all hardware threads (order preserved).
pub fn par_map<A: Sync, B: Send + Default + Clone>(items: &[A], f: impl Fn(&A) -> B + Sync) -> Vec<B> {
    let nt = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let per = items.len().div_ceil(nt).max(1);
    let mut out = vec![B::default(); items.len()];
    thread::scope(|sc| {
        for (src, dst) in items.chunks(per).zip(out.chunks_mut(per)) {
            let f = &f;
            sc.spawn(move || {
                for (a, b) in src.iter().zip(dst.iter_mut()) {
                    *b = f(a);
                }
            });
        }
    });
    out
}
