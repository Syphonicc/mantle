//! Input checks on the STL triangle soup (before meshing) and under-resolution warnings from the chosen sizing.
//!
//! `analyze` welds vertices (relative tolerance 1e-8 of the bbox length), then reports: open / non-manifold edges, inconsistent
//! winding, inward normals (negative signed volume), degenerate and duplicate triangles, self-intersections (edges of one triangle
//! piercing a non-adjacent triangle, BVH segment queries; coplanar overlaps are not seen), number of parts, sharp edges.
//! It also stores per-triangle data used later: inward ray thickness, outward facing-gap distance, convex radius of curvature.
//! `sizing_findings` compares those with `h_surf` / `delta`.
use crate::bvh::Bvh;
use crate::dexel::Tri;
use crate::tree::par_map;
use std::collections::{HashMap, HashSet};

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

#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub enum Severity {
    Info,
    Warning,
    Error,
}

impl Severity {
    pub fn name(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Error => "error",
        }
    }
}

/// One finding: what, how many, where (up to 5 examples), why it matters and what to change.
#[derive(Clone, Debug)]
pub struct Finding {
    pub id: &'static str,
    pub severity: Severity,
    pub count: usize,
    pub title: String,
    pub examples: Vec<V3>,
    pub explain: String,
    pub suggestion: String,
}

pub fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push(' '),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

pub fn jpt(p: V3) -> String {
    format!("[{:.6e}, {:.6e}, {:.6e}]", p[0], p[1], p[2])
}

impl Finding {
    pub fn json(&self) -> String {
        let ex: Vec<String> = self.examples.iter().map(|&p| jpt(p)).collect();
        format!(
            "{{\"id\": {}, \"severity\": {}, \"count\": {}, \"title\": {}, \"examples\": [{}], \"explanation\": {}, \"suggestion\": {}}}",
            jstr(self.id),
            jstr(self.severity.name()),
            self.count,
            jstr(&self.title),
            ex.join(", "),
            jstr(&self.explain),
            jstr(&self.suggestion)
        )
    }
}

/// Result of the geometry analysis of the soup.
pub struct Geometry {
    pub n_tris: usize,
    pub bbox: [V3; 2],
    pub length: f64,
    pub area: f64,
    /// signed volume (positive = outward normals)
    pub volume: f64,
    pub n_parts: usize,
    pub n_sharp_edges: usize,
    pub sharp_length: f64,
    /// topology / validity findings
    pub findings: Vec<Finding>,
    /// per triangle: distance along the inward normal to the next surface (inf when none within `probe`)
    pub tri_thick: Vec<f64>,
    /// per triangle: distance along the outward normal to a surface facing it (inf when none within `probe`)
    pub tri_gap: Vec<f64>,
    /// per triangle: smallest convex radius of curvature at its vertices (inf = none)
    pub tri_r: Vec<f64>,
    pub tri_area: Vec<f64>,
    pub tri_part: Vec<u32>,
    pub probe: f64,
    pub feature_angle: f64,
}

impl Geometry {
    pub fn has_errors(&self) -> bool {
        self.findings.iter().any(|f| f.severity == Severity::Error)
    }
}

fn centroid(t: &Tri) -> V3 {
    [0, 1, 2].map(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0)
}

fn tri_normal(t: &Tri) -> V3 {
    cross(sub(t[1], t[0]), sub(t[2], t[0]))
}

/// Up to `n` examples with the highest score, mutually at least `sep` apart.
pub fn pick_examples(mut c: Vec<(f64, V3)>, sep: f64, n: usize) -> Vec<V3> {
    c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<V3> = Vec::new();
    for (_, p) in c {
        if out.len() >= n {
            break;
        }
        if out.iter().all(|&q| norm(sub(p, q)) >= sep) {
            out.push(p);
        }
    }
    out
}

fn find(id: &'static str, severity: Severity, count: usize, title: String, examples: Vec<V3>, explain: &str, suggestion: &str) -> Finding {
    Finding { id, severity, count, title, examples, explain: explain.to_string(), suggestion: suggestion.to_string() }
}

/// Analyse the soup. `feature_angle` (degrees): edges with a larger dihedral are counted as sharp features.
pub fn analyze(tris: &[Tri], feature_angle: f64) -> Geometry {
    let nt = tris.len();
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in tris.iter().flatten() {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let length = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
    let q = (1e-8 * length).max(1e-300);
    // weld
    let mut pos: Vec<V3> = Vec::new();
    let mut idx: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut vid: Vec<[u32; 3]> = Vec::with_capacity(nt);
    for t in tris {
        let mut v = [0u32; 3];
        for (k, p) in t.iter().enumerate() {
            let key = ((p[0] / q).round() as i64, (p[1] / q).round() as i64, (p[2] / q).round() as i64);
            v[k] = *idx.entry(key).or_insert_with(|| {
                pos.push(*p);
                (pos.len() - 1) as u32
            });
        }
        vid.push(v);
    }
    let tri_area: Vec<f64> = tris.iter().map(|t| 0.5 * norm(tri_normal(t))).collect();
    let area: f64 = tri_area.iter().sum();
    let mut findings: Vec<Finding> = Vec::new();

    // degenerate (collapsed or zero-area) and duplicate triangles
    let tiny = 1e-14 * length * length;
    let (mut collapsed, mut zero_area) = (vec![false; nt], Vec::new());
    for i in 0..nt {
        let v = vid[i];
        if v[0] == v[1] || v[1] == v[2] || v[0] == v[2] {
            collapsed[i] = true;
        } else if tri_area[i] <= tiny {
            zero_area.push(i);
        }
    }
    let ncol = collapsed.iter().filter(|&&c| c).count();
    if ncol + zero_area.len() > 0 {
        let cand: Vec<(f64, V3)> = (0..nt).filter(|&i| collapsed[i] || tri_area[i] <= tiny).map(|i| (1.0, centroid(&tris[i]))).collect();
        findings.push(find(
            "degenerate_triangles",
            Severity::Warning,
            ncol + zero_area.len(),
            format!("{} degenerate triangles ({} with coincident vertices, {} with zero area)", ncol + zero_area.len(), ncol, zero_area.len()),
            pick_examples(cand, 1e-3 * length, 5),
            "Zero-area triangles have no normal; they can break the inside/outside test and the feature detection locally.",
            "Remove them in your CAD / mesh tool (merge vertices, delete zero-area facets) and re-export the STL.",
        ));
    }
    let mut seen: HashSet<[u32; 3]> = HashSet::new();
    let mut dup = vec![false; nt];
    for i in 0..nt {
        if collapsed[i] {
            continue;
        }
        let mut k = vid[i];
        k.sort_unstable();
        if !seen.insert(k) {
            dup[i] = true;
        }
    }
    let ndup = dup.iter().filter(|&&d| d).count();
    if ndup > 0 {
        let cand: Vec<(f64, V3)> = (0..nt).filter(|&i| dup[i]).map(|i| (1.0, centroid(&tris[i]))).collect();
        findings.push(find(
            "duplicate_triangles",
            Severity::Warning,
            ndup,
            format!("{ndup} duplicate triangles (same three vertices as another triangle)"),
            pick_examples(cand, 1e-3 * length, 5),
            "Duplicated facets double-count the surface (the winding number of the region is wrong there) and create non-manifold edges.",
            "Delete the duplicated facets (e.g. remove duplicate faces in MeshLab / Blender) and re-export.",
        ));
    }

    // edge analysis on the usable triangles (not collapsed, not duplicate)
    let usable: Vec<bool> = (0..nt).map(|i| !collapsed[i] && !dup[i]).collect();
    // value: (forward uses, backward uses, first tri, second tri)
    let mut edges: HashMap<(u32, u32), (u8, u8, u32, u32)> = HashMap::new();
    for i in 0..nt {
        if !usable[i] {
            continue;
        }
        for e in 0..3 {
            let (a, b) = (vid[i][e], vid[i][(e + 1) % 3]);
            let ent = edges.entry((a.min(b), a.max(b))).or_insert((0, 0, i as u32, u32::MAX));
            if a < b {
                ent.0 += 1;
            } else {
                ent.1 += 1;
            }
            if ent.2 != i as u32 && ent.3 == u32::MAX {
                ent.3 = i as u32;
            }
        }
    }
    let mid = |a: u32, b: u32| mul(add(pos[a as usize], pos[b as usize]), 0.5);
    let (mut open, mut nonman, mut incons) = (Vec::new(), Vec::new(), Vec::new());
    for (&(a, b), &(f, r, _, _)) in &edges {
        let n = f + r;
        let el = norm(sub(pos[a as usize], pos[b as usize]));
        if n == 1 {
            open.push((el, mid(a, b)));
        } else if n > 2 {
            nonman.push((n as f64, mid(a, b)));
        } else if f == 2 || r == 2 {
            incons.push((el, mid(a, b)));
        }
    }
    let sep = 0.02 * length;
    if !open.is_empty() {
        let tot: f64 = open.iter().map(|x| x.0).sum();
        findings.push(find(
            "open_edges",
            Severity::Error,
            open.len(),
            format!("open surface: {} boundary edges used by a single triangle (total length {:.3e})", open.len(), tot),
            pick_examples(open.clone(), sep, 5),
            "The surface has holes (or gaps between patches): there is no well-defined inside, so the fluid region cannot be cut reliably.",
            "Close the holes / stitch the gaps (tolerance-based vertex merge, hole filling) so that every edge is shared by exactly two triangles.",
        ));
    }
    if !nonman.is_empty() {
        findings.push(find(
            "nonmanifold_edges",
            Severity::Error,
            nonman.len(),
            format!("{} non-manifold edges (shared by more than two triangles)", nonman.len()),
            pick_examples(nonman, sep, 5),
            "Several sheets touch along an edge (or surfaces are glued together): the enclosed volume is ambiguous.",
            "Separate the touching parts (small gap) or merge them into one watertight solid with a boolean union, then re-export.",
        ));
    }
    if !incons.is_empty() {
        let cand: Vec<(f64, V3)> = incons.clone();
        findings.push(find(
            "inconsistent_orientation",
            Severity::Error,
            incons.len(),
            format!("inconsistent orientation: {} edges where neighbouring triangles have opposite winding", incons.len()),
            pick_examples(cand, sep, 5),
            "Some triangles are flipped relative to their neighbours, so their normals point into the body; the inside/outside test is wrong around them.",
            "Re-orient the facets consistently (unify normals / 'recalculate outside' in your mesh tool) and re-export.",
        ));
    }

    // signed volume
    // about the bbox centre for accuracy
    let volume = {
        let c = mul(add(lo, hi), 0.5);
        let mut v = 0.0;
        for (i, t) in tris.iter().enumerate() {
            if usable[i] {
                v += dot(sub(t[0], c), cross(sub(t[1], c), sub(t[2], c))) / 6.0;
            }
        }
        v
    };
    if open.is_empty() && incons.is_empty() && volume < 0.0 {
        findings.push(find(
            "inward_normals",
            Severity::Error,
            nt,
            format!("negative enclosed volume ({:.4e}): the triangle normals point into the body", volume),
            vec![],
            "The surface is closed and consistent but inside-out, so mantle would mesh the interior of the body instead of the fluid around it.",
            "Flip all normals (reverse the winding of every triangle) and re-export.",
        ));
    }

    // parts (vertex-connected components)
    let mut parent: Vec<u32> = (0..pos.len() as u32).collect();
    fn root(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for i in 0..nt {
        if collapsed[i] {
            continue;
        }
        let (a, b, c) = (root(&mut parent, vid[i][0]), root(&mut parent, vid[i][1]), root(&mut parent, vid[i][2]));
        parent[b as usize] = a;
        parent[c as usize] = a;
    }
    let mut part_of: HashMap<u32, u32> = HashMap::new();
    let mut tri_part = vec![0u32; nt];
    let mut part_n: Vec<usize> = Vec::new();
    for i in 0..nt {
        let r = root(&mut parent, vid[i][0]);
        let np = part_of.len() as u32;
        let id = *part_of.entry(r).or_insert(np);
        tri_part[i] = id;
        if id as usize >= part_n.len() {
            part_n.push(0);
        }
        part_n[id as usize] += 1;
    }
    let n_parts = part_n.len();
    if n_parts > 1 {
        let mut s = part_n.clone();
        s.sort_unstable_by(|a, b| b.cmp(a));
        let list: Vec<String> = s.iter().take(6).map(|x| x.to_string()).collect();
        findings.push(find(
            "multiple_parts",
            Severity::Info,
            n_parts,
            format!("{n_parts} separate parts (triangles per part: {}{})", list.join(", "), if s.len() > 6 { ", ..." } else { "" }),
            vec![],
            "Disconnected closed surfaces are meshed together as one fluid domain; stray tiny parts are usually leftovers.",
            "If a part is not meant to be there (dust shells, stray facets), delete it; otherwise ignore.",
        ));
    }

    // normals, sharp edges, vertex normals / curvature
    let nrm: Vec<V3> = tris.iter().map(|t| {
        let n = tri_normal(t);
        let l = norm(n);
        if l > 0.0 { mul(n, 1.0 / l) } else { [0.0; 3] }
    }).collect();
    let sharp_cos = (feature_angle.max(1.0)).to_radians().cos();
    let curv_sharp_cos = 40f64.to_radians().cos();
    let (mut sharp_cand, mut n_sharp, mut sharp_len) = (Vec::new(), 0usize, 0.0);
    let mut sharp_vert = vec![false; pos.len()];
    for (&(a, b), &(f, r, t1, t2)) in &edges {
        if f + r != 2 || f == 2 || r == 2 || t2 == u32::MAX {
            continue;
        }
        let c = dot(nrm[t1 as usize], nrm[t2 as usize]);
        let el = norm(sub(pos[a as usize], pos[b as usize]));
        if c < sharp_cos {
            n_sharp += 1;
            sharp_len += el;
            sharp_cand.push((el, mid(a, b)));
        }
        if c < curv_sharp_cos {
            sharp_vert[a as usize] = true;
            sharp_vert[b as usize] = true;
        }
    }
    if n_sharp > 0 {
        findings.push(find(
            "sharp_edges",
            Severity::Info,
            n_sharp,
            format!("{n_sharp} sharp feature edges (dihedral > {feature_angle:.0} deg, total length {sharp_len:.3e})"),
            pick_examples(sharp_cand, sep, 5),
            "Sharp edges (wing trailing edges, box corners) get wall-node fans in the layers; convex ones are where layer cells are most distorted.",
            "Nothing to do; round the edge in CAD if the sharp corner is not physical.",
        ));
    }
    // angle-weighted vertex normals
    let mut vn = vec![[0.0f64; 3]; pos.len()];
    for i in 0..nt {
        if collapsed[i] || tri_area[i] <= tiny {
            continue;
        }
        let t = &tris[i];
        for k in 0..3 {
            let (p, a, b) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
            let (u, w) = (sub(a, p), sub(b, p));
            let ang = (dot(u, w) / (norm(u) * norm(w)).max(1e-300)).clamp(-1.0, 1.0).acos();
            let v = &mut vn[vid[i][k] as usize];
            *v = add(*v, mul(nrm[i], ang));
        }
    }
    for v in vn.iter_mut() {
        let l = norm(*v);
        if l > 0.0 {
            *v = mul(*v, 1.0 / l);
        }
    }
    let mut vr = vec![f64::INFINITY; pos.len()];
    let mut curv_pts: Vec<(f64, V3)> = Vec::new();
    for (&(a, b), &(f, r, _, t2)) in &edges {
        if f + r != 2 || f == 2 || r == 2 || t2 == u32::MAX || sharp_vert[a as usize] || sharp_vert[b as usize] {
            continue;
        }
        let d = sub(pos[b as usize], pos[a as usize]);
        let k = dot(sub(vn[b as usize], vn[a as usize]), d) / dot(d, d).max(1e-300);
        if k > 1e-12 {
            let rad = 1.0 / k;
            vr[a as usize] = vr[a as usize].min(rad);
            vr[b as usize] = vr[b as usize].min(rad);
            curv_pts.push((rad, mid(a, b)));
        }
    }
    let tri_r: Vec<f64> = (0..nt).map(|i| vid[i].iter().map(|&v| vr[v as usize]).fold(f64::INFINITY, f64::min)).collect();
    let _ = curv_pts;

    // rays: inward thickness, outward facing gap; self-intersections
    let bvh = Bvh::new(tris);
    let probe = 0.25 * length;
    let idxs: Vec<u32> = (0..nt as u32).collect();
    let rays: Vec<[f64; 2]> = par_map(&idxs, |&i| {
        let i = i as usize;
        let n = nrm[i];
        if collapsed[i] || tri_area[i] <= tiny {
            return [f64::INFINITY; 2];
        }
        let c = centroid(&tris[i]);
        let mut out = [f64::INFINITY; 2];
        // inward
        let b = add(c, mul(n, -probe));
        for (t, id) in bvh.segment_hits(c, b) {
            if id as usize != i && t * probe > 1e-9 * length {
                out[0] = t * probe;
                break;
            }
        }
        let b = add(c, mul(n, probe));
        for (t, id) in bvh.segment_hits(c, b) {
            if id as usize != i && t * probe > 1e-9 * length && dot(nrm[id as usize], n) < -0.5 {
                out[1] = t * probe;
                break;
            }
        }
        out
    });
    let tri_thick: Vec<f64> = rays.iter().map(|r| r[0]).collect();
    let tri_gap: Vec<f64> = rays.iter().map(|r| r[1]).collect();

    let hits: Vec<Vec<u32>> = par_map(&idxs, |&i| {
        let i = i as usize;
        let mut out = Vec::new();
        if collapsed[i] {
            return out;
        }
        let t = &tris[i];
        for e in 0..3 {
            let (a, b) = (t[e], t[(e + 1) % 3]);
            let len = norm(sub(b, a));
            if len <= 0.0 {
                continue;
            }
            for (s, id) in bvh.segment_hits(a, b) {
                let u = id as usize;
                if u == i || s < 1e-7 || s > 1.0 - 1e-7 || collapsed[u] {
                    continue;
                }
                if vid[u].iter().any(|v| vid[i].contains(v)) {
                    continue;
                }
                out.push(id);
            }
        }
        out
    });
    let mut pairs: HashSet<(u32, u32)> = HashSet::new();
    let (mut same, mut cross_part) = (0usize, 0usize);
    let (mut ex_same, mut ex_cross): (Vec<(f64, V3)>, Vec<(f64, V3)>) = (Vec::new(), Vec::new());
    for (i, h) in hits.iter().enumerate() {
        for &u in h {
            let key = (i.min(u as usize) as u32, i.max(u as usize) as u32);
            if pairs.insert(key) {
                let p = centroid(&tris[i]);
                if tri_part[i] == tri_part[u as usize] {
                    same += 1;
                    ex_same.push((tri_area[i], p));
                } else {
                    cross_part += 1;
                    ex_cross.push((tri_area[i], p));
                }
            }
        }
    }
    if cross_part > 0 {
        findings.push(find(
            "overlapping_parts",
            Severity::Warning,
            cross_part,
            format!("overlapping parts: {cross_part} triangle pairs of different parts intersect"),
            pick_examples(ex_cross, sep, 5),
            "Two closed parts penetrate each other (e.g. a wing pushed into a fuselage without a boolean union). The fluid region is their union, but the intersection curve is not a mesh feature, so cells and layers there are distorted or missing.",
            "Make a boolean union of the parts in CAD (or leave a clear gap between them) and re-export a single watertight surface.",
        ));
    }
    if same > 0 {
        findings.push(find(
            "self_intersections",
            Severity::Warning,
            same,
            format!("self-intersections: {same} triangle pairs of the same part intersect"),
            pick_examples(ex_same, sep, 5),
            "The surface passes through itself (folded or overlapping facets), which makes the local inside/outside test and the surface normals unreliable.",
            "Repair the surface (remove folds / self-intersections in your mesh tool) and re-export.",
        ));
    }

    Geometry {
        n_tris: nt,
        bbox: [lo, hi],
        length,
        area,
        volume,
        n_parts,
        n_sharp_edges: n_sharp,
        sharp_length: sharp_len,
        findings,
        tri_thick,
        tri_gap,
        tri_r,
        tri_area,
        tri_part,
        probe,
        feature_angle,
    }
}

fn sig1(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return x;
    }
    let e = 10f64.powf(x.log10().floor());
    (x / e).round() * e
}

/// Under-resolution warnings of the chosen sizing against the geometry.
pub fn sizing_findings(g: &Geometry, tris: &[Tri], h_surf: f64, delta: f64) -> Vec<Finding> {
    let mut out = Vec::new();
    let nt = g.n_tris;
    let sep = 0.02 * g.length;
    // 1. convex radius of curvature vs surface cell size
    let small: Vec<usize> = (0..nt).filter(|&i| g.tri_r[i] < h_surf).collect();
    if small.len() >= 3 {
        let rmin = small.iter().map(|&i| g.tri_r[i]).fold(f64::INFINITY, f64::min);
        let cand: Vec<(f64, V3)> = small.iter().map(|&i| (1.0 / g.tri_r[i], centroid(&tris[i]))).collect();
        let sug = sig1(rmin / 1.5);
        let fine = g.length / 80.0;
        let preset = if fine <= sug * 1.2 { " (or --preset fine)" } else { "" };
        out.push(find(
            "curvature_vs_hsurf",
            Severity::Warning,
            small.len(),
            format!(
                "Convex radius of curvature down to {:.3e} (e.g. a leading-edge nose) is smaller than the surface cell size {:.3e}: layers and cells there are poorly resolved ({} triangles affected)",
                rmin, h_surf, small.len()
            ),
            pick_examples(cand, sep, 5),
            "Fewer than about one surface cell per radius cannot follow the round edge; the shell and layer columns bunch up there and the first cells get skewed.",
            &format!("Use --h-surf {sug:.3e}{preset}, or accept it if the edge region is not critical (note: refining raises the cell count roughly with 1/h^2)."),
        ));
    } else {
        let tight: Vec<usize> = (0..nt).filter(|&i| g.tri_r[i] < 2.0 * h_surf).collect();
        if tight.len() >= 3 {
            let rmin = tight.iter().map(|&i| g.tri_r[i]).fold(f64::INFINITY, f64::min);
            let cand: Vec<(f64, V3)> = tight.iter().map(|&i| (1.0 / g.tri_r[i], centroid(&tris[i]))).collect();
            out.push(find(
                "curvature_vs_hsurf",
                Severity::Info,
                tight.len(),
                format!("Convex radius of curvature down to {:.3e} is only {:.1}x the surface cell size {:.3e}", rmin, rmin / h_surf, h_surf),
                pick_examples(cand, sep, 5),
                "About two surface cells per radius: acceptable for most cases, but the curved region is resolved coarsely.",
                "Use a finer --h-surf if the flow around this edge matters.",
            ));
        }
    }
    // 2. thin regions: body thickness well below the layer thickness (a layer stack several times thicker than the body: the shell
    // forms a fat cap around the edge, e.g. a sharp trailing edge). The offset shell lies outside the body, so a body thinner than
    // 2 delta alone is no problem; only the part where t < delta / 4 is flagged.
    let thin: Vec<usize> = (0..nt).filter(|&i| g.tri_thick[i] < 0.25 * delta).collect();
    if !thin.is_empty() {
        let a_thin: f64 = thin.iter().map(|&i| g.tri_area[i]).sum();
        let frac = a_thin / g.area.max(1e-300);
        let tmin = thin.iter().map(|&i| g.tri_thick[i]).fold(f64::INFINITY, f64::min);
        let cand: Vec<(f64, V3)> = thin.iter().map(|&i| (1.0 / g.tri_thick[i].max(1e-300), centroid(&tris[i]))).collect();
        let sev = if frac > 0.002 { Severity::Warning } else { Severity::Info };
        out.push(find(
            "thin_regions",
            sev,
            thin.len(),
            format!(
                "Thin regions (e.g. trailing edges, thin plates): body thickness down to {:.3e}, below a quarter of the layer thickness delta = {:.3e}, over {:.1}% of the surface ({} triangles)",
                tmin, delta, 100.0 * frac, thin.len()
            ),
            pick_examples(cand, sep, 5),
            "The layer stack there is several times thicker than the body itself, so the layers from the two sides meet at the thin edge and its cells are squeezed or sheared; this is where the worst cells of thin wings usually are.",
            &format!("Reduce --delta (the thinnest part you want layered is about delta/4; try {:.2e}) or use fewer / thinner layers (--dy0, --layers); a sharp trailing edge is always thinner than delta near its tip, so a small remainder is expected.", sig1(0.5 * delta)),
        ));
    }
    // 3. narrow gaps between facing surfaces vs 2 delta
    let gap: Vec<usize> = (0..nt).filter(|&i| g.tri_gap[i] < 2.0 * delta * 0.98).collect();
    if !gap.is_empty() {
        let gmin = gap.iter().map(|&i| g.tri_gap[i]).fold(f64::INFINITY, f64::min);
        let cand: Vec<(f64, V3)> = gap.iter().map(|&i| (1.0 / g.tri_gap[i].max(1e-300), centroid(&tris[i]))).collect();
        out.push(find(
            "narrow_gaps",
            Severity::Warning,
            gap.len(),
            format!("Narrow gaps: facing surfaces only {:.3e} apart, below 2 x delta = {:.3e} ({} triangles)", gmin, 2.0 * delta, gap.len()),
            pick_examples(cand, sep, 5),
            "The layers of the two facing walls (delta each) would overlap in the gap, and the core cannot place cells between them.",
            &format!("Reduce --delta to below {:.2e} (half the gap), widen the gap in CAD, or fill it if it is not physical.", sig1(0.45 * gmin)),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(lo: V3, hi: V3, flip: bool) -> Vec<Tri> {
        let c = |i: usize| [if i & 1 == 0 { lo[0] } else { hi[0] }, if i & 2 == 0 { lo[1] } else { hi[1] }, if i & 4 == 0 { lo[2] } else { hi[2] }];
        // outward-oriented quads (corner bit order x, y, z)
        let q = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
        let mut t = Vec::new();
        for f in q {
            let (a, b, cc, d) = (c(f[0]), c(f[1]), c(f[2]), c(f[3]));
            if flip {
                t.push([a, cc, b]);
                t.push([a, d, cc]);
            } else {
                t.push([a, b, cc]);
                t.push([a, cc, d]);
            }
        }
        t
    }

    fn ids(g: &Geometry) -> Vec<&'static str> {
        g.findings.iter().map(|f| f.id).collect()
    }

    #[test]
    fn clean_box_has_no_errors() {
        let g = analyze(&cube([0.0; 3], [1.0, 0.6, 0.4], false), 25.0);
        assert!(!g.has_errors() && g.findings.iter().all(|f| f.severity == Severity::Info), "{:?}", ids(&g));
        assert!((g.volume - 0.24).abs() < 1e-12 && g.n_parts == 1 && g.n_sharp_edges == 12);
        assert!(g.tri_thick.iter().all(|&t| t > 0.39));
    }

    #[test]
    fn broken_boxes_are_errors() {
        let mut t = cube([0.0; 3], [1.0; 3], false);
        let open = {
            t.remove(0);
            analyze(&t, 25.0)
        };
        assert!(ids(&open).contains(&"open_edges") && open.has_errors());
        let mut t = cube([0.0; 3], [1.0; 3], false);
        t[0].swap(1, 2);
        let g = analyze(&t, 25.0);
        assert!(ids(&g).contains(&"inconsistent_orientation") && g.has_errors());
        let g = analyze(&cube([0.0; 3], [1.0; 3], true), 25.0);
        assert!(ids(&g).contains(&"inward_normals") && g.volume < 0.0);
    }

    #[test]
    fn duplicates_and_overlap() {
        let mut t = cube([0.0; 3], [1.0; 3], false);
        t.push(t[3]);
        assert!(ids(&analyze(&t, 25.0)).contains(&"duplicate_triangles"));
        let mut t = cube([0.0; 3], [1.0; 3], false);
        t.extend(cube([0.5; 3], [1.5; 3], false));
        let g = analyze(&t, 25.0);
        assert!(g.n_parts == 2 && ids(&g).contains(&"overlapping_parts"));
        let mut t = cube([0.0; 3], [1.0; 3], false);
        t.extend(cube([3.0, 0.0, 0.0], [4.0, 1.0, 1.0], false));
        assert!(!ids(&analyze(&t, 25.0)).contains(&"overlapping_parts"));
    }

    #[test]
    fn gap_and_thin_warnings() {
        // two slabs 0.05 apart facing each other; delta = 0.1
        let mut t = cube([0.0, 0.0, 0.0], [1.0, 1.0, 0.5], false);
        t.extend(cube([0.0, 0.0, 0.55], [1.0, 1.0, 1.05], false));
        let g = analyze(&t, 25.0);
        let f = sizing_findings(&g, &t, 0.05, 0.1);
        assert!(f.iter().any(|x| x.id == "narrow_gaps" && x.severity == Severity::Warning), "{:?}", f.iter().map(|x| x.id).collect::<Vec<_>>());
        assert!(sizing_findings(&g, &t, 0.05, 0.01).iter().all(|x| x.id != "narrow_gaps"));
        // a thin plate (t = 0.01) against delta 0.2
        let t = cube([0.0; 3], [1.0, 1.0, 0.01], false);
        let g = analyze(&t, 25.0);
        assert!(sizing_findings(&g, &t, 0.1, 0.2).iter().any(|x| x.id == "thin_regions" && x.severity == Severity::Warning));
    }
}
