//! Localised mesh issue report: bad cells (any failing OpenFOAM-style check, see `quality`, plus non-positive layer cells) are grouped
//! into spatial clusters; each cluster gets centroid / extent, the failing checks, the stage that built the cells (core / pillow / layer k),
//! the nearby geometric feature (sharp convex edge, concave crease, high curvature, thin region), a likely cause and a suggestion.
//! Also assembles `mantle_report.md` and the JSON fields, and writes the OpenFOAM cellSet `mantleBadCells`.
use crate::bvh::Bvh;
use crate::checks::{self, jpt, jstr, Finding, Geometry, Severity};
use crate::cutmesh::CutMesh;
use crate::dexel::Tri;
use crate::features::{FeatureOptions, Features, Kind};
use crate::quality::{self, Metrics, FLAG_NAMES, F_JACOBIAN, F_NEG_VOL, F_PYRAMID};
use std::collections::HashMap;
use std::path::Path;

type V3 = [f64; 3];
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn norm(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Stage of a cell.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Stage {
    Core,
    Pillow,
    Layer,
}

#[derive(Clone, Debug)]
pub struct Cluster {
    pub severity: Severity,
    pub n_cells: usize,
    pub centroid: V3,
    pub lo: V3,
    pub hi: V3,
    /// (check name, number of cells failing it)
    pub checks: Vec<(&'static str, usize)>,
    pub n_core: usize,
    pub n_pillow: usize,
    pub n_layer: usize,
    /// first / last layer index (1-based, from the wall) of the layer cells
    pub layers: Option<(usize, usize)>,
    pub max_skew: f64,
    pub max_nonorth: f64,
    pub max_aspect: f64,
    pub min_vol: f64,
    /// extra layer cells with non-positive Jacobian known only from the layer statistics (prisms / pyramids)
    pub n_extra: usize,
    pub wall_dist: f64,
    pub feature: String,
    pub cause: String,
    pub suggestion: String,
    pub rank: f64,
}

#[derive(Clone, Debug, Default)]
pub struct MeshReport {
    pub metrics: Metrics,
    pub clusters: Vec<Cluster>,
    pub bad_cells: Vec<u32>,
    pub n_core: usize,
    pub n_pillow: usize,
    pub n_layer: usize,
    /// bad cells per stage
    pub bad_stage: [usize; 3],
}

/// Everything the localisation needs besides the mesh.
pub struct Locate<'a> {
    pub tris: &'a [Tri],
    pub geom: Option<&'a Geometry>,
    pub h_surf: f64,
    pub delta: f64,
    pub nlayers: usize,
    pub growth: f64,
    pub n_core: usize,
    pub n_pillow: usize,
    /// (wall foot, layer index, scaled Jacobian) of non-positive layer cells from the layer statistics
    pub layer_bad: &'a [(V3, usize, f64)],
}

pub fn stage_of(c: usize, n_core: usize, n_pillow: usize) -> Stage {
    if c < n_core {
        Stage::Core
    } else if c < n_core + n_pillow {
        Stage::Pillow
    } else {
        Stage::Layer
    }
}

/// Layer index (1-based) of a point at wall distance `d` in a geometric stack of `n` layers filling `delta` with growth `g`.
fn layer_index(d: f64, delta: f64, n: usize, g: f64) -> usize {
    let f = (d / delta).clamp(0.0, 1.0);
    let cum = |j: usize| if (g - 1.0).abs() < 1e-9 { j as f64 / n as f64 } else { (g.powi(j as i32) - 1.0) / (g.powi(n as i32) - 1.0) };
    (1..=n).find(|&j| f <= cum(j)).unwrap_or(n)
}

struct Dsu(Vec<u32>);
impl Dsu {
    fn find(&mut self, mut x: u32) -> u32 {
        while self.0[x as usize] != x {
            self.0[x as usize] = self.0[self.0[x as usize] as usize];
            x = self.0[x as usize];
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[b as usize] = a;
        }
    }
}

/// Compute the metrics, find and describe the issue clusters.
pub fn analyze_mesh(m: &CutMesh, lc: &Locate) -> MeshReport {
    let (metrics, cd) = quality::compute(m);
    let mut rep = MeshReport { metrics, ..Default::default() };
    let nc = m.cells.len();
    rep.n_core = lc.n_core.min(nc);
    rep.n_pillow = lc.n_pillow.min(nc - rep.n_core);
    rep.n_layer = nc - rep.n_core - rep.n_pillow;
    let bad: Vec<u32> = (0..nc as u32).filter(|&c| cd.flags[c as usize] != 0 && !quality::info_only(cd.flags[c as usize], cd.max_nonorth[c as usize] as f64)).collect();
    for &c in &bad {
        let s = stage_of(c as usize, lc.n_core, lc.n_pillow);
        rep.bad_stage[s as usize] += 1;
    }
    rep.bad_cells = bad.clone();
    // items: bad cells, plus layer-stat points with no flagged cell nearby
    let rl = 2.0 * lc.h_surf;
    let mut items: Vec<(V3, Option<u32>)> = bad.iter().map(|&c| (cd.centre[c as usize], Some(c))).collect();
    let n_cell_items = items.len();
    let mut extra_pts: Vec<(V3, usize)> = Vec::new();
    if !lc.layer_bad.is_empty() {
        // flagged cells with a Jacobian / volume failure already explain the point if one lies within 1.5 h_surf of its wall foot
        let hard: Vec<V3> = bad.iter().filter(|&&c| cd.flags[c as usize] & (F_NEG_VOL | F_JACOBIAN | F_PYRAMID) != 0).map(|&c| cd.centre[c as usize]).collect();
        for &(w, k, _) in lc.layer_bad {
            if !hard.iter().any(|&q| norm(sub(q, w)) < 1.5 * lc.h_surf + lc.delta) {
                extra_pts.push((w, k));
            }
        }
    }
    for &(w, _) in &extra_pts {
        items.push((w, None));
    }
    if items.is_empty() {
        return rep;
    }
    // spatial clustering: link items closer than rl (uniform grid hash)
    let key = |p: V3| ((p[0] / rl).floor() as i64, (p[1] / rl).floor() as i64, (p[2] / rl).floor() as i64);
    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    for (i, it) in items.iter().enumerate() {
        grid.entry(key(it.0)).or_default().push(i as u32);
    }
    let mut dsu = Dsu((0..items.len() as u32).collect());
    for (i, it) in items.iter().enumerate() {
        let (a, b, c) = key(it.0);
        for x in a - 1..=a + 1 {
            for y in b - 1..=b + 1 {
                for z in c - 1..=c + 1 {
                    if let Some(l) = grid.get(&(x, y, z)) {
                        for &j in l {
                            if (j as usize) > i && norm(sub(items[j as usize].0, it.0)) < rl {
                                dsu.union(i as u32, j);
                            }
                        }
                    }
                }
            }
        }
    }
    let mut groups: HashMap<u32, Vec<u32>> = HashMap::new();
    for i in 0..items.len() as u32 {
        let r = dsu.find(i);
        groups.entry(r).or_default().push(i);
    }
    // context for the description
    let bvh = Bvh::new(lc.tris);
    let owned;
    let geom: &Geometry = match lc.geom {
        Some(g) => g,
        None => {
            owned = checks::analyze(lc.tris, 25.0);
            &owned
        }
    };
    let feats = Features::detect(lc.tris, &FeatureOptions::default());
    let mut clusters: Vec<Cluster> = Vec::new();
    for (_, g) in groups {
        let cells: Vec<u32> = g.iter().filter(|&&i| (i as usize) < n_cell_items).map(|&i| items[i as usize].1.unwrap()).collect();
        let pts: Vec<usize> = g.iter().filter(|&&i| (i as usize) >= n_cell_items).map(|&i| i as usize - n_cell_items).collect();
        let mut cen = [0.0; 3];
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for &i in &g {
            let p = items[i as usize].0;
            for k in 0..3 {
                cen[k] += p[k] / g.len() as f64;
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let mut cl = Cluster {
            severity: Severity::Warning,
            n_cells: cells.len(),
            centroid: cen,
            lo,
            hi,
            checks: Vec::new(),
            n_core: 0,
            n_pillow: 0,
            n_layer: 0,
            layers: None,
            max_skew: 0.0,
            max_nonorth: 0.0,
            max_aspect: 0.0,
            min_vol: f64::INFINITY,
            n_extra: pts.len(),
            wall_dist: 0.0,
            feature: String::new(),
            cause: String::new(),
            suggestion: String::new(),
            rank: 0.0,
        };
        let mut cnt = [0usize; 6];
        let mut lrange: Option<(usize, usize)> = None;
        let mut hard = !pts.is_empty();
        for &c in &cells {
            let ci = c as usize;
            let f = cd.flags[ci];
            for (k, &(bit, _)) in FLAG_NAMES.iter().enumerate() {
                cnt[k] += (f & bit != 0) as usize;
            }
            hard |= f & (F_NEG_VOL | F_PYRAMID | F_JACOBIAN) != 0;
            cl.max_skew = cl.max_skew.max(cd.max_skew[ci] as f64);
            cl.max_nonorth = cl.max_nonorth.max(cd.max_nonorth[ci] as f64);
            cl.max_aspect = cl.max_aspect.max(cd.aspect[ci] as f64);
            cl.min_vol = cl.min_vol.min(cd.vol[ci]);
            match stage_of(ci, lc.n_core, lc.n_pillow) {
                Stage::Core => cl.n_core += 1,
                Stage::Pillow => cl.n_pillow += 1,
                Stage::Layer => {
                    cl.n_layer += 1;
                    let d = bvh.closest(cd.centre[ci]).0;
                    let k = layer_index(d, lc.delta, lc.nlayers.max(1), lc.growth);
                    lrange = Some(lrange.map_or((k, k), |(a, b)| (a.min(k), b.max(k))));
                }
            }
        }
        for &p in &pts {
            let k = extra_pts[p].1 + 1;
            cl.n_layer += 0;
            lrange = Some(lrange.map_or((k, k), |(a, b)| (a.min(k), b.max(k))));
        }
        cl.layers = lrange;
        for (k, &(_, name)) in FLAG_NAMES.iter().enumerate() {
            if cnt[k] > 0 {
                cl.checks.push((name, cnt[k]));
            }
        }
        if !pts.is_empty() {
            cl.checks.push(("non-positive layer Jacobian (prism / pyramid, from layer statistics)", pts.len()));
        }
        cl.severity = if hard { Severity::Error } else { Severity::Warning };
        if cl.min_vol == f64::INFINITY {
            cl.min_vol = 0.0;
        }
        describe(&mut cl, &bvh, &feats, geom, lc);
        let worst = (cl.max_skew / quality::SKEW_MAX).max(cl.max_nonorth / quality::NONORTH_MAX).max(if hard { 10.0 } else { 0.0 });
        cl.rank = (if hard { 1e9 } else { 0.0 }) + worst * 1e4 + (cl.n_cells + cl.n_extra) as f64;
        clusters.push(cl);
    }
    clusters.sort_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap_or(std::cmp::Ordering::Equal));
    rep.clusters = clusters;
    rep
}

fn describe(cl: &mut Cluster, bvh: &Bvh, feats: &Features, g: &Geometry, lc: &Locate) {
    let (d, w, tri) = bvh.closest(cl.centroid);
    cl.wall_dist = d;
    let ti = tri as usize;
    let reach = (2.0 * lc.delta).max(3.0 * lc.h_surf);
    let curve = feats.nearest_curve(w, reach);
    let thin = g.tri_thick.get(ti).copied().unwrap_or(f64::INFINITY);
    let rad = g.tri_r.get(ti).copied().unwrap_or(f64::INFINITY);
    let gap = g.tri_gap.get(ti).copied().unwrap_or(f64::INFINITY);
    let is_thin = thin < 0.25 * lc.delta;
    let is_gap = gap < 2.0 * lc.delta * 0.98;
    let is_curved = rad < lc.h_surf;
    let (mut convex, mut concave) = (false, false);
    let mut parts: Vec<String> = Vec::new();
    if let Some(h) = &curve {
        match feats.curves[h.curve].kind {
            Kind::Convex => {
                convex = true;
                parts.push(format!("sharp convex edge ({:.2e} from the wall foot)", h.dist));
            }
            Kind::Concave => {
                concave = true;
                parts.push(format!("concave crease ({:.2e} from the wall foot)", h.dist));
            }
            Kind::Mixed => {
                convex = true;
                concave = true;
                parts.push(format!("mixed convex / concave feature curve ({:.2e} away)", h.dist));
            }
        }
        let corners = feats.corners_near(w, reach);
        if let Some(&(_, dc)) = corners.first() {
            parts.push(format!("feature corner {dc:.2e} away"));
        }
    }
    if is_thin {
        parts.push(format!("thin region (body thickness {thin:.2e} < delta/4 = {:.2e})", 0.25 * lc.delta));
    }
    if is_gap {
        parts.push(format!("narrow gap to a facing surface ({gap:.2e} < 2 delta)"));
    }
    if is_curved {
        parts.push(format!("high curvature (radius {rad:.2e} < h_surf {:.2e})", lc.h_surf));
    }
    cl.feature = if parts.is_empty() { format!("no sharp feature within {reach:.2e} of the wall; nearest wall point at {:.2e}", cl.wall_dist) } else { parts.join("; ") };
    let layer_dominant = cl.n_layer + cl.n_extra > cl.n_core + cl.n_pillow;
    let (cause, sug): (String, String) = if is_gap {
        ("The layers of two facing walls overlap in this narrow gap, so the cells between them are squeezed.".into(), format!("Reduce --delta below {:.2e} or open the gap in CAD.", 0.45 * gap))
    } else if is_thin {
        (
            "The body is much thinner than the layer stack here (typically a trailing edge or thin plate tip): layers from the two sides meet at the edge and the cells are squeezed or sheared.".into(),
            format!("Reduce --delta (try {:.2e}) or use fewer layers; a sharp trailing edge always stays thinner than delta near its tip, so a small remainder is expected.", 0.5 * lc.delta),
        )
    } else if convex && layer_dominant {
        (
            "Layer columns fan out around a sharp convex edge; cells at the edge are strongly sheared (large skewness / non-orthogonality) or invert when the wedge angle is very small.".into(),
            "Reduce --delta or the number of layers, lower --growth, or round the edge in CAD; the shell resolution near the edge (--h-surf) also sets how well the fan is formed.".into(),
        )
    } else if concave && layer_dominant {
        (
            "Layer columns from the two flanks converge in a concave crease and pinch; cells there are skewed or squeezed.".into(),
            "Increase --fillet (shell fillet radius in units of delta) or reduce --delta; a refined --h-surf near the crease helps.".into(),
        )
    } else if is_curved {
        (
            "The radius of curvature here is below the surface cell size, so the shell and its cells cannot follow the surface (typical for a leading-edge nose).".into(),
            format!("Use a finer --h-surf (about {:.1e} or below) or --preset fine.", rad / 1.5),
        )
    } else if layer_dominant {
        (
            "No sharp feature or thin region was found nearby: the layer cells are likely distorted where the shell mesh changes (hanging / irregular shell vertices) or by the column smoothing.".into(),
            "Try a slightly smaller --delta or --growth, or a finer --h-surf; if it persists on smooth geometry, please report the case.".into(),
        )
    } else if convex || concave {
        (
            "Pillow / core cells sit between the hex core and the offset shell; the shell is strongly curved or folded at this feature, so the projected pillow cells are stretched and skewed.".into(),
            "Reduce --delta (the shell then follows the surface more closely) or use a finer --h-surf near the feature; round the edge in CAD if possible.".into(),
        )
    } else {
        (
            "Core / pillow cells at the shell: the projection of the core boundary onto the offset shell is locally irregular (strong size change or a fold of the shell).".into(),
            "Try a finer --h-surf or a smaller --delta; if it persists on smooth geometry, please report the case.".into(),
        )
    };
    cl.cause = cause;
    cl.suggestion = sug;
}

/// Write `constant/polyMesh/sets/mantleBadCells` (OpenFOAM cellSet; same cell numbering as the polyMesh).
pub fn write_cellset(dir: &Path, cells: &[u32]) -> std::io::Result<()> {
    let sets = dir.join("constant").join("polyMesh").join("sets");
    std::fs::create_dir_all(&sets)?;
    let mut s = String::from("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       cellSet;\n    location    \"constant/polyMesh/sets\";\n    object      mantleBadCells;\n}\n\n");
    s += &format!("{}\n(\n", cells.len());
    for c in cells {
        s += &format!("{c}\n");
    }
    s += ")\n";
    std::fs::write(sets.join("mantleBadCells"), s)
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Verdict {
    Ok,
    Warnings,
    Problems,
}

impl Verdict {
    pub fn text(self) -> &'static str {
        match self {
            Verdict::Ok => "OK",
            Verdict::Warnings => "OK with warnings",
            Verdict::Problems => "problems found",
        }
    }
}

/// Counts and parameters printed in the report.
pub struct Params {
    pub input: String,
    pub preset: String,
    pub h_surf: f64,
    pub delta: f64,
    pub dy0: f64,
    pub layers: usize,
    pub growth: f64,
    pub growth_used: [f64; 2],
    pub fillet: f64,
    pub level: u8,
    pub domain: f64,
    pub note: String,
    pub cells: usize,
    pub hex: usize,
    pub poly: usize,
    pub prism: usize,
    pub pyramid: usize,
    pub points: usize,
    pub core: usize,
    pub pillow: usize,
    pub layer_cells: usize,
    pub orphan_faces: usize,
    pub layer_nonpositive: usize,
    pub layer_sj_min: f64,
    pub su2_nonpositive: usize,
    pub first: [f64; 3],
    pub volume: f64,
}

pub struct Full<'a> {
    pub geom: &'a Geometry,
    pub sizing: &'a [Finding],
    pub params: &'a Params,
    pub mesh: &'a MeshReport,
}

pub fn verdict(f: &Full) -> Verdict {
    let m = &f.mesh.metrics;
    let p = f.params;
    if m.failed_checks > 0 || p.orphan_faces > 0 || p.layer_nonpositive > 0 || p.su2_nonpositive > 0 || f.geom.findings.iter().any(|x| x.severity == Severity::Error) {
        return Verdict::Problems;
    }
    if f.mesh.clusters.iter().any(|c| c.severity == Severity::Error) {
        return Verdict::Problems;
    }
    if f.geom.findings.iter().chain(f.sizing.iter()).any(|x| x.severity == Severity::Warning) || !f.mesh.clusters.is_empty() {
        return Verdict::Warnings;
    }
    Verdict::Ok
}

fn sev_tag(s: Severity) -> &'static str {
    match s {
        Severity::Info => "INFO",
        Severity::Warning => "WARNING",
        Severity::Error => "ERROR",
    }
}

fn p3(p: V3) -> String {
    format!("({:.4e}, {:.4e}, {:.4e})", p[0], p[1], p[2])
}

fn findings_md(fs: &[&Finding]) -> String {
    if fs.is_empty() {
        return "No findings.\n".into();
    }
    let mut s = String::new();
    for f in fs {
        s += &format!("- **{}** {}\n", sev_tag(f.severity), f.title);
        if !f.examples.is_empty() {
            s += &format!("  - where: {}\n", f.examples.iter().map(|&p| p3(p)).collect::<Vec<_>>().join(", "));
        }
        s += &format!("  - why: {}\n  - fix: {}\n", f.explain, f.suggestion);
    }
    s
}

/// Input-only report (for `--report-only`).
pub fn input_markdown(input: &str, g: &Geometry, sizing: Option<&[Finding]>) -> String {
    let mut fs: Vec<&Finding> = g.findings.iter().collect();
    if let Some(s) = sizing {
        fs.extend(s.iter());
    }
    fs.sort_by(|a, b| b.severity.cmp(&a.severity));
    let worst = fs.iter().map(|f| f.severity).max();
    let v = match worst {
        Some(Severity::Error) => "problems found: the input cannot be meshed reliably",
        Some(Severity::Warning) => "OK with warnings",
        _ => "OK",
    };
    let mut s = format!("# mantle input report\n\nInput: `{input}`\n\n## Summary\n\nVerdict: **{v}**\n\n## Input checks\n\n");
    s += &geom_table(g);
    s += &findings_md(&fs);
    s
}

fn geom_table(g: &Geometry) -> String {
    format!(
        "- triangles: {}\n- parts: {}\n- bbox: {} .. {} (length {:.4e})\n- surface area: {:.4e}, enclosed volume: {:.4e}\n\n",
        g.n_tris, g.n_parts, p3(g.bbox[0]), p3(g.bbox[1]), g.length, g.area, g.volume
    )
}

pub fn markdown(f: &Full) -> String {
    let (g, p, mr) = (f.geom, f.params, f.mesh);
    let m = &mr.metrics;
    let v = verdict(f);
    let mut s = format!("# mantle report\n\nInput: `{}`\n\n## Summary\n\nVerdict: **{}**\n\n", p.input, v.text());
    let ok = |b: bool| if b { "OK" } else { "FAIL" };
    s += &format!(
        "- mesh: {} cells ({} hex, {} polyhedra), {} points; core {} + pillow {} + layers {}\n- checkMesh-equivalent checks failed: {} (non-orthogonality, skewness, face pyramids, volumes, aspect ratio, closed cells)\n- bad cells: {} ({} core, {} pillow, {} layer) in {} issue clusters\n",
        p.cells, p.hex, p.poly, p.points, p.core, p.pillow, p.layer_cells, m.failed_checks, m.n_bad_cells, mr.bad_stage[0], mr.bad_stage[1], mr.bad_stage[2], mr.clusters.len()
    );
    if p.orphan_faces > 0 {
        s += &format!("- ERROR: {} orphan boundary faces (holes in the mesh; SU2 will reject it)\n", p.orphan_faces);
    }
    if p.su2_nonpositive > 0 {
        s += &format!("- WARNING: {} SU2 elements fail the validity check after decomposition\n", p.su2_nonpositive);
    }
    s += "\n## Input checks\n\n";
    s += &geom_table(g);
    let mut inp: Vec<&Finding> = g.findings.iter().collect();
    inp.sort_by(|a, b| b.severity.cmp(&a.severity));
    s += &findings_md(&inp);
    s += "\n## Sizing\n\n";
    s += &format!(
        "- preset {}; h_surf {:.4e}, delta {:.4e} ({:.1} h_surf), layers {}, growth {} (used {:.3}..{:.3}), first cell {:.4e} (min {:.4e}, max {:.4e}), fillet {}\n- octree level {}, domain side {:.4e}\n",
        p.preset, p.h_surf, p.delta, p.delta / p.h_surf, p.layers, p.growth, p.growth_used[0], p.growth_used[1], p.dy0, p.first[0], p.first[2], p.fillet, p.level, p.domain
    );
    if !p.note.is_empty() {
        s += &format!("- note: {}\n", p.note);
    }
    s += "\nUnder-resolution checks against this sizing:\n\n";
    let mut sz: Vec<&Finding> = f.sizing.iter().collect();
    sz.sort_by(|a, b| b.severity.cmp(&a.severity));
    s += &findings_md(&sz);
    s += "\n## Mesh statistics\n\n";
    s += &format!(
        "| item | value |\n|---|---|\n| cells | {} |\n| hex / polyhedra | {} / {} |\n| SU2 prisms / pyramids (layers) | {} / {} |\n| points | {} |\n| faces (internal) | {} ({}) |\n| core / pillow / layer cells | {} / {} / {} |\n| fluid volume | {:.6e} |\n| min / max cell volume | {:.4e} / {:.4e} |\n| min / max face area | {:.4e} / {:.4e} |\n| layer min scaled Jacobian | {:.3} |\n\n",
        p.cells, p.hex, p.poly, p.prism, p.pyramid, p.points, m.n_faces, m.n_internal, mr.n_core, mr.n_pillow, mr.n_layer, p.volume, m.vol_min, m.vol_max, m.area_min, m.area_max, p.layer_sj_min
    );
    s += "## Quality\n\nOpenFOAM definitions (as in checkMesh), computed by mantle on the final mesh:\n\n";
    s += &format!(
        "| check | value | limit | status |\n|---|---|---|---|\n| max non-orthogonality | {:.2} deg (avg {:.2}) | 90 (checkMesh: > 70 is 'severe') | {} |\n| faces > 70 deg (severe) | {} | - | {} |\n| faces >= 90 deg (error) | {} | 0 | {} |\n| faces > 60 deg | {} | - | - |\n| max skewness | {:.3} | 4 | {} |\n| highly skew faces (> 4) | {} | 0 | {} |\n| face-pyramid orientation errors | {} | 0 | {} |\n| non-positive volume cells | {} | 0 | {} |\n| max aspect ratio | {:.2} | 1000 | {} |\n| closed-cell residual (max) | {:.2e} | 1e-9 | {} |\n| hex cells with non-positive corner Jacobian | {} | 0 | {} |\n\n",
        m.non_orth_max, m.non_orth_avg, ok(m.n_non_orth_ge90 == 0), m.n_non_orth_gt70, if m.n_non_orth_gt70 == 0 { "OK" } else { "note: accepted by checkMesh, use nNonOrthogonalCorrectors >= 2" }, m.n_non_orth_ge90, ok(m.n_non_orth_ge90 == 0), m.n_non_orth_gt60, m.skew_max, ok(m.skew_max <= 4.0), m.n_skew_gt4, ok(m.n_skew_gt4 == 0), m.n_pyramid_err, ok(m.n_pyramid_err == 0), m.n_neg_vol, ok(m.n_neg_vol == 0), m.aspect_max, ok(m.aspect_max <= 1000.0), m.closed_max, ok(m.n_unclosed == 0), m.n_jacobian, ok(m.n_jacobian == 0)
    );
    s += "## Issues\n\n";
    if mr.clusters.is_empty() {
        s += "No bad cells: no cell fails any check.\n\n";
    } else {
        s += &format!("{} bad cells in {} clusters, sorted by severity. In OpenFOAM, open `constant/polyMesh/sets/mantleBadCells` (cellSet of all bad cells): `foamToVTK -cellSet mantleBadCells` then load `VTK/*/mantleBadCells.vtk` in ParaView, or `cellSet`-aware tools such as `topoSet`/`subsetMesh`.\n\n", m.n_bad_cells, mr.clusters.len());
        for (i, c) in mr.clusters.iter().enumerate() {
            s += &cluster_md(i + 1, c);
        }
    }
    s += "## Suggestions\n\n";
    let mut sug: Vec<String> = Vec::new();
    let mut push = |x: &str| {
        if !sug.iter().any(|y| y == x) {
            sug.push(x.to_string());
        }
    };
    for x in g.findings.iter().chain(f.sizing.iter()).filter(|x| x.severity >= Severity::Warning) {
        push(&x.suggestion);
    }
    for c in mr.clusters.iter().take(5) {
        push(&c.suggestion);
    }
    if sug.is_empty() {
        s += "None: the mesh passes all checks.\n";
    } else {
        for x in &sug {
            s += &format!("- {x}\n");
        }
    }
    s
}

fn cluster_md(i: usize, c: &Cluster) -> String {
    let nn = c.n_cells + c.n_extra;
    let mut s = format!("### Issue {i} ({}): {nn} {} near {}\n\n", sev_tag(c.severity), if nn == 1 { "cell" } else { "cells" }, p3(c.centroid));
    s += &format!("- extent: x {:.3e}..{:.3e}, y {:.3e}..{:.3e}, z {:.3e}..{:.3e}\n", c.lo[0], c.hi[0], c.lo[1], c.hi[1], c.lo[2], c.hi[2]);
    s += &format!("- failing checks: {}\n", c.checks.iter().map(|(n, k)| format!("{n} ({k})")).collect::<Vec<_>>().join(", "));
    s += &format!("- worst values: skewness {:.2}, non-orthogonality {:.1} deg, aspect {:.1}, min volume {:.2e}\n", c.max_skew, c.max_nonorth, c.max_aspect, c.min_vol);
    let lay = c.layers.map_or(String::new(), |(a, b)| if a == b { format!(" (layer {a})") } else { format!(" (layers {a}..{b})") });
    s += &format!("- stage: core {}, pillow {}, layer {}{}\n", c.n_core, c.n_pillow, c.n_layer + c.n_extra, lay);
    s += &format!("- nearby geometry: {}\n- likely cause: {}\n- suggestion: {}\n\n", c.feature, c.cause, c.suggestion);
    s
}

fn cluster_json(c: &Cluster) -> String {
    let ch: Vec<String> = c.checks.iter().map(|(n, k)| format!("{{\"check\": {}, \"cells\": {}}}", jstr(n), k)).collect();
    format!(
        "{{\"severity\": {}, \"cells\": {}, \"centroid\": {}, \"lo\": {}, \"hi\": {}, \"checks\": [{}], \"stage\": {{\"core\": {}, \"pillow\": {}, \"layer\": {}, \"layer_first\": {}, \"layer_last\": {}}}, \"max_skewness\": {:.4}, \"max_nonorthogonality\": {:.3}, \"max_aspect\": {:.3}, \"min_volume\": {:e}, \"wall_distance\": {:e}, \"nearby_geometry\": {}, \"likely_cause\": {}, \"suggestion\": {}}}",
        jstr(c.severity.name()),
        c.n_cells + c.n_extra,
        jpt(c.centroid),
        jpt(c.lo),
        jpt(c.hi),
        ch.join(", "),
        c.n_core,
        c.n_pillow,
        c.n_layer + c.n_extra,
        c.layers.map_or(0, |l| l.0),
        c.layers.map_or(0, |l| l.1),
        c.max_skew,
        c.max_nonorth,
        c.max_aspect,
        c.min_vol,
        c.wall_dist,
        jstr(&c.feature),
        jstr(&c.cause),
        jstr(&c.suggestion)
    )
}

/// JSON members (without braces) appended to `mantle_summary.json`.
pub fn json_fields(f: &Full) -> String {
    let (g, mr) = (f.geom, f.mesh);
    let m = &mr.metrics;
    let inp: Vec<String> = g.findings.iter().map(|x| x.json()).collect();
    let siz: Vec<String> = f.sizing.iter().map(|x| x.json()).collect();
    let cl: Vec<String> = mr.clusters.iter().map(cluster_json).collect();
    format!(
        "\"mantle_verdict\": {}, \"input_checks\": {{\"n_tris\": {}, \"n_parts\": {}, \"area\": {:e}, \"volume\": {:e}, \"n_sharp_edges\": {}, \"findings\": [{}]}}, \"sizing_checks\": [{}], \"quality\": {{\"non_orthogonality_max\": {:.4}, \"non_orthogonality_avg\": {:.4}, \"n_non_orthogonal_gt70\": {}, \"n_non_orthogonal_ge90\": {}, \"n_non_orthogonal_gt60\": {}, \"skewness_max\": {:.5}, \"n_skew_gt4\": {}, \"n_face_pyramid_errors\": {}, \"n_nonpositive_volume\": {}, \"aspect_ratio_max\": {:.4}, \"n_aspect_gt1000\": {}, \"closed_cell_residual_max\": {:e}, \"n_unclosed_cells\": {}, \"n_hex_nonpositive_jacobian\": {}, \"min_cell_volume\": {:e}, \"max_cell_volume\": {:e}, \"failed_checks\": {}, \"n_bad_cells\": {}, \"bad_cells_by_stage\": {{\"core\": {}, \"pillow\": {}, \"layer\": {}}}, \"seconds\": {:.3}}}, \"issues\": [{}]",
        jstr(verdict(f).text()),
        g.n_tris,
        g.n_parts,
        g.area,
        g.volume,
        g.n_sharp_edges,
        inp.join(", "),
        siz.join(", "),
        m.non_orth_max,
        m.non_orth_avg,
        m.n_non_orth_gt70,
        m.n_non_orth_ge90,
        m.n_non_orth_gt60,
        m.skew_max,
        m.n_skew_gt4,
        m.n_pyramid_err,
        m.n_neg_vol,
        m.aspect_max,
        m.n_aspect_gt1000,
        m.closed_max,
        m.n_unclosed,
        m.n_jacobian,
        m.vol_min,
        m.vol_max,
        m.failed_checks,
        m.n_bad_cells,
        mr.bad_stage[0],
        mr.bad_stage[1],
        mr.bad_stage[2],
        m.seconds,
        cl.join(", ")
    )
}

/// Concise terminal summary: verdict, headline metrics, the top 3 issues with suggestions.
pub fn terminal(f: &Full) -> String {
    let (g, mr) = (f.geom, f.mesh);
    let m = &mr.metrics;
    let mut s = format!("report     verdict: {}\n", verdict(f).text());
    s += &format!(
        "           non-orth max {:.1} (avg {:.1}, {} severe > 70, {} >= 90), skewness max {:.2} ({} > 4), pyramid errors {}, non-positive volumes {}, aspect max {:.1}\n",
        m.non_orth_max, m.non_orth_avg, m.n_non_orth_gt70, m.n_non_orth_ge90, m.skew_max, m.n_skew_gt4, m.n_pyramid_err, m.n_neg_vol, m.aspect_max
    );
    let mut fs: Vec<&Finding> = g.findings.iter().chain(f.sizing.iter()).filter(|x| x.severity >= Severity::Warning).collect();
    fs.sort_by(|a, b| b.severity.cmp(&a.severity));
    for x in fs {
        s += &format!("           {}: {}\n             -> {}\n", sev_tag(x.severity), x.title, x.suggestion);
    }
    if !mr.clusters.is_empty() {
        s += &format!("           {} bad cells in {} clusters (cellSet constant/polyMesh/sets/mantleBadCells, details in mantle_report.md); top issues:\n", m.n_bad_cells, mr.clusters.len());
        for (i, c) in mr.clusters.iter().take(3).enumerate() {
            let chk: Vec<String> = c.checks.iter().map(|(n, _)| n.to_string()).collect();
            s += &format!(
                "           {}. {} {} near {} [{}]: {}\n              cause: {}\n              -> {}\n",
                i + 1,
                c.n_cells + c.n_extra,
                if c.n_cells + c.n_extra == 1 { "cell" } else { "cells" },
                p3(c.centroid),
                chk.join(", "),
                c.feature,
                c.cause,
                c.suggestion
            );
        }
    }
    s
}
