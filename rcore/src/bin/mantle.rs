//! mantle: STL -> shell-first hybrid CFD mesh (OpenFOAM polyMesh + SU2). See `--help`.
use rcore::dexel::Tri;
use rcore::hexcore::Options;
use rcore::checks;
use rcore::pipeline::{self, MeshSpec, Summary};
use rcore::report;
use rcore::sizing::CurvOptions;
use rcore::stl;
use std::path::{Path, PathBuf};

const HELP: &str = "mantle: shell-first hybrid CFD mesher (hex core + boundary layers) from a closed STL surface

usage: mantle INPUT.stl -o CASE_DIR [options]

  -o, --out DIR        output case directory (required)
  --yplus Y --re RE    first cell from target y+ and Reynolds number (Re on body length; flat-plate Cf = 0.026 Re^(-1/7))
  --dy0 D              first cell height, absolute (overrides --yplus/--re)
  --layers N           boundary layers (default 12; raised automatically, max 60, if dy0 needs growth > 1.3)
  --growth G           layer growth ratio (default 1.2)
  --delta D            layer / shell thickness (default 4 h_surf; caps h_surf at delta/2 when --h-surf is not given)
  --preset P           coarse | medium | fine: h_surf = length / 32 | 48 | 80 (default medium; explicit flags override)
  --h-surf H           surface cell size (default bbox length / preset divisor)
  --farfield F         domain side = F * bbox length, cube centred on the body (default 5)
  --level L            octree depth (default: smallest with side/2^L <= h_surf)
  --fillet F           shell fillet radius at concave creases, in units of delta (default 1.0; 0 = plain offset)
  --feature-angle DEG  wall-node fans only at edges sharper than this (default 25; shallower facet edges are treated as smooth)
  --edge-nodes on|off  structured fans: wall nodes of columns on sharp convex edges from a 1D discretisation of the edge curve (experimental, default off = cluster feet by proximity)
  --curvature on|off   curvature-based surface sizing: h = clamp(c R, h_min, h_surf) per triangle (default OFF, experimental: the hex core / pillow needs one uniform shell level, so size transitions on the shell break it, see README of the sizing module); sharp edges (dihedral > 60 deg) are ignored
  --curv-c C           cells per radius of curvature, h = C R (default 0.25, ~25 cells per circle)
  --h-min H            smallest surface size (default h_surf / 8; the octree gets at most 3 extra levels)
  --su2-only           write only CASE_DIR/mesh.su2
  --foam-only          write only the OpenFOAM case
  --force              mesh even if the input checks find fatal problems (open / non-manifold / inconsistent / inward surface)
  --report-only        run the input checks and sizing warnings only, print them (and write CASE_DIR/mantle_report.md if -o is given), exit
  --quiet              no stdout summary
  --json               print the JSON summary instead of the text summary
  -h, --help           this text

output: CASE_DIR/constant/polyMesh (patches wall, farfield), CASE_DIR/system/*, CASE_DIR/mesh.su2, CASE_DIR/mantle_summary.json,
        CASE_DIR/mantle_report.md (input checks, sizing, checkMesh-equivalent quality, localised issues, suggestions),
        CASE_DIR/constant/polyMesh/sets/mantleBadCells (cellSet of bad cells: foamToVTK -cellSet mantleBadCells, open in ParaView)
The STL must be a closed, consistently oriented surface (any units; the mesh uses the same units).";

struct Cli {
    input: PathBuf,
    out: PathBuf,
    yplus: Option<f64>,
    re: Option<f64>,
    dy0: Option<f64>,
    layers: Option<usize>,
    growth: f64,
    delta: Option<f64>,
    h_surf: Option<f64>,
    farfield: f64,
    preset: String,
    level: Option<u8>,
    fillet: f64,
    angle: f64,
    edge_nodes: bool,
    curvature: bool,
    curv_c: f64,
    h_min: Option<f64>,
    su2: bool,
    foam: bool,
    quiet: bool,
    json: bool,
    force: bool,
    report_only: bool,
}

fn parse_args(args: &[String]) -> Result<Option<Cli>, String> {
    let mut c = Cli { input: PathBuf::new(), out: PathBuf::new(), yplus: None, re: None, dy0: None, layers: None, growth: 1.2, delta: None, h_surf: None, farfield: 5.0, preset: "medium".into(), level: None, fillet: 1.0, angle: 25.0, edge_nodes: false, curvature: false, curv_c: 0.25, h_min: None, su2: true, foam: true, quiet: false, json: false, force: false, report_only: false };
    let (mut have_in, mut have_out) = (false, false);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |what: &str| it.next().ok_or_else(|| format!("{a} needs a value ({what})")).cloned();
        let num = |s: String, a: &str| s.parse::<f64>().ok().filter(|x| x.is_finite() && *x > 0.0 || (a == "--fillet" || a == "--feature-angle") && *x == 0.0).ok_or_else(|| format!("{a}: need a positive number, got '{s}'"));
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "-o" | "--out" => (c.out, have_out) = (PathBuf::from(val("directory")?), true),
            "--yplus" => c.yplus = Some(num(val("Y")?, a)?),
            "--re" => c.re = Some(num(val("RE")?, a)?),
            "--dy0" => c.dy0 = Some(num(val("D")?, a)?),
            "--layers" => c.layers = Some(val("N")?.parse().ok().filter(|n| (1..=60).contains(n)).ok_or("--layers: integer 1..60")?),
            "--growth" => c.growth = num(val("G")?, a).and_then(|g| if g >= 1.0 && g <= 1.5 { Ok(g) } else { Err("--growth: need 1.0..1.5".to_string()) })?,
            "--delta" => c.delta = Some(num(val("D")?, a)?),
            "--h-surf" => c.h_surf = Some(num(val("H")?, a)?),
            "--farfield" => c.farfield = num(val("F")?, a).and_then(|f| if f >= 2.0 { Ok(f) } else { Err("--farfield: need >= 2".to_string()) })?,
            "--preset" => c.preset = val("coarse|medium|fine")?.to_lowercase(),
            "--level" => c.level = Some(val("L")?.parse().ok().filter(|l| (1..=16).contains(l)).ok_or("--level: integer 1..16")?),
            "--fillet" => c.fillet = num(val("F")?, a)?,
            "--feature-angle" => c.angle = num(val("DEG")?, a)?,
            "--edge-nodes" => c.edge_nodes = match val("on|off")?.as_str() { "on" => true, "off" => false, o => return Err(format!("--edge-nodes: on|off, got '{o}'")) },
            "--curvature" => c.curvature = match val("on|off")?.as_str() { "on" => true, "off" => false, o => return Err(format!("--curvature: on|off, got '{o}'")) },
            "--curv-c" => c.curv_c = num(val("C")?, a)?,
            "--h-min" => c.h_min = Some(num(val("H")?, a)?),
            "--su2-only" => c.foam = false,
            "--foam-only" => c.su2 = false,
            "--quiet" => c.quiet = true,
            "--json" => c.json = true,
            "--force" => c.force = true,
            "--report-only" => c.report_only = true,
            s if s.starts_with('-') => return Err(format!("unknown option {s} (see --help)")),
            s => {
                if have_in {
                    return Err(format!("unexpected extra argument '{s}'"));
                }
                (c.input, have_in) = (PathBuf::from(s), true)
            }
        }
    }
    if !have_in || (!have_out && !c.report_only) {
        return Err("need INPUT.stl and -o CASE_DIR (see --help)".into());
    }
    if !c.su2 && !c.foam {
        return Err("--su2-only and --foam-only are mutually exclusive".into());
    }
    if !["coarse", "medium", "fine"].contains(&c.preset.as_str()) {
        return Err("--preset: coarse|medium|fine".into());
    }
    if c.yplus.is_some() != c.re.is_some() {
        return Err("--yplus and --re must be given together".into());
    }
    Ok(Some(c))
}

/// Growth ratio r with dy0 (r^n - 1) / (r - 1) = h (1 if dy0 n >= h).
fn growth_needed(h: f64, n: usize, dy0: f64) -> f64 {
    let sum = |r: f64| if (r - 1.0).abs() < 1e-12 { dy0 * n as f64 } else { dy0 * (r.powi(n as i32) - 1.0) / (r - 1.0) };
    if sum(1.0) >= h {
        return 1.0;
    }
    let (mut lo, mut hi) = (1.0, 4.0);
    for _ in 0..60 {
        let m = 0.5 * (lo + hi);
        if sum(m) < h { lo = m } else { hi = m }
    }
    hi
}

/// Thickness scale of the body: the smaller of the shortest bbox side and twice the median inward-ray thickness (rays along -n from up to 300 triangle
/// centroids, first hit distance, Moller-Trumbore); robust against thin trailing edges because only the median is used.
fn thickness(tris: &[Tri], bbox_min_side: f64) -> f64 {
    let (sub, dot, cross) = (|a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]], |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2], |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]);
    let step = (tris.len() / 300).max(1);
    let mut th: Vec<f64> = Vec::new();
    for t in tris.iter().step_by(step) {
        let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        let nl = dot(n, n).sqrt();
        if nl < 1e-300 {
            continue;
        }
        let o = [0, 1, 2].map(|k| (t[0][k] + t[1][k] + t[2][k]) / 3.0);
        let d = n.map(|x| -x / nl);
        let mut best = f64::INFINITY;
        for u in tris {
            let (e1, e2) = (sub(u[1], u[0]), sub(u[2], u[0]));
            let pv = cross(d, e2);
            let det = dot(e1, pv);
            if det.abs() < 1e-300 {
                continue;
            }
            let tv = sub(o, u[0]);
            let a = dot(tv, pv) / det;
            let q = cross(tv, e1);
            let b = dot(d, q) / det;
            if a < 0.0 || b < 0.0 || a + b > 1.0 {
                continue;
            }
            let dist = dot(e2, q) / det;
            if dist > 1e-9 * bbox_min_side.max(1e-300) && dist < best {
                best = dist;
            }
        }
        if best.is_finite() {
            th.push(best);
        }
    }
    th.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if th.is_empty() { bbox_min_side } else { bbox_min_side.min(2.0 * th[th.len() / 2]) }
}

fn t(x: f64) -> String {
    format!("{x:.2}s")
}

fn run(c: &Cli) -> Result<(), String> {
    let tris = stl::read(&c.input)?;
    let geom = checks::analyze(&tris, c.angle);
    let input = c.input.display().to_string();
    if geom.has_errors() && !c.force && !c.report_only {
        let md = report::input_markdown(&input, &geom, None);
        if !c.out.as_os_str().is_empty() && std::fs::create_dir_all(&c.out).is_ok() {
            let _ = std::fs::write(c.out.join("mantle_report.md"), &md);
        }
        let msgs: Vec<String> = geom.findings.iter().filter(|f| f.severity == checks::Severity::Error).map(|f| format!("  - {}\n    fix: {}", f.title, f.suggestion)).collect();
        return Err(format!("the input surface has fatal problems, not meshing (use --force to try anyway; details in {}):\n{}", c.out.join("mantle_report.md").display(), msgs.join("\n")));
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in tris.iter().flatten() {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let lb = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
    if lb <= 0.0 {
        return Err("degenerate STL: zero extent".into());
    }
    let div = match c.preset.as_str() { "coarse" => 32.0, "fine" => 80.0, _ => 48.0 };
    let tscale = thickness(&tris, (0..3).map(|k| hi[k] - lo[k]).fold(f64::INFINITY, f64::min));
    let mut h_surf = c.h_surf.unwrap_or(lb / div);
    let mut delta = c.delta.unwrap_or(4.0 * h_surf);
    let mut cap = String::new();
    if c.delta.is_none() && delta > 0.5 * tscale {
        delta = 0.5 * tscale;
        if c.h_surf.is_none() {
            h_surf = delta / 2.0;
        }
        cap = format!("thin body (thickness scale {tscale:.4e}): delta capped to {delta:.4e} (0.5 t), h_surf {h_surf:.4e} (= delta/2)");
    } else if delta > 0.5 * tscale && !c.quiet && !c.json {
        eprintln!("warning: delta {delta:.3e} > 0.5 x thickness scale {tscale:.3e}: thin parts may produce an invalid shell");
    }
    // the shell must span at least two surface cells: an explicit --delta without --h-surf caps the surface size, like the thin-body rule
    if c.delta.is_some() && c.h_surf.is_none() && h_surf > 0.5 * delta {
        h_surf = 0.5 * delta;
        cap = format!("h_surf {h_surf:.4e} (= delta/2) from the explicit --delta, so the layer shell spans at least two surface cells");
    } else if c.delta.is_some() && c.h_surf.is_some() && delta < 2.0 * h_surf && !c.quiet && !c.json {
        eprintln!("warning: delta {delta:.3e} < 2 x h_surf {h_surf:.3e}: the layer shell is thinner than two surface cells; expect a poor hex core / pillow layer");
    }
    let sizing = checks::sizing_findings(&geom, &tris, h_surf, delta);
    if c.report_only {
        let md = report::input_markdown(&input, &geom, Some(&sizing));
        if !c.out.as_os_str().is_empty() {
            std::fs::create_dir_all(&c.out).map_err(|e| e.to_string())?;
            std::fs::write(c.out.join("mantle_report.md"), &md).map_err(|e| e.to_string())?;
        }
        print!("{md}");
        if geom.has_errors() {
            return Err("input checks found fatal problems".into());
        }
        return Ok(());
    }
    let mut side = c.farfield * lb;
    let level = c.level.unwrap_or_else(|| (side / h_surf).log2().ceil().clamp(1.0, 16.0) as u8);
    if c.level.is_none() {
        // the finest octree cell is exactly h_surf (the domain grows to h_surf 2^level >= the requested far field): a thin body with a shell several
        // cells thick (delta / cell > 2) folds at its trailing edge, which a cell size drifting below h_surf with the level rounding caused
        side = h_surf * (1u64 << level) as f64;
    }
    let ctr: Vec<f64> = (0..3).map(|k| 0.5 * (lo[k] + hi[k])).collect();
    let dy0_req = c.dy0.or_else(|| c.yplus.zip(c.re).map(|(y, re)| y * lb * (2.0 / (0.026 * re.powf(-1.0 / 7.0))).sqrt() / re));
    let mut n = c.layers.unwrap_or(12);
    let mut note = cap.clone();
    if let Some(d) = dy0_req {
        if d >= delta {
            return Err(format!("first cell {d:.3e} >= delta {delta:.3e}: increase --delta or reduce y+/dy0"));
        }
        let n0 = n;

        while growth_needed(delta, n, d) > 1.3 && n < 60 {
            n += 1;
        }
        if n != n0 {
            if !note.is_empty() {
                note += "; ";
            }
            note += &format!("layers raised {n0} -> {n} (dy0 {d:.3e} needs growth <= 1.3 to fill delta {delta:.3e})");
        }
        if growth_needed(delta, n, d) > 1.3 {
            note += &format!("; warning: growth {:.2} still > 1.3 at the 60-layer cap", growth_needed(delta, n, d));
        }
    }
    if !note.is_empty() && !c.quiet && !c.json {
        eprintln!("note: {note}");
    }
    let spec = MeshSpec {
        origin: [ctr[0] - side / 2.0, ctr[1] - side / 2.0, ctr[2] - side / 2.0],
        size: side,
        lmax: level,
        h_surf,
        grade: 0.25,
        delta,
        opt: Options { fillet: c.fillet, ..Options::default() },
        nlayers: n,
        dy0: dy0_req.unwrap_or(-1.0),
        growth: c.growth,
        feature_angle: c.angle,
        edge_nodes: c.edge_nodes,
        curvature: c.curvature,
        curv: CurvOptions { c: c.curv_c, h_min: c.h_min.unwrap_or(0.0), ..CurvOptions::default() },
        write_foam: c.foam,
        write_su2: c.su2,
        ..MeshSpec::default()
    };
    let r = pipeline::run_with(&tris, &spec, &c.out, Some(&geom))?;
    let params = report::Params {
        input: input.replace('\\', "/"),
        preset: if c.h_surf.is_some() { format!("{} (h_surf set explicitly)", c.preset) } else { c.preset.clone() },
        h_surf,
        delta,
        dy0: r.dy0,
        layers: n,
        growth: c.growth,
        growth_used: r.growth_used,
        fillet: c.fillet,
        level,
        domain: side,
        note: note.clone(),
        cells: r.cells,
        hex: r.hex_cells,
        poly: r.poly_cells,
        prism: r.n_prism,
        pyramid: r.n_pyramid,
        points: r.n_points,
        core: r.core_cells,
        pillow: r.pillow_hexes,
        layer_cells: r.layer_cells,
        orphan_faces: r.orphan_faces,
        layer_nonpositive: r.layer_nonpositive,
        layer_sj_min: r.layer_sj_min,
        su2_nonpositive: r.su2_nonpositive,
        first: [r.first_min, r.first_median, r.first_max],
        volume: r.volume,
    };
    let full = report::Full { geom: &geom, sizing: &sizing, params: &params, mesh: &r.mesh };
    let verdict = report::verdict(&full).text().to_string();
    std::fs::create_dir_all(&c.out).map_err(|e| e.to_string())?;
    std::fs::write(c.out.join("mantle_report.md"), report::markdown(&full)).map_err(|e| e.to_string())?;
    let curv_json = r.curv.as_ref().map_or("null".to_string(), |s| format!("{{\"c\": {}, \"h_min\": {:e}, \"h_max\": {:e}, \"refined_tris\": {}, \"sharp_tris\": {}, \"r_min\": {:e}}}", c.curv_c, s.h_min_used, s.h_max_used, s.n_refined, s.n_sharp_tris, s.r_min));
    let hdr = format!(
        "{{\"input\": \"{}\", \"n_tris\": {}, \"bbox\": [[{:e}, {:e}, {:e}], [{:e}, {:e}, {:e}]], \"length\": {:e}, \"preset\": \"{}\", \"h_surf\": {:e}, \"delta\": {:e}, \"dy0\": {:e}, \"layers\": {}, \"growth\": {}, \"fillet\": {}, \"level\": {}, \"domain_side\": {:e}, \"cells\": {}, \"hex\": {}, \"poly\": {}, \"prism\": {}, \"pyramid\": {}, \"n_points\": {}, \"nonpositive_cells\": {}, \"orphan_faces\": {}, \"layer_nonpositive\": {}, \"layer_sj_min\": {:e}, \"core_sj_min\": {:e}, \"first_cell\": [{:e}, {:e}, {:e}], \"t\": {{\"octree\": {:.3}, \"hexcore\": {:.3}, \"layers\": {:.3}, \"foam\": {:.3}, \"su2\": {:.3}, \"total\": {:.3}}}, \"verdict\": \"{}\", \"curvature\": {}, \"leaves_per_level\": {:?}, \"pipeline\": ",
        c.input.display().to_string().replace('\\', "/").replace('"', "'"), tris.len(), lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], lb, c.preset, h_surf, delta, r.dy0, n, c.growth, c.fillet, level, side, r.cells, r.hex_cells, r.poly_cells, r.n_prism, r.n_pyramid, r.n_points, r.nonpositive_cells, r.orphan_faces, r.layer_nonpositive, r.layer_sj_min, r.sj_min_core, r.first_min, r.first_median, r.first_max, r.t_octree, r.t_hexcore, r.t_layers, r.t_foam, r.t_su2, r.t_total, verdict.replace('"', "'"), curv_json, r.leaves_per_level
    );
    let json = format!("{hdr}{}, {}}}\n", r.json, report::json_fields(&full));
    std::fs::write(c.out.join("mantle_summary.json"), &json).map_err(|e| e.to_string())?;
    if c.json {
        print!("{json}");
    } else if !c.quiet {
        print_text(c, &r, tris.len(), lo, hi, [lb, h_surf, delta, side], n, level, &note);
        print!("{}", report::terminal(&full));
        println!("report file {}", c.out.join("mantle_report.md").display());
    }
    if r.orphan_faces > 0 {
        return Err(format!("{} orphan boundary faces (outputs written, but the mesh has holes)", r.orphan_faces));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_text(c: &Cli, r: &Summary, nt: usize, lo: [f64; 3], hi: [f64; 3], g: [f64; 4], n: usize, level: u8, note: &str) {
    let [lb, h, d, side] = g;
    println!("input      {} ({nt} triangles)", c.input.display());
    println!("bbox       [{:.4e} {:.4e} {:.4e}] .. [{:.4e} {:.4e} {:.4e}]  (length {lb:.4e})", lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
    println!("preset     {}\nparams     h_surf {h:.4e}  delta {d:.4e}  dy0 {:.4e}  layers {n}  growth {} (used {:.3}..{:.3})  fillet {}  level {level}  domain {side:.4e}", if c.h_surf.is_some() { format!("{} (h_surf set explicitly)", c.preset) } else { c.preset.clone() }, r.dy0, c.growth, r.growth_used[0], r.growth_used[1], c.fillet);
    if !note.is_empty() {
        println!("note       {note}");
    }
    println!("cells      {} total: {} hex, {} polyhedra (SU2: {} prism, {} pyramid); core {} + pillow {} + layers {}", r.cells, r.hex_cells, r.poly_cells, r.n_prism, r.n_pyramid, r.core_cells, r.pillow_hexes, r.layer_cells);
    match &r.curv {
        Some(s) => println!("curvature  on (c {}): surface size {:.4e} .. {:.4e} (h_surf {h:.4e}), {} of {} triangles refined, {} touch sharp edges, min R {:.3e}; leaves/level {:?}", c.curv_c, s.h_min_used, s.h_max_used, s.n_refined, s.n_tris, s.n_sharp_tris, s.r_min, r.leaves_per_level),
        None => println!("curvature  off; leaves/level {:?}", r.leaves_per_level),
    }
    println!("quality    non-positive cells {} (layers {}); layer min scaled Jacobian {:.3}; core min scaled Jacobian {:.3}", r.nonpositive_cells, r.layer_nonpositive, r.layer_sj_min, r.sj_min_core);
    println!("first cell min/median/max {:.4e} / {:.4e} / {:.4e}", r.first_min, r.first_median, r.first_max);
    println!("timings    octree {}  hexcore {}  layers {}  foam {}  su2 {}  total {}", t(r.t_octree), t(r.t_hexcore), t(r.t_layers), t(r.t_foam), t(r.t_su2), t(r.t_total));
    println!("output     {}", Path::new(&c.out).display());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&args) {
        Ok(None) => println!("{HELP}"),
        Ok(Some(c)) => {
            if let Err(e) = run(&c) {
                eprintln!("mantle: error: {e}");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("mantle: {e}");
            std::process::exit(2);
        }
    }
}
