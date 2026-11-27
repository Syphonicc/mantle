//! End-to-end shell-first pipeline: shell-clipped octree -> hex core + pillow -> boundary layers -> OpenFOAM / SU2 writers.
//! Shared by `hexcore_bench` and the `mantle` CLI.
use crate::dexel::Tri;
use crate::hexcore::{self, Options};
use crate::layers::{self, Columns, LayerOptions};
use crate::octree::{Clip, Octree, SignMode, SizeField};
use crate::sizing::{self, CurvOptions, CurvStats};
use std::path::Path;
use std::time::Instant;

/// Everything the pipeline needs; `delta < 0` means 4 h_surf, `dy0 <= 0` the stack fit for `growth`, `h_max` infinite = domain size.
#[derive(Clone, Debug)]
pub struct MeshSpec {
    pub origin: [f64; 3],
    pub size: f64,
    pub lmax: u8,
    pub h_surf: f64,
    pub grade: f64,
    pub h_max: f64,
    pub delta: f64,
    pub mode: SignMode,
    pub opt: Options,
    pub nlayers: usize,
    pub dy0: f64,
    pub growth: f64,
    pub snap: f64,
    pub repair_sj: f64,
    pub features: bool,
    pub cont: u8,
    pub cont_rings: usize,
    pub zip: bool,
    pub columns: Columns,
    pub hg_k: f64,
    /// see `LayerOptions::feature_angle` (degrees, 0 = off)
    pub feature_angle: f64,
    /// see `LayerOptions::edge_nodes`
    pub edge_nodes: bool,
    /// smoothing sweeps; negative = default (25 harmonic, 0 nearest)
    pub smooth: i64,
    pub write_foam: bool,
    pub write_su2: bool,
    pub write_shell_quads: bool,
    /// curvature-based per-triangle surface size (`sizing`); the octree gets up to `extra_levels` finer levels
    pub curvature: bool,
    pub curv: CurvOptions,
    pub extra_levels: u8,
}

impl Default for MeshSpec {
    fn default() -> Self {
        MeshSpec { origin: [0.0; 3], size: 1.0, lmax: 6, h_surf: 0.0, grade: 0.2, h_max: f64::INFINITY, delta: -1.0, mode: SignMode::Dexel, opt: Options::default(), nlayers: 0, dy0: -1.0, growth: 1.25, snap: 0.4, repair_sj: 0.05, features: false, cont: 0, cont_rings: 3, zip: false, columns: Columns::Harmonic, hg_k: 8.0, feature_angle: 0.0, edge_nodes: false, smooth: -1, write_foam: true, write_su2: true, write_shell_quads: false, curvature: false, curv: CurvOptions::default(), extra_levels: 3 }
    }
}

/// Result of one run: the full stats JSON (bench format) and the headline numbers.
#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub json: String,
    pub delta: f64,
    pub dy0: f64,
    pub cells: usize,
    pub hex_cells: usize,
    pub poly_cells: usize,
    pub n_prism: usize,
    pub n_pyramid: usize,
    pub n_points: usize,
    pub nonpositive_cells: usize,
    pub orphan_faces: usize,
    pub core_cells: usize,
    pub pillow_hexes: usize,
    pub layer_cells: usize,
    pub layer_nonpositive: usize,
    pub layer_sj_min: f64,
    pub first_min: f64,
    pub first_median: f64,
    pub first_max: f64,
    pub growth_used: [f64; 2],
    pub sj_min_core: f64,
    pub su2_nonpositive: usize,
    pub t_octree: f64,
    pub t_hexcore: f64,
    pub t_layers: f64,
    pub t_foam: f64,
    pub t_su2: f64,
    pub t_total: f64,
    /// curvature sizing: (stats, octree leaves per level); `None` when off
    pub curv: Option<CurvStats>,
    pub leaves_per_level: Vec<usize>,
    pub t_sizing: f64,
    /// OpenFOAM-equivalent metrics and localised issue clusters of the final mesh
    pub mesh: crate::report::MeshReport,
    pub n_faces: usize,
    pub n_internal_faces: usize,
    pub volume: f64,
    pub growth_min: f64,
    pub growth_max: f64,
    pub t_report: f64,
}

/// Run the pipeline on `tris` and write the requested outputs under `out`.
pub fn run(tris: &[Tri], spec: &MeshSpec, out: &Path) -> Result<Summary, String> {
    run_with(tris, spec, out, None)
}

/// As `run`, with the precomputed input geometry analysis (`checks::analyze`) reused for the issue localisation.
pub fn run_with(tris: &[Tri], spec: &MeshSpec, out: &Path, geom: Option<&crate::checks::Geometry>) -> Result<Summary, String> {
    let io = |e: std::io::Error| e.to_string();
    let t0 = Instant::now();
    let (mut opt, mut dy0) = (spec.opt.clone(), spec.dy0);
    let (nlayers, growth, columns) = (spec.nlayers, spec.growth, spec.columns);
    let h_surf = if spec.h_surf <= 0.0 { spec.size / (1u64 << spec.lmax) as f64 } else { spec.h_surf };
    let h_max = if spec.h_max.is_finite() { spec.h_max } else { spec.size };
    let delta = if spec.delta < 0.0 { 4.0 * h_surf } else { spec.delta };
    opt.delta = delta;
    let mut sf = SizeField::new(h_surf, spec.grade, h_max);
    let (mut lvl, mut curv) = (spec.lmax, None);
    if spec.curvature {
        // finest allowed size: h_min, and at most `extra_levels` below the base level (a leaf is never finer than that)
        let cs = spec.size / (1u64 << spec.lmax) as f64;
        let mut co = spec.curv.clone();
        if co.h_min <= 0.0 {
            co.h_min = h_surf / 8.0;
        }
        co.h_min = co.h_min.max(cs / (1u64 << spec.extra_levels) as f64);
        co.offset = delta;
        let (th, st) = sizing::curvature_sizes(tris, h_surf, &co);
        // octree depth: just enough for the finest size actually requested
        let extra = ((cs / st.h_min_used).log2().ceil().max(0.0) as u8).min(spec.extra_levels);
        lvl = spec.lmax + extra;
        sf.tri_h = Some(th);
        curv = Some(st);
    }
    let t_sizing = t0.elapsed().as_secs_f64();
    let oc = Octree::build_clipped(tris, spec.origin, spec.size, lvl, &sf, 0, spec.mode, Clip::Shell { delta });
    let t_octree = t0.elapsed().as_secs_f64();
    let viol = oc.balance_violations();

    let t = Instant::now();
    let hc = hexcore::build(&oc, tris, &opt);
    let t_hexcore = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let lm = (nlayers > 0).then(|| {
        if dy0 <= 0.0 {
            dy0 = layers::fit_dy0(delta, nlayers, growth);
        }
        layers::build(&hc, tris, &LayerOptions { features: spec.features, cont: spec.cont, cont_rings: spec.cont_rings, zip: spec.zip, n: nlayers, dy0, growth, snap: spec.snap, repair_sj: spec.repair_sj, columns, hg_k: spec.hg_k, feature_angle: spec.feature_angle, edge_nodes: spec.edge_nodes, smooth_sweeps: if spec.smooth >= 0 { spec.smooth as usize } else if columns == Columns::Harmonic { 25 } else { 0 } })
    });
    let t_layers = t.elapsed().as_secs_f64();
    let (m, s) = (lm.as_ref().map_or(&hc.mesh, |l| &l.mesh), &hc.stats);

    std::fs::create_dir_all(out).map_err(io)?;
    if spec.write_shell_quads {
        hexcore::write_shell_quads(&hc.shell, &out.join("shell_quads.bin")).map_err(io)?;
    }
    let t = Instant::now();
    let (mut ni, mut patches) = (0, Vec::new());
    if spec.write_foam {
        (ni, patches) = crate::foam::write_polymesh(m, &out.join("constant").join("polyMesh")).map_err(io)?;
        std::fs::create_dir_all(out.join("system")).map_err(io)?;
        std::fs::write(
            out.join("system").join("controlDict"),
            "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object controlDict;\n}\n\napplication simpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\nwritePrecision 15;\nwriteFormat ascii;\n",
        )
        .map_err(io)?;
        let hdr = |f: &str| format!("FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object {f};\n}}\n\n");
        std::fs::write(
            out.join("system").join("fvSchemes"),
            hdr("fvSchemes") + "ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n",
        )
        .map_err(io)?;
        std::fs::write(out.join("system").join("fvSolution"), hdr("fvSolution") + "solvers {}\n").map_err(io)?;
    }
    let t_foam = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let lbad: &[([f64; 3], usize, f64)] = lm.as_ref().map_or(&[], |l| &l.stats.bad_cells);
    let mesh = crate::report::analyze_mesh(m, &crate::report::Locate { tris, geom, h_surf, delta, nlayers, growth: lm.as_ref().map_or(growth, |l| 0.5 * (l.stats.growth_min + l.stats.growth_max)), n_core: s.core_cells, n_pillow: s.pillow_hexes, layer_bad: lbad, cell_layer: lm.as_ref().map_or(&[], |l| &l.cell_layer) });
    if spec.write_foam {
        crate::report::write_cellset(out, &mesh.bad_cells).map_err(io)?;
    }
    let t_report = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let s2 = if spec.write_su2 { crate::su2::write_su2(m, &out.join("mesh.su2")).map_err(io)? } else { Default::default() };
    let t_su2 = t.elapsed().as_secs_f64();

    let pj: Vec<String> = patches.iter().map(|(n, c)| format!("\"{n}\": {c}")).collect();
    let stg = |x: &hexcore::SjStage| format!("{{\"min\": {:e}, \"mean\": {:.6}, \"n_lt_0p1\": {}, \"n_lt_0p2\": {}, \"n_lt_0p3\": {}}}", x.min, x.mean, x.n_lt_0p1, x.n_lt_0p2, x.n_lt_0p3);
    let lj = lm.as_ref().map_or(String::new(), |l| format!(", \"layers\": {}, \"t_layers_total\": {:.3}", l.stats.json(), t_layers));
    let ext = format!(
        ", \"sj_min_initial\": {:e}, \"sj_min_after_smooth\": {:e}, \"sj_min_after_opt\": {:e}, \"n_sj_below_0p1\": {}, \"n_sj_below_0p3\": {}, \"opt_sweeps\": {}, \"moved_core_vertices\": {}, \"core_sj_min\": {:e}, \"t_opt\": {:.3}, \"fillet\": {}, \"ridge_nodes\": {}, \"grid_nodes\": {}, \"t_shell\": {:.3}, \"stage_initial\": {}, \"stage_smooth\": {}, \"stage_opt\": {}",
        s.stage_initial.min, s.stage_smooth.min, s.stage_opt.min, s.n_sj_below_0p1, s.n_sj_below_0p3, s.opt_sweeps, s.moved_core_vertices, s.core_sj_min, s.t_opt, opt.fillet, s.ridge_nodes, s.grid_nodes, s.t_shell, stg(&s.stage_initial), stg(&s.stage_smooth), stg(&s.stage_opt)
    ) + &lj;
    let h = |a: &[usize; 5]| format!("[{}, {}, {}, {}, {}]", a[0], a[1], a[2], a[3], a[4]);
    let json = format!(
        "{{\"leaves\": {}, \"balance_violations\": {}, \"delta\": {}, \"gamma\": {}, \"theta_max\": {}, \"core_initial\": {}, \"core_cells\": {}, \"removed_margin\": {}, \"removed_nonfluid\": {}, \"removed_nonmanifold\": {}, \"removed_hanging\": {}, \"removed_split\": {}, \"removed_normal\": {}, \"added_fill\": {}, \"removed_jacobian\": {}, \"rounds\": {}, \"post_rounds\": {}, \"budget_hit\": {}, \"unprojected\": {}, \"ray_targets\": {}, \"smooth_ray_moves\": {}, \"manifold_iters\": {}, \"core_boundary_quads\": {}, \"core_boundary_verts\": {}, \"open_edges\": {}, \"nonmanifold_edges_final\": {}, \"nonmanifold_verts_final\": {}, \"normal_violations_final\": {}, \"jacobian_violations_final\": {}, \"pillow_hexes\": {}, \"core_hex\": {}, \"core_poly\": {}, \"core_split_faces\": {}, \"sj_min_before\": {:e}, \"sj_min_after\": {:e}, \"sj_mean\": {:.6}, \"n_sj_below_0p2\": {}, \"n_sj_below_0\": {}, \"sj_hist_before\": {}, \"sj_hist_after\": {}, \"max_proj_dist_over_h\": {:.4}, \"smooth_accepted\": {}, \"smooth_rejected\": {}, \"core_move_rejected\": {}, \"cells\": {}, \"hex_cells\": {}, \"poly_cells\": {}, \"orphan_faces\": {}, \"nonmanifold_faces\": {}, \"nonpositive_cells\": {}, \"min_vol_ratio\": {:e}, \"volume\": {:.15e}, \"n_points\": {}, \"n_faces\": {}, \"n_internal_faces\": {}, \"patches\": {{{}}}, \"su2\": {{\"tet\": {}, \"hex\": {}, \"prism\": {}, \"pyramid\": {}, \"points\": {}, \"shell_elems\": {}, \"farfield_elems\": {}, \"hex_decomposed\": {}, \"quads_split\": {}, \"nonpositive\": {}}}, \"t_octree\": {:.3}, \"t_hexcore\": {:.3}, \"t_core\": {:.3}, \"t_manifold\": {:.3}, \"t_normal\": {:.3}, \"t_pillow\": {:.3}, \"t_smooth\": {:.3}, \"t_assemble\": {:.3}, \"t_foam\": {:.3}, \"t_su2\": {:.3}, \"t_total\": {:.3}{}}}",
        s.leaves, viol, delta, opt.gamma, opt.theta_max, s.core_initial, s.core_cells, s.removed_margin, s.removed_nonfluid, s.removed_nonmanifold, s.removed_hanging, s.removed_split, s.removed_normal, s.added_fill, s.removed_jacobian, s.rounds, s.post_rounds, s.budget_hit, s.unprojected, s.ray_targets, s.smooth_ray_moves, s.manifold_iters, s.core_boundary_quads, s.core_boundary_verts, s.open_edges, s.nonmanifold_edges_final, s.nonmanifold_verts_final, s.normal_violations_final, s.jacobian_violations_final, s.pillow_hexes, s.core_hex, s.core_poly, s.core_split_faces, s.sj_min_before, s.sj_min_after, s.sj_mean, s.n_sj_below_0p2, s.n_sj_below_0, h(&s.sj_hist_before), h(&s.sj_hist_after), s.max_proj_dist_over_h, s.smooth_accepted, s.smooth_rejected, s.core_move_rejected,
        m.cells.len(), m.stats.hex_cells, m.stats.poly_cells, m.stats.orphan_faces, m.stats.nonmanifold_faces, m.stats.n_nonpositive_cells, m.stats.min_vol_ratio, m.stats.volume, m.points.len(), m.faces.len(), ni, pj.join(", "),
        s2.n_tet, s2.n_hex, s2.n_prism, s2.n_pyramid, s2.n_points, s2.n_wall_elems, s2.n_far_elems, s2.hex_decomposed, s2.quads_split, s2.nonpositive,
        t_octree, t_hexcore, s.t_core, s.t_manifold, s.t_normal, s.t_pillow, s.t_smooth, s.t_assemble, t_foam, t_su2, t0.elapsed().as_secs_f64(), ext
    );
    let ls = lm.as_ref().map(|l| &l.stats);
    Ok(Summary {
        json,
        delta,
        dy0: ls.map_or(0.0, |l| l.dy0),
        cells: m.cells.len(),
        hex_cells: m.stats.hex_cells,
        poly_cells: m.stats.poly_cells,
        n_prism: s2.n_prism,
        n_pyramid: s2.n_pyramid,
        n_points: m.points.len(),
        nonpositive_cells: m.stats.n_nonpositive_cells,
        orphan_faces: m.stats.orphan_faces,
        core_cells: s.core_cells,
        pillow_hexes: s.pillow_hexes,
        layer_cells: ls.map_or(0, |l| l.hex + l.prism + l.pyramid + l.poly),
        layer_nonpositive: ls.map_or(0, |l| l.n_nonpositive),
        layer_sj_min: ls.map_or(f64::NAN, |l| l.sj_min),
        first_min: ls.map_or(0.0, |l| l.first_min),
        first_median: ls.map_or(0.0, |l| l.first_median),
        first_max: ls.map_or(0.0, |l| l.first_max),
        growth_used: ls.map_or([0.0; 2], |l| [l.growth_min, l.growth_max]),
        sj_min_core: s.sj_min_after,
        su2_nonpositive: s2.nonpositive,
        t_octree,
        t_hexcore,
        t_layers,
        t_foam,
        t_su2,
        t_total: t0.elapsed().as_secs_f64(),
        curv,
        leaves_per_level: {
            let mut v = oc.leaves_per_level();
            while v.last() == Some(&0) {
                v.pop();
            }
            v
        },
        t_sizing,
        mesh,
        n_faces: m.faces.len(),
        n_internal_faces: ni,
        volume: m.stats.volume,
        growth_min: ls.map_or(0.0, |l| l.growth_min),
        growth_max: ls.map_or(0.0, |l| l.growth_max),
        t_report,
    })
}
