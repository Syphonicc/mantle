//! hexcore_bench SOUP.bin origin=x,y,z size=S L=N h_surf=H grade=G h_max=M [delta=D (4*h_surf)] [gamma=0.3] [theta_max=60] [max_proj_h=2.5]
//! [smooth_iters=5] [core_move=0] [max_rounds=5] [sj_remove=0] [opt_sweeps=0] [opt_core=1] [opt_ring=0] [opt_init=1] [max_removed_frac=0.03] [signs=fwn|dexel] [layers=0] [dy0=] [growth=1.25] [snap=0.4] [repair_sj=0.05] OUT_DIR
//! `layers=N` adds N boundary layers between wall and shell (`rcore::layers`; dy0 default: stack fit for `growth`); patches `wall`, `farfield`.
//! SOUP.bin: u64 ntri, then ntri*9 f64. Builds the shell-clipped buffered octree and the hex-core + pillow mesh (`rcore::hexcore`),
//! writes OUT_DIR/constant/polyMesh/*, OUT_DIR/system/*, OUT_DIR/mesh.su2, OUT_DIR/shell_quads.bin and prints one JSON line.
use rcore::dexel::Tri;
use rcore::hexcore::{self, Options};
use rcore::layers::{self, LayerOptions};
use rcore::octree::{Clip, Octree, SignMode, SizeField};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: hexcore_bench SOUP.bin origin=x,y,z size= L= h_surf= grade= h_max= [delta=D] [gamma=0.3] [theta_max=60] [max_proj_h=2.5] [smooth_iters=5] [core_move=0] [max_rounds=5] [sj_remove=0] [opt_sweeps=0] [opt_core=1] [opt_ring=0] [opt_init=1] [max_removed_frac=0.03] [signs=fwn|dexel] OUT_DIR");
        std::process::exit(2);
    }
    let t0 = Instant::now();
    let mut sb = Vec::new();
    std::fs::File::open(&args[1]).unwrap().read_to_end(&mut sb).unwrap();
    let nt = u64::from_le_bytes(sb[0..8].try_into().unwrap()) as usize;
    let tris: Vec<Tri> = (0..nt)
        .map(|t| {
            let mut tr = [[0.0; 3]; 3];
            for (n, c) in tr.iter_mut().flatten().enumerate() {
                let o = 8 + (t * 9 + n) * 8;
                *c = f64::from_le_bytes(sb[o..o + 8].try_into().unwrap());
            }
            tr
        })
        .collect();
    let out = Path::new(args.last().unwrap()).to_path_buf();
    let mut mode = SignMode::Dexel;
    let (mut origin, mut size, mut lmax, mut h_surf, mut grade, mut h_max, mut delta) = ([0.0; 3], 1.0, 6u8, 0.0, 0.2, f64::INFINITY, -1.0f64);
    let mut opt = Options::default();
    let (mut nlayers, mut dy0, mut growth, mut snap) = (0usize, -1.0f64, 1.25f64, 0.4f64);
    let mut repair_sj = 0.05f64;
    for a in &args[2..args.len() - 1] {
        let (k, v) = a.split_once('=').unwrap_or_else(|| panic!("bad arg {a}"));
        match k {
            "origin" => {
                let p: Vec<f64> = v.split(',').map(|x| x.parse().unwrap()).collect();
                origin = [p[0], p[1], p[2]];
            }
            "size" => size = v.parse().unwrap(),
            "L" => lmax = v.parse().unwrap(),
            "h_surf" => h_surf = v.parse().unwrap(),
            "grade" => grade = v.parse().unwrap(),
            "h_max" => h_max = v.parse().unwrap(),
            "delta" => delta = v.parse().unwrap(),
            "gamma" => opt.gamma = v.parse().unwrap(),
            "theta_max" => opt.theta_max = v.parse().unwrap(),
            "max_proj_h" => opt.max_proj_h = v.parse().unwrap(),
            "smooth_iters" => opt.smooth_iters = v.parse().unwrap(),
            "core_move" => opt.core_move = v.parse().unwrap(),
            "max_rounds" => opt.max_rounds = v.parse().unwrap(),
            "max_removed_frac" => opt.max_removed_frac = v.parse().unwrap(),
            "remove_all_incident" => opt.remove_all_incident = v == "1",
            "opt_sweeps" => opt.opt_sweeps = v.parse().unwrap(),
            "opt_core" => opt.opt_core = v == "1",
            "opt_ring" => opt.opt_ring = v == "1",
            "opt_cap" => opt.opt_cap = v.parse().unwrap(),
            "opt_min_step" => opt.opt_min_step = v.parse().unwrap(),
            "opt_init" => opt.opt_init = v == "1",
            "sj_remove" => opt.sj_remove = v.parse().unwrap(),
            "layers" => nlayers = v.parse().unwrap(),
            "dy0" => dy0 = v.parse().unwrap(),
            "growth" => growth = v.parse().unwrap(),
            "snap" => snap = v.parse().unwrap(),
            "repair_sj" => repair_sj = v.parse().unwrap(),
            "signs" => {
                mode = match v {
                    "fwn" => SignMode::Fwn,
                    "dexel" => SignMode::Dexel,
                    _ => panic!("signs=fwn|dexel"),
                }
            }
            _ => panic!("unknown key {k}"),
        }
    }
    if h_surf <= 0.0 {
        h_surf = size / (1u64 << lmax) as f64;
    }
    if !h_max.is_finite() {
        h_max = size;
    }
    if delta < 0.0 {
        delta = 4.0 * h_surf;
    }
    opt.delta = delta;
    let oc = Octree::build_clipped(&tris, origin, size, lmax, &SizeField::new(h_surf, grade, h_max), 0, mode, Clip::Shell { delta });
    let t_octree = t0.elapsed().as_secs_f64();
    let viol = oc.balance_violations();

    let t = Instant::now();
    let hc = hexcore::build(&oc, &tris, &opt);
    let t_hexcore = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let lm = (nlayers > 0).then(|| {
        if dy0 <= 0.0 {
            dy0 = layers::fit_dy0(delta, nlayers, growth);
        }
        layers::build(&hc, &tris, &LayerOptions { n: nlayers, dy0, growth, snap, repair_sj })
    });
    let t_layers = t.elapsed().as_secs_f64();
    let (m, s) = (lm.as_ref().map_or(&hc.mesh, |l| &l.mesh), &hc.stats);

    std::fs::create_dir_all(&out).unwrap();
    hexcore::write_shell_quads(&hc.shell, &out.join("shell_quads.bin")).unwrap();
    let t = Instant::now();
    let (ni, patches) = rcore::foam::write_polymesh(m, &out.join("constant").join("polyMesh")).unwrap();
    std::fs::create_dir_all(out.join("system")).unwrap();
    std::fs::write(
        out.join("system").join("controlDict"),
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object controlDict;\n}\n\napplication simpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\nwritePrecision 15;\nwriteFormat ascii;\n",
    )
    .unwrap();
    let hdr = |f: &str| format!("FoamFile\n{{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object {f};\n}}\n\n");
    std::fs::write(
        out.join("system").join("fvSchemes"),
        hdr("fvSchemes") + "ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n",
    )
    .unwrap();
    std::fs::write(out.join("system").join("fvSolution"), hdr("fvSolution") + "solvers {}\n").unwrap();
    let t_foam = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let s2 = rcore::su2::write_su2(m, &out.join("mesh.su2")).unwrap();
    let t_su2 = t.elapsed().as_secs_f64();
    std::io::stdout().flush().unwrap();

    let pj: Vec<String> = patches.iter().map(|(n, c)| format!("\"{n}\": {c}")).collect();
    let stg = |x: &hexcore::SjStage| format!("{{\"min\": {:e}, \"mean\": {:.6}, \"n_lt_0p1\": {}, \"n_lt_0p2\": {}, \"n_lt_0p3\": {}}}", x.min, x.mean, x.n_lt_0p1, x.n_lt_0p2, x.n_lt_0p3);
    let lj = lm.as_ref().map_or(String::new(), |l| format!(", \"layers\": {}, \"t_layers_total\": {:.3}", l.stats.json(), t_layers));
    let ext = format!(
        ", \"sj_min_initial\": {:e}, \"sj_min_after_smooth\": {:e}, \"sj_min_after_opt\": {:e}, \"n_sj_below_0p1\": {}, \"n_sj_below_0p3\": {}, \"opt_sweeps\": {}, \"moved_core_vertices\": {}, \"core_sj_min\": {:e}, \"t_opt\": {:.3}, \"stage_initial\": {}, \"stage_smooth\": {}, \"stage_opt\": {}",
        s.stage_initial.min, s.stage_smooth.min, s.stage_opt.min, s.n_sj_below_0p1, s.n_sj_below_0p3, s.opt_sweeps, s.moved_core_vertices, s.core_sj_min, s.t_opt, stg(&s.stage_initial), stg(&s.stage_smooth), stg(&s.stage_opt)
    ) + &lj;
    let h = |a: &[usize; 5]| format!("[{}, {}, {}, {}, {}]", a[0], a[1], a[2], a[3], a[4]);
    println!(
        "{{\"leaves\": {}, \"balance_violations\": {}, \"delta\": {}, \"gamma\": {}, \"theta_max\": {}, \"core_initial\": {}, \"core_cells\": {}, \"removed_margin\": {}, \"removed_nonfluid\": {}, \"removed_nonmanifold\": {}, \"removed_hanging\": {}, \"removed_split\": {}, \"removed_normal\": {}, \"added_fill\": {}, \"removed_jacobian\": {}, \"rounds\": {}, \"post_rounds\": {}, \"budget_hit\": {}, \"unprojected\": {}, \"ray_targets\": {}, \"smooth_ray_moves\": {}, \"manifold_iters\": {}, \"core_boundary_quads\": {}, \"core_boundary_verts\": {}, \"open_edges\": {}, \"nonmanifold_edges_final\": {}, \"nonmanifold_verts_final\": {}, \"normal_violations_final\": {}, \"jacobian_violations_final\": {}, \"pillow_hexes\": {}, \"core_hex\": {}, \"core_poly\": {}, \"core_split_faces\": {}, \"sj_min_before\": {:e}, \"sj_min_after\": {:e}, \"sj_mean\": {:.6}, \"n_sj_below_0p2\": {}, \"n_sj_below_0\": {}, \"sj_hist_before\": {}, \"sj_hist_after\": {}, \"max_proj_dist_over_h\": {:.4}, \"smooth_accepted\": {}, \"smooth_rejected\": {}, \"core_move_rejected\": {}, \"cells\": {}, \"hex_cells\": {}, \"poly_cells\": {}, \"orphan_faces\": {}, \"nonmanifold_faces\": {}, \"nonpositive_cells\": {}, \"min_vol_ratio\": {:e}, \"volume\": {:.15e}, \"n_points\": {}, \"n_faces\": {}, \"n_internal_faces\": {}, \"patches\": {{{}}}, \"su2\": {{\"tet\": {}, \"hex\": {}, \"prism\": {}, \"pyramid\": {}, \"points\": {}, \"shell_elems\": {}, \"farfield_elems\": {}, \"hex_decomposed\": {}, \"quads_split\": {}, \"nonpositive\": {}}}, \"t_octree\": {:.3}, \"t_hexcore\": {:.3}, \"t_core\": {:.3}, \"t_manifold\": {:.3}, \"t_normal\": {:.3}, \"t_pillow\": {:.3}, \"t_smooth\": {:.3}, \"t_assemble\": {:.3}, \"t_foam\": {:.3}, \"t_su2\": {:.3}, \"t_total\": {:.3}{}}}",
        s.leaves, viol, delta, opt.gamma, opt.theta_max, s.core_initial, s.core_cells, s.removed_margin, s.removed_nonfluid, s.removed_nonmanifold, s.removed_hanging, s.removed_split, s.removed_normal, s.added_fill, s.removed_jacobian, s.rounds, s.post_rounds, s.budget_hit, s.unprojected, s.ray_targets, s.smooth_ray_moves, s.manifold_iters, s.core_boundary_quads, s.core_boundary_verts, s.open_edges, s.nonmanifold_edges_final, s.nonmanifold_verts_final, s.normal_violations_final, s.jacobian_violations_final, s.pillow_hexes, s.core_hex, s.core_poly, s.core_split_faces, s.sj_min_before, s.sj_min_after, s.sj_mean, s.n_sj_below_0p2, s.n_sj_below_0, h(&s.sj_hist_before), h(&s.sj_hist_after), s.max_proj_dist_over_h, s.smooth_accepted, s.smooth_rejected, s.core_move_rejected,
        m.cells.len(), m.stats.hex_cells, m.stats.poly_cells, m.stats.orphan_faces, m.stats.nonmanifold_faces, m.stats.n_nonpositive_cells, m.stats.min_vol_ratio, m.stats.volume, m.points.len(), m.faces.len(), ni, pj.join(", "),
        s2.n_tet, s2.n_hex, s2.n_prism, s2.n_pyramid, s2.n_points, s2.n_wall_elems, s2.n_far_elems, s2.hex_decomposed, s2.quads_split, s2.nonpositive,
        t_octree, t_hexcore, s.t_core, s.t_manifold, s.t_normal, s.t_pillow, s.t_smooth, s.t_assemble, t_foam, t_su2, t0.elapsed().as_secs_f64(), ext
    );
}
