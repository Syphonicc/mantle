//! cutmesh_bench SOUP.bin origin=x,y,z size=S L=N h_surf=H grade=G h_max=M [eps=0.05] [merge_tau=0.1; 0 when clip=shell] [signs=fwn|dexel] [clip=wall|shell] [delta=D] OUT_DIR
//! clip=shell clips against the offset surface at distance delta (default 4*h_surf), patch `shell`, and writes OUT_DIR/shell_surface.bin.
//! SOUP.bin: u64 ntri, then ntri*9 f64. Builds the buffered octree, the fluid cut-cell mesh, writes
//! OUT_DIR/constant/polyMesh/*, OUT_DIR/mesh.su2 and OUT_DIR/system/controlDict, prints one JSON line.
use rcore::cutmesh::{self, Options};
use rcore::dexel::Tri;
use rcore::octree::{Clip, Octree, SignMode, SizeField};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: cutmesh_bench SOUP.bin origin=x,y,z size= L= h_surf= grade= h_max= [eps=0.05] [merge_tau=0.1 (0 for shell)] [signs=fwn|dexel] [clip=wall|shell] [delta=D] OUT_DIR");
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
    let (mut origin, mut size, mut lmax, mut h_surf, mut grade, mut h_max, mut eps, mut merge_tau) = ([0.0; 3], 1.0, 6u8, 0.0, 0.2, f64::INFINITY, 0.05, -1.0);
    let (mut shell, mut delta) = (false, -1.0f64);
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
            "eps" => eps = v.parse().unwrap(),
            "merge_tau" => merge_tau = v.parse().unwrap(),
            "clip" => {
                shell = match v {
                    "wall" => false,
                    "shell" => true,
                    _ => panic!("clip=wall|shell"),
                }
            }
            "delta" => delta = v.parse().unwrap(),
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
    if merge_tau < 0.0 {
        merge_tau = if shell { 0.0 } else { 0.1 };
    }
    let clip = if shell { Clip::Shell { delta } } else { Clip::Wall };
    let oc = Octree::build_clipped(&tris, origin, size, lmax, &SizeField::new(h_surf, grade, h_max), 0, mode, clip);
    let t_octree = t0.elapsed().as_secs_f64();
    let viol = oc.balance_violations();

    let t = Instant::now();
    let m = cutmesh::build(&oc, &tris, &Options { eps, merge_tau });
    let t_cutmesh = t.elapsed().as_secs_f64();

    let (mut sv, mut sp) = (0usize, 0usize);
    if shell {
        std::fs::create_dir_all(&out).unwrap();
        let ss = cutmesh::shell_surface(&m, &tris);
        cutmesh::write_shell_surface(&ss, &out.join("shell_surface.bin")).unwrap();
        (sv, sp) = (ss.pts.len(), ss.polys.len());
    }
    let t = Instant::now();
    let (ni, patches) = rcore::foam::write_polymesh(&m, &out.join("constant").join("polyMesh")).unwrap();
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
    let s2 = rcore::su2::write_su2(&m, &out.join("mesh.su2")).unwrap();
    let t_su2 = t.elapsed().as_secs_f64();
    std::io::stdout().flush().unwrap();

    let s = &m.stats;
    let pj: Vec<String> = patches.iter().map(|(n, c)| format!("\"{n}\": {c}")).collect();
    let line = format!(
        "{{\"leaves\": {}, \"balance_violations\": {}, \"fluid_cells\": {}, \"hex_cells\": {}, \"poly_cells\": {}, \"cut_leaves\": {}, \"cut_cells\": {}, \"multi_component_cells\": {}, \"unresolved_edges\": {}, \"multi_hit_nonfirst\": {}, \"snapped_points\": {}, \"flipped_corners\": {}, \"ambiguous_faces\": {}, \"ambiguous_connected\": {}, \"dropped_faces\": {}, \"open_wall_chains\": {}, \"unclosed_cells\": {}, \"degenerate_cells\": {}, \"nonmanifold_faces\": {}, \"orphan_faces\": {}, \"min_vol_ratio\": {:e}, \"nonpositive_cells\": {}, \"volume\": {:.15e}, \"volume_foam_def\": {:.15e}, \"n_points\": {}, \"n_faces\": {}, \"n_internal_faces\": {}, \"patches\": {{{}}}, \"su2\": {{\"tet\": {}, \"hex\": {}, \"prism\": {}, \"pyramid\": {}, \"points\": {}, \"wall_elems\": {}, \"farfield_elems\": {}, \"apex_fallbacks\": {}}}, \"nonpositive_su2_elements\": {}, \"t_octree\": {:.3}, \"t_cutmesh\": {:.3}, \"t_cut_edges\": {:.3}, \"t_cut_faces\": {:.3}, \"t_cut_cells\": {:.3}, \"t_cut_final\": {:.3}, \"t_foam\": {:.3}, \"t_su2\": {:.3}, \"t_total\": {:.3}}}",
        s.leaves, viol, m.cells.len(), s.hex_cells, s.poly_cells, s.cut_leaves, s.cut_cells, s.multi_component_cells, s.unresolved_edges, s.multi_hit_nonfirst, s.snapped_points, s.flipped_corners, s.ambiguous_faces, s.ambiguous_connected, s.dropped_faces, s.open_wall_chains, s.unclosed_cells, s.degenerate_cells, s.nonmanifold_faces, s.orphan_faces, s.min_vol_ratio, s.n_nonpositive_cells, s.volume, s.volume_foam_def, m.points.len(), m.faces.len(), ni, pj.join(", "),
        s2.n_tet, s2.n_hex, s2.n_prism, s2.n_pyramid, s2.n_points, s2.n_wall_elems, s2.n_far_elems, s2.apex_fallbacks, s2.nonpositive,
        t_octree, t_cutmesh, s.t_edges, s.t_faces, s.t_cells, s.t_final, t_foam, t_su2, t0.elapsed().as_secs_f64()
    );
    let q = |q: &cutmesh::Quality| format!("{{\"bad_cut_cells\": {}, \"small\": {}, \"concave\": {}, \"concave_in_merged_cells\": {}, \"foam_pyr_tet_fail\": {}, \"not_star_shaped\": {}, \"low_weight_faces\": {}, \"vol_ratio_faces\": {}}}", q.bad, q.small, q.concave, q.concave_merged, q.bad_foam_tets, q.not_star_shaped, q.low_weight_faces, q.vol_ratio_faces);
    println!(
        "{}, \"clip\": \"{}\", \"delta\": {}, \"shell_cut_edges\": {}, \"root_iterations_max\": {}, \"shell_surface_verts\": {}, \"shell_surface_polys\": {}, \"merge_tau\": {}, \"cells_merged\": {}, \"merge_groups\": {}, \"merge_passes\": {}, \"unmergeable_bad_cells\": {}, \"bad_cells_before\": {}, \"bad_cells_after\": {}, \"min_vol_ratio_before\": {:e}, \"min_vol_ratio_after\": {:e}, \"quality_before\": {}, \"quality_after\": {}, \"t_merge\": {:.3}, \"su2_elements\": {{\"invalid\": {}, \"hex_decomposed\": {}, \"quads_split\": {}}}}}",
        line.trim_end().trim_end_matches('}'), if shell { "shell" } else { "wall" }, if shell { delta } else { 0.0 }, if shell { oc.cut_edges().len() } else { 0 }, oc.root_iterations_max, sv, sp, merge_tau, s.cells_merged, s.merge_groups, s.merge_passes, s.unmergeable_bad_cells, s.bad_cells_before, s.bad_cells_after, s.min_vol_ratio_before, s.min_vol_ratio, q(&s.q_before), q(&s.q_after), s.t_merge, s2.nonpositive, s2.hex_decomposed, s2.quads_split
    );
}
