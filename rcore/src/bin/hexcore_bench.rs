//! hexcore_bench SOUP.bin origin=x,y,z size=S L=N h_surf=H grade=G h_max=M [delta=D (4*h_surf)] [gamma=0.3] [theta_max=60] [max_proj_h=2.5]
//! [smooth_iters=5] [core_move=0] [max_rounds=5] [sj_remove=0] [opt_sweeps=0] [opt_core=1] [opt_ring=0] [opt_init=1] [max_removed_frac=0.03] [fillet=0 (rho/delta)] [fillet_k=8] [graded_shell=0|1 (graded conforming shell, default 0)] [grade_z=Z grade_fac=0.5 (synthetic grading: surface size grade_fac * h_surf for triangles with centroid z > Z)] [signs=fwn|dexel] [layers=0] [dy0=] [growth=1.25] [snap=0.4] [repair_sj=0.05] [features=0|1 (default 0)] [cont=0|1|2 (continuous foot map near collision seams / everywhere)] [cont_rings=3] [columns=harmonic|nearest] [hg_k=8] [smooth=25 (harmonic) | 0 (nearest)] OUT_DIR
//! `fillet=F` opens the shell at concave creases with fillet radius rho = F delta (`rcore::shell`, grid spacing delta/fillet_k); 0 = plain offset shell.
//! `layers=N` adds N boundary layers between wall and shell (`rcore::layers`; dy0 default: stack fit for `growth`); patches `wall`, `farfield`.
//! `columns=harmonic` (default): streamline columns of a harmonic field in the band (grid spacing delta/hg_k); `columns=nearest`: straight nearest-point columns.
//! SOUP.bin: u64 ntri, then ntri*9 f64. Builds the shell-clipped buffered octree and the hex-core + pillow mesh (`rcore::hexcore`),
//! writes OUT_DIR/constant/polyMesh/*, OUT_DIR/system/*, OUT_DIR/mesh.su2, OUT_DIR/shell_quads.bin and prints one JSON line.
use rcore::dexel::Tri;
use rcore::hexcore::Options;
use rcore::layers::Columns;
use rcore::octree::SignMode;
use rcore::pipeline::{self, MeshSpec};
use std::io::Read;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: hexcore_bench SOUP.bin origin=x,y,z size= L= h_surf= grade= h_max= [delta=D] [gamma=0.3] [theta_max=60] [max_proj_h=2.5] [smooth_iters=5] [core_move=0] [max_rounds=5] [sj_remove=0] [opt_sweeps=0] [opt_core=1] [opt_ring=0] [opt_init=1] [max_removed_frac=0.03] [fillet=0] [fillet_k=8] [signs=fwn|dexel] OUT_DIR");
        std::process::exit(2);
    }
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
    let mut features_on = false;
    let mut cont_mode = 0u8;
    let mut cont_rings = 3usize;
    let mut zip_on = false;
    let mut feature_angle = 0.0f64;
    let mut edge_nodes = false;
    let (mut curvature, mut curv_c) = (false, 0.25f64);
    let (mut grade_z, mut grade_fac) = (None::<f64>, 0.5f64);
    let (mut columns, mut hg_k, mut smooth) = (Columns::Harmonic, 8.0f64, -1i64);
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
            "edge_nodes" => edge_nodes = v == "1",
            "curvature" => curvature = v == "1",
            "curv_c" => curv_c = v.parse().unwrap(),
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
            "fillet" => opt.fillet = v.parse().unwrap(),
            "fillet_k" => opt.fillet_k = v.parse().unwrap(),
            "graded_shell" => opt.graded_shell = v == "1",
            "grade_z" => grade_z = Some(v.parse().unwrap()),
            "grade_fac" => grade_fac = v.parse().unwrap(),
            "sj_remove" => opt.sj_remove = v.parse().unwrap(),
            "layers" => nlayers = v.parse().unwrap(),
            "dy0" => dy0 = v.parse().unwrap(),
            "growth" => growth = v.parse().unwrap(),
            "snap" => snap = v.parse().unwrap(),
            "repair_sj" => repair_sj = v.parse().unwrap(),
            "features" => features_on = v != "0",
            "cont" => cont_mode = v.parse().unwrap(),
            "cont_rings" => cont_rings = v.parse().unwrap(),
            "zip" => zip_on = v != "0",
            "feature_angle" => feature_angle = v.parse().unwrap(),
            "hg_k" => hg_k = v.parse().unwrap(),
            "smooth" => smooth = v.parse().unwrap(),
            "columns" => {
                columns = match v {
                    "harmonic" => Columns::Harmonic,
                    "nearest" => Columns::Nearest,
                    _ => panic!("columns=harmonic|nearest"),
                }
            }
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
    let spec = MeshSpec { origin, size, lmax, h_surf, grade, h_max, delta, mode, opt, nlayers, dy0, growth, snap, repair_sj, features: features_on, cont: cont_mode, cont_rings, zip: zip_on, columns, hg_k, feature_angle, edge_nodes, smooth, write_foam: true, write_su2: true, write_shell_quads: true, curvature, curv: rcore::sizing::CurvOptions { c: curv_c, ..Default::default() }, extra_levels: 3, grade_z, grade_fac };
    match pipeline::run(&tris, &spec, &out) {
        Ok(r) => println!("{}", r.json),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
