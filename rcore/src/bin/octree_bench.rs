//! octree_bench SOUP.bin origin=x,y,z size=S L=N h_surf=H grade=G h_max=M OUT_PREFIX
//! SOUP.bin: u64 ntri, then ntri*9 f64. Builds the balanced octree + corner signs + cut edges, prints one JSON line.
//! Writes OUT_PREFIX.leaves.bin (u64 n, n * (u8 level, u32 i, u32 j, u32 k) little-endian packed, 13 bytes each)
//! and OUT_PREFIX.cuts.bin (u64 n, n * 6 f64: point xyz, normal xyz).
use rcore::dexel::Tri;
use rcore::octree::{Octree, SignMode, SizeField};
use std::io::{Read, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: octree_bench SOUP.bin origin=x,y,z size= L= h_surf= grade= h_max= [signs=fwn|dexel] OUT_PREFIX");
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
    let prefix = args.last().unwrap().clone();
    let mut mode = SignMode::Dexel;
    let (mut origin, mut size, mut lmax, mut h_surf, mut grade, mut h_max) = ([0.0; 3], 1.0, 6u8, 0.0, 0.2, f64::INFINITY);
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
    let oc = Octree::build_with(&tris, origin, size, lmax, &SizeField::new(h_surf, grade, h_max), 0, mode);
    let tb = Instant::now();
    let viol = oc.balance_violations();
    let t_check = tb.elapsed().as_secs_f64();

    let mut lb = Vec::with_capacity(8 + 13 * oc.leaves().len());
    lb.extend((oc.leaves().len() as u64).to_le_bytes());
    for l in oc.leaves() {
        lb.push(l.level);
        for v in [l.i, l.j, l.k] {
            lb.extend(v.to_le_bytes());
        }
    }
    std::fs::File::create(format!("{prefix}.leaves.bin")).unwrap().write_all(&lb).unwrap();
    let mut cb = Vec::with_capacity(8 + 48 * oc.cut_edges().len());
    cb.extend((oc.cut_edges().len() as u64).to_le_bytes());
    for c in oc.cut_edges() {
        for v in c.point.iter().chain(c.normal.iter()) {
            cb.extend(v.to_le_bytes());
        }
    }
    std::fs::File::create(format!("{prefix}.cuts.bin")).unwrap().write_all(&cb).unwrap();

    let tm = oc.timings;
    let per: Vec<String> = oc.leaves_per_level().iter().map(|x| x.to_string()).collect();
    println!(
        "{{\"tris\": {}, \"leaves\": {}, \"leaves_per_level\": [{}], \"unique_vertices\": {}, \"cut_edges\": {}, \"multi_hit_edges\": {}, \"unresolved_edges\": {}, \"balance_violations\": {}, \"t_bvh\": {:.4}, \"t_build\": {:.4}, \"t_balance\": {:.4}, \"t_signs\": {:.4}, \"fallback_vertices\": {}, \"signs\": \"{}\", \"t_cuts\": {:.4}, \"t_check\": {:.4}, \"t_total\": {:.4}, \"threads\": {}}}",
        nt,
        oc.leaves().len(),
        per.join(", "),
        oc.vertices().len(),
        oc.cut_edges().len(),
        oc.cut_edges().iter().filter(|c| c.n_hits > 1).count(),
        oc.unresolved,
        viol,
        tm.bvh,
        tm.build,
        tm.balance,
        tm.signs,
        oc.fallback_vertices,
        if mode == SignMode::Fwn { "fwn" } else { "dexel" },
        tm.cuts,
        t_check,
        t0.elapsed().as_secs_f64(),
        std::thread::available_parallelism().map(|x| x.get()).unwrap_or(0)
    );
}
