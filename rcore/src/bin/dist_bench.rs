//! dist_bench SOUP.bin PTS.bin OUT.bin
//! SOUP.bin: u64 ntri, then ntri*9 f64. PTS.bin: u64 npts, then npts*3 f64. OUT.bin: npts f64 unsigned distances.
use rcore::bvh::Bvh;
use rcore::dexel::Tri;
use std::io::{Read, Write};
use std::time::Instant;

fn read_all(p: &str) -> Vec<u8> {
    let mut b = Vec::new();
    std::fs::File::open(p).unwrap().read_to_end(&mut b).unwrap();
    b
}
fn f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: dist_bench SOUP.bin PTS.bin OUT.bin");
        std::process::exit(2);
    }
    let sb = read_all(&args[1]);
    let nt = u64::from_le_bytes(sb[..8].try_into().unwrap()) as usize;
    let v = f64s(&sb[8..8 + nt * 72]);
    let tris: Vec<Tri> = v.chunks_exact(9).map(|c| [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]]).collect();
    let pb = read_all(&args[2]);
    let np = u64::from_le_bytes(pb[..8].try_into().unwrap()) as usize;
    let v = f64s(&pb[8..8 + np * 24]);
    let pts: Vec<[f64; 3]> = v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let t0 = Instant::now();
    let bvh = Bvh::new(&tris);
    let t_build = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let hits = bvh.closest_many(&pts);
    let t_query = t1.elapsed().as_secs_f64();
    let bytes: Vec<u8> = hits.iter().flat_map(|h| h.0.to_le_bytes()).collect();
    std::fs::File::create(&args[3]).unwrap().write_all(&bytes).unwrap();
    println!("{{\"t_build\": {:.4}, \"t_query\": {:.4}, \"npts\": {}, \"tris\": {}}}", t_build, t_query, np, nt);
}
