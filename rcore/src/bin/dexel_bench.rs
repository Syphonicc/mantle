//! dexel_bench SOUP.bin GRID.bin OUT_PREFIX [--resolve]
//! SOUP.bin: u64 ntri, then ntri*9 f64 (triangle soup). GRID.bin: u64 n0 n1 n2, then the three axis arrays (f64).
//! Writes OUT_PREFIX.inside.u8 and OUT_PREFIX.fallback.u8 (node order (ix*n1+iy)*n2+iz); prints JSON timings.
use rcore::dexel::{classify, resolve_fallback, Tri};
use std::io::{Read, Write};
use std::time::Instant;

fn read_u64(b: &[u8], o: &mut usize) -> u64 {
    let v = u64::from_le_bytes(b[*o..*o + 8].try_into().unwrap());
    *o += 8;
    v
}
fn read_f64(b: &[u8], o: &mut usize) -> f64 {
    let v = f64::from_le_bytes(b[*o..*o + 8].try_into().unwrap());
    *o += 8;
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: dexel_bench SOUP.bin GRID.bin OUT_PREFIX [--resolve]");
        std::process::exit(2);
    }
    let resolve = args.iter().any(|a| a == "--resolve");
    let t0 = Instant::now();
    let mut sb = Vec::new();
    std::fs::File::open(&args[1]).unwrap().read_to_end(&mut sb).unwrap();
    let mut o = 0;
    let nt = read_u64(&sb, &mut o) as usize;
    let mut tris: Vec<Tri> = Vec::with_capacity(nt);
    for _ in 0..nt {
        let mut t = [[0.0; 3]; 3];
        for v in t.iter_mut() {
            for c in v.iter_mut() {
                *c = read_f64(&sb, &mut o);
            }
        }
        tris.push(t);
    }
    let mut gb = Vec::new();
    std::fs::File::open(&args[2]).unwrap().read_to_end(&mut gb).unwrap();
    let mut o = 0;
    let n: Vec<usize> = (0..3).map(|_| read_u64(&gb, &mut o) as usize).collect();
    let g: Vec<Vec<f64>> = n.iter().map(|&k| (0..k).map(|_| read_f64(&gb, &mut o)).collect()).collect();
    let t_read = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut c = classify(&tris, [&g[0], &g[1], &g[2]]);
    let t_classify = t1.elapsed().as_secs_f64();
    let nfb = c.fallback.iter().filter(|&&f| f).count();
    let t2 = Instant::now();
    if resolve {
        resolve_fallback(&tris, [&g[0], &g[1], &g[2]], &mut c);
    }
    let t_fallback = t2.elapsed().as_secs_f64();

    let wr = |name: &str, v: &[bool]| {
        let bytes: Vec<u8> = v.iter().map(|&b| b as u8).collect();
        std::fs::File::create(format!("{}{}", args[3], name)).unwrap().write_all(&bytes).unwrap();
    };
    wr(".inside.u8", &c.inside);
    wr(".fallback.u8", &c.fallback);
    println!(
        "{{\"tris\": {}, \"nodes\": {}, \"crossings\": {}, \"threads\": {}, \"t_read\": {:.3}, \"t_classify\": {:.4}, \"fallback_nodes\": {}, \"t_fallback\": {:.3}, \"resolved\": {}}}",
        nt,
        n[0] * n[1] * n[2],
        c.n_crossings,
        std::thread::available_parallelism().map(|x| x.get()).unwrap_or(0),
        t_read,
        t_classify,
        nfb,
        t_fallback,
        resolve
    );
}
