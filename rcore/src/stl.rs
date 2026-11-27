//! STL reader: ASCII and binary (auto-detected), multiple `solid` blocks, no panics (errors are `Err(String)`).
use crate::dexel::Tri;
use std::path::Path;

/// Read an STL file.
pub fn read(path: &Path) -> Result<Vec<Tri>, String> {
    let b = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    parse(&b)
}

/// Parse STL bytes: binary iff `len == 84 + 50 * ntri` (ntri from the header), else ASCII.
pub fn parse(b: &[u8]) -> Result<Vec<Tri>, String> {
    if b.len() >= 84 {
        let n = u32::from_le_bytes([b[80], b[81], b[82], b[83]]) as usize;
        if n.checked_mul(50).and_then(|x| x.checked_add(84)) == Some(b.len()) {
            let f = |o: usize| f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as f64;
            let tris: Vec<Tri> = (0..n)
                .map(|t| {
                    let o = 84 + 50 * t + 12;
                    let mut tr = [[0.0; 3]; 3];
                    for (k, c) in tr.iter_mut().flatten().enumerate() {
                        *c = f(o + 4 * k);
                    }
                    tr
                })
                .collect();
            return finish(tris);
        }
    }
    let s = std::str::from_utf8(b).map_err(|_| "not an STL file (neither valid binary size nor ASCII text)".to_string())?;
    let mut v: Vec<[f64; 3]> = Vec::new();
    for (ln, line) in s.lines().enumerate() {
        let mut w = line.split_whitespace();
        if w.next().is_some_and(|k| k.eq_ignore_ascii_case("vertex")) {
            let c: Vec<f64> = w.map(|x| x.parse::<f64>()).collect::<Result<_, _>>().map_err(|_| format!("line {}: bad vertex", ln + 1))?;
            if c.len() != 3 || c.iter().any(|x| !x.is_finite()) {
                return Err(format!("line {}: vertex needs 3 finite numbers", ln + 1));
            }
            v.push([c[0], c[1], c[2]]);
        }
    }
    if v.len() % 3 != 0 {
        return Err(format!("ASCII STL: {} vertices is not a multiple of 3", v.len()));
    }
    finish(v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect())
}

fn finish(tris: Vec<Tri>) -> Result<Vec<Tri>, String> {
    if tris.is_empty() {
        return Err("no triangles found".into());
    }
    if tris.iter().flatten().flatten().any(|x| !x.is_finite()) {
        return Err("non-finite coordinate in STL".into());
    }
    Ok(tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: [[f32; 3]; 4] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    const F: [[usize; 3]; 4] = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];

    fn binary() -> Vec<u8> {
        let mut b = vec![0u8; 80];
        b.extend(4u32.to_le_bytes());
        for f in F {
            b.extend([0u8; 12]);
            for i in f {
                V[i].iter().for_each(|c| b.extend(c.to_le_bytes()));
            }
            b.extend([0u8; 2]);
        }
        b
    }

    fn ascii() -> String {
        let mut s = String::from("solid a\n");
        for (k, f) in F.iter().enumerate() {
            if k == 2 {
                s += "endsolid a\nsolid b\n";
            }
            s += "facet normal 0 0 0\n outer loop\n";
            for &i in f {
                s += &format!("  vertex {} {} {}\n", V[i][0], V[i][1], V[i][2]);
            }
            s += " endloop\nendfacet\n";
        }
        s + "endsolid b\n"
    }

    #[test]
    fn binary_and_ascii_agree() {
        let (a, b) = (parse(&binary()).unwrap(), parse(ascii().as_bytes()).unwrap());
        assert_eq!(a.len(), 4);
        assert_eq!(a, b);
        assert_eq!(a[2][2], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn malformed_is_error() {
        assert!(parse(b"").is_err());
        assert!(parse(b"solid x\nvertex 0 0 0\nvertex 1 0 0\nendsolid x\n").is_err());
        assert!(parse(b"solid x\nvertex 0 0 a\nendsolid\n").is_err());
        assert!(parse(&[0xff, 0xfe, 0x80]).is_err());
        let mut b = binary();
        b.truncate(b.len() - 7);
        assert!(parse(&b).is_err());
    }
}
