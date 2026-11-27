//! OpenFOAM polyMesh writer (ascii): points, faces, owner, neighbour, boundary (`wall`, `farfield`, optional `orphan`).
//! Internal faces are upper-triangular (owner < neighbour, sorted by owner then neighbour); boundary faces are grouped per
//! patch; every face is oriented by the right-hand rule from owner to neighbour / out of the domain.

use crate::cutmesh::{CutMesh, FARFIELD, INTERNAL, NONE, WALL};
use std::io::{BufWriter, Write};
use std::path::Path;

fn header<W: Write>(w: &mut W, class: &str, object: &str, note: &str) -> std::io::Result<()> {
    writeln!(w, "FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       {class};\n    {note}object      {object};\n}}\n")
}

/// Write `dir/{points,faces,owner,neighbour,boundary}`; returns (n_internal_faces, [(patch, n_faces)]).
pub fn write_polymesh(m: &CutMesh, dir: &Path) -> std::io::Result<(usize, Vec<(&'static str, usize)>)> {
    std::fs::create_dir_all(dir)?;
    let nc = m.cells.len();
    // oriented copies: (owner, neighbour, face index, flip)
    let mut internal: Vec<(u32, u32, u32, bool)> = Vec::new();
    let mut bnd: [Vec<(u32, u32)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (i, f) in m.faces.iter().enumerate() {
        match f.patch {
            INTERNAL => {
                let (o, n) = (f.owner, f.neigh);
                if o < n {
                    internal.push((o, n, i as u32, false));
                } else {
                    internal.push((n, o, i as u32, true));
                }
            }
            WALL => bnd[0].push((f.owner, i as u32)),
            FARFIELD => bnd[1].push((f.owner, i as u32)),
            _ => bnd[2].push((f.owner, i as u32)),
        }
    }
    internal.sort_unstable();
    for b in bnd.iter_mut() {
        b.sort_unstable();
    }
    let ni = internal.len();
    let nf = ni + bnd.iter().map(|b| b.len()).sum::<usize>();
    let mut order: Vec<(u32, u32, bool)> = internal.iter().map(|&(o, n, i, fl)| { let _ = i; (o, n, fl) }).collect();
    let mut fidx: Vec<u32> = internal.iter().map(|x| x.2).collect();
    for b in &bnd {
        for &(o, i) in b {
            order.push((o, NONE, false));
            fidx.push(i);
        }
    }

    let mut w = BufWriter::new(std::fs::File::create(dir.join("points"))?);
    header(&mut w, "vectorField", "points", "")?;
    writeln!(w, "{}\n(", m.points.len())?;
    for p in &m.points {
        writeln!(w, "({} {} {})", p[0], p[1], p[2])?;
    }
    writeln!(w, ")")?;
    w.flush()?;

    let mut w = BufWriter::new(std::fs::File::create(dir.join("faces"))?);
    header(&mut w, "faceList", "faces", "")?;
    writeln!(w, "{nf}\n(")?;
    for (k, &fi) in fidx.iter().enumerate() {
        let f = &m.faces[fi as usize];
        write!(w, "{}(", f.verts.len())?;
        let mut vs: Vec<u32> = f.verts.clone();
        if order[k].2 {
            vs.reverse();
        }
        for (t, v) in vs.iter().enumerate() {
            if t > 0 {
                write!(w, " ")?;
            }
            write!(w, "{v}")?;
        }
        writeln!(w, ")")?;
    }
    writeln!(w, ")")?;
    w.flush()?;

    let note = format!("note        \"nPoints:{} nCells:{} nFaces:{nf} nInternalFaces:{ni}\";\n    ", m.points.len(), nc);
    let mut w = BufWriter::new(std::fs::File::create(dir.join("owner"))?);
    header(&mut w, "labelList", "owner", &note)?;
    writeln!(w, "{nf}\n(")?;
    for o in &order {
        writeln!(w, "{}", o.0)?;
    }
    writeln!(w, ")")?;
    w.flush()?;

    let mut w = BufWriter::new(std::fs::File::create(dir.join("neighbour"))?);
    header(&mut w, "labelList", "neighbour", &note)?;
    writeln!(w, "{ni}\n(")?;
    for o in &order[..ni] {
        writeln!(w, "{}", o.1)?;
    }
    writeln!(w, ")")?;
    w.flush()?;

    let mut patches: Vec<(&'static str, usize)> = vec![(m.patch_name, bnd[0].len()), ("farfield", bnd[1].len())];
    if !bnd[2].is_empty() {
        patches.push(("orphan", bnd[2].len()));
    }
    let mut w = BufWriter::new(std::fs::File::create(dir.join("boundary"))?);
    header(&mut w, "polyBoundaryMesh", "boundary", "")?;
    writeln!(w, "{}\n(", patches.len())?;
    let mut start = ni;
    for &(name, n) in &patches {
        let is_clip = name == m.patch_name;
        let ty = if is_clip { "wall" } else { "patch" };
        writeln!(w, "    {name}\n    {{\n        type            {ty};")?;
        if is_clip {
            writeln!(w, "        inGroups        List<word> 1(wall);")?;
        }
        writeln!(w, "        nFaces          {n};\n        startFace       {start};\n    }}")?;
        start += n;
    }
    writeln!(w, ")")?;
    w.flush()?;
    Ok((ni, patches))
}
