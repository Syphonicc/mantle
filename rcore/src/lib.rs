//! rcore: compiled core of the Auto-Mesh shell-first mesher.
//!
//! Module 1, `dexel`: tri-dexel sampling of a triangle soup on a Cartesian grid. Rays run along x, y
//! and z through every grid line; every crossing is recorded (position along the ray, sign of the face
//! normal component, triangle id). Inside/outside of each node comes from the signed crossing count
//! along its rays (1D winding number). Rays with a non-zero net winding (holes, open soups) are
//! inconsistent; nodes touched by any inconsistent ray, or whose consistent rays disagree, are flagged
//! for an exact generalized-winding-number fallback.
//! Module 2, `fwn`: fast generalized winding number (BVH + second-order dipole far field), used to
//! resolve fallback nodes quickly. Module 3, `bvh`: closest point / unsigned distance / segment hits. `tree` is the
//! shared BVH builder.
//! Module 4, `octree`: size-field-driven linear octree with strict 2:1 balance (26-neighbourhood), corner
//! inside/outside signs and exact Hermite cut data (point, triangle, normal) on sign-changing leaf edges.
//!
//! Module 5, `cutmesh`: cut-cell fluid volume mesh (hexahedra + clipped polyhedra) from the buffered octree
//! (`Octree::build_with_buffer`); `foam` writes it as an OpenFOAM polyMesh, `su2` as a conforming SU2 mesh
//! (hex / pyramid / tet decomposition, validated per element, surface elements in SU2's inward orientation). Bad cut cells
//! (small, concave, low interpolation weight) are agglomerated into face neighbours (`CutMesh::agglomerate`).
//!
//! Shell-first clipping (`octree::Clip::Shell { delta }`, `Octree::build_clipped`): the octree is clipped not against the wall but
//! against the offset surface psi = s - delta = 0 (s = signed distance); size field, surface-leaf test, buffer, corner signs
//! (kept = psi > 0) and cut points (1D roots of psi, with nearest wall point / triangle / shell normal) all refer to the shell;
//! the band between wall and shell is dropped (future layer region). The clipped patch is named `shell` and
//! `cutmesh::shell_surface` / `write_shell_surface` export it for the layer step. `Clip::Wall` is the unchanged default.
//!
//! Module 6, `hexcore`: projection + pillow hex-core mesher for the shell-first pipeline (core octree hexes outside the shell,
//! a one-hex pillow layer from the core boundary to the projected shell quads instead of cut cells; cleanup of non-manifold
//! boundary and bad projection normals, Jacobian-guarded smoothing); assembled as a `CutMesh` so `foam` / `su2` write it.
//!
//! `surfq`: graded, conforming shell quad meshes (polygon surface of the core boundary, surface T-join, face templates; `hexcore::Options::graded_shell`).
//!
//! Module 7, `layers`: boundary layers between wall and the offset shell (columns: streamlines of a harmonic field in the band, solved with PCG
//! on a sparse grid, or straight nearest-point columns; geometric node distribution along the arc length, hexes / prisms / pyramids at merged wall nodes, patch `wall`).
//!
//! Module 8, `shell`: opened shell (offset surface with concave-crease ridges replaced by fillets of radius rho; `ShellField`: psi, projection, ray
//! tracing; fast-marching field on a sparse grid), used by `hexcore` (`Options::fillet`) and `layers` (harmonic band bounded by psi = 0).
//!
//! `stl` reads ASCII / binary STL; `pipeline` runs the whole shell-first chain (used by `hexcore_bench` and the `mantle` CLI).
//!
//! Dependency-free (std only); parallel with `std::thread::scope`.

pub mod checks;
pub mod dexel;
pub mod tree;
pub mod fwn;
pub mod bvh;
pub mod octree;
pub mod cutmesh;
pub mod foam;
pub mod su2;
pub mod surfq;
pub mod hexcore;
pub mod layers;
pub mod shell;
pub mod features;
pub mod stl;
pub mod sizing;
pub mod pipeline;
pub mod quality;
pub mod report;
