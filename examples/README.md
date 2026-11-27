# mantle: command-line mesher

Closed STL surface in, hybrid CFD mesh out (boundary layers + hex core, OpenFOAM and SU2).

## Build

    cd rcore && cargo build --release        # std only, binary: rcore/target/release/mantle

## Examples

    python3 examples/make_examples.py        # writes sphere.stl and box.stl (numpy)
    mantle examples/sphere.stl -o runs/sphere
    mantle examples/sphere.stl -o runs/sphere_y1 --yplus 1 --re 1e6
    mantle --help

Defaults come from the STL bounding box (length Lb): `--preset medium` (surface size Lb/48; coarse Lb/32, fine Lb/80),
layer thickness 4 h_surf, 12 layers (growth 1.2), cubic domain of side about 5 Lb centred on the body, filleted shell.
For thin bodies (wings) the sizing is capped from an estimated thickness so the layer stack stays below half the local thickness.
Shallow facet edges of a coarse tessellation (below `--feature-angle`, default 25 deg) are treated as a smooth wall;
only sharper edges get fans (prisms / pyramids) of merged wall nodes. Explicit flags override the preset. The input must be a closed,
consistently oriented (outward normals) surface; units are kept.

## Example geometries

| File | Body | Notes |
|---|---|---|
| `sphere.stl` | sphere, R 0.5, 1,944 triangles | deliberately coarse tessellation |
| `box.stl` | box 1 x 0.6 x 0.4, 12 triangles | sharp edges and corners (layer fans) |
| `oneram6_full.stl` | ONERA M6 wing, 11,280 triangles | mirrored to full span from the SU2 test case (`proto2d/m6_extract_v1.py`) |

## What you get in CASE_DIR

- `constant/polyMesh`, `system/{controlDict,fvSchemes,fvSolution}`: OpenFOAM case, patches `wall` and `farfield`
- `mesh.su2`: SU2 mesh, markers `wall` and `farfield`
- `mantle_summary.json`: all parameters and statistics (cell counts, quality, first cell, timings)

## Check

    source /usr/lib/openfoam/openfoam2512/etc/bashrc
    checkMesh -case CASE_DIR
    SU2_CFD your_config.cfg                  # MESH_FILENAME= CASE_DIR/mesh.su2, MARKER_HEATFLUX/ISOTHERMAL= ( wall ), MARKER_FAR= ( farfield )
