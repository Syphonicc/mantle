# proto2d: shell-first hybrid mesher, 2D prototype

See `../RESEARCH_volume_meshing_v1.md` sections 4, 10, 11.

```bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
python3 sfhm2d_v1.py --geom airfoil --out runs/sfhm_airfoil --plot            # geometries: airfoil square twoelement dirty
python3 sfhm2d_v1.py --geom twoelement --out runs/sfhm_twoelement --variable  # variable-thickness shell (gap clipping)
python3 snappy_compare_v1.py --geom airfoil --out runs/snappy_airfoil         # snappyHexMesh baseline, same spec
python3 foam2d_v1.py metrics runs/sfhm_airfoil 0.00101                        # first-cell height stats vs target
python3 foam2d_v1.py plot runs/snappy_airfoil 1.0 0.0 0.06 out.png            # window plot of any 2D polyMesh
python3 flow_case_v1.py runs/sfhm_airfoil 4.0 1e6 && (cd runs/sfhm_airfoil && simpleFoam)
```
v2 (eikonal shell + safety net), v3 (y+-aware layers), v4 (ingest defeaturing, current); runs in `runs/v2..v4/`, each version reproduces the previous via `--shell warped` / `--layers fixedN` / `--no-clean`:
```bash
python3 sfhm2d_v4.py --geom dirty --out runs/v4/sfhm_dirty --plot   # repairs are printed and stored in sfhm_info.json
```
Outputs: `runs/<case>/constant/polyMesh`, `sfhm_info.json`, `mesh_*.png`, `log.checkMesh`.
