# Volume meshing for complex geometry: landscape, debate, and a ranked proposal

Searched 13 Nov 2026 (arXiv, vendor docs, IMR, JCP, CGF, patents). Companion to README.md, which covers the *sizing / y+ / ML-spec* side. This document covers the *core volume-meshing algorithm*: how to turn a dirty STL into a CFD-quality body-fitted mesh fast, and what is actually new.

---

## 1. What industry does today (and where each one breaks)

| Tool | Bulk | Boundary layer (BL) | Transition | Known weak spot |
|---|---|---|---|---|
| Ansys Fluent **Mosaic poly-hexcore** | octree hex | poly-prisms grown from a wrapped surface | general polyhedra (patented family: US10803661B1, US12094059B1; "Mosaic" itself patent-pending in the white paper) | prisms still come from advancing layers; collapse is hidden by "last-ratio" tapering; surface wrap is serial-ish |
| Siemens **STAR-CCM+ trimmer** | octree hex, cut at wall | prisms via inflation | hanging-node polyhedra | wrapper + remesher is the time sink; MPI wrapper only arrived in 2506/2510 (≈2–5.6× vs legacy) |
| Cadence **Hexpress** | octree hex, snapped | "viscous layer insertion" = inflate mesh away from wall, insert layers (linear-elastic smoothing) | none needed | snapping quality at sharp features; concave corners |
| Cadence **Pointwise T-Rex + Voxel** | voxel hex far field | anisotropic tet extrusion, recombined into prisms/hexes | pyramids/tets | extrusion stops early at collisions → pyramids; needs a good surface mesh first |
| **snappyHexMesh** | octree hex, castellate → snap | shrink-and-add layers (medial-axis or displacementMotionSolver) | 2:1 polyhedra | layers collapse at sharp edges and in gaps; 60+ parameters; practitioners call its BL "objectively worse" than structured |
| **cfMesh** | octree, polyhedral transitions | extrusion of surface | polyhedra | fewer knobs, same BL failure class |
| **Converge** | runtime Cartesian cut-cell + AMR | "inlaid" extruded BL mesh (optional) | none | cut-cell wall accuracy; inlaid mesh is user-supplied |
| **Cart3D / BoXeR / Kleb-Fidkowski-Martins 2026** | cut-cell, embedded boundary | none: ODE wall function with momentum balance, y+ > 600 | n/a | accuracy with separation/transition; not body-fitted |
| **PowerFLOW / ultraFluidX / Octoplus (LBM)** | octree voxels | none: wall model | n/a | "progress needed on wall models" (AutoCFD workshop) |
| Flexcompute **Flow360 + GeometryAI** | automated unstructured pipeline, GridPro partnership for structured | prisms | tets | AI is used for CAD prep, not for the volume algorithm |

**Pattern.** Everyone converged on the same skeleton: *octree bulk + extruded BL + something to glue them*. Every one of them generates the BL **outward from the wall** by advancing layers, then fights collisions, collapses, and the glue. That outward step is where all the failures and all the parameters live.

Throughput reference points (so we know what "fast" means): Fluent rapid-octree 124 M cells/min on 3,960 cores; 1 B cells in 24 min on 768 cores. Any new method must be per-cell parallel or it loses.

## 2. What academia has that industry has not absorbed yet

**Robust geometry ingest, no repair step**
- Generalized winding numbers (Jacobson 2013; fast version Barill 2018): inside/outside for triangle soup with holes, overlaps, self-intersections. Sign × unsigned distance gives a usable SDF for dirty STL.
- Tusqh (2025, Sandia): background grid + winding number + persistent homology to *control topology* of the meshed domain (close pinholes, drop slivers) by one threshold. Warns that volume-fraction topology does not always converge under refinement.
- GPU-native geometry embedding in octrees (Dec 2025, arXiv 2512.01251): voxelization + near-wall refinement + cut-link tables entirely on device for 7 M-triangle models.

**Guaranteed-valid grid-based hex meshing**
- **MCHex** (arXiv 2511.02064, 2026): Marching Cubes inside each octree cell, constrained to 3-regular polyhedra, midpoint-subdivided to hexes. *Proven positive Jacobian for every hex*, second-order boundary convergence, fastest wall-clock of all grid-based methods on 202 models, linear in #hex. Weaknesses: no sharp features (MC limitation), isotropic padding only, ~8× element multiplication from midpoint subdivision. Code to be released.
- Gao–Shen–Panozzo 2019: feature-preserving octree hex with bijective feature-line mapping; Maréchal Hexotic/MeshGems-Hexa: industrial octree all-hex with sharp-angle handling; HybridOctree_Hex 2024: Jacobian-controlled.

**Boundary-layer robustness**
- Ye, Liu, Ni, Chen (JCP 2025): *full-layer* prisms with no pyramids by deforming a thin initial mesh toward an orthogonal target with an ARAP bijective-mapping energy. Handles complete aircraft. Still outward, still optimization-heavy.
- Reberol et al. (ACM TOMS 2023): robust *topological* construction of all-hex BL meshes (fans at convex/concave features by combinatorics, not geometry).
- Harmonic-field prisms (Adv. Aerodynamics 2021) and vector-Laplace / eikonal marching directions (Löhner, Loseille–Alauzet): PDE fields give non-crossing extrusion directions globally instead of smoothed normals.
- Zint et al. (CGF 2023) "Feature-preserving offset mesh generation from topology-adapted octrees": explicitly produces an *offset B-rep as the coupling interface between BL mesh and background mesh*. **Topological Offsets** (Zint, Panozzo et al. 2024/25): provably manifold, intersection-free offsets homeomorphic to an infinitesimal offset. **Robust & feature-preserving offset meshing** (arXiv 2412.15564): mitered offsets that keep sharp features.

**Where ML really sits**
- Sizing fields: AMBER, LAMG, GCN sizing (arXiv 2510.08645), Lock et al. anisotropic metrics (inviscid only).
- Mesh-quality-before-meshing: Sandia/Utah 2025 thesis (defeaturing decisions).
- Learned isosurfacing: Neural Dual Contouring / Self-Supervised DC — MCHex authors name this as their path to sharp features.
- Generative hex (DDPM-Polycube, PolycubeNet 2026): polycube topology for IGA/solids; not applicable to external aero domains.
- RL meshing (Dmsh 2026, SAC quads): 2D only.
- LLM agents (Foam-Agent 2.0, OpenFOAMGPT 2.0): parameter writers; no quality guarantees.
- Owen et al. survey (Dec 2025): no AI work on prisms/BL/y+ at all. Confirms the README gap.

## 3. The debate

**Q: Is the body-fitted BL mesh even worth keeping?** Cut-cell + ODE wall function (Kleb 2026) and LBM wall models give meshing-free pipelines in minutes. *Against:* both groups report wall-model limits at separation, strong pressure gradients, transition; y+ > 600 is far outside what the README's y+-spec promises. *Verdict:* keep the body-fitted BL as the primary product, but keep cut-cell/wall-function as **mode 0 fallback** when the geometry is so dirty or thin that a BL shell cannot exist. The ML gate chooses the mode.

**Q: Why not just re-implement poly-hexcore in the open?** It is proven and OpenFOAM eats polyhedra. *Against:* patent exposure, zero novelty, and it inherits the outward-advancing BL failure class. Only worth doing as a baseline.

**Q: Why not a learned mesher (diffusion/transformer emits cells)?** *Against:* no validity guarantee, no 1e8-cell scaling, no datasets, and the README already rules that "the network never builds the mesh". Rejected as core.

**Q: Polycube / frame-field all-hex?** Excellent for solids and IGA. External aero domains (fluid around a car with mirrors, wheels, underbody) have far too complex a topology; singularity structure explodes. Rejected.

**Q: So what is the structural flaw we can attack?** Every production tool builds layers *outward from the wall into unknown space*, which is why collisions, gaps and pyramids appear and why glue elements are needed. If the **outer surface of the boundary layer is computed first, as a guaranteed object**, then (a) the bulk mesher sees a *smooth* target, which is exactly the case where grid-based hex meshing has proofs, and (b) the layers can be filled by a **bijection from that smooth outer surface down to the wall**, which is a well-posed problem (thin shell, known topology) rather than a collision-avoidance heuristic. Nobody assembles the pipeline in that order.

## 4. Proposed method: **Shell-first hex meshing (SFHM)**

```
dirty STL ──▶ (1) GPU octree SDF, sign by fast winding number  [no repair]
          ──▶ (2) sizing: curvature/feature/medial-gap rules + ML spec (README) → octree levels, δ_BL(x), Δy0(x), N
          ──▶ (3) BL outer shell Γ = topological offset at distance δ_BL(x), clipped to 0.4·local gap (medial axis from SDF)
          ──▶ (4) bulk: MCHex on {SDF > δ_BL} bounded by Γ → all-hex, positive Jacobian proven, per-cell parallel
          ──▶ (5) BL fill: solve Laplace φ in shell (φ=0 wall, φ=1 Γ); trace ∇φ streamlines from every
                 quad vertex of the bulk boundary down to the wall; place N nodes by growth ratio;
                 snap last layer to exact STL + feature lines; convex wall edges → combinatorial fans (Reberol)
          ──▶ (6) verify: checkMesh + short solve + y+ (README loop) → feed back to (2)
```

**Why each step is defensible**
1. Winding-number SDF is the only ingest that needs no wrapper and no repair; the GPU octree paper already shows device-side voxelization + near-wall refinement at 7 M triangles.
2. Sizing is where the README's ML lives; it is unchanged.
3. Topological Offsets give a manifold, non-self-intersecting Γ with the topology of an infinitesimal offset; Γ is C¹ across convex wall features (rounded by the offset) and creased only where two walls are < 2δ apart, which is exactly the "gap" case and is now an explicit surface we can inspect instead of a collision we discover late.
4. MCHex's only published weaknesses are "no sharp features" and "isotropic padding". Γ has no convex sharp features, and we do not want padding there. So the bulk gets the fastest method with a Jacobian proof, applied where its proof is strongest. Bulk boundary is a quad mesh Q on Γ with 2:1 templates already resolved.
5. In a shell homotopy-equivalent to wall × [0,1] (which Topological Offsets guarantee), a harmonic φ with monotone boundary data has no interior critical points, so ∇φ streamlines are a bijection from Γ to the wall. Layers are hexes that share Q exactly: **no transition elements, no pyramids, no hanging polyhedra at the BL interface**. Concave creases of Γ fan out onto the wall (larger wall quads; refine with 2:1 hex templates or accept polyhedra there, both OpenFOAM-legal). Convex sharp wall edges make streamlines converge; use Reberol's topological fans rather than degenerate hexes. Anisotropic diffusivity in the Laplace operator (the 2021 harmonic-field trick) keeps layers near-orthogonal.
6. Same verification loop as README.

**Why it is fast.** Steps 1, 3-evaluation, 4 and the streamline tracing in 5 are per-cell / per-vertex parallel (GPU). Step 5's Laplace solve is a single sparse SPD solve on the thin shell only (≈10–20 % of cells), a few CG iterations with AMG on a GPU. There is no surface wrapping, no snapping iteration, no shrink-layer iteration, no collision detection. Expected cost scaling: linear in cells, dominated by the SDF build.

**Guarantees we can actually claim**
- Bulk: every hex has positive Jacobian (MCHex theorem).
- BL: full layer coverage with no pyramids on every wall patch where the shell exists (bijection argument); explicit, inspectable list of places where the shell was clipped (gaps) *before* any volume mesh is generated.
- Interface: conformal by construction.

## 5. Self-check: novelty audit

| Component | Prior art | New? |
|---|---|---|
| Offset surface as BL/background interface | Zint 2023, PostBL, T-Rex→Voxel | no |
| Guaranteed manifold offset | Topological Offsets 2024/25 | no (we are first to use it for CFD BL shells, as far as searched) |
| Harmonic/Laplace extrusion directions | Adv. Aero 2021, Löhner, Loseille | no, but all run *outward* |
| MC-based all-hex with Jacobian proof | MCHex 2026 | no |
| Meshing the bulk *to the offset* so MC's sharp-feature weakness is moot | not found | **yes** |
| Inverse (shell → wall) bijective layer fill, conformal to a hex bulk, no transition elements | not found (Ye 2025 is outward + optimization) | **yes** |
| Winding-number ingest + topological offset + MCHex + inverse harmonic fill + ML spec + verify loop | not found | **yes (pipeline)** |

Self-assigned novelty: **7/10** (two structural ideas plus an unseen combination; every building block has a paper or code, which is a feature, not a bug). Feasibility: high. Patent exposure: low; none of the building blocks is Mosaic's polyhedral-transition claim, and the design needs no transition cells at all.

Weaknesses I could not argue away:
- Variable-radius offsets: Topological Offsets is stated for constant d; a spatially varying δ(x) needs the SDF-level-set route plus Tusqh-style topological anti-aliasing. Must be prototyped.
- MCHex multiplies element count ≈8× by midpoint subdivision. Cell count may lose to poly-hexcore on the bulk. Mitigation: dual-octree hex (Maréchal) in the interior, MCHex only in the band touching Γ; or accept polyhedra in the band.
- Layer orthogonality degrades near concave creases of Γ; needs the anisotropic-diffusivity Laplacian and a quality smoother.
- Wall tangential resolution is inherited from the bulk at Γ; the sizing field must be set on the wall and transported to Γ, or features will be under-resolved.
- Topological anti-aliasing can silently delete real small features (Tusqh non-convergence caveat). Needs an explicit report to the user.

## 6. Runner-up methods, ranked

1. **SFHM (above)** — novelty 7, speed high, risk medium.
2. **GPU cut-cell + ODE wall function + runtime AMR** (Converge-style, Kleb 2026 physics) — novelty 4, speed highest, accuracy limited by the wall model; keep as mode 0.
3. **Open poly-hexcore + LLM/ML parameter agent** (Foam-Agent-style) — novelty 2, patent risk; keep as baseline for benchmarks.
4. **Learned MC vertex placement (NDC) on top of MCHex** for sharp features — novelty 5 but MCHex authors already list it; worth doing only if we want sharp features on the bulk, which SFHM avoids.
5. Polycube/frame-field all-hex — not applicable to external aero.

## 7. Where the "AI bot" goes

- Sizing / δ_BL / Δy0 / N prediction from geometry + Re/Ma (README Phases 1–2).
- **Mode gate**: from SDF-octree statistics (gap histogram, feature density, winding-number ambiguity) predict whether the shell will be clipped, and choose SFHM vs mode 0 per region.
- Learned snap of the last layer (NDC-style) for feature-preserving wall conformity.
- Verification agent reading checkMesh / y+ and adjusting the spec (README Phase 3).
The mesher itself stays deterministic.

## 8. Fastest path to evidence

1. **2D prototype (days):** polygon soup → winding-number SDF on quadtree → offset curve → marching-squares quads bulk → Laplace in the annulus → streamline layers → OpenFOAM 2D case via extrudeMesh. Geometries: multi-element airfoil (gap case), square with sharp corners (fan case), dirty self-intersecting airfoil (ingest case). Compare with snappyHexMesh on layer coverage, non-orthogonality, cell count, wall-clock.
2. **3D band test:** use MCHex code when released (or HybridOctree_Hex / Gao–Panozzo as stand-in) on an offset of DrivAerNet geometry; measure Jacobian, cell count vs a poly-hexcore reference from the literature.
3. **Shell robustness study:** Topological Offsets code (NYU GCL) on 50 dirty DrivAerNet/ShapeNet-car STLs; count clipped regions.
4. Only then: GPU port of SDF + streamlines.

## 9. Sources

- Ansys Mosaic white paper — https://www.ansys.com/content/dam/resource-center/white-paper/ansys-fluent-mosaic-technology-wp.pdf ; Poly-hexcore high-lift — https://nal.res.in/themes/stark/cfdimgs/FullPaper/P41-ANSYS-Mosaic-Poly-Hexcore-Mesh-For-High-Lift-Aircraft-Configuration.pdf ; Rapid octree — https://www.ansys.com/blog/what-is-rapid-octree-meshing
- Ansys patents: US10803661B1 — https://patents.google.com/patent/US10803661B1/en ; US12094059B1 — https://patents.google.com/patent/US12094059B1/en
- STAR-CCM+ 2510/2602 meshing — https://volupe.com/?p=27650 , https://volupe.com/?p=29211
- Pointwise T-Rex / Voxel — https://resources.system-analysis.cadence.com/blog/evolution-of-fidelity-pointwise-t-rex-meshing-for-resolving-viscous-layers , https://community.cadence.com/cadence_blogs_8/b/cfd/posts/hex-core-voxel-meshes-the-best-of-structured-and-unstructured-meshing
- Converge inlaid mesh — https://convergecfd.com/blog/converge-3-0-from-specialized-software-to-cfd-powerhouse
- Kleb, Fidkowski, Martins 2026 (cut-cell ODE wall function) — https://mdolab.engin.umich.edu/bibliography/Kleb2026
- Octoplus / LBM GPU octree — https://arxiv.org/abs/2504.04465 ; GPU-native geometry embedding — https://arxiv.org/abs/2512.01251
- Flow360 / GridPro / GeometryAI — https://www.digitalengineering247.com/article/gpu-native-cfd-integrated-with-structured-mesh-generation
- MCHex — https://arxiv.org/abs/2511.02064 ; HybridOctree_Hex — https://arxiv.org/abs/2401.05984 ; Gao–Shen–Panozzo 2019 — https://cims.nyu.edu/gcl/papers/2019-OctreeMeshing.pdf ; Maréchal/Hexotic — https://link.springer.com/doi/10.1007/978-3-642-04319-2_5 ; Hex survey — https://arxiv.org/abs/2202.12670
- Tusqh — https://arxiv.org/abs/2502.10609 ; Generalized winding numbers — https://igl.ethz.ch/projects/winding-number/ ; Fast winding numbers — https://www.dgp.toronto.edu/projects/fast-winding-numbers/
- Topological Offsets — https://arxiv.org/abs/2407.07725 ; Feature-preserving offset meshing — https://arxiv.org/abs/2412.15564 ; Zint 2023 octree offsets — https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14906
- Ye et al. JCP 2025 bijective full-layer prisms — https://www.sciencedirect.com/science/article/abs/pii/S0021999125000270 ; IMR 2023 rigid mapping — https://internationalmeshingroundtable.com/assets/papers/2023/06-Ye-compressed.pdf
- Reberol 2023 all-hex BL topology — https://dl.acm.org/doi/10.1145/3577196 ; Harmonic-field prisms 2021 — https://www.sciopen.com/article_pdf/10.1186/s42774-021-00065-y.pdf ; PostBL — https://www.mcs.anl.gov/papers/P4092-0713_1.pdf ; Loseille–Löhner robust BL — https://scholar.archive.org/work/jtxumn5f3rekzdtv246s5npvxa
- BL generation survey list — https://github.com/Bigger-and-Stronger/boundary-layer-generation-survey
- snappyHexMesh layers doc — https://doc.openfoam.com/2606/tools/pre-processing/mesh/generation/snappyhexmesh/layers ; practitioner view — https://blttkgl.github.io/posts/2026/01/meshing
- ML: AMBER — https://arxiv.org/abs/2505.23663 ; LAMG — https://arxiv.org/abs/2505.20457 ; GCN sizing — https://arxiv.org/abs/2510.08645 ; Self-supervised DC — https://arxiv.org/abs/2405.18131 ; DDPM-Polycube — https://arxiv.org/abs/2503.13541 ; PolycubeNet — https://arxiv.org/abs/2605.20274 ; Dmsh — https://arxiv.org/abs/2606.10601 ; Owen et al. survey — https://arxiv.org/abs/2512.23719 ; Foam-Agent 2.0 — https://arxiv.org/abs/2509.18178

---

## 10. 2D prototype results (13 Nov 2026)

Code: `proto2d/sfhm2d_v1.py` (mesher), `proto2d/snappy_compare_v1.py` (baseline), `proto2d/foam2d_v1.py` (metrics/plots), `proto2d/flow_case_v1.py` (RANS setup). Same spec for both meshers: shell thickness δ = 0.02c, 8 layers, growth 1.25 (first cell 0.00101c), finest cell 0.0078c, 16c box. snappyHexMesh v2512 run on a 3D slab (no 2D mode exists) and reduced to 2D with extrudeMesh; its wall-clock is therefore not comparable and is listed only for completeness.

| Case | Mesher | cells | checkMesh hard failures | first-cell height within ±20 % of spec | y+ max (RANS) | Cl | Cd | mesh time |
|---|---|---|---|---|---|---|---|---|
| NACA0012 | SFHM | 4182 | 0 | 100 % | 31 | 0.396 | 0.0153 | 6.7 s (python) |
| NACA0012 | snappy | 5518 | 0 | 95 % (layers collapse at TE) | 132 | 0.391 | 0.0149 | 132 s (8 cores, 3D slab) |
| square | SFHM | 7600 | 0 | 100 % | – | – | – | 14 s |
| square | snappy | 10288 | 0 | 100 % | – | – | – | 188 s |
| two-element (gap 0.03c) | SFHM | 5412 | 0 | 95 % (gap clipped on purpose) | 35 | 1.62–1.66 | 0.132–0.148 | 9 s |
| two-element | snappy | 7002 | 0 | 88 % | 217 | 1.48–1.74 | 0.10–0.17 | 175 s |
| dirty (hole, fold, zero-width sliver) | SFHM | 4115 | inverted cells at the sliver only | 50 % | – | – | – | 4 s |
| dirty | snappy | 8530 | 0, **but meshes the inside of the body through the hole and cuts off the TE** | 46 % | – | – | – | 197 s |

RANS: simpleFoam, k-ω SST, Re 1e6, α 4°, wall functions, identical dictionaries. Ranges are over the last 200 iterations. Reference NACA0012 at this Re/α is Cl ≈ 0.42–0.44 in a large domain; both meshes agree with each other to 1 % in Cl and 2.5 % in Cd, so the comparison is about mesh quality, not solver accuracy.

**What the prototype established**
- The shell-first order works end to end: implicit ingest → shell → clipped bulk → inverse layer fill → conformal mesh with no transition cells, no pyramids.
- Full layer coverage and a flat first-cell-height distribution by construction; y+ max 4–6× lower than snappy on the same geometry because snappy's layers collapse at trailing edges and in the gap.
- 25–50 % fewer cells than snappy at the same wall spacing.
- Winding-number ingest bridges a hole in the surface and ignores a backtracking fold. snappy leaks through the hole and meshes the solid interior.
- Convex corners need an explicit fan: seeds on the shell's rounded cap all have the corner as nearest point, so they map radially onto the corner (apex triangles). With that, zero non-convex cells; without it, a few inverted cells at every sharp trailing edge. Smoothing was tried first and made things worse.
- Gaps between bodies and sharp coves are locally the same geometry (medial pair separation 2d·cos(θ/2)), so the clip rule must be angle based (clip when separation > 1.7·d, i.e. cove angle < ~64°), and a parallel-gap test based on "far apart along the wall" never fires.

**Known limitations of v1**
- Zero-width slivers produce zero-area faces (needs an ε-envelope or defeaturing in the ingest, not meshing).
- First-layer height is derived from the streamline length (N fixed), so where δ is clipped the first cell is thinner than spec; a per-region N or a y+-aware clip is needed.
- Cut cells can be thin polygons; vertex snapping at 0.25h removes most but not all.
- Pure Python; the streamline tracing and the winding evaluation dominate. All steps are per-cell or per-seed parallel.

## 11. Variable-thickness offsets: options

The shell must have δ(x) varying along the wall (y+ spec, Re_x growth, gaps). Candidate constructions, with what each guarantees:

| # | Construction | Guarantee | Cost | Verdict |
|---|---|---|---|---|
| A | Warped level set ψ = d(x) − δ(π(x)), π = nearest wall point; δ made 1-Lipschitz along the wall by a lower envelope | manifold iff |dδ/ds| < 1 and δ < gap half-width; needs explicit medial/gap clipping | one nearest-point query per node | used in v1; fine in 2D, fragile where nearest-point labels jump between bodies with different δ |
| B | **Eikonal with variable speed**: solve \|∇T\| = 1/δ(x), T = 0 on the wall, shell = {T = 1} | T is a single-valued viscosity solution, so the level set is always closed and never self-intersects, for any positive δ(x); no Lipschitz condition | fast marching O(N log N) or fast iterative method on GPU, on the same octree as the SDF | **recommended for 3D** |
| C | Topological Offsets (Zint, Panozzo 2024/25) with per-vertex target distance in the inflation step | provably manifold and intersection free by combinatorial construction | needs a background tet mesh and exact arithmetic | strongest guarantee; use as the reference implementation to validate B |
| D | Metric offset: unit distance in the metric induced by the sizing/anisotropy field | same as B (tensor speed); shell is thinner where the tangential spacing is fine | same as B | natural extension of B once the README's metric predictor exists |
| E | Nested shells at fractions of δ (δ/4, δ/2, δ) | each shell is a valid B-shell; the layer fill stops at the outermost shell that survives clipping | K eikonal level sets | handles gaps without collapsing all layers: the explicit, geometric version of Mosaic's "last-ratio" tapering |
| F | Physics prior for δ(x): turbulent flat plate δ ≈ 0.37·x·Re_x^(−1/5) with x the arc length from the stagnation point (min n·U∞), clipped, then smoothed by a screened Poisson on the wall | consistent with the y+ spec; smoothing provides the Lipschitz property for A | trivial | use as the default δ(x) before ML |

Gap handling that goes with any of these:
- Clip rule: a medial pair with separation > 1.7·d is a gap or a sharp cove (angle < ~64°). Pairs on different bodies, or far apart in arc length (> 10·d) on the same body, are always gaps. Everything else is a corner and is left to the harmonic fill, which fans into it.
- Safety net: after extracting shell loops, require exactly one loop per body and no crossing of the wall (both are cheap winding-number checks). On failure, reduce δ locally by 0.7 and rebuild the shell. Deterministic, no parameter hunting.
- Where a gap is narrower than 2·δ_min the shell vanishes and the gap is all boundary layer. Then the fill needs a two-sided treatment: solve φ with φ = 1 on the medial ridge instead of on Γ, so layers from both walls meet at the ridge. Not implemented.

## 12. Next steps

1. Replace A by B (fast marching on the quadtree) and add the loop safety net.
2. ε-envelope in the ingest for slivers; report deleted features.
3. Per-region N and y+-aware clipping so the first-cell height stays on spec where δ is clipped.
4. Port the hot loops (winding, tracing) to numba or CUDA; target < 1 s for these cases.
5. 3D: octree SDF with fast winding numbers, eikonal shell, MCHex (or dual-octree hex) to the shell, Laplace on the shell cells, streamline tracing from shell quads; test on a DrivAerNet car.
