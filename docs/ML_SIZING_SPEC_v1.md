# Auto-Mesh

**Constraint-aware, ML-assisted mesh generation for CFD.**

You give the geometry, the flow conditions and a mesh spec (target y+, number of layers, growth ratio, domain extents, cell budget, quality limits). Auto-Mesh predicts *where* and *how* to resolve the flow, builds the mesh with a classical mesher, and checks that the spec was actually met.

> Status: concept / early research. Nothing is implemented yet. Core volume-meshing algorithm study: see RESEARCH_volume_meshing_v1.md.

---

## 1. Motivation

SU2 adjoint shape optimization automated the design loop. Meshing is still mostly manual trial and error: choosing refinement regions, getting the first-cell height right for the target y+, and fighting layer collapse in snappyHexMesh. The NASA CFD Vision 2030 study names meshing as a major bottleneck.

The classical analogue to the adjoint idea already exists: **goal-oriented, adjoint-weighted mesh adaptation** (Venditti & Darmofal; metric-based tools such as NASA `refine`, Mmg/ParMmg, Inria AMG). It works but needs several solve → adapt cycles.

**Core idea:** replace those cycles with *one prediction*, conditioned on the user's hard constraints, and backed by a deterministic verification loop.

## 2. Design philosophy

1. **The neural network never builds the mesh itself.** A classical mesher (snappyHexMesh, Gmsh, Mmg, refine, or a custom extruder) does that, so mesh validity is guaranteed.
2. **Hard constraints are computed or checked, never learned.**
3. **ML is only used where it adds value:** spatially varying decisions that are hard to make by hand.
4. **Verify everything** with checkMesh plus a short solve and y+ check, with feedback when the spec isn't met.

### Constraint classes

| Type | Examples | Handled by |
|---|---|---|
| Computable | y+ → first-layer height, growth ratio, layer count, BL thickness, far-field distance | Formulas |
| Checkable | non-orthogonality, skewness, aspect ratio, negative volumes, layer coverage % | `checkMesh`, SU2 mesh-quality output (pass/fail gate) |
| Hard to get right | refinement placement, **local** first-cell height along the wall, snappy layer settings, cell-budget allocation, wake/shock regions | **ML model** |

### The y+ problem (example)

Flat-plate estimate of the first-cell height:

```
Δy = y⁺ · ν / u_τ,   u_τ = U·√(Cf/2),   Cf ≈ 0.026 · Re_x^(-1/7)
```

This gives one uniform Δy₀, but the real y+ varies a lot along the wall: high at stagnation and suction peaks, near zero at separation. Auto-Mesh predicts the **local Cf(s) distribution** from geometry plus Re/Ma, and from it a spatially varying Δy₀(s).

- Δy₀ ∝ 1/√Cf, so a 20 % error in Cf becomes only about 10 % in Δy₀.
- Cf → 0 at separation sends Δy₀ to infinity, so it needs a clip or a Reynolds-based floor.
- The target y+ band depends on wall treatment (≈1 for low-Re SA/SST, 30–300 for wall functions) and is part of the spec.

## 3. Architecture

```
User spec (y+, layers, growth, domain, budget, quality limits) ──┐
Geometry (STL / airfoil) + flow conditions (Re, Ma, AoA) ────────┤
                                                                 ▼
   [Physics formulas] → baseline Δy₀, BL thickness, domain box
   [ML model]         → local Δy₀(s), refinement regions / metric field, layer settings
                                                                 ▼
   Classical mesher (custom extruder / snappyHexMesh / Gmsh / Mmg / refine)
                                                                 ▼
   Verify: checkMesh + short solve → yPlus, layer coverage, cell count
          └── fail → feed back to model/rules → retry
```

Ways to predict the metric safely: output log-eigenvalues plus an angle so the tensor is always symmetric positive-definite; use conformal prediction to refine conservatively on geometry unlike the training data (as in AutoSurf).

## 4. Literature review (searched Nov 2026)

### What exists

**ML sizing / metric-field prediction, with a classical mesher**
- MeshingNet (2020), MeshingNet3D (2021): 2D or tetrahedral size fields from geometry and boundary conditions.
- Lock, Hassan, Sevilla, Jones (Swansea, 2025), *Anisotropic mesh spacing prediction using neural networks*: 3D aircraft / ONERA M6 metric fields. **Inviscid only**, stretching capped at 5, predicting directions is harder than sizes.
- AMBER (2025): hierarchical GNN that refines the sizing field over several passes from expert meshes; generalizes to unseen geometry.
- AutoSurf (MIT, 2023): GNN surface classification for aircraft mesh settings with conformal prediction; about 5× faster than adaptive remeshing. Open source.

**ML for goal-oriented (adjoint) adaptation**
- Fidkowski & Chen (JCP 2020): neural network predicts anisotropy from primal and adjoint features.
- Ojha, Chen, Fidkowski (2022): ML anisotropic *initial* mesh for output-based adaptation; resolves boundary layer and wake in 2D.
- E2N (2022): neural network replaces the adjoint error-estimation step.

**Mesh movement / RL / structured meshes**
- M2N (NeurIPS 2022), UM2N (NeurIPS 2024): zero-shot Monge–Ampère mesh movement.
- RL quad meshing (SAC, 2022), RL block decomposition (2024), Dmsh (2026, all-quad), MeshDQN, Swarm-RL adaptive refinement.
- MeshONet, 3DMeshNet, PINN-MG: structured mesh generation.

**LLM agents for OpenFOAM**
- Foam-Agent / Foam-Agent 2.0: a meshing agent writes blockMesh/snappy/Gmsh setups; 88.2 % success on 110 tasks.
- OpenFOAMGPT 2.0: chooses the meshing strategy.
- **Neither verifies y+ or layer quality.**

**Classical y+ control (non-AI)**
- Centaur boundary-layer layer redistribution to a y+ target; a community OpenFOAM yPlus-based wall-layer refinement utility; SimScale and STAR-CCM+ y+ settings. All are *reactive*: they work after a solve.

**Datasets with wall shear (for training Δy₀ prediction)**
- AirfRANS (about 1,000 2D RANS airfoil cases), DrivAerNet / DrivAerNet++ (about 4,000 3D cars), DrivAerStar (12,000 cases).

**Commercial tools**
- Ansys SimAI / Fluent octree meshing; Siemens ML part grouping. These are surrogates and geometry-prep helpers, not constraint-aware meshers.

**Survey:** Owen et al. (Sandia, LANL, CEA, CMU …), *A Survey of AI Methods for Geometry Preparation and Mesh Generation in Engineering Simulation* (Dec 2025). **Does not cover boundary layers, prisms or y+.** It names dataset scarcity, the lack of benchmarks and the lack of closed-loop workflows as gaps.

### Gaps (none of these were found)

| Gap | Status |
|---|---|
| Spec-driven meshing with y+ / layers / growth ratio / quality limits as hard inputs | Not found |
| Local first-cell-height Δy₀(s) prediction from geometry + Re/Ma | Not found (wall-shear surrogates exist but aren't linked to meshing) |
| ML for viscous BL / prism meshing in 3D | Not found (existing 3D work is inviscid) |
| Predict *and* verify loop (checkMesh + yPlus feedback) | Not found |
| Predicting snappyHexMesh layer-collapse failures | Not found |
| Standard ML-meshing benchmark | Field-wide gap |

Caveat: recent AIAA SciTech/Aviation 2025–26 proceedings still need a direct search.

## 5. Expected problems and ways around them

1. **Meshers can't take a varying first-cell height.** snappy sets `firstLayerThickness` per patch; Gmsh's BL field is basically uniform; snappy layers collapse at sharp edges. Fix: own extruder in 2D; patch zoning or custom extrusion in 3D; consider cfMesh.
2. **Circular dependency:** y+ ← τ_w ← solution ← mesh. Errors in predicted Cf go straight into y+. Separation (Cf → 0) and transition need special handling.
3. **No single correct mesh to train on.** Labels inherit the habits of the adaptation tool; labels are expensive (mesh-converged runs); dropping failed cases biases the data toward easy ones.
4. **Can't train end-to-end:** the mesher and solver aren't differentiable. Use supervised pre-training plus a cheap correction step; RL/black-box is costly.
5. **Geometry representation and generalization** (SDF / point cloud / graph). 3D costs about 100× more than 2D.
6. **Solver dependence:** SU2 and OpenFOAM treat walls differently. Use one solver per study.
7. **Data-pipeline engineering:** thousands of automated mesh → solve → post-process runs with failure detection. Usually more work than the ML.
8. **Strong baseline:** flat-plate y+ estimate plus one yPlus correction pass. Must beat it on y+ spread, cell count or wall-clock time *including* corrections.
9. **Fast-moving field:** keep the first contribution narrow and quick.

### Evaluation metrics
- % of the wall inside the target y+ band
- Layer coverage %
- `checkMesh` failures
- Cell count
- CL/CD error against a mesh-converged reference
- Number of solves / wall-clock time to an acceptable mesh

## 6. Roadmap

**Phase 0: rule-based baseline**
- Spec file → `blockMeshDict` + `snappyHexMeshDict`, with the y+ → Δy₀ calculation built in.
- Verify with `checkMesh` + short `simpleFoam` + `yPlus`.

**Phase 1: 2D proof of concept (lowest risk)**
- 2D airfoils with a custom extrusion / C-mesh generator, which avoids problem 1.
- Train a Cf(s) → Δy₀(s) predictor on AirfRANS wall shear.
- One solve → measure the y+ distribution → compare with the flat-plate baseline.

**Phase 2: goal-oriented metric prediction**
- UIUC airfoils (about 1,600) × Ma/Re/AoA sweeps.
- Labels: SU2 adjoint/Hessian metric adaptation (refine / Mmg2d) until CL/CD converge; metric interpolated onto a background mesh.
- Model: GNN or neural operator; inputs SDF + flow parameters; output SPD metric.
- Test: same CL/CD accuracy with one solve instead of N cycles; airfoil families and transonic shock cases held out of training.

**Phase 3: 3D and OpenFOAM**
- snappyHexMesh refinement regions/levels plus layer settings, with patch zoning for varying Δy₀.
- Learned prediction of layer-collapse risk.
- Optional: an LLM agent layer (editing dictionaries, reading checkMesh/yPlus) on top of the learned predictor.

## 7. Toolchain

- OpenFOAM v2512 (ESI): blockMesh, snappyHexMesh, checkMesh, simpleFoam, yPlus
- SU2: RANS, adjoint, mesh adaptation
- Gmsh, Mmg/ParMmg, NASA refine
- PyTorch / PyTorch Geometric (GNNs), optionally neural-operator libraries

## 8. References

- MeshingNet — https://arxiv.org/abs/2004.07016
- MeshingNet3D — https://www.sciencedirect.com/science/article/abs/pii/S0965997821000508
- Lock et al. 2025, Anisotropic mesh spacing prediction — https://arxiv.org/abs/2504.00456
- AMBER — https://arxiv.org/abs/2505.23663
- Iterative sizing field prediction from expert demonstrations — https://arxiv.org/abs/2406.14161
- AutoSurf / conformal GNN meshing — https://arxiv.org/abs/2308.07358 , https://github.com/ahnobari/AutoSurf
- ML-based optimal mesh generation in CFD — https://arxiv.org/abs/2102.12923
- Fidkowski & Chen 2020 (JCP) — https://public.websites.umich.edu/~kfid/MYPUBS/Fidkowski_Chen_2020_JCP.pdf
- Ojha, Chen, Fidkowski 2022 — https://websites.umich.edu/~kfid/MYPUBS/Ojha_Chen_Fidkowski_2022.pdf
- E2N — https://arxiv.org/abs/2207.11233
- UM2N — https://arxiv.org/abs/2407.00382
- RL quad meshing (SAC) — https://arxiv.org/abs/2203.11203
- MeshDQN — https://arxiv.org/abs/2212.01428
- Swarm RL for AMR — https://arxiv.org/abs/2304.00818
- MeshONet — https://arxiv.org/abs/2501.11937
- 3DMeshNet — https://arxiv.org/abs/2407.01560
- PINN-MG — https://arxiv.org/abs/2503.00814
- Survey of AI for geometry prep & meshing (Owen et al., 2025) — https://arxiv.org/abs/2512.23719
- Foam-Agent — https://arxiv.org/abs/2505.04997
- Foam-Agent 2.0 — https://arxiv.org/abs/2509.18178
- OpenFOAMGPT 2.0 — https://arxiv.org/abs/2504.19338
- Centaur mesh redistribution — https://centaur.netplanet.gr/?p=1128
- OpenFOAM yPlus wall-layer refinement utility — https://cfd-online.com/Forums/openfoam-community-contributions/109885-contribution-new-utility-refine-wall-layer-mesh-based-yplus-field.html
- AirfRANS — https://proceedings.neurips.cc/paper_files/paper/2022/hash/94ab7b23a345f93333eac8748a66c763-Abstract.html
- DrivAerNet — https://decode.mit.edu/assets/papers/DrivAerNet.pdf
- DrivAerStar — https://arxiv.org/abs/2510.16857
