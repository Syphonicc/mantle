# Certified fold-free optimisation for the pillow and boundary layers (design v1)

Status labels: **proved** / **sketch** / **conjecture** / **to verify**.

## Problem

- **Graded shell.** Pillow hexes over 2:1 core-boundary steps fold (initial SJ ≈ −1). The local Gauss–Seidel optimiser cannot recover them.
- **Layers.** Whole columns invert at thin sharp trailing edges and at under-resolved (faceted) leading edges. Repair rounds rescue the default meshes but diverge at fine sizing.

## Prior art

- **Garanzha, Kaporin, Kudryavtseva, Protais, Ray, Sokolov, *Foldover-free maps in 50 lines of code*** (ACM TOG 2021, arXiv 2102.03069):
  - Energy F(U,ε) = Σ_t (f_ε(J_t) + λ g_ε(J_t)) vol_ref(t), with f_ε = tr(JᵀJ)/χ(det J, ε)^{2/d}, g_ε = (det²J + 1)/χ(det J, ε) and χ(D,ε) = (D + √(ε² + D²))/2.
  - ε-continuation with an L-BFGS or Newton inner solver.
  - Guarantee: positive Jacobians are reached in finitely many steps if an admissible configuration exists.
  - Not covered: hexahedra, surface-constrained vertices, very anisotropic cells, and existence itself.
- **Livesu, Sheffer, Vining, Tarini, *Practical hex-mesh optimization via edge-cone rectification*** (ACM TOG 2015): per-edge convex cone QPs, no guarantee.
- **Supporting techniques:**
  - target-matrix optimisation (Knupp, TMOP);
  - barrier-aware line search (Smith & Schaefer 2015);
  - projected Newton (Teran et al. 2005);
  - retractions (Absil et al.);
  - pillowing (Mitchell & Tautges);
  - degree argument for injectivity (Lipman 2014).

## 1. Quadrature matched to solver validity

| Cell | Tets with positive volume required | Count |
|---|---|---|
| Hex | 8 corner tets + 24 fan tets (f_k, f_{k+1}, face vertex mean; cell vertex mean) | 32 |
| Prism | corner tets + fan tets | 24 |
| Pyramid | the 4 base-diagonal tets with the apex (SU2) + fan tets | 20 |
| Polyhedron | fan tets of every face + the SU2 diagonal tets of each quad face | varies |

Results:

- **V1 (proved).** A twisted hex can have all corner Jacobians positive while fan tets, OpenFOAM face pyramids and SU2 dual sub-volumes are degenerate or negative. So corner tets alone are insufficient.
- **V2 (proved).** Positive fan tets do not imply positive corner tets (dart-shaped face). Both families are needed.
- **V3 (proved).** The fan tets of a face sum exactly to OpenFOAM's face pyramid with the vertex-mean face centre. This gives a certificate for warped faces using the difference between the area-weighted and vertex-mean centres.
- **V4 (proved).** Positive pyramids on both sides of a face imply non-orthogonality < 90°.
- **V5 (proved, given SU2's median-dual convention; to verify against the SU2 version).** Fan tets split exactly into SU2's median-dual sub-tets.

Admissible means det J_t ≥ τ_adm (relative to the reference, default 1e-6).

## 2. Energy with per-cell targets

- Garanzha's energy is used with J_t = D_t W_t⁻¹, where W_t is the same tet on the cell's **target** shape:
  - pillow: a right column over the template quad;
  - layer cell: the ideal column cell with heights from dy0 and the growth ratio (aspect up to 1000);
  - collapsed fan cells: ideal sectors;
  - core and transition cells: their lattice geometry.
- **A1 (proved).** The target mesh is a global minimiser.
- **Why targets matter.** With a cube reference, valid thin cells have det J ≈ 1e-6, the same order as the ε floor, so the barrier cannot separate valid from inverted cells. With targets, valid cells have det J ≈ 1, and the remaining anisotropy is handled by a column-block (line) preconditioner. Aspect-independent iteration count is **conjecture C4**.

## 3. Surface constraints

- Shell vertices use the retraction `ShellField::project`, with a tangent projector.
- Wall feet use the BVH closest point, constrained to feature curves when on them, and fixed at feature corners.
- Constraint penalties are rejected, so iterates stay feasible.
- **Interior-point guarantee.** In interior-point mode (ε = 0, admissible start), every accepted step keeps all tets admissible after the retraction, so the output is admissible by induction.
- Untangling guarantees with a retraction inner solver remain **conjecture C5**.

## 4. Existence of an admissible pillow (constructive)

Setting: the core boundary B is the conforming quad mesh from the graded-shell design, with axis normals and minimum angle 45°. The fibres are v* = v + t_v w_v onto the shell S.

- **E0 (proved).** The bottom corner tet of a pillow hex at v equals t_v |e_1 × e_2| (n_q · w_v). Admissibility therefore requires w_v in the open cone K(v) = {u : n_q · u > 0 for all quads q at v}.
  - Closest-point fibres are ≈ −∇psi. They are near-tangent to 2:1 risers, so those hexes start degenerate or inverted. This explains the observed folds, and why flat graded faces (the box) work.
- **Lemma C, column lemma (proved).** A hex or prism whose fibres share one direction w, with convex positively oriented projected base, has all 32 tets positive, even with non-planar ends.
- **Lemma R (proved identity).** For rotating fibres, the end-corner determinant expands multilinearly in the fibre differences. Sufficient bound: T·Δ < ℓ·(√(1 + c sin α) − 1). The bound is uniform in height, so one per-quad condition certifies whole layer stacks.
  - Fan-tet constant σ_fan exists by compactness; its explicit value is **C1**.
  - One-sided (diverging-only) relaxation is **C2**.
- **Theorem E.** Assume:
  - (E1) planar convex quads;
  - (E2) cone condition n_q · w_v ≥ c > 0;
  - (E3) ray hits S at t ∈ [t_min, T];
  - (E4) bounded rotation T_q Δ_q ≤ min(σ, σ_fan) ℓ_q;
  - (E5) disjoint fibres and no top/bottom intersection.

  Then the pillow is admissible, and embedded by the degree argument (Lipman 2014; applicability to a shell to verify). The corner-tet part is proved; the fan-tet part follows via σ_fan.
- **Lemma H and Proposition W (proved, constructive).**
  - A patch whose normals contain no opposite pair has a common cone direction: the normalised sum of its distinct normals, with n · w ≥ 1/√3.
  - Patch fields blend through a partition of unity and keep the cone condition.
  - Rotation must be spread over a distance proportional to the pillow thickness.
- **Pockets (2D proved; quality bound sketch).** Narrow pockets (width g ≲ 2c₀T) obstruct a good single layer.

Minimal fixes, in order of preference:

1. Cone-field ray targets instead of closest-point targets.
2. Locally thinner pillows: move core-boundary vertices along w ("inflate").
3. A pocket-fill cleanup rule (terminating).
4. A second padding layer along the same rays provably does not help, and is not used.
5. A topology check: χ per shell component.

**Cone-field algorithm (deterministic).**

1. Start from the closest-point directions.
2. Project into the cones.
3. Run Jacobi smoothing, projecting after each iteration (about (2T/(σℓ))² iterations).
4. Ray-cast the targets.
5. Evaluate the certificates.
6. Where a certificate fails, apply the fixes locally.

The output is an admissible start, or an explicit obstruction list.

## 5. Layers

- **L1 (proved for corner tets).** The column-field bound with T = column length certifies every layer cell of a stack, for any dy0 and growth ratio.
- **L2 (proved).** Closest-point columns have their focal set exactly on convex wall edges, at distance δ. This includes facet edges of under-resolved leading edges. So first-layer cells there are degenerate by construction. This is why whole columns invert and repair diverges at fine sizing.
- **Fixes:**
  - exact collapse at sharp feature edges (merged nodes are single variables with their own quadrature);
  - ray-cast feet along a smooth cone-projected field (or harmonic streamlines) at faceted edges;
  - a "projection classification" that chooses hex, prism, pyramid or merge per quad, replacing the repair rounds.

## 6. Algorithm

- **Per-tet gradient:**
  - D = det J, T = ‖J‖_F², ρ = √(ε² + D²), χ = (D + ρ)/2;
  - ∂f/∂J = χ^{−2/3}[2J − (2T/(3ρ)) cof J];
  - ∂g/∂J = [2D/χ − (D² + 1)/(χρ)] cof J;
  - chain rule via W_t⁻ᵀ; derived points distribute their gradient with fixed weights.
- **Hessian.** Convexified per tet, by eigenvalue clamping of the 9×9 block or by dropping the indefinite terms. Used for the preconditioner (3×3 vertex blocks; block-tridiagonal per layer column) and optionally for Newton-PCG.
- **Solver.** Preconditioned L-BFGS (m = 6) with Armijo line search. In barrier mode, the maximum step is the smallest positive root of the per-tet cubic det(D + αΔD), solved with a safeguarded method.
- **Phases:**
  - A: constructive start;
  - B: interior-point polish at ε = 0;
  - C: ε-continuation fallback (ε_k = √(1e-12 + 0.04 min(0, D_min)²), floored) only if the start is not admissible.
- **Active set:** two rings around bad cells, plus transition zones.
- **Parallelism and determinism:** a fixed chunk size independent of the thread count, ordered reductions, no hash-order dependence. Results must be bit-identical across thread counts.
- **Cost:** about 6.4 kflop per hex per gradient evaluation, ≈ 0.3 s per evaluation for 1M cells on 8–16 cores. Active-set polish takes seconds. Memory is ≈ 0.4 GB per 1M cells.

## 7. Integration and stages

New modules: `foldfree.rs` (quadrature, energy, gradient, preconditioner, L-BFGS, barrier line search, certificate) and `conefield.rs` (cones, field, E2–E5 checks, ray targets, pocket rule). Flags default off; uniform runs stay bit-identical.

| Stage | Content | Success bar |
|---|---|---|
| S0 | baselines, new statistics | stored |
| S1 | energy, gradients, quadrature | finite-difference gradients (rel. err < 1e-6); V1 counterexample; Lemma C property test (10⁴ random cases); Lemma R identity |
| S2 | solver and line search | single folded hex fixed in < 50 iterations; random tangled 6³ patch admissible (100 seeds); deterministic across thread counts |
| S3 | constraints | vertices on S to 1e-12·size; feet on the wall |
| S4 | cone field | synthetic stepped 2:1 patch: closest-point init folds (negative test), cone init certified, min SJ ≥ 0.3 after polish; pocket rule triggers; σ_fan computed |
| S5 | hexcore integration | graded sphere: 0 negative cells, SU2 clean, pillow SJ ≥ 0.3; graded box ≥ 0.27; M6 `--curvature on --graded-shell on`: pillow SJ ≥ 0.2, 0 inverted layer cells |
| S6 | layers | A = 1000 column within 1e-3 of target; M6 fine sizing with 0 non-positive cells and no repair rounds; layer runtime ≤ 1.5× today |
| S7 | defaults, documentation | full regression |

## Novelty (relative to the surveyed literature)

- (N1) A quadrature that coincides with solver validity: the V3/V5 identities, the warped-face certificate, and the V1 counterexample.
- (N2) The generalised column lemma and the multilinear rotation identity, giving one per-quad certificate for whole layer stacks.
- (N3) A constructive existence theorem for single-layer pillows over 2:1 orthogonal core boundaries via cone fields, with the pocket obstruction and a minimal cleanup rule.
- (N4) The focal-distance criterion unifying the pillow and layer failures.
- (N5) Garanzha's energy with exact surface constraints and an admissible-start interior-point mode. The output is either a certified admissible mesh or an explicit obstruction list.

Open items: C1, C2, C4, C5, plus the two "to verify" items (SU2's dual sub-volume construction, Lipman's degree argument applied to a shell).
