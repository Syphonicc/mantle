# Graded, conforming shell quad meshes (design v1)

**Goal.** Let the octree resolution vary along the boundary-layer shell (2:1 transitions), so curvature sizing can refine leading edges, tips and trailing edges, while the pillow layer and the boundary-layer columns keep working on a conforming all-quad shell mesh.

**Today.** The pillow and the layers need a conforming quad shell, which forces one uniform octree level along the whole shell. Core cells whose boundary quad has a hanging vertex are removed (`removed_hanging` in `hexcore.rs`). An attempt with one extra level near a wing leading edge removed 7782 cells and folded the pillow.

## Prior art and positioning

| Method | Idea | Limitation for mantle |
|---|---|---|
| Schneiders (1996); Ito, Shih, Soni (IJNME 2009) | 3-refinement templates | 3:1 size jumps, 27x cells per level, poor template shapes |
| Maréchal (IMR 2009); Livesu, Pitzalis, Cherchi, *Optimal Dual Schemes for Adaptive Grid Based Hexmeshing* (ACM TOG 2021, arXiv 2103.07745) | dual of a strongly balanced grid with 8 atomic transition schemes, all-hex | full core rewrite; reference code (CinoLib) uses hard-coded orientation tables; transition-vertex selection uses an ILP (Gurobi) in the companion code; over-refinement is the stated bottleneck |
| Pitzalis et al., *Generalized adaptive refinement for grid-based hexahedral meshing* (ACM TOG 2021) | weaker pairing criterion | still refines the octree |
| Tong, Halilaj, Zhang, *HybridOctree_Hex* (2024, arXiv 2401.05984) | balancing + sibling pairing, 5 templates | pairing refines leaves; min SJ 0.258 before smoothing |
| Tong, Zhang, *MCHex* (arXiv 2511.02064) | marching-cubes boundary, positive Jacobian | all-hex volume focus, boundary from the cut |
| snappyHexMesh, cfMesh | split-hex polyhedra with hanging nodes | hanging nodes on the boundary, so layers are polyhedral there |

**Observation.** mantle needs conformity only on the core boundary, where the pillow and the layers attach. The interior can keep 2:1 hanging-node polyhedra: OpenFOAM accepts them, and mantle's SU2 writer decomposes them. So the 3D transition problem reduces to a **2D parity problem on one surface**.

## Design B1: shell-confined transitions with a surface T-join

1. **Keep the core as it is.** The octree core stays as it is (vertex 2:1 balanced, already enforced by `Octree::balance`).
2. **Build the boundary polygon surface P.** Each core boundary face is one polygon: its 4 corners plus the midpoint of any side that is a *hanging* side (the side's midpoint is a vertex of a finer core leaf).
3. **Parity.** A polygon with n sides can be quad-meshed only if n is even (4F = 2E_int + n). A face with an odd number of hanging sides is *odd*.
4. **Restore parity with a T-join.** Choose a set J of *full* edges (same-level edges shared by two same-level faces). Splitting every J edge at its midpoint makes every face even. J is found in linear time: greedy pairing of neighbouring odd faces, then BFS paths with XOR, then a spanning-forest fallback.
5. **Templates.** Every face is subdivided by one of 4 face-local templates according to its (even) split mask:

   | Mask (canonical) | Template | Quads | Planar SJ min |
   |---|---|---|---|
   | none | T0 | the face | 1 |
   | two adjacent sides | Tc | 3 quads around the centre Z | 0.707 |
   | two opposite sides | Ts | 2 quads | 1 |
   | all four sides | T4 | 4 quads | 1 |

   Eight even masks map onto these 4 templates by rotation. All points are exact doubled-lattice keys.
6. **Pillow and layers.** One pillow hex per template quad; the layers consume the resulting shell quad mesh Q unchanged.

### Guarantees

- **Lemma F (fan lemma).** Under vertex 2:1 balance and a manifold P, a hanging point m on a level-L face F has either:
  - (i) exactly one other face, of level L, sharing both half edges (concave case); or
  - (ii) a fan of level-(L+1) faces with m as a corner, consecutive ones sharing full edges. These lie in one component of the full-edge graph.
- **Theorem 1.** Every component of the full-edge graph contains an even number of odd faces. Hence a T-join made of full edges always exists (Edmonds & Johnson 1973). No octree pairing and no extra refinement are needed.
  - Proof idea: counting sides gives |odd ∩ R| ≡ #(fine-side half edges in R) (mod 2). By Lemma F those half edges come in pairs within one component.
  - With an open boundary (shell touching the domain box), use a virtual sink node.
- **Conformity.** A full edge is split in both incident faces if and only if it is in J. Half edges are never split again. So Q is a closed, conforming all-quad mesh with χ(Q) = χ(P).
- **Volume validity.**
  - **Core cells owning a transition face:** their vertices stay frozen on the lattice, so they are exact cubes with planar convex sub-faces, i.e. convex polyhedra. The SU2 apex decomposition about the cube centre is then positive by construction. Polygons are fanned about their vertex mean, shared by both neighbours.
  - **Pillow hexes:** validity is enforced as today (corner-Jacobian test, guarded smoothing, optimiser, repair). There is no a-priori proof; the templates only change the bottom shapes (angles of at least 45 deg).

### What is new (relative to the surveyed literature)

- **Not new:** T-joins; parity in quad meshing (Mitchell, IMR 1997; Müller-Hannemann, IMR 1997); 2D transition templates and pairing; pillowing.
- **New (not found in the surveyed literature):**
  - (N1) Reducing 3D transitions to a 2D parity problem on the core boundary only, with a polyhedral interior.
  - (N2) Theorem 1 with Lemma F, covering the concave case: parity is always restorable by splitting same-level edges, with zero refinement beyond the balance already present.
  - (N3) Placing the templates on the core faces, so the pillow and the layers are all-hex and the transition cells are convex cubes.

## Staged implementation

| Stage | Content | Test |
|---|---|---|
| S0 | Baselines; synthetic grading helper; optional VTK dump of P and Q | stored baseline JSON |
| S1 | `surfq.rs`: polygon surface P, edge kinds (full/full, half/full, half/half), manifold check; `scan` records hanging masks (behind a flag) | P manifold on graded sphere/box; random balanced leaf sets; Lemma F debug assertion |
| S2 | T-join (greedy, BFS-XOR, spanning-forest fallback) | all final masks even on random balanced sets |
| S3 | Templates, Q mesh, doubled-lattice keys, pillow on Q, assembly with inserted edge points, optimiser freeze; `graded_shell` flag | templates tile the square; Q closed; cells closed; uniform runs bit-identical; graded sphere/box valid; SU2 load |
| S4 | Layers on graded Q; CLI wiring; ONERA M6 with curvature sizing | success bar below |
| S5 | Quarter faces instead of `removed_split`; Tc/Ts cost tuning; shell-side smoothing of 2:1 jumps | same tests with `removed_split` = 0 |
| S6 | Default on with curvature sizing; docs | full regression |

## Success bar (versus the uniform-shell baseline)

- **Correctness:**
  - `removed_hanging` = 0;
  - Q closed and conforming;
  - 0 non-positive cells (OpenFOAM) and 0 SU2 element failures;
  - checkMesh "Mesh OK" on sphere and box;
  - no new failure classes on the M6.
- **Zero extra octree refinement:** identical leaf counts with the flag on and off; at most 5 % extra shell quads.
- **Quality:**
  - pillow SJ min ≥ 0.30;
  - layer SJ min at least the baseline minus 0.05.
- **ONERA M6 with curvature sizing:**
  - leading-edge shell size ≤ h_surf/4;
  - 0 non-positive layer cells at the leading edge;
  - at least 10 columns across the nose half-arc;
  - at most 2x the uniform medium cell count, and at most 25 % of a uniform mesh at the leading-edge size.
