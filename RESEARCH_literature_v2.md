# Literature hunt v2 (16 Nov 2026)

Three parallel searches (boundary-layer/hybrid meshing; speed/GPU kernels; SU2 ecosystem + multi-view idea). Items marked [U] were only seen as search snippets; verify before citing.

## 1. Closest prior work / novelty
- **No paper found combining** eikonal (variable-δ) shell first + octree clipped to the shell + harmonic-streamline layers with per-column layer counts. Frame novelty as that combination; run a citation-chain check before publishing.
- Partial overlaps: harmonic-field prism directions (Adv. Aerodynamics 2021, doi 10.1186/s42774-021-00065-y); eikonal layer-by-layer BL [U]; **Topological Offsets** (Zint et al., arXiv 2407.07725) — guaranteed manifold nested offsets/cages, nearest competitor on "offset first"; **HybridOctree_Hex** (arXiv 2401.05984) — core octree + buffer fill by nearest-point projection.
- No peer-reviewed 2024–26 benchmark of snappy/cfMesh/Gmsh/T-Rex/Mosaic found → our y+ ~ 1 comparison is itself publishable material.

## 2. Ideas to adopt (ranked)
1. **MCHex** (Tong, Zhang, arXiv 2511.02064): marching-cubes cut cells → 3-regular polyhedra → midpoint subdivision = all-hex, positive Jacobian. Relevant because SU2 needs standard elements (no polyhedra, no hanging nodes); plus **3-refinement templates** (arXiv 2512.14862) for conforming transitions.
2. Shell validity oracle from Topological Offsets (manifold, nested, non-intersecting); clamp δ to a fraction of local thickness.
3. Bijective / no-flip criteria for layer columns (JCP 2025 "full-layer prismatic mesh via bijective mapping" [U]).
4. Lipschitz (K-Lipschitz) sizing from local feature size (curvature + thickness), solved as a gradient limiter on the same grid as T.
5. Dirty-geometry topology control: **Tusqh** (arXiv 2502.10609, winding numbers + persistent homology) for sub-layer gaps.
- CAD front end: Topology-First B-Rep Meshing (arXiv 2604.02141).

## 3. Speed stack (target ~30–60 s CPU for a 1M-cell 3D mesh, estimates)
- Winding number + closest point: **NVIDIA Warp** (Apache-2.0, `pip install warp-lang`, CPU+CUDA, winding-number sign + mesh_query_point); **fcpw** (MIT) for 2D segments/3D triangles; Barill fast winding numbers (MIT); Antipodal method (arXiv 2605.01536) as exact reference.
- Eikonal: fast sweeping in numba (O(N), no heap) or GPU FIM (arXiv 2106.15869); skip the solve when δ is plain wall distance (BVH query).
- Octree + 2:1 balance: Morton-key linear octree in numba (p4est-style balance: < 10 s per 1M octants); GPU option Ren et al. SIGGRAPH Asia 2025, Cornerstone (arXiv 2307.06345).
- Laplace: pyamg/scipy; streamlines as one batched RK2 kernel (numba prange / Warp).
- Writers: binary, flat arrays + offsets, no per-cell Python loops.
- Release stack: numpy, scipy, numba, warp-lang (+pyamg). No C++/Rust core needed for v1.

## 4. SU2 side
- Format (https://su2code.github.io/docs/Mesh-File): tri 5, quad 9, tet 10, hex 12, prism 13, pyramid 14; conforming meshes only. Polygons/polyhedra must be split.
- Meshers used with SU2: Gmsh (2D BoundaryLayer field; weak/experimental in 3D, TE intersections reported), Pointwise T-Rex (commercial reference), CENTAUR; adaptation via INRIA AMG (pyAMG, `mesh_adaptation.py`).
- Grid-convergence reference: Diskin et al., NASA NTRS 20150005716 (NACA 0012 α 10, TE resolution dominates error).

## 5. Multi-view ("look from top/bottom/sides") idea — prior art
- Closest: **axis-aligned height-field block decomposition** (Muntoni, Livesu, Scateni, Sheffer, Panozzo, ACM TOG 2018; code github.com/cg3hci/heightfielddecomposition); tri-dexel / layered depth images; polycube maps (DL-Polycube arXiv 2410.18852, CE-PolyCubeMaps 2020); sweepable decomposition (Cubit).
- Known failure modes: regions occluded from every view (undercuts, internal cavities), features thinner than sampling, topology changes; visual hull cannot recover hidden concavities.
