# SimCraft Roadmap

How the remaining Ansys-style capabilities should be built in this codebase. Each section was
researched in parallel by short-lived Haiku sub-agents reading the actual source, then checked and
combined by the lead model; corrections made during review are marked **Review note**.

Order is by value for effort. Efforts assume one developer.

## Status (2026-11-23)

| Step | State |
|---|---|
| §0 Bridge security | ✅ done — token auth, 4 MB line limit |
| §1 Per-body heat sources, HTML reports, Python access | ✅ done |
| §2 Response spectrum, harmonic, random vibration, fatigue | ✅ done — validated in `tests/vibration.rs` |
| §3 GPU FDTD + microwave heating | ✅ done — PEC/absorbing boundaries, TE10 ports, loss tangent, GPU, near-to-far-field transform (pattern, directivity/gain, radiated power, field at 3 m / 10 m vs FCC / CISPR 32 class B); validated in `tests/fdtd.rs` (short dipole D = 1.52). Still open: Debye dispersion, turntable, ground-plane EMI sites, S-parameters, sub-cell traces |
| §4 Electronics cooling (CHT + turbulence) | ✅ done — conjugate heat transfer, Boussinesq buoyancy, LVEL, moving walls, multigrid pressure, implicit SIMPLE (default, 21× faster than projection on a 92k-cell electronics box) with projection fallback for unsteady flows; validated in `tests/cfd.rs` (Ghia, de Vahl Davis, Dean). Still open: radiation, transient CFD, k-ε |
| §5 KiCad PCB import | ✅ done — outline/cut-outs, stack-up, tracks/arcs/pads/zones/vias rasterised, orthotropic homogenised conductivity, component boxes with power, per-net DC solve (`pcb_dc`: resistance, IR drop, current density, IPC-2221 rise, via currents); validated in `tests/kicad.rs`. Still open: AC / skin effect, warpage, 3D models |
| §6 Accurate curved surfaces | ✅ done for structural, thermal, modal and vibration — conservative voxels + pillow layer on curved/off-grid patches, nodes projected onto the true surfaces with tangential smoothing, general hex8 with Taylor-corrected incompatible modes, consistent face loads; fitted conduction elements, true face areas and consistent nodal volumes (mass, heat capacity); validated in `tests/fitting.rs` (structural and thermal patch tests, Kirsch 1.7 %, Lamé 1.7 %, pipe-wall conduction 0.1 %, rod frequency 0.8 %). Still open: buckling, plasticity, EM fields, CFD |
| Tier 3: large models | ✅ done — matrix-free geometric-multigrid CG for static structural / thermal stress (element colouring, octant Galerkin coarsening, Chebyshev smoothing), aggregation multigrid for steady thermal; element limit 400 k → 2 M for those analyses; 1.07 M DOF in 3 s (Jacobi-PCG 60 s CPU / 10 s GPU). Still open: modal / transient / nonlinear on multigrid, GPU multigrid |
| §7 Surrogates | ✅ done — Gaussian-process response surfaces (ARD kernel, marginal likelihood, closed-form LOO/Q²), what-if sliders and `predict_surrogate`, Bayesian optimisation (EI × probability of feasibility) with every proposal verified by a real solve; validated in `tests/surrogate.rs`. Still open: field-level surrogates |


## Phase 2 — industrial parity (plan, 2026-11-24)

Phase 1 (§0–§7 and Tier 1–3) is done. Phase 2 closes the gaps that keep SimCraft from replacing
Ansys / COMSOL / Icepak / HFSS for everyday product work: real CAD, assemblies, radiation, broadband
EMC, transient and turbulent flow, large deformation. Items are built one at a time, each in its own
branch and PR.

**Every item ships with:** (1) validation tests against closed forms or published benchmarks in
`tests/`, with the numbers in the README table; (2) a GUI entry that follows the existing UX rules
(analysis settings in SOLVE ▸ SETTINGS, loads in SETUP, computations that need inputs as a dialog
launched from the ribbon, contextual groups only when they apply); (3) an MCP tool or `set_analysis`
field with a schema description; (4) user-guide text and, when it is a new workflow, a tutorial;
(5) a robustness check (cancellable, no panics on bad input, warnings for out-of-range use).

Sizes: **S** ≈ one session, **M** ≈ 2–3, **L** ≈ 4–6, **XL** > 6.

| # | Item | Size | Why first / depends on | State |
|---|---|---|---|---|
| P2.0 | Benchmark suite (`simcraft bench`): 24 reference cases, mesh sweeps, HTML report, other-solver column, CI gate | S–M | measures every later item against references and Ansys | done |
| P2.0b | Fix the fitting defects the benchmark found (see below) | S–M | wrong peak stresses at some element sizes; must come before more physics | done |
| P2.1 | Thermal radiation + temperature-dependent properties | M | biggest missing physics for electronics cooling; self-contained | done |
| P2.2 | Path plots, cylindrical results, mesh-convergence assistant | S–M | cheap, raises trust in every result | done |
| P2.3 | STEP import (spike first) | M–L | unlocks real CAD parts and assemblies | planned |
| P2.4 | Contact, bolt pretension, springs, remote points | L | assemblies; best after P2.3 | planned |
| P2.5 | Local mesh refinement (octree) + adaptive error control | XL | real parts with small features; touches every FE solver | planned |
| P2.6 | Broadband FDTD: lumped ports, S-parameters, emissions spectrum, ground plane | M–L | completes the EMC story started in Tier 3 | planned |
| P2.7 | Fans, grilles / porous media, compact package models, PCB Joule heat | M | Icepak parity for boards and enclosures | planned |
| P2.8 | Transient CFD + k-ω SST turbulence | L | unsteady flows, better separated-flow accuracy | planned |
| P2.9 | Large deformation + hyperelastic materials | L | seals, snap fits, rubber | planned |
| P2.10 | Eddy currents and induction heating | L | needs edge elements | planned |
| P2.11 | Shell elements for sheet bodies | L | depends on P2.3 (STEP sheet bodies) | planned |

See also the **Platform, Security, UX & Architecture track** below (`P2.S`, `P2.A1`, `P2.A2`, `P2.U`, `P2.W`; checklist in `todo.md`) — `P2.S` and `P2.A1` are prerequisites before `P2.3` (STEP) and `P2.5` (octree).

### P2.0 Benchmark suite (S–M) — done

`src/bench.rs`, `simcraft bench`: 24 cases (structural incl. NAFEMS LE1, thermal incl. NAFEMS T4,
fluids, electromagnetics, electronics) driven through the tool API at 1–3 mesh sizes each; JSON +
HTML report with error-vs-unknowns plots and observed convergence order; `--compare` adds another
solver's results from a `--template` file; `--quick` (19 cases, one size) runs in CI as an accuracy
gate and uploads the report. Every Phase 2 item adds its cases here.

### P2.0b Fitting defects found by the first full benchmark run (S–M) — done

Fixed in `mesh.rs` / `geometry.rs`: (a) grid lines within 0.45 h of an axis-aligned flat face (box
faces, cylinder caps, flat STL faces) snap onto it, so flat faces are meshed exactly and only curved
or oblique surfaces get a pillow layer; (b) fitting is decided per face, not per surface; (c) flat
surfaces only take voxel faces facing the same way; (d) the pillow inward step only adds missing
thickness; (e) STL node inside-tests count surface nodes as inside (the top layer of STL parts was
being dropped); (f) bodies the conservative voxelisation would drop keep centre-inside voxels with a
warning. Results: Lamé within 1.8–3.7 % at every size from 0.7 to 2 mm (was +79 to +132 % at
1.25 / 1.5 / 1.75 mm); NAFEMS LE1 0.5–1.2 % at 50 / 35 / 25 mm (was 33–73 %; plain voxels 14 %);
all 27 benchmark metrics pass with no convergence anomalies. Original findings:

1. **Curved fitted surface meeting an off-grid flat face.** When a part's thickness is not a whole
   number of cells, the flat top/bottom get a fitted layer too, and where it meets a fitted curved
   side the elements distort: Lamé bore hoop stress is +79 % at h = 1.5 and 1.25 mm (t = 4 mm) but
   within 1.5 % for t = 4.5 or 3 mm at the same h, and within 1.7 % at h = 1 or 2 mm. NAFEMS LE1
   (imported STL, 100 mm thick) fails the same way: σyy(D) 62 MPa vs 92.7 (plain voxels: 80), with
   ~17 % of the outer face's quads tilted out of plane and the fitted outer face carrying 18 % too
   little load. Fix in `mesh.rs::fit_boundary` (edge nodes shared by two pillow layers, quality
   relaxation at convex edges) and/or align the voxel lattice to planar faces.
2. **Thin parts vanish with fitting.** A 2 mm plate at h = 1.5 mm gives an empty mesh (conservative
   voxels need whole cells inside). Fall back to centre-inside voxels for bodies the conservative
   voxelisation would drop, with a warning.

Done when `simcraft bench` passes LE1 and shows no convergence anomalies.

### P2.1 Thermal radiation + temperature-dependent properties (M) — done

`radiation.rs` (radiation to ambient in secant form; enclosure radiation: patches ≤ 1 500, BVH ray
casting from every patch against all boundary faces, reciprocity, LU-factorised grey radiosity),
`Load::Radiation` (`add_radiation`, SETUP ▸ THERMAL ▸ Radiation), material `conductivity_table` /
`specific_heat_table`, Picard iterations in steady and transient thermal (also thermal stress and
microwave heating), radiation to ambient in conjugate heat transfer. Validated in
`tests/radiation.rs` and three bench cases: radiating cube 0.02 %, black plates 0.10 %, grey plates
0.23 %, k(T) slab exact, c(T) 0.05 %, radiative cooling 0.4 %, CHT energy balance exact. Still open:
surface-to-surface radiation across the CHT fluid, spectral / specular surfaces. Original plan:

- **Radiation to ambient**: new `Load::Radiation { faces, emissivity, ambient }`,
  q = εσ(T⁴ − T∞⁴), linearised per iteration (h_r = εσ(T²+T∞²)(T+T∞)) in steady and transient
  thermal, thermal-stress and CHT (solid faces only).
- **Surface-to-surface**: an "enclosure radiation" option on selected faces. Boundary quads are
  agglomerated into ≤ ~5 000 patches; view factors by ray casting against a BVH of the surface
  (hemisphere sampling, reciprocity enforced, rows normalised); radiosity system solved per
  iteration; blocking by the parts is automatic.
- **k(T), c(T)**: materials gain optional tables `[[T, k], …]` (and c, and for electronics
  materials sensible defaults); steady solves iterate (Picard, then Newton if slow); transient
  updates per step. Library entries for FR-4, aluminium, copper, air (k and μ for CFD too).
- **Validate**: grey parallel plates and concentric cylinders (analytic exchange); view factor of
  coaxial squares and perpendicular rectangles (Howell catalogue); radiating fin efficiency; slab with
  k = k₀(1+βT) (Kirchhoff transform); energy balance in a sealed box with natural convection +
  radiation.
- **Files**: `model.rs`, `material.rs`, `fem.rs`, `dynamics.rs`, new `radiation.rs`, `cfd.rs`
  (energy equation), GUI load form and material editor.
- **Risks**: view-factor cost (patching + BVH keep it O(N log N)); non-linear convergence at high
  emissivity (relaxation, fall back to Newton).

### P2.2 Results you can trust (S–M) — done

Trilinear hex-mesh point locator and straight-line path sampling (`Mesh::locate_points` / `sample_path` / `sample_path_tensor`, `Results::path_plot`, CSV export, void-gap handling across holes, viewport line overlay, `RESULTS ▸ TOOLS ▸ Path`, `Charts & Path Plot` window, MCP `plot_path`), cylindrical stress tensor rotation (`CylindricalAxis`, `hoop_stress` / `radial_stress` / `axial_stress` in `RESULTS ▸ STRESS`, contextual `CYLINDRICAL AXIS` group, `set_display` / `probe` / `plot_path`), and general 3-grid Richardson extrapolation (`richardson_extrapolate`, `mesh_convergence`, `OPTIMIZE ▸ Convergence`, `apply_design` element-size adoption + undo, MCP `mesh_convergence`). Validated in `tests/results_p2_2.rs` and `src/bench.rs` (`lame_path` 0.63 % hoop / 2.38 % radial along 45° diagonal, `cantilever_conv` Richardson-extrapolated deflection 0.38 % from Timoshenko theory from 3 coarse meshes, plus hole gaps, custom axis orientation, invalid inputs, workspace-jailed CSV, and cancellation). Original plan:

- **Path plots**: pick two points (or a probe line) → any result along it in the chart window, CSV
  export; MCP `plot_path`.
- **Cylindrical results**: radial / hoop / axial stress about a chosen axis (pressure vessels, shafts).
- **Mesh-convergence assistant**: solve at h, h/1.4, h/2, Richardson-extrapolate the chosen
  responses, report the discretisation error and the size needed for a target accuracy; OPTIMIZE ▸
  Convergence.
- **Validate**: Lamé hoop stress along the wall vs closed form; extrapolated cantilever deflection
  within 0.5 % of theory from three coarse meshes.

### P2.3 STEP import (M–L, spike first)

- **Spike (S)**: try the pure-Rust `truck-stepio` reader on five real files (SolidWorks, Fusion 360,
  Onshape, FreeCAD, a KiCad 3D model). If at least four import as closed solids with correct volume,
  use it; otherwise wrap OpenCASCADE behind an optional `step` cargo feature (CI builds it once and
  caches), with a fallback that converts through a locally installed FreeCAD / gmsh.
- **Design**: each STEP solid becomes a body (name from the product tree); B-rep faces become named
  patches with their analytic type (plane, cylinder r, cone, sphere, torus, spline) so boundary
  fitting projects onto the exact surface, not the facets; units read from the file.
- **Validate**: volume and centroid vs the CAD mass properties (< 0.1 %); plate-with-hole Kt from a
  STEP plate equals the CSG result; an assembly of three parts keeps its names and contacts.
- **Risks**: STEP dialect coverage; tolerance gaps between faces (sew with a tolerance, warn).

### P2.4 Assemblies: contact, pretension, connections (L)

- Today bodies that touch share voxel nodes, so every interface is **bonded**. Add per-pair
  interface types — bonded, no separation, frictionless, frictional (μ) — by duplicating the
  interface nodes. On the voxel lattice the duplicated nodes stay paired, so contact is
  node-to-node (small sliding): augmented-Lagrange gap and Coulomb friction with an active-set loop,
  extending today's compression-only supports.
- **Bolt pretension**: a cylinder body or selected bolt body with a pretension section (force or
  adjustment), solved as step 1, external loads as step 2; reports preload loss.
- **Connections**: springs (point-to-ground and between faces), remote points / remote forces and
  moments (distributing coupling), point masses for modal.
- **Validate**: stacked blocks under pressure (separation where expected); block on a plane sliding
  at F = μN; Hertz cylinder-on-flat with fitting (peak pressure within 10 %); bolted joint stiffness
  ratio vs VDI 2230; remote moment vs beam theory.

### P2.5 Local mesh refinement (XL)

- **Octree voxels** with 2:1 balance for the FE solvers (structural, thermal, modal, fields); CFD
  and FDTD stay on uniform grids. Refinement from user sizing per body / face, automatic feature size
  (thin walls, small holes, curvature) and an **adaptive loop** driven by a Zienkiewicz–Zhu error
  estimate.
- **Hanging nodes** are eliminated by linear constraints during assembly; boundary fitting runs per
  leaf; the geometric multigrid coarsens the octree (leaf merging) instead of the uniform lattice.
- **Validate**: patch tests with hanging nodes (exact); Kirsch plate with refinement at the hole:
  same Kt with ≥ 5× fewer elements; adaptive loop shows the expected energy-norm convergence rate.
- **Risks**: the largest refactor of the plan (mesh, assembly, multigrid, fitting, rendering,
  probes); done behind a mesh option until every solver passes its tests on octree meshes.

### P2.6 Broadband FDTD and EMC (M–L)

- Gaussian-modulated pulse excitation; DFTs at many frequencies on monitor surfaces only (ports and
  the far-field box) to keep memory bounded.
- **Lumped ports** (50 Ω, any impedance) with voltage / current integration → S11 / S21, input
  impedance, VSWR, resonances; the existing waveguide ports gain S-parameters too.
- **Emissions spectrum**: field at 3 m / 10 m vs frequency with FCC / CISPR limit lines;
  **ground plane** (image theory) and an antenna-height scan for a test-site estimate.
- **Validate**: half-wave dipole Zin ≈ 73 + j42 Ω at resonance; microstrip Z₀ vs Hammerstad–Jensen;
  patch-antenna resonance vs the cavity model; S11 of a matched waveguide < −25 dB.

### P2.7 Electronics-cooling components (M)

- **Fans**: fan curve (Δp vs flow) on an inlet or an internal fan face; operating point reported.
- **Grilles and porous media**: perforated plates (loss coefficient from open area) and porous
  bodies (Darcy–Forchheimer) for filters and simplified heat sinks.
- **Compact package models**: two-resistor (JEDEC θJC / θJB) parts instead of a uniform box.
- **PCB Joule heat**: `pcb_dc` copper loss mapped onto the board as a heat source in thermal / CHT.
- **Validate**: operating point = fan curve ∩ system curve; perforated-plate Δp vs Idelchik; 2R model
  junction temperature vs the hand calculation.

### P2.8 Transient CFD + k-ω SST (L)

- Implicit transient SIMPLE/PISO with a time step and time-averaged and instantaneous results;
  transient conjugate heat transfer; k-ω SST with wall functions next to LVEL.
- **Validate**: cylinder at Re 100 (Strouhal 0.164, Cd 1.33); backward-facing step reattachment
  length (Driver & Seegmiller); turbulent channel friction closer to Dean than LVEL; heat-up of a
  block in a duct vs a lumped model.

### P2.9 Large deformation + hyperelasticity (L)

- Total-Lagrangian Newton solver with load stepping and line search; follower pressure; Neo-Hooke
  and Mooney–Rivlin materials from test data.
- **Validate**: large-deflection cantilever (Bisshopp–Drucker), uniaxial / biaxial rubber tests vs
  the closed forms, post-buckling of a column (arc length later).

### P2.10 Eddy currents and induction heating (L)

- Frequency-domain A–φ formulation on edge (Nédélec) elements; Joule loss into transient thermal for
  induction heating; coil impedance.
- **Validate**: skin depth and loss in a conducting slab; induction heating of a billet vs the
  analytic power; coil inductance vs the magnetostatic solver.

### P2.11 Shells (L)

- MITC4 shells for sheet bodies from STEP (P2.3) or a plate primitive, connected to solids by
  bonded edges; thickness as a property.
- **Validate**: Scordelis–Lo roof, pinched cylinder, plate bending vs Kirchhoff theory.

## Platform, Security, UX & Architecture (audit, 2026-11-26)

Full line-by-line checklists live in `todo.md`. Two deployment targets were audited:
- **Native laptop/desktop app (current target):** Pure-Rust native RAM/CPU + hybrid iGPU/dGPU
  power design (`main.rs::prefer_integrated_gpu`, on-demand Vulkan dGPU in `gpu.rs`) + `Core::call`
  are strong foundations, needing a hardware iGPU viewport (`P2.A2`), background meshing (`P2.A1`),
  and cross-platform `f32`/`f64` GPU adapter splitting.
- **Browser web app (laptop/desktop):** Compiling `eframe`/`egui` + CPU `render.rs` + `SHADER_F64`
  `gpu.rs` directly to WASM is **not** viable long-term (browser WebGPU lacks `f64`, WASM hits 4 GB
  before 2M–12M cells, CPU rasterization drains battery). `Core::call` instead enables a clean
  split (`P2.W`): Web DOM + Three.js/WebGL2 frontend over a Rust compute backend (or Tauri).

| # | Item | Size | Why / depends on | State |
|---|---|---|---|---|
| P2.S | Security, path sandboxing & parser DoS hardening | S–M | blocks untrusted `.simcraft`/`.stl`/`.kicad_pcb`/MCP risk; prerequisite for web/STEP | done |
| P2.A1 | Non-blocking meshing, STL BVH, `f32`/`f64` GPU split, undo & cancel cleanup | M | prevents UI freezes and panic-unwind hazards before `P2.3` (STEP) and `P2.5` (octree) | done |
| P2.A2 | Hardware iGPU 3D viewport (`glow`/`wgpu`) + modular `gui.rs` / `egui_dock` | M–L | replaces 0.6× CPU `render.rs` drag rasterizer; unlocks smooth animations, transparency, gizmos | planned |
| P2.U | CAE UX overhaul (3D glyphs, body pick/hide, unified edit forms, pre-flight, trackpad, Copilot) | M–L | best after `P2.A2` (hardware viewport + docked drawers) | planned |
| P2.W | Web / Tauri split architecture (DOM + WebGL2/WebGPU UI + `axum` Rust backend / demo WASM) | L–XL | required for browser laptop/desktop delivery; depends on `P2.S` and `P2.A1` | planned |

### P2.S Security, path sandboxing & parser hardening (S–M) — done

- **Path trust boundaries (`session.rs`)**: Enforce an explicit workspace root (`--workspace` /
  `SIMCRAFT_WORKSPACE`) for MCP/headless file tools (`open_project`, `save_project`, `import_stl`,
  `import_kicad`, `pcb_nets`, `pcb_dc`, `export_vtk`, `make_report`, `topology_optimize`
  `export_path`), restrict extensions, and prompt before `Session::open` loads external
  `Shape::Stl` / `Shape::Pcb` paths embedded inside untrusted `.simcraft` files.
- **IPC & temp-file hardening (`bridge.rs`, `mcp.rs`, `pysimcraft`)**: Replace `/dev/urandom` + FNV
  fallback in `random_token()` with `getrandom`; enforce `0o600` on existing token files and drop
  the shared `temp_dir()` token fallback; cap concurrent bridge threads, add auth/read timeouts and
  client line-length limits (or migrate to `0o600` Unix domain sockets / named pipes); replace
  predictable `simcraft-handoff-{pid}.simcraft` in `mcp::launch_gui` with `NamedTempFile`.
- **Parser, mesh & study bounds (`geometry.rs`, `kicad.rs`, `mesh.rs`, `study.rs`)**:
  - Reject `!is_finite()` (`NaN`/`Inf`) in `Shape::validate()`, `mesh::build_mesh_ex` (`h`), and all
    tool inputs; use `checked_mul` for `dims[0] * dims[1] * dims[2]` and binary STL `n * 50`.
  - Cap STL / `.kicad_pcb` file sizes; bound S-expression nesting depth + iterative `Drop` on `Sx`
    to prevent stack overflow; clamp KiCad 5 arc subdivisions (`kicad.rs:236`) and enforce a
    `MAX_PIXELS` cap on `PcbData::from_text` (`nx * ny`) for extreme aspect ratios.
  - Require `run.responses[0].is_some()` in `study::optimize`'s `feasible` check (`study.rs:460`).
- **Escaping, SSRF & shared-state safety (`bench.rs`, `assistant.rs`, `gpu.rs`, `elastic_mg.rs`)**:
  - Escape `"` and `'` in `bench::esc` and escape all report metadata fields.
  - Move `gpu::MODE`, `LAST_DEVICE`, `FALLBACK` from process-global statics to per-session/solve
    context; allowlist/validate LLM URLs and cap `ureq` response bodies in `assistant.rs`; add
    `debug_assert!` bounds/disjointness checks around `SyncPtr` in `elastic_mg.rs`.
- **Validate**: Regression tests with malformed/extreme-aspect/deeply-nested `.kicad_pcb`, `NaN`/`Inf`
  and huge-dim `.stl`/CSG inputs, path-traversal rejections, and `cargo test --release`.

### P2.A1 Non-blocking engine, STL BVH, GPU adapter split & undo/cancel cleanup (M) — done

- **Off-UI-thread meshing & KiCad inspection (`gui.rs`, `geometry.rs`)**: Move `ensure_mesh()` out
  of `viewport()` (`gui.rs:3796`) and `net_pads()` out of `open_pcb_dc()` (`gui.rs:2325`) onto
  cancellable background jobs so large meshes/CAD never freeze the window; build a triangle BVH in
  `load_stl` so `Shape::Stl::{project, best_patch}` and `x_crossings` run in O(log T) instead of
  O(T).
- **Cross-platform laptop GPU selection (`gpu.rs`, `fdtd.rs`)**: Split `pick_adapter(require_f64)`
  so `f32` `fdtd.rs` uses `wgpu::Backends::PRIMARY` without requiring `SHADER_F64` (unlocking Apple
  Silicon Metal, Windows DX12, and integrated GPUs), reserving `SHADER_F64` for `gpu.rs` PCG (with
  a future mixed-precision `f32` fallback).
- **Undo & cancellation refactor (`session.rs`)**: Replace double `serde_json::to_string(&s.model)`
  undo diffing (`session.rs:517-523`) with explicit mutation/command tracking before `P2.3` (STEP
  B-reps) lands; migrate `check_cancel()` (`session.rs:1761-1765`) from `panic::resume_unwind` to
  cooperative `&AtomicBool` checks returning `Result` so `P2.5` (octree) and `P2.8` (transient CFD)
  can cancel cleanly inside parallel `rayon` loops.
- **Validate**: Non-blocking UI responsiveness during >500k-cell mesh generation; GPU FDTD passes
  on Metal/DX12/non-`f64` adapters; cancellation tests inside parallel loops leave no poisoned state.

### P2.A2 Hardware iGPU 3D viewport & modular UI shell (M–L)

- **Hardware 3D viewport (`gui.rs`, `render.rs`)**: Replace CPU RGBA texture uploads (`render::rasterize`
  + `0.6×` drag downscaling) with an `egui::PaintCallback` (`glow` or `wgpu` on the integrated GPU,
  keeping `prefer_integrated_gpu()` battery savings). Upload mesh vertex buffers, nodal scalars, and
  deformations once so orbit, mode animation (currently rebuilding `Scene` every frame via `skey` at
  `gui.rs:3890`), MSAA, perspective toggle, translucent enclosures, and section cuts run at 60+ FPS.
  Keep `render::render_png` for headless MCP `render_image` and HTML reports.
- **Modularize `gui.rs`**: Split the 4,421-line file into `ui/{header,ribbon,browser,properties,viewport,charts,dialogs}.rs`
  and replace overlapping floating `egui::Window` popups (`Study`, `Help`, `Planner`, `PCB DC`,
  `History`) with dockable/bottom drawers (`egui_dock` + `egui_plot`).

### P2.U CAE UX overhaul (M–L)

- **3D clarity & direct manipulation**: Instanced 3D load/BC glyphs (force vectors, pressure normals,
  constraint pins, fluid inlet arrows, coil axes); hover face/body highlight + body selection mode +
  right-click depth-peeling for hidden faces; per-body hide/ghost/isolate toggles in the Browser
  (essential for enclosures and `P2.3` assemblies); live ghost previews + 3D transform gizmos before
  `Create` on primitives/enclosures; ViewCube alongside axis triad.
- **Unified forms & pre-flight diagnostics**: Replace the raw `json_editor` fallback in `item_editor`
  (`gui.rs:390, 2934`) with the same unit-aware, face-picker forms used during creation; add a live
  pre-solve readiness checklist (`✓ Solid · ✓ Material · ✗ Needs Support · ⚠ Mesh > λ/10`); group
  KiCad component bodies under their board in a searchable Browser.
- **Laptop trackpad ergonomics & unified AI Copilot**: Support `Space+drag` / two-finger pan,
  pinch-to-zoom, view/tool shortcuts (`F`, `1–6`, `Del`, `Esc`), unit-system switcher, and unify
  `✨ Planner` + `Connect AI` into an in-app Copilot drawer proposing undoable tool actions.

### P2.W Web / Tauri split architecture (L–XL)

- **Why not WASM-only `egui`**: Browser WebGPU WGSL lacks `f64` (`gpu.rs`), `wasm32` hits < 4 GB
  RAM before 2M–12M cells, `rayon` requires COOP/COEP `SharedArrayBuffer`, and CPU `render.rs`
  drains laptop battery in WASM.
- **Target split**:
  1. **Web Frontend (TypeScript + React/Svelte + Three.js/WebGL2 or WebGPU):** Uses `session::tool_defs()`
     JSON schemas for UI parity, renders client-side binary mesh/field buffers (`Float32Array`) with
     hardware picking, gizmos, and native DOM accessibility/charts.
  2. **Rust Compute Backend (`axum` HTTP + WebSocket + binary frame streaming):** Reuses `Session` /
     `Core::call` (after `P2.S` and `P2.A1`) for full 2M–12M cell solves and native Vulkan `f64` GPU
     acceleration on a server or local companion daemon — or packaged with the exact same Web UI as a
     **Tauri** native laptop app.
  3. **Optional zero-server WASM worker:** Strip `egui`, `render.rs`, `gpu.rs`, `bridge.rs`, and `rfd`
     behind cargo features so small-mesh demos (< ~100k cells) run offline in a WebWorker.

### Not planned (for now)

Explicit dynamics (drop tests), composites lay-ups, multiphase / compressible CFD, motors (transient
magnetics with motion), acoustics, piezoelectrics, battery electrochemistry and distributed (cluster)
solving. Each is large and serves a narrower audience; revisit after P2.8.

## 0. Do first: security of the local bridge (S)

The GUI's MCP bridge (`src/bridge.rs`, `127.0.0.1:47800`) accepts any tool call from any local
process with no authentication, and tools such as `export_vtk`, `save_project` and `import_stl` take
arbitrary file paths. Fix before adding more automation:

- random token in `~/.config/simcraft/token` (mode 0600), required as the first line of a connection
  (the MCP server reads the same file);
- maximum request line length (≈ 1 MB);
- document file paths as a trust boundary.

## 1. Quick wins (S–M)

| Item | Design | Files |
|---|---|---|
| **Per-body / per-element heat sources** | `Load::HeatGeneration` gains an optional `body`; `transient_thermal` accepts an element heat vector like `thermal` already does. Prerequisite for microwave heating, PCB component power and Joule-coupled transients. | `model.rs`, `fem.rs`, `dynamics.rs` |
| **HTML reports** (Dynamic Reporting) | One self-contained HTML file: model, materials, loads table, mesh stats, PNG views from `render_png` as data URIs, results with user-set pass/fail checks, study tables, inline SVG charts. No new dependency; PDF via the browser's print. | new `report.rs`, `session.rs` (`make_report`), GUI button |
| **Python access** (PyAnsys / SAF) | Keep the JSON bridge as the protocol (no HTTP server). `simcraft serve --headless --port N`; `simcraft schema` dumps `tool_defs()`; a stdlib-only `pysimcraft` package with methods generated from the schema. | `main.rs`, `bridge.rs`, `python/` |

## 2. Dynamics on top of modal (S–M each)

All reuse the existing modal solver and validate against single-degree-of-freedom closed forms.

1. **Response spectrum** (S): participation factors Γ = φᵀMr, modal response from a spectrum table,
   SRSS/CQC combination.
2. **Harmonic response, mode superposition** (S–M): modal damping, frequency sweep stored as a
   history; validate |X| = (F/k)/√((1−r²)² + (2ζr)²).
3. **Random vibration / PSD** (M): 1σ and 3σ responses from modal PSD integration; needs per-mode
   stress fields in `Results`.
4. **Fatigue** (M): post-processing only: S-N curves per material, Goodman/Gerber mean-stress
   correction, Miner damage; works on static, harmonic and random results.

*Review note:* modal uses lumped mass; add a consistent-mass option before trusting high modes in
random and spectrum results. Full complex harmonic solves (non-proportional damping), large
deformation and two-body contact are large rewrites and come later.

## 3. Full-wave electromagnetics on the GPU (L, ≈ 8–10 weeks)

**Method:** FDTD (Yee grid) in fp32 WGSL on the existing voxel grid; frequency-domain edge elements
would need complex indefinite solves the codebase doesn't have.

- Grid: ≥ 10 cells per wavelength in the densest material (food εr ≈ 50 at 2.45 GHz → λ ≈ 17 mm →
  h ≈ 1.5 mm); fp32 fields, f64 accumulation of DFT phasors and absorbed power.
- Boundaries: PEC (oven walls), CPML (8–10 cells) on open faces; sources: WR-340 TE10 waveguide port,
  lumped ports, plane wave.
- Steady state: ramped source, run to steady amplitude, running DFT over whole periods.
- Lossy food: Debye model via auxiliary differential equations; FR-4 via loss tangent.
- Coupling: P = ½ωε₀ε''|E|² per element → transient thermal (uses §1 heat sources); re-solve EM when
  ΔT > ~3 K; turntable as an average over several rotated positions.
- EMI: Huygens box + near-to-far-field transform at CISPR frequencies.
- Validation: empty-cavity resonances (TE101), plane wave on a lossy half-space, slab absorption depth,
  WR-340 guide wavelength, 50 Ω microstrip vs Hammerstad.

*Review note:* the sub-agent sized the time step from the slowest medium (food). FDTD stability is
set by the fastest one (air): Δt = h/(c√3) ≈ 2.9 ps at 1.5 mm, ≈ 140 steps per 2.45 GHz period, not
≈ 1100. An oven at 1.5 mm (~5 M cells) should therefore solve in well under a minute on the RTX 3070 Ti.
PCB EMI needs sub-cell trace models on top (largest risk); microwave heating does not.

## 4. Electronics cooling: CFD + heat transfer (M–L)

Order: validation harness (lid-driven cavity vs Ghia, de Vahl Davis natural convection) →
**conjugate heat transfer with Boussinesq buoyancy** (energy equation on the same voxel cells,
harmonic-mean conductance at solid/fluid faces) → **zero-equation turbulence (LVEL)**, which suits a
staircase voxel grid → **geometric multigrid pressure solver** + GPU kernels → linearised **radiation**
BC, then surface-to-surface radiosity → second-order/implicit **transient flow** (validate cylinder at
Re = 100: St ≈ 0.165, Cd ≈ 1.33).

## 5. KiCad PCB import (M, ≈ 6–8 weeks)

Parse `.kicad_pcb` (S-expressions) in pure Rust: board outline, stack-up, filled zones, tracks, vias,
pads, footprints. 35 µm copper can't be voxelised directly, so each copper layer is rasterised to a
per-cell copper fraction and homogenised into **orthotropic** properties (in-plane k = Σkᵢtᵢ/Σtᵢ,
through-plane k = Σtᵢ/Σ(tᵢ/kᵢ), via density adds to k_z), quantised into a few derived materials.
Needs orthotropic conductivity/stiffness/CTE in `material.rs` and `fem.rs`. Components become heat
sources (§1); pads become named faces for current terminals. Warpage from Cu/FR-4 CTE mismatch.
Validate with slab conduction, trace resistance ρL/(wt), IPC-2152 temperature rise and bimetal
curvature. Pairs with the Irona KiCad tools for component power.

## 6. Accurate curved surfaces (L, ≈ 6–8 weeks)

**Boundary node projection**: move the voxel mesh's surface nodes onto the true CAD surface and smooth
the first layer inward. Connectivity, the sparse pattern and the GPU ELLPACK layout are unchanged.
Interior elements remain identical cubes (one cached stiffness), and only the boundary shell uses
general isoparametric hex8 with Taylor-corrected incompatible modes. Check det J > 0 and fall back
per element. Validate with the Kirsch plate (Kt = 3) and Lamé thick sphere/cylinder; target: peak
stress error from ~15 % to < 3 % at the same element size. CFD, topology and field solvers keep the
plain voxel grid.

## 7. Surrogates (SimAI-style) (M)

Gaussian-process response surfaces fitted to study runs (ARD squared-exponential kernel, inputs
normalised to [0,1], hyper-parameters by marginal likelihood, closed-form leave-one-out error).
Used for instant "what-if" sliders and expected-improvement optimisation that verifies each proposal
with a real solve. Always labelled "surrogate, not verified". Field-level surrogates later.

## Recommended sequence

1. Bridge security (§0) and per-element heat sources (§1).
2. HTML reports and Python access (§1).
3. Response spectrum → harmonic → random → fatigue (§2).
4. GPU FDTD + microwave heating (§3), then PCB EMI.
5. Conjugate heat transfer + LVEL + multigrid (§4).
6. KiCad import (§5), which feeds §3 and §4 with real boards.
7. Boundary projection meshing (§6) and surrogates (§7).
