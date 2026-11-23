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
| §3 GPU FDTD + microwave heating | ✅ done — PEC/absorbing boundaries, TE10 ports, loss tangent, GPU; validated in `tests/fdtd.rs`. Still open: Debye dispersion, turntable, near-to-far-field / EMI, sub-cell traces |
| §4 Electronics cooling (CHT + turbulence) | ✅ done — conjugate heat transfer, Boussinesq buoyancy, LVEL, moving walls, multigrid pressure; validated in `tests/cfd.rs` (Ghia, de Vahl Davis, Dean). Still open: radiation, transient CFD, k-ε |
| §5 KiCad PCB import | ✅ done — outline/cut-outs, stack-up, tracks/arcs/pads/zones/vias rasterised, orthotropic homogenised conductivity, component boxes with power; validated in `tests/kicad.rs`. Still open: electrical (trace resistance / IPC-2152), warpage, 3D models |
| §6 Accurate curved surfaces | ✅ done for Static Structural — conservative voxels + pillow layer on curved/off-grid patches, nodes projected onto the true surfaces with tangential smoothing, general hex8 with Taylor-corrected incompatible modes, consistent face loads; validated in `tests/fitting.rs` (patch test, Kirsch 1.7 %, Lamé 1.7 %). Still open: fitted meshes for thermal/modal/vibration |
| §7 Surrogates | ✅ done — Gaussian-process response surfaces (ARD kernel, marginal likelihood, closed-form LOO/Q²), what-if sliders and `predict_surrogate`, Bayesian optimisation (EI × probability of feasibility) with every proposal verified by a real solve; validated in `tests/surrogate.rs`. Still open: field-level surrogates |


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
