# SimCraft User Guide

SimCraft is a desktop finite-element simulation app for structural, thermal, electromagnetic and
fluid problems. You build a simple 3D model, pick an analysis, apply supports and loads, solve, and
read the results — by hand in the app, or by asking an AI agent (Claude Code, Antigravity, …) to
drive it for you.

**Contents**

1. [Installing and starting](#1-installing-and-starting)
2. [A tour of the window](#2-a-tour-of-the-window)
3. [Your first simulation in 5 minutes](#3-your-first-simulation-in-5-minutes)
4. [Geometry](#4-geometry)
5. [Materials](#5-materials)
6. [Meshing](#6-meshing)
7. [The analyses](#7-the-analyses)
8. [Loads and supports reference](#8-loads-and-supports-reference)
9. [Reading results](#9-reading-results)
10. [Optimization studies](#10-optimization-studies)
11. [CPU and GPU compute](#11-cpu-and-gpu-compute)
12. [The Simulation Planner (local AI)](#12-the-simulation-planner-local-ai)
13. [Driving SimCraft from AI agents (MCP)](#13-driving-simcraft-from-ai-agents-mcp)
14. [Units cheat sheet](#14-units-cheat-sheet)
15. [Accuracy and validation](#15-accuracy-and-validation)
16. [Troubleshooting](#16-troubleshooting)
17. [Limitations](#17-limitations)

---

## 1. Installing and starting

```bash
git clone https://github.com/sushil-sundaram/simcraft
cd simcraft
cargo build --release          # needs a recent Rust toolchain
./target/release/simcraft      # start the app
```

On Linux, add a launcher entry and icon (shows up in your app menu / dock):

```bash
scripts/install-desktop.sh
```

Command-line options:

| Command | What it does |
|---|---|
| `simcraft` | Open the app |
| `simcraft model.simcraft` | Open the app with a project |
| `simcraft mcp` | Run the MCP server for AI agents (see §13) |
| `simcraft serve --port N` | Headless server for Python scripts (no window) |
| `simcraft schema` | Print every tool with its JSON schema |
| `SIMCRAFT_GPU=discrete simcraft` | Let the window itself render on the discrete GPU (normally it stays on the integrated GPU) |
| `SIMCRAFT_PORT=…` | Change the local port used between the app and the MCP server (default 47800) |
| `SIMCRAFT_LLM_URL=…` | Address of the local model server for the Planner (default `http://127.0.0.1:11434`) |

---

## 2. A tour of the window

```
┌ SimCraft  File  Examples  Connect AI  ✨ Planner ───────────── ● MCP ready  ☀/🌙 ┐
│ GEOMETRY  MATERIAL  SETUP  MESH  SOLVE  RESULTS          ← ribbon tabs           │
│ [tools for the selected tab, grouped CREATE · MODIFY · INSERT …]                 │
├──────────────┬───────────────────────────────────────────────┬──────────────────┤
│ BROWSER      │                 3D VIEWPORT                   │ PROPERTIES       │
│ Geometry     │  legend · title             [view toolbar]    │ forms for the    │
│ Analysis +   │                                               │ current tool,    │
│  loads       │                                               │ selected faces   │
│ Mesh         │  axis triad                                   │                  │
│ Solution     │                                               │                  │
├──────────────┴───────────────────────────────────────────────┴──────────────────┤
│ status bar: progress, messages, errors                                          │
└──────────────────────────────────────────────────────────────────────────────────┘
```

- **Ribbon** — the workflow from left to right: build geometry, assign material, set up the analysis
  and its loads, mesh, solve, look at results. Each tab shows only the tools that make sense.
- **Browser** (left) — everything in the model: bodies (🗑 to delete), the analysis with its supports
  and loads (coloured dots match the colours in the viewport), mesh statistics and a solution summary
  (key maxima, reactions, frequencies, derived quantities such as capacitance or drag).
- **Properties** (right) — the form for the tool you picked (e.g. Box size, Force vector) and the
  faces you have selected.
- **Viewport** — left-drag to orbit, right/middle-drag to pan, scroll to zoom (towards the cursor).
  Click a face to select it, Shift+click to add more. Hovering shows the face name. The toolbar
  (top-right, or bottom-right in narrow windows) has Fit, Iso, Front, Top, Right and Left views.
- **Header** — **Connect AI** (MCP setup), **✨ Planner** (local AI to-do lists), the MCP status
  dot and the light/dark switch (remembered between runs).
- **OPTIMIZE** tab — parameter studies, design optimization and topology optimization (§10).
- **Examples** menu — ready-made models for every kind of analysis; open one and press Solve.

Model colours before solving: grey = solid, blue = fixed support, purple = displacement, light blue =
compression-only, red = force, orange = pressure, pink = temperature, yellow = heat flux, cyan =
convection, violet = voltage/current, brown = coil, crimson = magnet, green = inlet/outlet, green
highlight = your current selection.

---

## 3. Your first simulation in 5 minutes

A 120 × 40 × 10 mm aluminium bracket with a hole, fixed at one end, pushed down at the other.

1. **GEOMETRY ▸ Box** — corner `0, 0, 0`, size `120, 40, 10`, name it `Bracket`, **Create**.
2. **GEOMETRY ▸ Hole** — base centre `80, 20, 0`, axis Z, radius `8`, height `10`, **Create**.
3. **MATERIAL** — choose *Aluminum 6061-T6*, **Assign to all bodies**.
4. **SETUP** — the analysis is already *Static Structural*.
   - Click the left end face (`Bracket.xmin`), press **Fixed**, **Apply**.
   - Click the right end face (`Bracket.xmax`), press **Force**, enter `0, 0, -500` N, **Apply**.
5. **MESH** — element size `2` mm, **Generate** (or leave it on Auto).
6. **SOLVE ▸ ⚡ Solve**.
7. **RESULTS** — the von Mises stress appears. Try *Total* deformation, *Safety* factor, the
   deformation scale and **Animate**. The Browser lists the maximum stress, deflection and the
   reaction force at the support (it should equal the 500 N you applied).

Save with **File ▸ Save** (`.simcraft` project file).

---

## 4. Geometry

SimCraft builds parts from simple shapes combined in order (constructive solid geometry):

| Tool | Creates | Faces you can select |
|---|---|---|
| **Box** | rectangular block from a corner and a size | `Name.xmin`, `xmax`, `ymin`, `ymax`, `zmin`, `zmax` |
| **Cylinder** | cylinder along X, Y or Z from its base centre | `Name.side`, `Name.bottom`, `Name.top` |
| **Sphere** | sphere | `Name.surface` |
| **Hole / Pocket** | a cylinder / box that *subtracts* material | the cut surfaces, e.g. `Hole.side` |
| **Import STL** | a closed STL solid (e.g. exported from SolidCraft, Fusion 360, FreeCAD, KiCad 3D) | smooth regions become `Name.face0`, `face1`, … |
| **Enclosure** | the air / water / oil box around your parts (see below) | `Enclosure.xmin` … `Enclosure.zmax` |

Bodies are applied in the order listed in the Browser: later bodies add to or cut from earlier ones.
Touching bodies merge into one connected part. Each body has its own material.

**STL tips** — the mesh must be closed (watertight). Use the `scale` option if your file is in metres
(scale 1000) or inches (25.4). Use **list_faces** (or hover in the viewport) to find the face names.

**Face selectors** — wherever a load asks for faces you can use:

- a face name: `Bracket.xmax`, `Hole.side`
- a whole body: `TopPlate` (all of its outer faces)
- the extreme flat faces of the whole model: `bbox.xmin` … `bbox.zmax` (handy for STL parts)
- `all` — every outer surface of the parts (e.g. convection everywhere)

**Enclosure** — electrostatic, magnetostatic and fluid analyses need the space *around* the parts.
**GEOMETRY ▸ Enclosure** adds a box (default: 50 % of the part size on every side) filled with Air,
Water or Engine Oil. It is ignored by structural and thermal analyses. Its outer faces are where you
put CFD inlets and outlets; magnetic fields are held at zero there, so leave enough room (≥ 1–2 part
sizes) around coils and magnets.

### Importing a KiCad board

**GEOMETRY ▸ KiCad PCB** (or **File ▸ Import KiCad board…**, MCP `import_kicad`) reads a
`.kicad_pcb` file from KiCad 5 to 10. **Fill all zones (B) and save in KiCad first** — unfilled pours
carry no copper and SimCraft warns about them.

- **The board** becomes one body (default name `Board`): the Edge.Cuts outline including cut-outs
  and rounded corners, extruded over the stack-up thickness, lying in XY from z = 0. Its faces are
  `Board.top`, `Board.bottom` and `Board.edge`.
- **Copper** — tracks, arcs, pads, filled zones and via rings on every copper layer, plus the plated
  via barrels — is rasterised at about 0.1 mm. A 35 µm copper layer is far thinner than any mesh
  cell, so each cell column gets a **homogenised orthotropic** conductivity: along the board
  k∥ = Σkᵢtᵢ / t (copper planes spread heat), through it k⊥ = t / Σ(tᵢ/kᵢ) in parallel with the
  via copper. A ground plane makes k∥ about 25 W/m·K versus 0.3 for bare FR-4; thermal vias raise
  k⊥ several times. Heat capacity is mixed by volume. Values are corrected for the voxelised
  thickness, so use any element size up to **half the board thickness** (0.8 mm for 1.6 mm boards;
  the automatic size does this).
- **Components** — footprints with a courtyard (or fab/pad outline) of at least 4 mm² become boxes
  named by reference (`U1`, `Q3`, `L1`…) of IC mould compound, 1 mm tall by default, on top of
  the board or below it for bottom-side parts. Options: `heights` {"U1": 2.5}, `component_height_mm`,
  `min_component_area_mm2`, `components: false`.
- **Power** — `power` {"U1": 1.5, "Q1": 0.8} (W) adds heat sources at import, or add **Heat
  Generation ▸ Total power** on a component body later.

Then either:
- **Steady/transient thermal**: add **Convection** on `all` (natural air ≈ 5–10 W/m²K, a small fan
  20–50) — fastest, good for comparing layouts and copper pours.
- **Conjugate heat transfer**: add an Air enclosure around the board, an inlet (fan speed and
  temperature) and an outlet, or gravity for natural convection — the airflow is computed.

Results show the board and part temperatures; the report lists the hottest part. Structural and
electrical analyses use the laminate's properties (copper is homogenised for heat only).

---

## 5. Materials

The library (SI properties) includes structural steel, stainless steel 304, aluminium 6061-T6 /
7075-T6, titanium Ti-6Al-4V, copper, brass, cast iron, electrical steel (M19), NdFeB N42 magnet,
concrete, ABS, PLA, nylon PA12, PTFE, FR-4, carbon fibre, oak, glass, and the fluids Air, Water and
Engine Oil.

Each material carries: Young's modulus, Poisson's ratio, density, yield and ultimate strength,
plastic tangent modulus, thermal conductivity, expansion and specific heat, relative permittivity,
electrical conductivity, relative permeability and (for fluids) viscosity.

- **MATERIAL** tab: pick a material and **Assign to all bodies**; the panel shows its key values.
- Per body: via the MCP tool `set_material` with `bodies`.
- Custom: `define_material` (start from an existing one with `based_on` and override values).

---

## 6. Meshing

SimCraft fills the model with equal cubes ("voxels") of the **element size** you choose.

- **Auto** picks a size giving ~40 elements across the largest dimension.
- Rule of thumb: at least **3–4 elements through the thinnest wall** you care about
  (e.g. 2.5 mm for a 10 mm thick plate in bending).
- Halving the size gives 8× more elements; solve time grows faster than that.
- Limit: 400 000 elements per mesh.
- Curved surfaces come out stair-stepped. Global results (deflection, frequency, temperature,
  capacitance) stay accurate; peak stress *on* a curved surface reads high — refine there, or read
  stress a little away from the edge.
- Size the cubes so thin layers line up with the grid (e.g. 1 mm cubes for a 4 mm dielectric):
  a 4 mm layer meshed with 1.5 mm cubes becomes 4.5 mm.

The MESH tab also toggles element edges in the viewport.

---

## 7. The analyses

Pick the analysis in **SETUP ▸ Analysis**. The ribbon then shows only the loads that apply. Extra
settings appear in **SOLVE ▸ Settings**.

### Structural

**Static Structural** — stress and deflection under steady loads.
Needs at least one support (Fixed, Displacement, Compression-only or enough Symmetry planes) and a
load (Force, Pressure, Gravity or a prescribed Displacement). Results: deformation (total and X/Y/Z),
von Mises, max/min principal stress, safety factor (yield ÷ von Mises, capped at 15), reaction forces.
*Pitfall:* "under-constrained" means the part can still slide or rotate — add supports.

**Modal** — natural frequencies and mode shapes. Needs supports only (no loads). Setting: number of
modes (1–20). Results: frequency list; pick a mode in RESULTS and press **Animate**. A free-free part
shows near-zero "rigid-body" frequencies.

**Linear Buckling** — the load factor at which a slender part buckles. Apply supports and the
compressive load; the result is a multiplier: critical load = multiplier × applied load. Results: one
multiplier and buckled shape per mode. A negative/absent result means the load does not compress it.

**Transient Structural** — response over time to a load applied suddenly at t = 0 (impact, step loads).
Settings: end time, time steps, damping ratio (fraction of critical, default 2 %). Results: animated
response (time slider in RESULTS), peak displacement and when it occurs, the history of the maximum
displacement, stresses at the final time. Use ≥ 20 time steps per vibration period (SimCraft warns
if the step is too coarse).

**Nonlinear Structural (plasticity)** — loads beyond yield. Uses each material's yield strength and
tangent modulus (bilinear hardening). Setting: load steps (default 10). Results: stresses, equivalent
plastic strain, reaction forces, load-factor history. If a load step cannot converge the structure is
collapsing (plastic hinge) — the results show the last load level reached.

**Harmonic Response** — steady-state vibration amplitude when the loads oscillate sinusoidally
(an unbalanced motor, a shaker, a pump). Forces and pressures you apply are the **amplitudes**; gravity
is ignored. Uses mode superposition (at least 12 modes, more for wide sweeps) with the damping ratio.
Settings (SOLVE): damping ratio, sweep From/To in Hz (0 = automatic around the modes), frequency points;
extra points are placed around every resonance automatically. Results: the frequency-response curve
of the largest amplitude (shown in the Browser and the report), and deformation/stress fields at the
peak frequency. Quantities: peak frequency, peak amplitude.

**Response Spectrum** — peak response to a shock or seismic base motion described by a design
spectrum. Supports only (the base is what moves). Settings: base direction (X/Y/Z), damping ratio,
combination **SRSS** (well-separated modes) or **CQC** (closely spaced modes), and the spectrum table
— one `frequency_Hz  acceleration_g` pair per line in the SPECTRUM box, then **Apply** (log-log
interpolated; **Example** loads a flat 1 g spectrum). Results: combined peak deformation and stresses
(always positive — they are envelopes, not a deformed shape), effective mass per mode and the
cumulative effective mass. *Pitfall:* if the cumulative effective mass is below 90 % SimCraft warns —
raise the number of modes.

**Random Vibration (PSD)** — statistical response to broadband vibration (vehicles, launch, transport
tests) given as an acceleration PSD. Settings: base direction, damping ratio, PSD table in
`frequency_Hz  g²/Hz` (**Example** loads a 20–2000 Hz profile). Results: **1σ (RMS)** deformation and
stresses; quantities: input Grms, 1σ and 3σ maxima, zero-crossing frequency ν₀. Multiply 1σ by 3 for
a 99.7 % peak estimate.

**Fatigue (stress-life)** — post-processing of a solved Static, Harmonic, Response Spectrum or Random
Vibration result: **RESULTS ▸ FATIGUE ▸ Evaluate** (MCP `fatigue_life`). Uses the Basquin S-N curve
σa = σf′(2N)ᵇ of each material and its endurance limit; when a material has no fatigue data,
σf′ ≈ Su + 345 MPa, b ≈ −0.085 and (steels) Se ≈ 0.5 Su are estimated, and a warning says so — enter
measured values with `define_material` for design work.
- *Static results*: choose how the solved load repeats — **Fully reversed** (±), **Zero-based** (0 → max)
  or **Ratio R** = σmin/σmax — and the mean-stress correction **Goodman** (default), **Gerber**,
  **Soderberg** or none. Compressive means are not credited.
- *Harmonic / spectrum*: the stresses are amplitudes, treated as fully reversed.
- *Random*: Steinberg's 3-band method (68.3 % of cycles at 1σ, 27.1 % at 2σ, 4.33 % at 3σ) at ν₀ over
  the exposure time (in hours in the window, `exposure_time_s` over MCP).
- *Load scale* multiplies the stresses (quick what-if without re-solving).

Fields: **Fatigue Safety Factor** (stress factor to reach the design life; capped at 15), **Fatigue
Life** (cycles; 10¹² = infinite) and **Fatigue Damage** (Miner sum for the design life or exposure;
≥ 1 means failure). Safety-factor and life plots use a reversed colour scale: **red = critical**.

**Contact (compression-only support)** — available in Static Structural: the faces rest on a rigid
surface that pushes back but cannot pull. SimCraft finds which nodes stay in contact. If everything
lifts off, the part is free and you will get an "under-constrained" message.

### Thermal

**Steady-State Thermal** — temperatures once everything has settled. Needs at least one
Temperature or Convection boundary (otherwise heat has nowhere to go). Heat inputs: Heat Flux,
Heat Generation (whole model or one body). Results: temperature, heat flux, heat flow through
fixed-temperature faces (positive = into the part, negative = out of it).

**Transient Thermal** — temperatures over time. Settings: end time, time steps, initial temperature.
Uses density × specific heat. No sink is required (an insulated part simply heats up). Results:
temperature at every saved time (time slider), max-temperature history.

**Thermal-Stress** — a thermal solve followed by a structural solve with thermal expansion. Apply
both thermal loads and structural supports. Setting: stress-free reference temperature.

### Electromagnetic

**Electrostatic** — voltages on electrodes → potential, electric field, charge on each electrode,
capacitance and stored energy. Metals are treated as perfect conductors (equipotential). Add an
Enclosure to include the fringing field in the surrounding air.

**DC Current Flow** — current in conductors. Apply a Voltage (one face at 0 V as ground) and either a
second Voltage or a Current. Results: potential, current density (A/mm²), Joule heat (W/m³), current
through each terminal, resistance, total power.

**Thermal-Electric (Joule heating)** — DC current flow whose losses heat the part, then a thermal
solve. Add electrical terminals *and* thermal boundaries (e.g. convection on `all`).

**Magnetostatic** — magnetic fields of coils and permanent magnets, with iron parts. Needs an
Enclosure. Sources: **Coil** (current density in a body, flowing around an axis for windings or along
it for bus bars) and **Magnet** (remanence vector, e.g. `0, 0, 1.3` T for N42 along +Z; use the
NdFeB material). Results: B (T), H (A/m), stored energy, peak |B| in the solids. View it with the cut
plane.

### Full-wave electromagnetics (RF and microwaves)

**Full-Wave EM (FDTD)** — Maxwell's equations in the time domain on the voxel grid (Yee scheme), run
to a steady state at one frequency. Use it for microwave ovens and applicators, shielding boxes,
waveguides and RF fields around parts.
- **Enclosure** = the space the waves live in. Its outer boundary (SOLVE ▸ Outer boundary) is either
  **Metal cavity** (`em_boundary: pec` — oven walls, shields) or **Open** (`open` — a 10-cell absorbing
  layer, for things that radiate; parts touching the enclosure wall continue into the absorber, so a
  waveguide that reaches the wall is matched).
- **RF Port** (SETUP ▸ ELECTROMAGNETIC ▸ RF Port, MCP `add_port`) — where the power enters: pick a face
  (usually an Enclosure wall), the power in W and the electric-field direction, which must lie in the
  face. The **TE10** profile (default) is a half-sine across the face like a rectangular waveguide
  feed; *uniform* is constant. Fields are scaled so the port **delivers** exactly that power.
- **Materials**: εr, conductivity and the **loss tangent** tan δ (the absorbing part, ε″ = εr·tan δ).
  Library values at 2.45 GHz: Water (εr 78, tan δ 0.157), *Food (water-rich, 2.45 GHz)* (εr 52,
  tan δ 0.3), glass, PTFE, FR-4 … Metals whose skin depth is much smaller than a cell are perfect
  conductors. Add your own with `define_material` (`relative_permittivity`, `loss_tangent`).
- **Mesh**: at least **10 cells per wavelength in the densest material** — λ = 300 mm / (f[GHz]·√εr);
  food at 2.45 GHz needs ≤ 1.7 mm. SimCraft warns below 10 and refuses below 3; the automatic element
  size follows this rule. Full-wave meshes may have up to 12 M cells (the air counts).
- Settings: frequency (GHz), outer boundary, max RF periods (default 400; most cavities settle in
  50–150). Results: **|E|** (peak amplitude, V/m — view it with the cut plane), **Absorbed power
  density** (W/m³), **SAR** (W/kg), and quantities: input and absorbed power, absorbed fraction, power
  absorbed in each body, peak |E|, periods simulated, cells per wavelength. In a closed metal cavity
  the absorbed fraction is ~100 % (a check of convergence).
- Runs on the GPU for grids ≥ 150 000 cells in Auto mode (single precision; same algorithm on the
  CPU). A 240 × 240 × 160 mm oven at 1.6 mm (2.25 M cells, 136 RF periods) takes ≈ 24 s on an
  RTX 3070 Ti Laptop.

**Microwave Heating** — Full-Wave EM followed by transient heat conduction in the solid bodies, with
the absorbed power as the heat source (the cavity air is held at the initial temperature). Settings
add heating time (end time), time steps and initial temperature; thermal loads (convection on the
food, for example) apply to the solids. Results: temperature over time, the RF fields above, mean
temperature and coldest/hottest spot per body, temperature spread, heating power and energy. Try
**Examples ▸ Microwave oven heating** (800 W, 30 s, food from 5 °C). *Limits:* evaporation and
temperature-dependent properties are not modelled — above 100 °C SimCraft warns that water would boil
there; no turntable (average a few food positions by hand).

### Fluids

**Fluid Flow (CFD)** — steady incompressible flow through and around parts. Needs an Enclosure
filled with Air, Water or Oil, and something that drives the flow: a **Velocity Inlet** with a
**Pressure Outlet** (usually on opposite enclosure faces) or a **Moving Wall** (a lid, a belt).
Walls are no-slip by default; **Symmetry** makes a face free-slip (use it on the enclosure sides to
mimic open flow or a 2D slice one cell thick). Results: velocity magnitude and X/Y/Z components,
pressure, drag and lift on the solids, flow rate and mean pressure at each inlet/outlet, Reynolds
number. Settings: maximum iterations and **Turbulence**:
- **Laminar** — Reynolds numbers below ~2000 (small gaps, slow flows).
- **LVEL** — an algebraic turbulence model built for electronics enclosures, ducts and heat sinks: the
  turbulent viscosity follows Spalding's law of the wall from the local speed and the distance to the
  nearest wall, and wall shear uses the law of the wall, so coarse meshes still give sensible
  friction and heat transfer (plane-channel friction within ~14 % of Dean's correlation at Re 20 000).

If the flow never settles (vortex shedding behind bluff bodies, plumes) SimCraft warns and shows the
last state. Advection is second-order (central) where cells resolve the flow and switches to upwind
where they do not; the pressure equation is solved with multigrid-preconditioned CG on large meshes.

**Conjugate Heat Transfer (CFD + thermal)** — flow *and* temperature in the fluid **and the solid
parts together**: heat conducts through the parts, crosses into the air or liquid and is carried
away, so you get chip and heat-sink temperatures without guessing a convection coefficient. Typical
electronics-cooling setup:
1. Model the parts (chip, spreader, heat sink, board) and an Enclosure of Air around them — the duct
   or box the air flows through.
2. **Heat Generation** in the parts that dissipate power — give the total in W.
3. Forced air: **Velocity Inlet** with an inflow temperature (fan or duct speed) + **Pressure Outlet**.
   Natural convection: add **Gravity** (buoyancy, Boussinesq) and give the enclosure walls a
   **Temperature** or **Convection** boundary so the heat can leave.
4. Turbulence **LVEL** for fans and ducts; **Laminar** for natural convection and slow flows.

Results: temperature in parts and fluid (parts show their own temperature at interfaces), velocity,
pressure; quantities: max and mean temperature of every part, heat carried away by the flow, heat
through each Temperature/Convection boundary, heat generated and the energy balance (out/in ≈ 100 %
at convergence). Try **Examples ▸ Heat sink in a duct**: a 10 W chip under a six-fin aluminium sink in
a 2 m/s air stream. Settings: reference temperature (default inflow temperature and buoyancy
reference), maximum iterations, turbulence. Thermal loads on faces *between* a part and the fluid
are ignored — that exchange is computed; use them on the enclosure's outer faces.

---

## 8. Loads and supports reference

| Load / support | Applies to | Inputs (units) | Notes |
|---|---|---|---|
| Fixed Support | faces | — | all directions fixed |
| Displacement | faces | x, y, z (mm), each optional | leave a component free for rollers |
| Compression-only | faces | — | rigid frictionless contact, static structural |
| Symmetry | faces | — | zero normal motion (structural) / free-slip (CFD) |
| Force | faces | vector (N) | total force shared over the faces |
| Pressure | faces | MPa | positive pushes into the surface |
| Gravity | whole model | acceleration (m/s²) | default −9.81 Z |
| Temperature | faces | °C | |
| Heat Flux | faces | W/m² | positive flows in |
| Convection | faces | h (W/m²K), ambient (°C) | natural air ≈ 5–25, forced air 25–250, water 500+ |
| Heat Generation | whole model or one body (e.g. a KiCad component `U1`) | total power (W) or W/m³ | total power is spread over the body (exact on any mesh); W/m³ = power ÷ body volume, e.g. 2 W in a 10×10×2 mm chip = 10⁷ W/m³ (corrected to the exact volume for boxes, cylinders and spheres) |
| Voltage | faces | V | electrostatic, current flow |
| Current | faces | A (into the body) | needs a 0 V ground elsewhere |
| Coil | body | current density (A/mm²), axis, around/along | magnetostatic; ampere-turns = J × coil cross-section |
| Magnet | body | remanence (T) vector | magnetostatic |
| Velocity Inlet | faces | m/s into the domain, inflow temperature (°C, conjugate heat transfer) | CFD |
| Moving Wall | faces | wall velocity (m/s), tangential | CFD: lids, belts |
| Pressure Outlet | faces | Pa | CFD, usually 0 |
| RF Port | faces | power (W), E-field direction, TE10 / uniform | full-wave EM, microwave heating; direction must lie in the face |

---

## 9. Reading results

- **RESULTS** tab buttons switch the field (deformation, stresses, safety, plastic strain,
  fatigue safety / life / damage, temperature, heat flux, potential, E-field, current density, Joule heat, absorbed power, SAR, B, H,
  velocity, pressure).
- The **legend** (top-left) shows the 9 colour bands with min/max. For safety factors and fatigue life the
  scale is reversed so that red is always the critical end.
- **Deformation scale**: Auto (≈ 5 % of the model size), True scale, or Undeformed.
- **Animate** — mode shapes, buckled shapes and the deformed shape.
- **Step / time slider** — modes, buckling modes, or transient time points.
- **Cut plane** — slice through the model along X, Y or Z and slide it; essential for fields in air
  and fluid flow.
- **Probe** (MCP `probe`) — values at a point.
- **Browser ▸ Solution** — maxima, reactions, frequencies, quantities (capacitance, resistance,
  drag, peak displacement…), warnings, solve time and which solver/GPU ran it.
- **RESULTS ▸ Report** (or **File ▸ Export report**) — a self-contained HTML report: model, materials,
  loads, mesh, four rendered views, key results, pass/fail checks, study tables and charts. Open it in any
  browser and print to PDF. Agents can set their own checks: `make_report` with
  `checks: [{response: "max:von_mises", op: "<=", value: 150}]` (default check: yield safety factor ≥ 1).
- **File ▸ Export results (VTK)** — open in ParaView for advanced post-processing.
- **File ▸ Save image (PNG)**.
- Changing setup after solving marks results **stale** (⚠ in the Browser and legend).

---

## 10. Optimization studies

The **OPTIMIZE** tab turns single simulations into design answers. It works on copies of your model,
so nothing changes until you press **Apply**.

**Parameters** are numbers in the model, picked from a list with their current values:

| Target | Meaning |
|---|---|
| `Bracket.size.z`, `Bracket.min.x` | box size / corner position |
| `Hole.radius`, `Hole.height`, `Hole.base.x` | cylinder dimensions / position |
| `Ball.radius`, `Ball.center.y` | sphere |
| `Part.scale`, `Part.offset.z` | imported STL |
| `load2.vector.z`, `load3.value_mpa`, `load4.value` | load magnitudes (numbered as in the Browser) |
| `element_size` | mesh size (for convergence studies) |

**Responses** are results: `mass`, `volume`, `max:<field>` / `min:<field>` (e.g. `max:von_mises`,
`min:safety_factor`, `max:temperature`), `frequency:1`, `load_multiplier:1`, `reaction:<load id>` (or
`reaction:1:z`), `quantity:<name>` (e.g. `quantity:Capacitance`). The form suggests the ones that fit
the current analysis.

### Parameter study
Choose parameters with a range and a number of steps, tick the responses, **Run study**. Every
combination is solved (up to 400 designs) and shown in a table; **Apply** any row to adopt it.
Agents can also request Latin-hypercube sampling (`samples`).

### Optimize
Pick parameters and ranges, an **objective** (minimise or maximise a response) and any number of
**constraints** (`≤` / `≥`), e.g. *minimise mass with max:von_mises ≤ 150 and min:safety_factor ≥ 2*.
SimCraft samples the design space (Latin hypercube), then refines the best design with a bounded
Nelder–Mead search; 20–40 evaluations is typical for 1–3 parameters. The results table marks the best
feasible design ★ and designs that violate a constraint. Because the mesh is made of cubes,
dimensions effectively change in steps of one element: use a finer element size for finer answers.

### Topology
For a **Static Structural** setup, **Topology** finds the stiffest layout using only a chosen
percentage of the material (SIMP method). Regions under supports and loads stay solid. Afterwards the
viewport shows the **Topology Density** field (red = keep, blue = remove; use the cut plane), and the
shape is exported as an STL (next to your project, or in `~/SimCraft/`). **Open optimized shape**
starts a new model from it so you can re-apply loads and verify the stresses, or import the STL into
your CAD as a starting point.

MCP tools: `list_parameter_targets`, `run_parameter_study`, `optimize_design`, `topology_optimize`,
`get_study`, `apply_design`.

---

## 11. CPU and GPU compute

**SOLVE ▸ Settings ▸ Compute**:

| Mode | Behaviour |
|---|---|
| **Auto** (default) | Small and medium models use the CPU direct solver (fast and exact). Large models that need the iterative solver (≥ 50 000 unknowns) run on the GPU. |
| **CPU** | Never use the GPU. |
| **GPU** | Every linear solve on the GPU. |

The GPU solver runs double-precision conjugate gradients as Vulkan compute shaders and gives the same
answers as the CPU. Measured on an RTX 3070 Ti Laptop vs Ryzen 9 6900HS:

| Model | Linear solve CPU | GPU | Speed-up |
|---|---|---|---|
| Structural plate, 1.06 M unknowns | 56.2 s | 9.4 s | 6.0× |
| Thermal plate, 275 k unknowns | 2.2 s | 0.3 s | 7.4× |

Notes:

- The GPU is only touched while a solve uses it — never when the app starts — and released afterwards.
- The app window itself always draws on the integrated GPU (it is light work); on hybrid laptops this
  avoids waking the discrete GPU, which can disturb Wi-Fi/Bluetooth combo cards.
- If the GPU is missing, busy or out of memory (for example a local AI model is loaded in its
  memory), SimCraft solves on the CPU and says so in the results' warnings.
- Meshing, assembly and stress recovery always run on the CPU (all cores).
- Full-wave EM (FDTD) uses the GPU in single precision for grids ≥ 150 000 cells in Auto mode;
  GPU and CPU give the same answer (validated), the GPU ≈ 6× faster.

---

## 12. The Simulation Planner (local AI)

Open **✨ Planner**, describe what you want to simulate — *"simulate EMI on my PCB"*, *"how hot does
food get in a microwave"*, *"will this bracket bend under 2 kN"* — and press **Draft to-do list**.

A language model running **on your own computer** (through [Ollama](https://ollama.com)) returns a
checklist:

- each step is **ready** (SimCraft can do it now), **workaround** (an honest approximation inside
  SimCraft) or **missing in SimCraft**;
- the analysis to start with (with a **Set up this analysis** button);
- the inputs you need to collect (dimensions, CAD, material data, operating conditions);
- the capabilities SimCraft would need to do the job properly.

Tick steps off as you go; the list is saved with the project and visible to AI agents (`get_plan`).

**Setup**

```bash
# install Ollama from https://ollama.com, then download a model, e.g.
ollama pull qwen2.5:7b        # good structured answers, ~4.7 GB
```

SimCraft picks the best installed model automatically; choose another in the Planner's Model list.
The planner is grounded in SimCraft's real capabilities, and a built-in check flags physics SimCraft
cannot do (radiated EMI, eddy currents, phase change, compressible flow, …) even if the model
misses it. Small local models still make mistakes — treat the list as a draft. The model runs on
your GPU if it has one, so it competes with GPU solves for graphics memory while it stays loaded.

---

## 13. Driving SimCraft from AI agents (MCP)

**Connect AI** in the header shows ready-to-copy setup:

```bash
claude mcp add --scope user simcraft -- /path/to/simcraft/target/release/simcraft mcp
```

or for Antigravity / Cursor / other MCP clients:

```json
{ "mcpServers": { "simcraft": { "command": "/path/to/simcraft/target/release/simcraft", "args": ["mcp"] } } }
```

When the SimCraft window is open, the agent drives it live (you watch it build and solve); otherwise
it runs headless, and `launch_gui` opens the window with the current model. Agents can see results
with `render_image` (contour PNG) and `screenshot_gui` (the whole window), read your face selection
with `get_selection`, and use the planner (`plan_simulation`, `get_plan`, `set_plan_step`).

**Security** — the window and `simcraft serve` only accept connections that present the token in
`~/.config/simcraft/token` (created automatically, readable only by you), so other programs on the machine
cannot drive SimCraft.

### Python scripting (`pysimcraft`)

```bash
pip install ./python
```

```python
import pysimcraft
with pysimcraft.connect() as sim:      # the open window, or a headless `simcraft serve` it starts
    sim.new_model()
    sim.add_box(name="Beam", size=[100, 10, 8])
    sim.add_fixed_support(faces=["Beam.xmin"])
    sim.add_force(faces=["Beam.xmax"], vector=[0, 0, -100])
    print(sim.solve()["fields"]["von_mises"]["max"])
    sim.make_report(path="beam.html")
```

Every tool is a method (`help(sim.add_force)` lists its arguments); errors raise `SimCraftError`. See
`python/examples/`. `simcraft schema` prints all tool definitions.

Example prompts:

- *"In SimCraft, model a 150 × 30 × 8 mm aluminium bracket with two 6 mm holes, fix the left end, load
  the right end with 800 N down, solve and tell me if it is safe."*
- *"Find the first three natural frequencies of this STL part (/home/me/part.stl) in steel, fixed on
  its bottom face."*
- *"Make a parallel-plate capacitor with FR-4 between 40 mm copper plates 4 mm apart and report the
  capacitance."*

---

## 14. Units cheat sheet

| Quantity | Unit |
|---|---|
| Length, displacement | mm |
| Force | N |
| Stress, pressure (structural) | MPa |
| Temperature | °C |
| Heat flux | W/m² |
| Film coefficient | W/(m²·K) |
| Heat generation, Joule heat | W/m³ |
| Voltage / current | V / A |
| Current density | A/mm² |
| Magnetic flux density | T |
| Fluid velocity / pressure | m/s / Pa |
| Material properties | SI (GPa for E, MPa for strengths, kg/m³, W/m·K, S/m) |

---

## 15. Accuracy and validation

`cargo test --release` checks every solver against textbook answers (cantilever deflection, stress
and frequency, Euler buckling, dynamic amplification, harmonic resonance amplification, response
spectrum vs. static 1 g, random-vibration RMS (Miles), Basquin/Goodman fatigue life, Steinberg damage,
WR-340 guide wavelength, closed-cavity power balance, microwave-heating energy balance, plastic yielding, conduction, transient heating,
thermal expansion, parallel-plate capacitance, resistance and Joule heating, ring-coil field,
laminar channel flow, lid-driven cavity (Ghia), natural convection (de Vahl Davis), LVEL turbulent
channel friction (Dean), conjugate-heat energy balance, contact reactions) and the GPU solver against the CPU. See the README for the
numbers.

---

## 16. Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| "under-constrained" / "matrix is singular" | The part can move freely: add supports (all 3 directions overall, and enough to stop rotation). |
| "face selector … matched nothing" | The face name doesn't exist in the current mesh; hover faces or use `list_faces`. Enclosure faces only exist in analyses that use the enclosure. |
| "needs an enclosure" | Add **GEOMETRY ▸ Enclosure** (electrostatic, magnetostatic, CFD). |
| "thermal model needs at least one Temperature or Convection" | Give the heat a way out. |
| CFD "did not reach a steady state" | Lower the velocity, enlarge the domain, or accept a snapshot of unsteady flow. |
| Nonlinear step "did not converge" | The structure is collapsing at that load; reduce the load or add support. |
| Results look blocky | Use a smaller element size. |
| Solve is slow | Use a coarser mesh first; try Compute ▸ GPU for very large models. |
| GPU warning in results | GPU unavailable or out of memory; the CPU was used instead. |
| Planner "cannot reach the local model server" | Install/start Ollama, or set `SIMCRAFT_LLM_URL`. |
| Planner "no local models" | `ollama pull qwen2.5:7b`. |
| MCP shows "bridge offline" | Another SimCraft window already uses the port; close it or set `SIMCRAFT_PORT`. |
| Dock shows an old icon | Re-run `scripts/install-desktop.sh`, or log out and back in. |

---

## 17. Limitations

- Voxel meshes stair-step curved surfaces (see §6).
- Magnetostatics uses nodal elements: very high permeability contrasts are approximate.
- CFD is steady and incompressible; turbulence is the algebraic LVEL model (no k-ε / k-ω), and
  there is no thermal radiation between surfaces.
- KiCad boards: copper is homogenised per cell for heat conduction only — trace resistance, current
  density in traces, warpage from copper/FR-4 expansion mismatch and component 3D models are not
  imported (components are boxes).
- Full-wave EM: one frequency at a time, near fields only (no far-field / radiated-emission
  transform, antenna gain or S-parameters yet), no sub-cell thin wires or PCB traces, μr = 1.
- Not included: eddy currents and induction heating, contact between two deformable bodies,
  radiation heat transfer, phase change,
  temperature-dependent properties, strain-life (ε-N) and weld fatigue, acoustics, moving parts.
- Vibration analyses use mode superposition with proportional (modal) damping and the lumped mass
  matrix; very high modes are approximate.
  The Planner flags these when your goal needs them.
