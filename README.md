# SimCraft

A finite-element simulation app written in Rust, with a simple Fusion-style UI and an MCP
server so AI agents (Claude Code, Antigravity, Cursor, …) can build models, run
simulations and look at the results.

```bash
cargo build --release
./target/release/simcraft            # desktop app
./target/release/simcraft mcp        # MCP server (stdio)
```

📖 **[User Guide](docs/USER_GUIDE.md)** · 🗺 **[Roadmap](docs/ROADMAP.md)** — installation, every analysis, loads, results, GPU compute,
the local-AI Planner and MCP setup.

## Highlights
- 19 analyses plus fatigue across structural, thermal, electromagnetic and fluid physics, validated against theory.
- **GPU acceleration**: large iterative solves run as double-precision Vulkan compute on the GPU
  (6–7× faster linear solves on an RTX 3070 Ti), with automatic CPU fallback.
- **Optimization studies**: parameter sweeps, constrained design optimization (minimise mass subject to
  stress/safety limits) and SIMP topology optimization with STL export.
- **✨ Simulation Planner**: describe a goal ("simulate EMI on my PCB") and a local model (Ollama)
  drafts a SimCraft to-do list, flagging steps SimCraft can do, workarounds and missing physics.
- **MCP server**: Claude Code, Antigravity and other agents can build, solve and inspect models live
  (token-protected local bridge).
- **Python**: `pysimcraft` (stdlib only) and `simcraft serve` for headless batch scripting.
- **HTML reports** with rendered views, pass/fail checks, study tables and charts.

## PCB thermal (KiCad)
`import_kicad` / **GEOMETRY ▸ KiCad PCB** reads a `.kicad_pcb` (KiCad 5–10): outline and cut-outs, stack-up,
tracks, pads, filled zones and vias become a board body with homogenised **orthotropic** copper
conductivity per cell; footprints become component boxes named by reference with optional power in W.
Solve it with thermal analyses (convection) or conjugate heat transfer (real airflow).

## Analyses

| Group | Analysis | Results |
|---|---|---|
| Structural | **Static Structural** (+ compression-only contact, symmetry) | deformation, von Mises, principal stresses, safety factor, reactions |
| | **Modal** | natural frequencies + mode shapes |
| | **Linear Buckling** | critical load multipliers + buckled shapes |
| | **Transient Structural** | time history, peak displacement, animated response |
| | **Nonlinear Structural** (von Mises plasticity, bilinear hardening) | plastic strain, load–displacement curve |
| | **Harmonic Response** (mode superposition) | frequency response, amplitude and stress at the peak |
| | **Response Spectrum** (SRSS / CQC) | peak deformation and stress, effective mass |
| | **Random Vibration** (acceleration PSD) | 1σ / 3σ deformation and stress, Grms, zero-crossing rate |
| | **Fatigue** (post-processing: Basquin S-N, Goodman/Gerber/Soderberg, Steinberg) | life, damage, fatigue safety factor |
| Thermal | **Steady-State Thermal**, **Transient Thermal** | temperature, heat flux, heat flow |
| | **Thermal-Stress** | thermal expansion stresses |
| Electromagnetic | **Electrostatic** | potential, E-field, electrode charge, capacitance |
| | **DC Current Flow** | potential, current density, Joule heat, resistance |
| | **Thermal-Electric** (Joule heating) | all of the above + temperature |
| | **Magnetostatic** (coils, permanent magnets, iron) | B and H fields, stored energy |
| | **Full-Wave EM (FDTD, GPU)** — metal cavity or open boundary, RF ports | E field, absorbed power density, SAR, power per body |
| | **Microwave Heating** (full-wave EM → transient heat) | temperature history, hot/cold spots, absorbed power |
| Fluids | **Fluid Flow (CFD)** — steady incompressible, laminar or LVEL turbulence, moving walls | velocity (+ components), pressure, drag/lift, flow rates, Reynolds number |
| | **Conjugate Heat Transfer** — flow + temperature in fluid and parts, forced or natural convection | part temperatures, heat carried by the flow, energy balance |

Field and flow analyses use an **enclosure** (GEOMETRY ▸ Enclosure / `add_enclosure`): an air, water or
oil box around the parts. Results inside it are shown on a **cut plane** (RESULTS ▸ Cut plane).

Loads and supports: fixed support, displacement (rollers), compression-only support, symmetry, force,
pressure, gravity, temperature, heat flux, convection, heat generation, voltage, current, coil current
density, permanent magnet, velocity inlet, pressure outlet.

### Numerics
- **Elements**: 8-node hexahedra with incompatible modes (no shear locking); nodal recovery of stresses.
- **Linear solvers**: direct sparse Cholesky (RCM + skyline) when the estimated work is small, otherwise
  Jacobi-preconditioned conjugate gradients on the GPU (Vulkan compute, f64, ELLPACK) or CPU.
  SOLVE ▸ Compute: Auto / CPU / GPU.
- **Modal / buckling**: shifted subspace iteration with Rayleigh–Ritz.
- **Transient**: implicit Euler (thermal), Newmark average acceleration + Rayleigh damping (structural).
- **Plasticity**: J2 radial return, Newton–Raphson with the consistent tangent, load stepping.
- **Contact**: active-set iteration on compression-only supports.
- **Magnetostatics**: nodal vector potential, Coulomb-gauged curl-curl + div-div form.
- **CFD**: MAC staggered grid, Chorin projection to steady state, factored pressure Poisson.
- **Mesher**: voxel hex mesh of a CSG model (add/subtract boxes, cylinders, spheres, closed STL files).

### Validation (`cargo test --release`)
| Case | SimCraft | Theory |
|---|---|---|
| Cantilever tip deflection | 0.1996 mm | 0.200 mm |
| Cantilever root bending stress | 61.1 MPa | 60 MPa |
| First bending frequency | 813.8 Hz | 815 Hz |
| Euler column buckling load | ×10.290 | ×10.281 |
| Step load dynamic amplification | 0.391 mm | 0.399 mm (2 × static) |
| Plastic bar force past yield | 25 544 N | 25 544 N |
| Heat flow through copper bar | 39.10 W | 39.1 W |
| Uniform transient heating | 29.352 K | 29.352 K |
| Free thermal expansion | 0.0600 mm, ~0 stress | 0.06 mm, 0 stress |
| Parallel-plate capacitance | 14.875 pF | 14.875 pF |
| Copper bar resistance / Joule power | 17.241 µΩ / 58.0 mW | 17.241 µΩ / 58.0 mW |
| Ring-coil centre field | 4.92 mT | 5.03 mT |
| Laminar channel peak velocity | 1.47 mm/s | 1.50 mm/s |
| Channel pressure drop (incl. entrance) | 7.8 mPa | 7.2 mPa fully developed |
| Compression-only reaction / lift-off | 1000 N / detected | 1000 N |
| Harmonic resonance amplification (ζ = 2 %) | 24.28 × static | 24.27 × |
| Response spectrum, flat 1 g, tip / static 1 g | 1.0147 | 1.0134 (mode 1) |
| Random vibration 1σ tip displacement | 0.02244 mm | 0.02245 mm (Miles) |
| Basquin fatigue life at 100 MPa | 1.743 × 10⁹ cycles | 1.743 × 10⁹ |
| Steinberg 3-band damage | 1.3194 × 10⁻¹² | 1.3194 × 10⁻¹² |
| WR-340 guide wavelength (λg/2 from standing-wave nulls) | 86.99 mm | 87.06 mm |
| Closed-cavity power balance (absorbed / delivered) | 100.45 % | 100 % |
| Microwave heating: mean temperature rise | 2.32526 K | 2.32525 K (P·t / m·c) |
| FDTD GPU vs CPU (absorbed power) | identical to 6 digits | — |
| Lid-driven cavity, Re 100 (16 profile points) | max deviation 0.0074 U | Ghia et al. 1982 |
| Natural convection, Ra 10⁴ / 10⁵: Nusselt | 2.257 / 4.556 | 2.243 / 4.519 (de Vahl Davis) |
| Turbulent channel (LVEL), Re 20 000: dp/dx | 94.6 Pa/m | 83.2 Pa/m (Dean) |
| Conjugate heat balance, heated block in a duct | 0.200 W carried | 0.200 W generated |
| Plate with hole (d/W = 0.1) Kt, h = r/10 | fitted 3.08 / voxel 3.31 | 3.03 (Heywood) |
| Lamé thick cylinder, bore hoop stress | fitted 16.95 / voxel 21.05 MPa | 16.67 MPa |
| Surrogate what-if (6 designs), beam deflection at unseen thickness | 0.65 % from the real solve | — |
| Bayesian optimisation, lightest beam with σ ≤ 150 MPa | 10 solves, σ = 149.1 MPa | direct search: 24 evaluations |
| KiCad board, copper plane in-plane conduction | 0.034817 W | 0.034861 W (Σkt) |
| KiCad board, through-plane with 40 thermal vias | 3.0395 W | 3.0333 W (series ∥ via copper) |

### Limitations
- Static Structural fits curved and off-grid surfaces (pillow layer + node projection, ~2 % peak
  stress); other analyses use stair-stepped voxels — global results stay accurate, peaks on curved
  surfaces read high.
- Magnetostatics uses nodal elements: very high permeability contrasts (iron cores) are approximate.
- CFD is steady and incompressible with laminar or LVEL turbulence (the solver reports when the flow
  does not settle, e.g. vortex shedding); no thermal radiation.
- Full-wave EM is single-frequency with near fields only (no far-field / EMI transform yet).
- Not included: contact between two
  deformable bodies (only contact against a rigid surface).

## Desktop app
Ribbon tabs **GEOMETRY · MATERIAL · SETUP · MESH · SOLVE · RESULTS**, a model browser, a
properties panel and a 3D viewport. Light/dark theme via the ☀/🌙 button (remembered between runs).

Linux desktop launcher + icon: `scripts/install-desktop.sh`.

- Left-drag orbit · right/middle-drag pan · wheel zoom · click a face to select (Shift adds).
- Try **Examples ▸** bracket, heat sink, tuning fork, column buckling, capacitor, bus-bar Joule heating,
  coil with iron core, or flow past a cylinder — then **SOLVE ▸ Solve**.
- **File** menu: open/save `.simcraft` projects, import STL, export VTK (ParaView), save PNG.

Units: **mm, N, MPa, °C** (heat flux W/m², film coefficient W/m²K, generation W/m³).

## MCP (control from Claude Code / Antigravity)

**Claude Code**
```bash
claude mcp add simcraft -- /path/to/simcraft/target/release/simcraft mcp
```
(or copy `.mcp.json.example` to `.mcp.json` with your path to register it for this project.)

**Antigravity / Cursor / other clients**: add this to the client's MCP config
(Antigravity: *Agent panel ▸ … ▸ MCP Servers ▸ Manage ▸ View raw config*):
```json
{ "mcpServers": { "simcraft": { "command": "/path/to/simcraft/target/release/simcraft", "args": ["mcp"] } } }
```

How it connects: when the SimCraft window is open, every MCP tool call is forwarded to it over
`127.0.0.1:47800` (change with `SIMCRAFT_PORT`), so you watch the model being built live and can click
faces that the agent reads with `get_selection`. When no window is open the MCP runs headless;
`launch_gui` opens the window and carries the current model over.

Tools: `get_model`, `new_model`, `add_box`, `add_cylinder`, `add_sphere`, `import_stl`, `add_enclosure`,
`remove_body`, `list_materials`, `define_material`, `set_material`, `set_analysis`, `list_faces`,
`add_fixed_support`, `add_displacement`, `add_compression_only_support`, `add_symmetry`, `add_force`,
`add_pressure`, `add_gravity`, `add_temperature`, `add_heat_flux`, `add_convection`, `add_heat_generation`,
`add_voltage`, `add_current`, `add_coil`, `add_magnet`, `add_inlet`, `add_outlet`, `add_moving_wall`, `remove_load`,
`import_kicad`, `add_port`, `clear_loads`, `set_mesh`, `solve`, `fatigue_life`, `make_report`, `get_results`, `probe`, `set_display` (incl. cut plane, dark mode),
`render_image` (PNG), `screenshot_gui` (PNG of the window), `export_vtk`, `save_project`, `open_project`,
`get_selection`, `get_status`, `launch_gui`, `plan_simulation`, `get_plan`, `set_plan_step`,
`list_local_models`, `list_parameter_targets`, `run_parameter_study`, `optimize_design`, `surrogate_optimize`, `fit_surrogate`, `predict_surrogate`,
`topology_optimize`, `get_study`, `apply_design`.

Example prompt: *"In SimCraft, make a 150×30×8 mm aluminium bracket with two 6 mm holes, fix the left
end, put 800 N down on the right end, solve, and show me the von Mises stress. Is it safe?"*

## Layout
```
src/geometry.rs  CSG primitives, STL loading + face segmentation
src/mesh.rs      voxel hex mesher, named CAD faces
src/fem.rs       element matrices, structural / thermal / modal solvers
src/sparse.rs    sparse matrices, direct Cholesky, PCG, dense helpers
src/session.rs   model state + the tool API shared by GUI and MCP
src/render.rs    software renderer (viewport + PNG export)
src/gui.rs       egui desktop app
src/bridge.rs    GUI ⇄ MCP local socket
src/mcp.rs       MCP stdio server
```

## License
MIT — see [LICENSE](LICENSE).
