# SimCraft

A finite-element simulation app written in Rust, with a simple Fusion-style UI and an MCP
server so AI agents (Claude Code, Antigravity, Cursor, …) can build models, run
simulations and look at the results.

```bash
cargo build --release
./target/release/simcraft            # desktop app
./target/release/simcraft mcp        # MCP server (stdio)
```

## Analyses

| Analysis | What it solves | Results |
|---|---|---|
| **Static Structural** | linear elasticity, K·u = f | deformation (total/X/Y/Z), von Mises, max/min principal stress, safety factor, reaction forces |
| **Steady-State Thermal** | heat conduction, K·T = q | temperature, heat flux, heat flow through fixed-temperature faces |
| **Modal** | natural frequencies, K·φ = ω²M·φ | up to 20 frequencies + mode shapes (animated) |
| **Thermal-Stress** | thermal solve → structural solve with thermal-expansion loads | everything from both |

Loads and supports: fixed support, prescribed displacement (rollers), force, pressure, gravity,
temperature, heat flux, convection, internal heat generation.

### Numerics
- **Elements**: 8-node hexahedra with Wilson–Taylor incompatible modes (bending without shear locking),
  stresses recovered at nodes and averaged. 8-node conduction hexahedra for thermal.
- **Linear solvers**: direct sparse Cholesky (Reverse Cuthill–McKee ordering + skyline factorisation),
  used automatically when it fits in memory; Jacobi-preconditioned conjugate gradients for larger models
  (force it with `SIMCRAFT_ITERATIVE=1`).
- **Eigen solver**: shifted subspace iteration with Rayleigh–Ritz (one factorisation, many back-solves).
- **Mesher**: voxel hex mesh of a CSG model (add/subtract boxes, cylinders, spheres, closed STL files).

### Validation (`cargo test --release`)
| Case | SimCraft | Theory |
|---|---|---|
| Cantilever tip deflection (100×10×10 steel, 100 N) | 0.1996 mm | 0.200 mm |
| Cantilever root bending stress | 61.1 MPa | 60 MPa |
| First bending frequency | 813.8 Hz | 815 Hz |
| Heat flow through copper bar | 39.10 W | 39.1 W |
| Free thermal expansion (ΔT = 100 K) | 0.0600 mm, ~0 stress | 0.06 mm, 0 stress |

**Known limitation:** curved surfaces are stair-stepped by the voxel mesh. Global results (deflection,
frequencies, temperatures) stay accurate, but peak stresses *on curved surfaces* read high (about
+15% on a round rod at 1 mm elements). Use a smaller element size where curved-surface stress matters.

## Desktop app
Ribbon tabs **GEOMETRY · MATERIAL · SETUP · MESH · SOLVE · RESULTS**, a model browser, a
properties panel and a 3D viewport. Light/dark theme via the ☀/🌙 button (remembered between runs).

- Left-drag orbit · right/middle-drag pan · wheel zoom · click a face to select (Shift adds).
- Try **Examples ▸ Cantilever bracket / Heat sink / Tuning fork**, then **SOLVE ▸ Solve**.
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

Tools: `get_model`, `new_model`, `add_box`, `add_cylinder`, `add_sphere`, `import_stl`,
`remove_body`, `list_materials`, `define_material`, `set_material`, `set_analysis`, `list_faces`,
`add_fixed_support`, `add_displacement`, `add_force`, `add_pressure`, `add_gravity`,
`add_temperature`, `add_heat_flux`, `add_convection`, `add_heat_generation`, `remove_load`,
`clear_loads`, `set_mesh`, `solve`, `get_results`, `probe`, `set_display`, `render_image` (PNG),
`screenshot_gui` (PNG of the window), `export_vtk`, `save_project`, `open_project`,
`get_selection`, `get_status`, `launch_gui`.

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
