# SimCraft Engineering Audit & Action Plan (`todo.md`)

**Date:** 2026-11-25  
**Scope:** Comprehensive audit of SimCraft (`a1727d5` on `p2-1-radiation`) across **Security**, **UX**, and **Architecture** — evaluating both:
1. **Web App** (usable in a browser on laptop/desktop)
2. **Native Laptop/Desktop App** (evolving the current Rust desktop app)

---

## 0. Context & Current Architecture Snapshot

SimCraft is a pure-Rust engineering simulation suite (~24k LOC across `src/`) featuring 19 physics analyses + fatigue, CSG/STL/KiCad PCB import, GPU-accelerated solvers, parameter/Bayesian/topology optimization, HTML dynamic reporting, an Ollama-backed simulation planner, Python scripting (`pysimcraft`), and an MCP server.

How the codebase is wired today:
- **Shared Core & Command API (`src/session.rs`):** `Session` holds the model, mesh, results, display state, undo/redo history, and study/plan state. `Core::call(tool, args)` is the single entry point shared by the GUI, the MCP server (`src/mcp.rs`), and the local TCP bridge (`src/bridge.rs`, `127.0.0.1:47800`).
- **UI & Viewport (`src/gui.rs`, `src/render.rs`):** A 4,421-line `eframe`/`egui` (`glow`) desktop app. The 3D viewport is **software-rasterized on the CPU** using `rayon` (`render::rasterize`) and uploaded each changed frame as an RGBA texture (`tex.set`), rather than drawn via hardware OpenGL/WebGPU shaders.
- **Compute & GPU (`src/gpu.rs`, `src/fdtd.rs`, `src/elastic_mg.rs`, `src/cfd.rs`):**
  - `src/main.rs::prefer_integrated_gpu()` pins the GUI window to Mesa/iGPU so the UI does not keep a hybrid laptop's discrete GPU awake.
  - `src/gpu.rs` acquires a Vulkan discrete GPU on demand (`wgpu::Backends::VULKAN`, requiring `wgpu::Features::SHADER_F64` for `f64` ELLPACK PCG) and releases it when finished.
  - `src/fdtd.rs` runs an `f32` WGSL compute pipeline, though it currently selects its adapter via `gpu::pick_adapter()`.
- **External I/O:** Local filesystem reads/writes (`std::fs`, `rfd`), local HTTP to Ollama (`src/assistant.rs` via `ureq`), and stdio/TCP IPC (`mcp.rs`, `bridge.rs`).

---

## 1. Security Audit

### Why This Matters
Today SimCraft runs as a single-user local desktop binary where `docs/ROADMAP.md` (`§0`) documents file paths as a trust boundary and protects the local bridge with a random token. That leaves **real risks even locally** (malicious `.simcraft` / `.stl` / `.kicad_pcb` files, untrusted MCP prompts / prompt injection, multi-user `/tmp` collisions, parser panics/OOM), and **critical blockers** if any part of the engine is ever exposed to untrusted users or hosted as a web backend.

### Checklist

#### High Priority
- [x] **Enforce path sandboxing / trust boundaries on all file-taking tools (`src/session.rs`)**
  - **Context:** `open_project`, `save_project`, `import_stl`, `import_kicad`, `pcb_nets`, `pcb_dc`, `export_vtk`, `make_report`, and `topology_optimize` (`export_path`) accept arbitrary file paths with no canonicalization, root directory jail, extension allowlist, or user confirmation.
  - **Indirect Attack Vector:** Opening an untrusted `.simcraft` JSON file (`Session::open`, `src/session.rs:295-306`) automatically calls `load_stl(path, ...)` and `PcbData::load(path)` on whatever external `path` strings are embedded in the JSON without prompting the user.
  - **Action:**
    - Canonicalize paths and restrict MCP/headless calls to an explicit workspace root (`--workspace` / `SIMCRAFT_WORKSPACE`) when configured.
    - Prompt the user in the GUI (or require explicit approval) before `Session::open` loads external STL/PCB paths outside the `.simcraft` file's directory, or optionally bundle/hash-verify external geometry inside `.simcraft` archives.
    - Restrict output extensions (`.simcraft`, `.vtk`, `.html`, `.stl`) and refuse overwriting existing files from non-interactive callers unless explicitly permitted.

#### Medium Priority — IPC, Bridge & Temp Files
- [x] **Replace weak PRNG fallback and harden token handling (`src/bridge.rs`)**
  - **Context:** `random_token()` (`src/bridge.rs:40-53`) falls back to an FNV-1a hash of `SystemTime::now()`, `process::id()`, and `&bytes` if `/dev/urandom` cannot be opened (always the case on Windows, and predictable locally). `token_path()` (`src/bridge.rs:29-38`) falls back to `std::env::temp_dir()` if `XDG_CONFIG_HOME` and `HOME` are unset, and `OpenOptionsExt::mode(0o600)` only sets permissions when creating a new file (an existing permissive file keeps its mode and is accepted as long as `t.len() >= 32`).
  - **Action:** Use `getrandom` for cryptographic tokens; verify/enforce `0o600` permissions on existing token files (refuse world/group-readable tokens); refuse falling back to shared `temp_dir()` for secrets.
- [x] **Bound TCP bridge threads, timeouts, and client reads (`src/bridge.rs`, `python/pysimcraft/__init__.py`)**
  - **Context:** `bridge::serve` (`src/bridge.rs:92-96`) spawns an unbounded `std::thread::spawn` per incoming TCP connection with no socket read timeout (`read_limited` blocks forever on slow/stalled clients) and no auth rate-limiting. Conversely, `bridge::connect()` (`src/bridge.rs:167`) and `pysimcraft` (`__init__.py:114`) use unbounded `read_line`/`readline`.
  - **Action:** Add a connection cap (or bounded thread pool), short auth read timeout (e.g., 2 s), idle read timeout, rate-limit failed auth attempts, cap response line sizes on clients, and prefer Unix domain sockets (`0o600` socket file) / Windows named pipes over TCP `127.0.0.1` for local IPC.
- [x] **Eliminate predictable `/tmp` handoff file in `launch_gui` (`src/mcp.rs`)**
  - **Context:** `src/mcp.rs:129-135` saves the handoff model to `temp_dir()/simcraft-handoff-{pid}.simcraft` with default umask and predictable PID filename (vulnerable to pre-created symlink / multi-user snooping).
  - **Action:** Create a private temp file via `tempfile::NamedTempFile` (mode `0o600`) or pass the model over an authenticated socket/stdin pipe, and delete the file after the GUI loads it.

#### Medium Priority — Parser, Mesh & Solver DoS / Crash Hardening
- [x] **Guard STL loader against oversized files, 32-bit overflows, `NaN`/`Inf`, and `O(T)` algorithmic DoS (`src/geometry.rs`)** _(file cap, `checked_mul`, finite checks done; STL BVH tracked in `P2.A1`)_
  - **Context:** `load_stl` (`src/geometry.rs:457-496`) reads the entire file into RAM with `std::fs::read`; binary STL validation (`84 + n * 50 == bytes.len()`, line 462) can overflow `usize` on 32-bit/WASM targets; `Shape::validate()` (`lines 178-207`) checks `<= 0.0` so `NaN` and `+Inf` pass; and `Shape::Stl::{project, best_patch}` / `StlData::x_crossings` scan all triangles linearly (`O(T)` per node/face/ray).
  - **Action:** Cap input file size, use `checked_mul` for triangle buffer checks, reject any non-finite (`!is_finite()`) coordinates/dimensions/parameters across `geometry.rs` and `session.rs`, and build a BVH over STL triangles (matching `radiation.rs`) for `O(log T)` queries.
- [x] **Guard KiCad S-expression parser & PCB rasterizer against stack overflow and OOM (`src/kicad.rs`, `src/kicad/dc.rs`)**
  - **Context:** `kicad::parse()` (`src/kicad.rs:86-135`) builds a nested `enum Sx` tree iteratively, but dropping `Sx::List(Vec<Sx>)` is recursive — deeply nested `((((...))))` input can overflow the call stack and abort. `gr_arc`/`fp_arc` KiCad 5 branch (`src/kicad.rs:236`) computes `((ang.abs() / 5.0).ceil() as usize).max(2)` without an upper clamp (`ang = 1e12` triggers OOM). `PcbData::from_text` (`src/kicad.rs:656-660`) computes `res` from `area`, so extreme aspect ratios (`size[0] >> size[1]`) produce tiny `res` and massive `nx * ny` causing OOM in `vec![0u8; nx * ny]`.
  - **Action:** Enforce a max file size, max S-expression nesting depth (e.g., 256) and iterative `Drop` for `Sx`, clamp KiCad 5 arc subdivisions (`4..=180` like `arc_points`), and enforce a `MAX_PIXELS` cap on `nx * ny` in `PcbData::from_text` just like `kicad/dc.rs:17`.
- [x] **Fix unchecked integer multiplication and `Inf` element size in mesher (`src/mesh.rs`)**
  - **Context:** `build_mesh_ex` (`src/mesh.rs:205, 216-219`) checks `if !(h > 0.0)` (allowing `h = f64::INFINITY`) and computes `dims[0] * dims[1] * dims[2]` without `checked_mul` before comparing to `max_cells * 4` (an integer wrap bypasses the guard).
  - **Action:** Check `h.is_finite() && h > 0.0`, validate finite bounding boxes, and use `dims[0].checked_mul(dims[1]).and_then(|v| v.checked_mul(dims[2]))`.
- [x] **Fix objective-`None` panic in constrained optimization (`src/study.rs`)**
  - **Context:** `study::optimize` (`src/study.rs:460`) sets `run.feasible` using `violation(&constraints, &run.responses, 1)` without requiring `run.responses[0].is_some()`, so line 532 (`responses[0].unwrap()`) panics if constraints pass when the objective failed.
  - **Action:** Require `run.responses[0].is_some_and(|v| v.is_finite())` for feasibility.

#### Low Priority / Multi-Tenant & Web Prerequisites
- [x] **Tighten HTML escaping in benchmark reports (`src/bench.rs`)**
  - **Context:** `bench::esc` (`src/bench.rs:1046-1048`) skips `"` and `'` (while `plot()` uses single-quoted SVG attributes), and `bench::html` (`lines 1182-1185`) interpolates `simcraft_version`, `commit`, `date`, `mode` raw.
  - **Action:** Escape `& < > " '` uniformly in both `report.rs` and `bench.rs`, and escape all fields derived from external `--compare` JSON or environment commands.
- [x] **Harden LLM client & eliminate global GPU state for server/multi-user use (`src/assistant.rs`, `src/gpu.rs`)**
  - **Context:** `assistant::list_models()` and `plan()` (`src/assistant.rs:54, 234`) use `SIMCRAFT_LLM_URL` with plaintext HTTP and no response size limit (SSRF risk if user-configurable on a web backend). `gpu.rs` stores `MODE`, `LAST_DEVICE`, and `FALLBACK` in process-global `static`s (`src/gpu.rs:26-36`), which race across concurrent sessions on a shared server.
  - **Action:** Move `ComputeMode`, device info, and fallback warnings into per-session/per-solve context structs; validate/allowlist any LLM backend URL on server deployments and cap response sizes (`take()`).
- [x] **Verify `unsafe` invariants (`src/elastic_mg.rs`)**
  - **Context:** `SyncPtr` (`lines 297, 413-415, 704`) relies on element colouring and 2×2×2 parity colouring guaranteeing disjoint write indices. No bug found, but invariants are unchecked at runtime.
  - **Action:** Add `debug_assert!` checks validating node bounds and colour disjointness during setup.

---

## 2. UX Audit & Redesign Plan

### What Works Well Today (Preserve)
- Left-to-right ribbon flow (`GEOMETRY → MATERIAL → SETUP → MESH → SOLVE → RESULTS → OPTIMIZE → HELP` in `src/gui.rs:1862`).
- Contextual filtering (`SETUP` hides irrelevant physics groups, warns when an enclosure is missing, `SOLVE` surfaces `PCB ▸ DC drop` only when a KiCad board is loaded, `RESULTS` dynamically shows solved fields).
- Tool-API parity (`GUI`, `MCP`, and Python operate on the exact same model with undo/redo and live selection sync), plus surrogate what-if sliders and `F1` context help.

### Friction Points Found & UX To-Do

#### 3D Viewport & Direct Manipulation
- [ ] **Upgrade from CPU software rasterization (`0.6×` drag blur) to hardware 3D rendering**
  - **Context:** `render::rasterize` (`src/render.rs`) draws flat-shaded quads on the CPU, drops to `0.6×` scale during drag/animation (`src/gui.rs:3887`), has no MSAA or perspective toggle, disables deformation whenever a section cut is active (`src/render.rs:258`), and lacks translucent body rendering.
  - **Action:** Add smooth/perspective camera toggle, MSAA, hardware depth-peeling or translucent enclosures (so internal CFD/EM/PCB heat is visible without hiding walls), deformed section cuts, and an interactive **ViewCube** (corner click-to-orient) alongside the axis triad.
- [ ] **Render 3D load & boundary condition glyphs**
  - **Context:** Currently loads only tint whole faces (`render::load_color`, `src/render.rs:133-153, 216-225`). Users cannot visually verify force vector direction/sign, pressure direction, fluid inlet flow direction, or coil axis in 3D.
  - **Action:** Draw directional 3D arrows, constraint pins, flow streamlines/arrows, and heat/radiation glyphs on scoped faces/bodies.
- [ ] **Add viewport body selection, hover highlighting, and per-body visibility toggles**
  - **Context:** Clicking in the viewport only selects faces (`src/gui.rs:3954`) and hovering only shows a text tooltip (`line 3979`). Bodies can only be selected for editing from the left Browser (`line 3263`), and cannot be hidden, isolated, or made wireframe/transparent.
  - **Action:** Highlight hovered faces/bodies directly in 3D, add a face/body selection mode toggle (plus right-click depth-cycle / "select behind" for inner faces), and add eye/ghost icons per body in the Browser.
- [ ] **Add live ghost previews & 3D gizmos for geometry creation and transforms**
  - **Context:** Box, Cylinder/Hole, Pocket, Sphere, and Enclosure forms (`src/gui.rs:3426-3540`) require blind numeric XYZ entry before clicking `Create` to see placement.
  - **Action:** Render a translucent ghost preview in 3D as values change, and add 3D drag handles + snap-to-face/center controls.

#### Forms, Workflows & Diagnostics
- [ ] **Replace `json_editor` in `item_editor` with the rich creation forms (`src/gui.rs:390-498, 2934-3067`)**
  - **Context:** Creating a load shows unit suffixes, hints, and click-to-pick face chips (`load_form`, `line 3550`), whereas clicking an existing body or load in the Browser falls back to `json_editor`, turning `faces` into a raw comma-separated string (`line 446`) and dropping unit labels.
  - **Action:** Share one unified, unit-aware form component with interactive viewport face-picking and inline validation for both creation and editing.
- [ ] **Add a live pre-solve readiness checklist**
  - **Context:** Missing fixed supports, missing fluid outlets, missing enclosures, or overly coarse mesh/FDTD wavelength cells are only discovered after pressing `Solve`.
  - **Action:** Display a live compact checklist in `SETUP` / `SOLVE` (e.g., `✓ Geometry · ✓ Material · ✗ Needs Support / Constraint · ⚠ Mesh > λ/10`) with one-click jump/fix links.
- [ ] **Scale the Browser for large PCB / assembly models (`src/gui.rs:3222-3380`)**
  - **Context:** KiCad boards create dozens of component bodies listed flatly in `Geometry` without grouping, search, or multi-select.
  - **Action:** Group PCB components under their parent board node (collapsible), add Browser search/filter, and support multi-select for batch material assignment or deletion.

#### Screen Ergonomics, Navigation & AI UX
- [ ] **Replace overlapping floating `egui::Window` modals with dockable/bottom drawers**
  - **Context:** `study_window`, `help_window` (`940×660`), `planner_window` (`620×640`), `pcb_window`, `chart_window`, `fatigue_window`, and `material_window` all float over the 3D viewport, crowding 13–14" laptop displays.
  - **Action:** Make side panels collapsible and host charts, optimization tables, surrogate what-if sliders, and KiCad DC track tables in a resizable bottom drawer (with click-to-highlight in 3D, e.g., clicking a hot track or optimization row).
- [ ] **Improve laptop trackpad navigation & keyboard shortcuts**
  - **Context:** Viewport panning currently requires right-drag or middle-drag (`src/gui.rs:3856`), which is awkward on laptop trackpads without a middle button.
  - **Action:** Support `Space+drag` / `Shift+middle` / two-finger trackpad pan, pinch-to-zoom (`zoom_delta()`), keyboard shortcuts (`F` fit, `1–6` standard views, `Del` delete selected, `Esc` close active form/clear selection), a right-click viewport context menu, and a unit-system preference (`mm/N/MPa`, `m/N/Pa`, `in/lbf/psi`).
- [ ] **Unify `✨ Planner` and `Connect AI` into an in-app executable Copilot**
  - **Context:** `Planner` (`src/assistant.rs`, `src/gui.rs:1609`) only outputs a static Ollama checklist (unable to execute steps beyond `Set up this analysis`), isolated from `Connect AI` (external MCP).
  - **Action:** Merge them into one AI Assistant drawer that inspects live selection/model state, runs or proposes step-by-step tool calls with preview/undo, and explains solver errors.

---

## 3. Architectural Assessment & Decision Plan

### A. For a Web App Used on Laptop/Desktop: **Current Stack Is *Not* the Best Approach**

Compiling the current `eframe`/`egui` + CPU software renderer (`render.rs`) + `wgpu` (`SHADER_F64`) binary directly to WebAssembly hits five fundamental walls:
1. **CPU Rasterizer in WASM (`src/render.rs`):** Software-rasterizing 100k–500k boundary quads on CPU inside a browser tab causes stuttering, high battery drain, and HiDPI lag, wasting the browser's hardware GPU (WebGL2/WebGPU).
2. **`egui` Canvas-Only Web UI (`src/gui.rs`):** Painting the entire app into one `<canvas>` loses native text selection, IME, browser accessibility, responsive CSS layouts, crisp scrolling, and mature charting/table widgets while coupling all UI growth to a 4,421-line immediate-mode file.
3. **WebGPU `f64` Incompatibility (`src/gpu.rs` & `src/fdtd.rs`):** `gpu::pick_adapter()` (`src/gpu.rs:198-209`) forces `wgpu::Backends::VULKAN` and `wgpu::Features::SHADER_F64`. **Standard browser WebGPU WGSL does not support `f64`** — and `fdtd.rs` (which only uses `f32`) inherits this block because it calls `gpu::pick_adapter()`. Both modules also call `pollster::block_on` and blocking `device.poll(Wait)`, which are illegal on browser main threads.
4. **WASM 4 GB Memory & Threading Limits:** SimCraft targets up to 2M hex elements (`MAX_CELLS_LARGE`), 12M FDTD cells (`MAX_CELLS_FDTD`), and 6M PCB DC pixels (`MAX_PIXELS`). `wasm32` hits browser tab memory ceilings far sooner, and `rayon` in WASM (`SharedArrayBuffer`) requires strict COOP/COEP HTTP headers.
5. **Native-Only Subsystems:** Local `TcpListener` (`bridge.rs`), stdio (`mcp.rs`), persistent local disk paths (`rfd`, `Shape::Stl`/`Shape::Pcb` re-reads), and localhost HTTP to Ollama (`assistant.rs`) cannot run directly inside a browser sandbox.

#### Recommended Web Architecture: **Split Modern Web Frontend + Rust Compute Engine**
- **Keep the Rust physics/meshing kernel** (`model`, `mesh`, `fem`, `elastic_mg`, `cfd`, `fdtd`, `kicad`, `study`, `session`) — `Core::call` already provides the clean command boundary.
- **Replace `gui.rs` + `render.rs`** with a **TypeScript + React/Svelte DOM UI** and a **Three.js / WebGL2 (or WebGPU) hardware 3D viewport**. The backend sends compact binary buffers (`Float32Array` vertices, patch IDs, nodal scalars, deformations) once per mesh/solve so orbit, zoom, field switching, mode animation, probing, and section cuts run at 60–120 FPS purely on the client GPU.
- **Execution Backends:**
  - *Primary (Full-Scale Cloud/Local Server):* Refactor `simcraft serve` into an async HTTP + WebSocket + binary-streaming server (`axum`) running on native Linux CPUs/Vulkan GPUs (retaining full 2M–12M cell capacity and `f64` GPU PCG, with per-session isolation and cooperative cancellation).
  - *Optional Zero-Server Demo Mode:* Compile a stripped `simcraft-wasm` WebWorker (excluding `egui`, `render.rs`, `gpu.rs`, `bridge.rs`, `rfd`) for instant offline/browser solves on small models (< ~100k cells), with an option to attach to a local `simcraft` daemon for heavy jobs.

---

### B. For a Proper Native Laptop/Desktop App: **Core Architecture Is Strong, With 4 Targeted Upgrades Needed**

If SimCraft remains a **dedicated native laptop/desktop application**, the current pure-Rust single-binary architecture (`rayon` + native RAM + hybrid iGPU/dGPU power management + `Core::call` + MCP) is a **strong foundation** — *provided* you address these four native bottlenecks:

#### Checklist for Native Laptop Architecture
- [ ] **Upgrade 1 — Hardware iGPU 3D Viewport (`src/gui.rs`, `src/render.rs`)**
  - Replace the CPU-rasterized RGBA upload in `viewport()` with an `egui::PaintCallback` (`glow` or `wgpu` on the integrated GPU, preserving `prefer_integrated_gpu()` battery savings).
  - Move deformation animation (`Scale * anim * disp`) into vertex shaders instead of rebuilding `Scene` allocations on CPU every frame (`src/gui.rs:3890-3899`).
  - Retain `render::render_png` strictly for headless MCP `render_image` and `report.rs` HTML exports.
- [ ] **Upgrade 2 — Non-Blocking Meshing & File Import (`src/gui.rs`, `src/session.rs`, `src/geometry.rs`)**
  - Move `s.ensure_mesh()` out of `viewport()` (`src/gui.rs:3796`) and `net_pads()` out of `open_pcb_dc()` (`src/gui.rs:2325`) onto background worker threads with progress + cancel support.
  - Add an STL triangle BVH in `geometry.rs` so boundary fitting and `x_crossings` don't stall on large CAD meshes.
- [ ] **Upgrade 3 — Cross-Platform Laptop GPU Adapter Selection (`src/gpu.rs`, `src/fdtd.rs`)**
  - Split `gpu::pick_adapter(require_f64: bool)`: use `wgpu::Backends::PRIMARY` (Vulkan + Metal + DX12) without `SHADER_F64` for `f32` `fdtd.rs`, unlocking Apple Silicon and Windows/integrated laptop GPUs; keep `SHADER_F64` strictly for double-precision `gpu.rs` PCG (or add a mixed-precision fallback).
- [ ] **Upgrade 4 — Core & UI Maintainability Refactors (`src/gui.rs`, `src/session.rs`)**
  - **Modularize `gui.rs`:** Split the 4,421-line file into `ui/{header,ribbon,browser,properties,viewport,charts,dialogs}.rs` and adopt `egui_dock` + `egui_plot` (or evaluate **Tauri + Web UI + Rust Core** if you eventually want commercial-grade multi-monitor docking, native trackpad/accessibility, *and* a shared codebase for a future web app).
  - **Remove double JSON serialization for undo (`src/session.rs:517-523`):** Track dirty mutations or command deltas instead of serializing the entire `Model` twice on every tool call (important before `P2.3` STEP B-reps land).
  - **Replace panic-unwinding cancellation (`src/session.rs:1761-1765`):** Migrate `check_cancel()` from `std::panic::resume_unwind(Box::new(Cancelled))` to cooperative `&AtomicBool` checks returning `Result` so parallel `rayon` loops (`P2.5` octree, `P2.8` transient CFD) and GPU poll loops can cancel cleanly without unwinding hazards.
