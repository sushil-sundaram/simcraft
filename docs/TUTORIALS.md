# Tutorials

Step-by-step walkthroughs, each 5–15 minutes. They use the ribbon names you see in the app
(**GEOMETRY ▸ Box** means: the GEOMETRY tab, then the Box tool). Faces are picked by clicking them in
the viewport (Shift+click adds more); their names, such as `Bracket.xmin`, appear in the form.
**Edit ▸ Undo** (Ctrl+Z) reverses any step.

## Tutorial 1 — Stress in a bracket with a hole

**You will learn:** the basic workflow — geometry, material, supports and loads, mesh, solve, read
the results. **Time:** 5 minutes.

1. **File ▸ New** to start from an empty model.
2. **GEOMETRY ▸ Box**: name `Bracket`, corner `0, 0, 0`, size `120, 40, 10` mm → **Create**.
3. **GEOMETRY ▸ Hole**: base centre `80, 20, 0`, axis **Z**, radius `8`, height `10` → **Create**.
   The hole is a cylinder cut from the bracket.
4. **MATERIAL**: pick *Aluminum 6061-T6* → **Assign to all bodies**.
5. **SETUP ▸ Fixed**: click the left end face (`Bracket.xmin`) → **Apply**. This is the wall the
   bracket is bolted to.
6. **SETUP ▸ Force**: click the right end face (`Bracket.xmax`), enter `0, 0, -500` N → **Apply**.
7. **MESH**: element size `2` mm → **Generate**. Rule of thumb: 3–4 elements through the thickness.
8. **SOLVE ▸ ⚡ Solve**. The RESULTS tab opens when it finishes.
9. **RESULTS ▸ Von Mises**: the stress peaks at the fixed end and around the hole (the hole edge is
   fitted to the true circle, so the peak there is accurate). **RESULTS ▸ Safety** shows yield
   strength ÷ stress; anything above about 1.5 is comfortable.
10. **RESULTS ▸ TOOLS ▸ Probe**, then click the model to read values at points.
11. **RESULTS ▸ EXPORT ▸ Report** saves an HTML report with views, numbers and checks.

**Try next:** halve the element size and solve again — if the peak stress barely changes, the mesh
is fine enough. Or move the hole: select `Hole` in the browser and edit its base centre.

## Tutorial 2 — Cooling a heat sink

**You will learn:** thermal boundary conditions, heat sources, and when to switch from a convection
coefficient to real airflow. **Time:** 10 minutes.

1. **Examples ▸ Heat sink (thermal)**: an aluminium base with six fins.
2. Look at the loads in the browser: a **Heat Flux** of 20 000 W/m² enters through the bottom of the
   base (48 W over 60 × 40 mm), and **Convection** on all faces removes it (a coefficient h in
   W/m²·K and the air temperature).
3. **SOLVE ▸ ⚡ Solve**, then **RESULTS ▸ Temperature**: hottest at the base, coolest at the fin
   tips. **RESULTS ▸ Heat Flux** shows the heat running up the fins.
4. Change the convection coefficient: click the load in the browser, edit `coefficient` (5–10 for
   still air, 20–50 with a small fan) and solve again. The peak temperature scales roughly with 1/h.
5. Now let SimCraft compute the airflow instead: **Examples ▸ Heat sink in a duct (conjugate heat
   transfer)** — a 10 W chip under the sink in a 2 m/s air stream.
6. **SOLVE ▸ ⚡ Solve** (SIMPLE CFD; about a minute). **RESULTS ▸ Temperature** shows chip, sink and
   air together; **RESULTS ▸ Velocity** shows how the air goes around and between the fins. The
   quantities give the chip temperature and the energy balance (≈ 100 %).

**Try next:** change the inlet velocity (edit the Inlet load) and plot chip temperature against
fan speed with **OPTIMIZE ▸ Parameters**.

## Tutorial 3 — Will it resonate? Modal, random vibration and fatigue

**You will learn:** natural frequencies, response to a vibration spectrum, and fatigue life.
**Time:** 10 minutes.

1. Build or load the bracket from Tutorial 1 (**Examples ▸ Cantilever bracket with hole**).
2. **SETUP ▸ ANALYSIS**: choose **Modal**. Delete the force (✖ next to it in the browser) and keep
   the fixed support — loads that do not apply are ignored with a warning anyway.
3. **SOLVE ▸ ⚡ Solve**. The results list the natural frequencies; in **RESULTS ▸ VIEW** pick a mode
   and tick **Animate** to see its shape. If a frequency sits inside the range your product is
   shaken at, expect large responses.
4. **SETUP ▸ ANALYSIS**: choose **Random Vibration**. In **SOLVE ▸ SETTINGS** pick the excitation
   **direction** (Z) and enter the acceleration PSD table (Hz, g²/Hz) — the **Example** button
   fills a NAVMAT-style 6 Grms profile.
5. **SOLVE ▸ ⚡ Solve**. Results are 1σ (RMS) stresses and displacements; the quantities give the
   RMS acceleration and the 3σ peak stress.
6. **RESULTS ▸ FATIGUE ▸ Evaluate**: enter the exposure time → **Evaluate fatigue**. SimCraft
   applies the Steinberg three-band method with the material's S-N curve; **Damage** below 1 means
   the part survives that exposure.

**Try next:** **Harmonic Response** sweeps a sinusoidal load through the resonances and plots the
amplitude (RESULTS ▸ TOOLS ▸ History).

## Tutorial 4 — A KiCad board: temperatures and DC drop

**You will learn:** importing a PCB, thermal analysis with copper spreading, and checking whether
a power net's traces are wide enough. **Time:** 10 minutes. You need a `.kicad_pcb` file — in KiCad,
press **B** (fill all zones) and save first.

1. **GEOMETRY ▸ KiCad PCB** and pick the file. The board becomes a `Board` body with its copper
   homogenised per cell; footprints of 4 mm² or more become component boxes named `U1`, `Q2` …
   SimCraft sets the element size to half the board thickness.
2. **SETUP ▸ ANALYSIS**: **Steady Thermal**.
3. **SETUP ▸ Heat Gen.**: body `U1`, total power e.g. `1.5` W → **Apply**. Repeat for the other
   parts that dissipate.
4. **SETUP ▸ Convection**: Shift+click the board's top and bottom faces and the component tops,
   h = `10` W/m²·K (still air), air 25 °C → **Apply**.
5. **SOLVE ▸ ⚡ Solve**, then **RESULTS ▸ Temperature**. Copper pours spread the heat; the report
   lists the hottest part.
6. **SOLVE ▸ PCB ▸ DC drop** opens the DC current check for one net. Pick the **net** (supply rails
   such as `+5V` or `VBUS` are offered first), the **From pad** (where the current enters, e.g. the
   connector pin `J1.1`) and the **To pad** (the load, e.g. `U3.4`), and the **current** in A →
   **Calculate**.
7. Read the result: **resistance** and **voltage drop** (does the load still get enough volts?), the
   **power** lost in the copper, the **peak current density** and where it is (neck-downs and
   corners), and the tracks carrying the most current with their **IPC-2221 temperature rise** —
   orange above 10 °C, red above 20 °C. *Width for 10 °C* is the width IPC-2221 recommends.

**Try next:** widen a hot track in KiCad, save, re-import and compare. For the board's real
temperature with that copper loss, add it as heat generation and solve the thermal model again.

## Tutorial 5 — Find the lightest design that is strong enough

**You will learn:** parameter studies, optimisation with constraints, and what-if surrogates.
**Time:** 10 minutes.

1. Start from the bracket of Tutorial 1 (static structural, solved once).
2. **OPTIMIZE ▸ Parameters**: **＋ Add parameter** `Bracket.size.z` (the thickness) from 6 to 14 mm
   in 5 steps; tick the responses `max:von_mises` and `mass` → **Run study**. The table and chart show
   stress falling and mass rising with thickness.
3. **OPTIMIZE ▸ Optimize**: same parameter, objective **minimise** `mass`, constraint
   `max:von_mises <= 150` MPa → **Optimize**. The default *Surrogate (Bayesian)* method fits a
   response surface and checks every proposal with a real solve; the best feasible design is
   marked ★.
4. **OPTIMIZE ▸ Study results**: under *What-if* drag the thickness slider — the surrogate predicts
   the responses instantly with a ±2σ band; **Verify with a real solve** solves that design.
5. **Apply** next to a run in the table writes that design into the model.

**Try next:** add a second parameter (the hole radius) — surrogates pay off most with 2–4
parameters. **OPTIMIZE ▸ Topology** shows where material is actually needed.

## Tutorial 6 — A radiated-emissions check

**You will learn:** full-wave EM with an open boundary, and reading far-field and EMI results.
**Time:** 10 minutes.

1. **File ▸ New**. **GEOMETRY ▸ Box**: name `Feed`, corner `-3, -3, -3`, size `6, 6, 6` mm →
   **Create**. **MATERIAL**: *PTFE* → **Assign to all bodies** (before adding the air) — a small
   dielectric block that carries the source.
2. **GEOMETRY ▸ Enclosure**: Air, tick **Explicit box**, corner `-60, -60, -60`, size
   `120, 120, 120` mm → **Create**. Keep at least λ/4 of air on every side and all parts ≥ 2 cells
   from its walls.
3. **SETUP ▸ ANALYSIS**: **Full-Wave EM**. In **SOLVE ▸ SETTINGS** set the frequency (2.45 GHz) and
   **Outer boundary ▸ Open** so the waves leave the box.
4. **SETUP ▸ RF Port**: click the top face of `Feed`, power `1` W, E-field along **X**, untick
   *Waveguide (TE10) profile* → **Apply**. This is a short current element — an electrically small
   antenna.
5. **MESH**: `3` mm (≥ 10 cells per wavelength; λ = 122 mm here) → **SOLVE ▸ ⚡ Solve**.
6. Read the quantities: **radiated power** (≈ the 1 W input), **max directivity** (≈ 1.76 dBi for a
   short dipole), the **field strength at 3 m and 10 m** and the **margin to FCC Part 15 / CISPR 32
   class B** — negative means the emission would exceed the limit. **RESULTS ▸ TOOLS ▸ History**
   plots the directivity pattern through the main lobe; **RESULTS ▸ E-Field** shows the near field.

**Try next:** replace the feed by your enclosure and cable geometry with the port at the noise
source. These are free-space estimates for pre-compliance — expect ±6 dB against a test site.

## Tutorial 7 — Let an AI agent run the simulation

**You will learn:** driving SimCraft from Claude or another MCP client. **Time:** 5 minutes.

1. Click **Connect AI** in the menu bar and follow the instructions for your client (Claude Code,
   Claude Desktop, Antigravity …). The header shows **● MCP ready** when the bridge is listening.
2. Ask the agent in plain words, for example: *"Model a 100 × 20 × 5 mm aluminium cantilever, fix
   one end, put 200 N down on the other, mesh it at 1 mm and tell me the tip deflection and the
   safety factor."* The agent calls the same tools the ribbon uses, and you watch the model build in
   the window.
3. Ask follow-ups: *"Make it the lightest thickness with a safety factor of at least 2"* (it runs
   `optimize_design`), *"check the DC drop on +5V from J1.1 to U2.3 at 3 A"* (`pcb_dc`), or *"how
   does boundary fitting work?"* (`get_help` reads this guide).
4. Everything the agent does can be undone (**Edit ▸ Undo**) and saved like your own work.

**Try next:** **✨ Planner** in the menu bar turns a goal into a checklist with a local AI model, and
`pysimcraft` scripts the same tools from Python (HELP ▸ MCP / Python).
