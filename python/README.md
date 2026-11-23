# pysimcraft

Drive [SimCraft](https://github.com/sushil-sundaram/simcraft) from Python — build models, solve,
run studies and export reports in scripts or notebooks. Standard library only.

```bash
pip install ./python            # from the SimCraft repository
```

```python
import pysimcraft

with pysimcraft.connect() as sim:        # uses the open SimCraft window, or starts a headless server
    sim.new_model()
    sim.add_box(name="Beam", size=[100, 10, 8])
    sim.set_material(material="Aluminum 6061-T6")
    sim.add_fixed_support(faces=["Beam.xmin"])
    sim.add_force(faces=["Beam.xmax"], vector=[0, 0, -100])
    r = sim.solve()
    print("max stress", r["fields"]["von_mises"]["max"], "MPa")
    sim.render_image(field="von_mises").save_image("beam.png")
    sim.make_report(path="beam-report.html")
```

- Every SimCraft tool is a method (`sim.tools()` lists them; `help(sim.add_force)` shows arguments).
- Results: `.text`, `.data` (parsed JSON), `.image` (PNG bytes), `r["key"]` shortcut.
- Errors raise `pysimcraft.SimCraftError`.
- `connect()` finds the executable via `$SIMCRAFT_BIN`, `PATH` or `~/simcraft/target/release/simcraft`,
  and authenticates with the token in `~/.config/simcraft/token`.

Examples: `examples/thickness_sweep.py`, `examples/batch_report.py`.
