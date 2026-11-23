"""Solve the bracket example and write an HTML report with pass/fail checks."""
import sys
import pysimcraft

out = sys.argv[1] if len(sys.argv) > 1 else "bracket-report.html"
with pysimcraft.connect() as sim:
    sim.new_model()
    sim.add_box(name="Bracket", size=[120, 40, 10])
    sim.add_cylinder(name="Hole", base=[80, 20, 0], radius=8, height=10, operation="subtract")
    sim.set_material(material="Aluminum 6061-T6")
    sim.set_mesh(element_size_mm=2)
    sim.add_fixed_support(faces=["Bracket.xmin"])
    sim.add_force(faces=["Bracket.xmax"], vector=[0, 0, -500])
    r = sim.solve()
    print("max von Mises:", r["fields"]["von_mises"]["max"], "MPa")
    print(sim.make_report(path=out, title="Bracket check",
                          checks=[{"response": "max:von_mises", "op": "<=", "value": 150},
                                  {"response": "min:safety_factor", "op": ">=", "value": 2}]).text)
