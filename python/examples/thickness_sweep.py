"""Sweep a beam's thickness and print stress, deflection and mass for each design."""
import pysimcraft

with pysimcraft.connect() as sim:
    sim.new_model()
    sim.add_box(name="Beam", size=[100, 10, 8])
    sim.set_material(material="Structural Steel")
    sim.set_mesh(element_size_mm=1)
    sim.add_fixed_support(faces=["Beam.xmin"])
    sim.add_force(faces=["Beam.xmax"], vector=[0, 0, -100])

    print(f"{'t [mm]':>7} {'max σ [MPa]':>12} {'tip δ [mm]':>11} {'mass [g]':>9}")
    for t in [4, 6, 8, 10, 12]:
        sim.call("run_parameter_study", parameters=[{"target": "Beam.size.z", "min": t, "max": t + 1e-9, "steps": 2}],
                 responses=["max:von_mises", "min:deformation_z", "mass"])
        row = sim.get_study()["runs"][0]
        print(f"{t:>7} {row['max:von_mises']:>12.1f} {abs(row['min:deformation_z']):>11.4f} {row['mass'] * 1000:>9.1f}")
