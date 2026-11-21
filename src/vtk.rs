//! Legacy VTK export (opens in ParaView).

use crate::mesh::Mesh;
use crate::results::{Field, Results};
use anyhow::Result;
use std::io::Write;

pub fn export(path: &str, mesh: &Mesh, res: Option<&Results>) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "# vtk DataFile Version 3.0\nSimCraft results\nASCII\nDATASET UNSTRUCTURED_GRID")?;
    writeln!(f, "POINTS {} double", mesh.nodes.len())?;
    for p in &mesh.nodes {
        writeln!(f, "{} {} {}", p[0], p[1], p[2])?;
    }
    writeln!(f, "CELLS {} {}", mesh.elems.len(), mesh.elems.len() * 9)?;
    for e in &mesh.elems {
        writeln!(f, "8 {} {} {} {} {} {} {} {}", e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7])?;
    }
    writeln!(f, "CELL_TYPES {}", mesh.elems.len())?;
    for _ in &mesh.elems {
        writeln!(f, "12")?;
    }
    writeln!(f, "CELL_DATA {}\nSCALARS material int 1\nLOOKUP_TABLE default", mesh.elems.len())?;
    for m in &mesh.elem_mat {
        writeln!(f, "{m}")?;
    }
    if let Some(r) = res {
        writeln!(f, "POINT_DATA {}", mesh.nodes.len())?;
        if let Some(d) = &r.disp {
            writeln!(f, "VECTORS displacement_mm double")?;
            for v in d {
                writeln!(f, "{} {} {}", v[0], v[1], v[2])?;
            }
        }
        for (i, m) in r.modes.iter().enumerate() {
            writeln!(f, "VECTORS mode{}_{:.1}Hz double", i + 1, m.frequency_hz)?;
            for v in &m.shape {
                writeln!(f, "{} {} {}", v[0], v[1], v[2])?;
            }
        }
        for fl in [Field::VonMises, Field::MaxPrincipal, Field::MinPrincipal, Field::SafetyFactor, Field::Temperature, Field::HeatFlux] {
            if r.analysis == crate::model::AnalysisType::Modal {
                break;
            }
            if let Some(v) = r.scalar(fl, 0) {
                writeln!(f, "SCALARS {} double 1\nLOOKUP_TABLE default", fl.key())?;
                for x in v {
                    writeln!(f, "{x}")?;
                }
            }
        }
    }
    Ok(())
}
