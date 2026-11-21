//! Solution fields and summaries.

use crate::geometry::V3;
use crate::mesh::Mesh;
use crate::model::AnalysisType;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Field {
    #[default]
    TotalDeformation,
    DeformationX,
    DeformationY,
    DeformationZ,
    VonMises,
    MaxPrincipal,
    MinPrincipal,
    SafetyFactor,
    Temperature,
    HeatFlux,
}

impl Field {
    pub fn all() -> [Field; 10] {
        use Field::*;
        [TotalDeformation, DeformationX, DeformationY, DeformationZ, VonMises, MaxPrincipal, MinPrincipal, SafetyFactor, Temperature, HeatFlux]
    }
    pub fn label(self) -> &'static str {
        match self {
            Field::TotalDeformation => "Total Deformation",
            Field::DeformationX => "Directional Deformation X",
            Field::DeformationY => "Directional Deformation Y",
            Field::DeformationZ => "Directional Deformation Z",
            Field::VonMises => "Equivalent (von Mises) Stress",
            Field::MaxPrincipal => "Maximum Principal Stress",
            Field::MinPrincipal => "Minimum Principal Stress",
            Field::SafetyFactor => "Safety Factor",
            Field::Temperature => "Temperature",
            Field::HeatFlux => "Total Heat Flux",
        }
    }
    pub fn unit(self) -> &'static str {
        match self {
            Field::TotalDeformation | Field::DeformationX | Field::DeformationY | Field::DeformationZ => "mm",
            Field::VonMises | Field::MaxPrincipal | Field::MinPrincipal => "MPa",
            Field::SafetyFactor => "",
            Field::Temperature => "°C",
            Field::HeatFlux => "W/m²",
        }
    }
    pub fn key(self) -> String {
        serde_json::to_value(self).unwrap().as_str().unwrap().to_string()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Mode {
    pub frequency_hz: f64,
    #[serde(skip)]
    pub shape: Vec<V3>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Reaction {
    pub load_id: u32,
    pub label: String,
    /// Reaction force [N] (structural) or [heat W, 0, 0] (thermal).
    pub value: V3,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct SolveInfo {
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    pub iterations: usize,
    pub seconds: f64,
    pub mass_kg: f64,
    pub element_size_mm: f64,
    pub solver: String,
}

#[derive(Clone, Debug)]
pub struct Results {
    pub analysis: AnalysisType,
    pub disp: Option<Vec<V3>>,
    pub stress: Option<Vec<[f64; 6]>>,
    pub vm: Option<Vec<f64>>,
    pub s1: Option<Vec<f64>>,
    pub s3: Option<Vec<f64>>,
    pub safety: Option<Vec<f64>>,
    pub temp: Option<Vec<f64>>,
    pub flux: Option<Vec<V3>>,
    pub modes: Vec<Mode>,
    pub reactions: Vec<Reaction>,
    pub info: SolveInfo,
    pub warnings: Vec<String>,
}

fn mag(v: &V3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

impl Results {
    pub fn available_fields(&self) -> Vec<Field> {
        Field::all().into_iter().filter(|f| self.has(*f)).collect()
    }
    pub fn has(&self, f: Field) -> bool {
        use Field::*;
        match f {
            TotalDeformation | DeformationX | DeformationY | DeformationZ => self.disp.is_some() || !self.modes.is_empty(),
            VonMises | MaxPrincipal | MinPrincipal | SafetyFactor => self.vm.is_some(),
            Temperature => self.temp.is_some(),
            HeatFlux => self.flux.is_some(),
        }
    }
    /// Displacement field to show the deformed shape: static result or mode shape.
    pub fn deformation(&self, mode: usize) -> Option<&[V3]> {
        if self.analysis == AnalysisType::Modal {
            return self.modes.get(mode).map(|m| m.shape.as_slice());
        }
        self.disp.as_deref()
    }
    pub fn scalar(&self, f: Field, mode: usize) -> Option<Vec<f64>> {
        use Field::*;
        match f {
            TotalDeformation => self.deformation(mode).map(|d| d.iter().map(mag).collect()),
            DeformationX => self.deformation(mode).map(|d| d.iter().map(|v| v[0]).collect()),
            DeformationY => self.deformation(mode).map(|d| d.iter().map(|v| v[1]).collect()),
            DeformationZ => self.deformation(mode).map(|d| d.iter().map(|v| v[2]).collect()),
            VonMises => self.vm.clone(),
            MaxPrincipal => self.s1.clone(),
            MinPrincipal => self.s3.clone(),
            SafetyFactor => self.safety.clone(),
            Temperature => self.temp.clone(),
            HeatFlux => self.flux.as_ref().map(|d| d.iter().map(mag).collect()),
        }
    }

    pub fn summary(&self, mesh: &Mesh) -> Value {
        let mut fields = serde_json::Map::new();
        for f in self.available_fields() {
            if self.analysis == AnalysisType::Modal {
                continue;
            }
            let Some(v) = self.scalar(f, 0) else { continue };
            let (mut imin, mut imax) = (0, 0);
            for i in 0..v.len() {
                if v[i] < v[imin] {
                    imin = i;
                }
                if v[i] > v[imax] {
                    imax = i;
                }
            }
            let r = |x: f64| float(x);
            let p = |i: usize| mesh.nodes[i].map(|c| (c * 1000.0).round() / 1000.0);
            fields.insert(
                f.key(),
                json!({"label": f.label(), "unit": f.unit(), "min": r(v[imin]), "min_at_mm": p(imin), "max": r(v[imax]), "max_at_mm": p(imax)}),
            );
        }
        let modes: Vec<Value> = self
            .modes
            .iter()
            .enumerate()
            .map(|(i, m)| json!({"mode": i + 1, "frequency_hz": float(m.frequency_hz)}))
            .collect();
        let reactions: Vec<Value> = self
            .reactions
            .iter()
            .map(|r| {
                if self.analysis == AnalysisType::SteadyThermal {
                    json!({"load_id": r.load_id, "label": r.label, "heat_flow_w": float(r.value[0])})
                } else {
                    json!({"load_id": r.load_id, "label": r.label, "force_n": r.value.map(float), "magnitude_n": float(mag(&r.value))})
                }
            })
            .collect();
        json!({
            "analysis": self.analysis.label(),
            "fields": fields,
            "modes": modes,
            "reactions": reactions,
            "info": self.info,
            "warnings": self.warnings,
        })
    }
}

/// Round to 6 significant digits for compact JSON.
pub fn float(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return if x.is_finite() { 0.0 } else { f64::MAX };
    }
    let d = 6 - x.abs().log10().ceil() as i32;
    let p = 10f64.powi(d);
    (x * p).round() / p
}
