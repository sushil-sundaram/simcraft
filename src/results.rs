//! Solution fields and summaries.

use crate::geometry::{V3, cross, dot, norm, normalize, scale, sub};
use crate::mesh::Mesh;
use crate::model::AnalysisType;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn default_z_axis() -> V3 {
    [0.0, 0.0, 1.0]
}

/// Cylindrical coordinate axis for decomposing stress into radial, hoop and axial components.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct CylindricalAxis {
    #[serde(default)]
    pub origin: V3,
    #[serde(default = "default_z_axis")]
    pub direction: V3,
}

impl Default for CylindricalAxis {
    fn default() -> Self {
        Self { origin: [0.0; 3], direction: [0.0, 0.0, 1.0] }
    }
}

impl CylindricalAxis {
    /// Orthonormal cylindrical basis `(e_r, e_theta, e_axial)` at position `p`.
    pub fn basis(self, p: V3) -> (V3, V3, V3) {
        let l = norm(self.direction);
        let ea = if l > 1e-12 && l.is_finite() { scale(self.direction, 1.0 / l) } else { [0.0, 0.0, 1.0] };
        let d = sub(p, self.origin);
        let rp = sub(d, scale(ea, dot(d, ea)));
        let er = if norm(rp) > 1e-12 {
            normalize(rp)
        } else {
            let refv = if ea[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            normalize(cross(refv, ea))
        };
        let et = cross(ea, er);
        (er, et, ea)
    }
}

/// Project symmetric stress `[xx, yy, zz, xy, yz, xz]` into `[sigma_r, sigma_theta, sigma_axial]`.
pub fn cylindrical_stress(p: V3, s: &[f64; 6], axis: CylindricalAxis) -> [f64; 3] {
    let (er, et, ea) = axis.basis(p);
    let proj = |v: V3| {
        s[0] * v[0] * v[0]
            + s[1] * v[1] * v[1]
            + s[2] * v[2] * v[2]
            + 2.0 * (s[3] * v[0] * v[1] + s[4] * v[1] * v[2] + s[5] * v[0] * v[2])
    };
    [proj(er), proj(et), proj(ea)]
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default, Hash, PartialOrd, Ord)]
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
    RadialStress,
    HoopStress,
    AxialStress,
    SafetyFactor,
    PlasticStrain,
    Temperature,
    HeatFlux,
    ElectricPotential,
    ElectricField,
    CurrentDensity,
    JouleHeat,
    MagneticFluxDensity,
    MagneticFieldIntensity,
    Velocity,
    FluidPressure,
    Density,
    FatigueLife,
    FatigueDamage,
    FatigueSafety,
    AbsorbedPower,
    Sar,
    VelocityX,
    VelocityY,
    VelocityZ,
    ContactPressure,
}

impl Field {
    pub fn all() -> [Field; 32] {
        use Field::*;
        [
            TotalDeformation,
            DeformationX,
            DeformationY,
            DeformationZ,
            VonMises,
            MaxPrincipal,
            MinPrincipal,
            RadialStress,
            HoopStress,
            AxialStress,
            SafetyFactor,
            PlasticStrain,
            ContactPressure,
            Temperature,
            HeatFlux,
            ElectricPotential,
            ElectricField,
            CurrentDensity,
            JouleHeat,
            MagneticFluxDensity,
            MagneticFieldIntensity,
            Velocity,
            FluidPressure,
            Density,
            FatigueLife,
            FatigueDamage,
            FatigueSafety,
            AbsorbedPower,
            Sar,
            VelocityX,
            VelocityY,
            VelocityZ,
        ]
    }
    pub fn label(self) -> &'static str {
        use Field::*;
        match self {
            TotalDeformation => "Total Deformation",
            DeformationX => "Directional Deformation X",
            DeformationY => "Directional Deformation Y",
            DeformationZ => "Directional Deformation Z",
            VonMises => "Equivalent (von Mises) Stress",
            MaxPrincipal => "Maximum Principal Stress",
            MinPrincipal => "Minimum Principal Stress",
            RadialStress => "Radial Stress",
            HoopStress => "Hoop Stress",
            AxialStress => "Axial Stress",
            SafetyFactor => "Safety Factor",
            PlasticStrain => "Equivalent Plastic Strain",
            ContactPressure => "Contact Pressure",
            Temperature => "Temperature",
            HeatFlux => "Total Heat Flux",
            ElectricPotential => "Electric Potential",
            ElectricField => "Electric Field",
            CurrentDensity => "Current Density",
            JouleHeat => "Joule Heat",
            MagneticFluxDensity => "Magnetic Flux Density",
            MagneticFieldIntensity => "Magnetic Field Intensity",
            Velocity => "Velocity",
            FluidPressure => "Pressure",
            Density => "Topology Density",
            FatigueLife => "Fatigue Life",
            FatigueDamage => "Fatigue Damage",
            FatigueSafety => "Fatigue Safety Factor",
            AbsorbedPower => "Absorbed Power Density",
            Sar => "Specific Absorption Rate",
            VelocityX => "Velocity X",
            VelocityY => "Velocity Y",
            VelocityZ => "Velocity Z",
        }
    }
    pub fn unit(self) -> &'static str {
        use Field::*;
        match self {
            TotalDeformation | DeformationX | DeformationY | DeformationZ => "mm",
            VonMises | MaxPrincipal | MinPrincipal | RadialStress | HoopStress | AxialStress | ContactPressure => "MPa",
            SafetyFactor => "",
            PlasticStrain => "mm/mm",
            Temperature => "°C",
            HeatFlux => "W/m²",
            ElectricPotential => "V",
            ElectricField => "V/m",
            CurrentDensity => "A/mm²",
            JouleHeat => "W/m³",
            MagneticFluxDensity => "T",
            MagneticFieldIntensity => "A/m",
            Velocity => "m/s",
            FluidPressure => "Pa",
            Density => "",
            FatigueLife => "cycles",
            FatigueDamage => "",
            FatigueSafety => "",
            AbsorbedPower => "W/m³",
            Sar => "W/kg",
            VelocityX | VelocityY | VelocityZ => "m/s",
        }
    }
    pub fn key(self) -> String {
        serde_json::to_value(self).unwrap().as_str().unwrap().to_string()
    }
    /// Fields where low values are critical: drawn with the colour scale reversed (low = red).
    pub fn reversed_scale(self) -> bool {
        matches!(self, Field::SafetyFactor | Field::FatigueSafety | Field::FatigueLife)
    }

    pub fn is_deformation(self) -> bool {
        matches!(self, Field::TotalDeformation | Field::DeformationX | Field::DeformationY | Field::DeformationZ)
    }

    pub(crate) fn cyl_comp(self) -> Option<usize> {
        match self {
            Field::RadialStress => Some(0),
            Field::HoopStress => Some(1),
            Field::AxialStress => Some(2),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Mode {
    pub frequency_hz: f64,
    /// Buckling load multiplier (critical load = multiplier × applied load).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_multiplier: Option<f64>,
    #[serde(skip)]
    pub shape: Vec<V3>,
}

/// One saved time point of a transient analysis.
#[derive(Clone, Debug, Default)]
pub struct TimeFrame {
    pub time: f64,
    pub disp: Option<Vec<V3>>,
    pub temp: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Reaction {
    pub load_id: u32,
    pub label: String,
    /// Reaction force [N] (structural), [heat W, 0, 0] (thermal), [charge C, 0, 0] (electrostatic),
    /// [current A, 0, 0] (current flow), [flow m³/s, mean pressure Pa, 0] (fluid).
    pub value: V3,
}

/// A named scalar result such as capacitance, resistance or drag.
#[derive(Clone, Debug, Serialize)]
pub struct Quantity {
    pub name: String,
    pub value: f64,
    pub unit: String,
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
    /// Wall time spent in linear solves (factorisation, CG, GPU).
    pub linear_solve_seconds: f64,
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
    /// Other nodal scalar fields (electric, magnetic, fluid, plastic strain…).
    pub extra: BTreeMap<Field, Vec<f64>>,
    pub modes: Vec<Mode>,
    pub frames: Vec<TimeFrame>,
    /// (time [s], value) history of the main response for transient analyses.
    pub history: Vec<(f64, f64)>,
    pub history_label: String,
    /// x-axis of `history` ("time [s]", "frequency [Hz]", "load factor"…).
    #[allow(dead_code)]
    pub history_x: String,
    /// What the displayed fields mean (e.g. "1σ RMS", "at 812 Hz", "SRSS combined").
    pub note: String,
    pub reactions: Vec<Reaction>,
    pub quantities: Vec<Quantity>,
    pub info: SolveInfo,
    pub warnings: Vec<String>,
}

pub fn mag(v: &V3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

impl Results {
    pub fn new(analysis: AnalysisType) -> Self {
        Results {
            analysis,
            disp: None,
            stress: None,
            vm: None,
            s1: None,
            s3: None,
            safety: None,
            temp: None,
            flux: None,
            extra: BTreeMap::new(),
            modes: vec![],
            frames: vec![],
            history: vec![],
            history_label: String::new(),
            history_x: String::new(),
            note: String::new(),
            reactions: vec![],
            quantities: vec![],
            info: SolveInfo::default(),
            warnings: vec![],
        }
    }

    pub fn quantity(&mut self, name: &str, value: f64, unit: &str) {
        self.quantities.push(Quantity { name: name.into(), value: float(value), unit: unit.into() });
    }

    /// Number of selectable "steps": modes, time frames, or 1.
    pub fn n_steps(&self) -> usize {
        if !self.modes.is_empty() {
            self.modes.len()
        } else if !self.frames.is_empty() {
            self.frames.len()
        } else {
            1
        }
    }
    pub fn step_label(&self, i: usize) -> String {
        if let Some(m) = self.modes.get(i) {
            return match m.load_multiplier {
                Some(l) => format!("Mode {} — load ×{:.4}", i + 1, l),
                None => format!("Mode {} — {:.2} Hz", i + 1, m.frequency_hz),
            };
        }
        if let Some(f) = self.frames.get(i) {
            return format!("t = {:.4} s", f.time);
        }
        "Final".into()
    }

    /// Number of nodes the nodal arrays refer to (None if there are none).
    pub fn node_len(&self) -> Option<usize> {
        self.disp.as_ref().map(|v| v.len()).or(self.temp.as_ref().map(|v| v.len())).or(self.vm.as_ref().map(|v| v.len())).or(self.extra.values().next().map(|v| v.len())).or(self.modes.first().map(|m| m.shape.len()))
    }
    pub fn available_fields(&self) -> Vec<Field> {
        Field::all().into_iter().filter(|f| self.has(*f)).collect()
    }
    pub fn has(&self, f: Field) -> bool {
        use Field::*;
        if self.extra.contains_key(&f) {
            return true;
        }
        match f {
            TotalDeformation | DeformationX | DeformationY | DeformationZ => self.disp.is_some() || !self.modes.is_empty(),
            VonMises | MaxPrincipal | MinPrincipal | RadialStress | HoopStress | AxialStress | SafetyFactor => self.vm.is_some(),
            Temperature => self.temp.is_some(),
            HeatFlux => self.flux.is_some(),
            _ => false,
        }
    }
    /// Displacement field for the deformed shape: static result, mode shape or time frame.
    pub fn deformation(&self, step: usize) -> Option<&[V3]> {
        if !self.modes.is_empty() {
            return self.modes.get(step.min(self.modes.len() - 1)).map(|m| m.shape.as_slice());
        }
        if !self.frames.is_empty() {
            if let Some(d) = &self.frames[step.min(self.frames.len() - 1)].disp {
                return Some(d.as_slice());
            }
        }
        self.disp.as_deref()
    }
    pub fn scalar(&self, f: Field, step: usize) -> Option<Vec<f64>> {
        use Field::*;
        if let Some(v) = self.extra.get(&f) {
            return Some(v.clone());
        }
        match f {
            TotalDeformation => self.deformation(step).map(|d| d.iter().map(mag).collect()),
            DeformationX => self.deformation(step).map(|d| d.iter().map(|v| v[0]).collect()),
            DeformationY => self.deformation(step).map(|d| d.iter().map(|v| v[1]).collect()),
            DeformationZ => self.deformation(step).map(|d| d.iter().map(|v| v[2]).collect()),
            VonMises => self.vm.clone(),
            MaxPrincipal => self.s1.clone(),
            MinPrincipal => self.s3.clone(),
            SafetyFactor => self.safety.clone(),
            Temperature => {
                if !self.frames.is_empty() {
                    if let Some(t) = &self.frames[step.min(self.frames.len() - 1)].temp {
                        return Some(t.clone());
                    }
                }
                self.temp.clone()
            }
            HeatFlux => self.flux.as_ref().map(|d| d.iter().map(mag).collect()),
            _ => None,
        }
    }

    /// Nodal scalar array including cylindrical stress transforms (evaluated around `cyl`).
    pub fn scalar_at(&self, mesh: &Mesh, f: Field, step: usize, cyl: CylindricalAxis) -> Option<Vec<f64>> {
        if let Some(k) = f.cyl_comp() {
            let st = self.stress.as_ref()?;
            return Some(mesh.nodes.iter().zip(st).map(|(p, s)| cylindrical_stress(*p, s, cyl)[k]).collect());
        }
        self.scalar(f, step)
    }

    pub fn summary(&self, mesh: &Mesh) -> Value {
        self.summary_with(mesh, CylindricalAxis::default())
    }

    pub fn summary_with(&self, mesh: &Mesh, cyl: CylindricalAxis) -> Value {
        let mut fields = serde_json::Map::new();
        let last = self.n_steps().saturating_sub(1);
        for f in self.available_fields() {
            if !self.modes.is_empty() && f.is_deformation() {
                continue;
            }
            let Some(v) = self.scalar_at(mesh, f, last, cyl) else { continue };
            if v.is_empty() {
                continue;
            }
            let (mut imin, mut imax) = (0, 0);
            for i in 0..v.len() {
                if v[i] < v[imin] {
                    imin = i;
                }
                if v[i] > v[imax] {
                    imax = i;
                }
            }
            let p = |i: usize| mesh.nodes[i].map(|c| (c * 1000.0).round() / 1000.0);
            fields.insert(f.key(), json!({"label": f.label(), "unit": f.unit(), "min": float(v[imin]), "min_at_mm": p(imin), "max": float(v[imax]), "max_at_mm": p(imax)}));
        }
        let modes: Vec<Value> = self
            .modes
            .iter()
            .enumerate()
            .map(|(i, m)| match m.load_multiplier {
                Some(l) => json!({"mode": i + 1, "load_multiplier": float(l)}),
                None => json!({"mode": i + 1, "frequency_hz": float(m.frequency_hz)}),
            })
            .collect();
        let reactions: Vec<Value> = self
            .reactions
            .iter()
            .map(|r| match self.analysis {
                AnalysisType::SteadyThermal | AnalysisType::TransientThermal => json!({"load_id": r.load_id, "label": r.label, "heat_flow_w": float(r.value[0])}),
                AnalysisType::Electrostatic => json!({"load_id": r.load_id, "label": r.label, "charge_c": float(r.value[0])}),
                AnalysisType::CurrentFlow | AnalysisType::ThermalElectric => json!({"load_id": r.load_id, "label": r.label, "current_a": float(r.value[0])}),
                AnalysisType::Fluid => json!({"load_id": r.load_id, "label": r.label, "flow_rate_m3_s": float(r.value[0]), "mean_pressure_pa": float(r.value[1])}),
                _ => json!({"load_id": r.load_id, "label": r.label, "force_n": r.value.map(float), "magnitude_n": float(mag(&r.value))}),
            })
            .collect();
        let mut out = json!({
            "analysis": self.analysis.label(),
            "note": self.note,
            "fields": fields,
            "reactions": reactions,
            "info": self.info,
            "warnings": self.warnings,
        });
        if !modes.is_empty() {
            out["modes"] = json!(modes);
        }
        if !self.quantities.is_empty() {
            out["quantities"] = json!(self.quantities);
        }
        if !self.frames.is_empty() {
            out["time_points"] = json!(self.frames.len());
            out["end_time_s"] = json!(self.frames.last().map(|f| f.time));
        }
        if !self.history.is_empty() {
            let stride = (self.history.len() / 25).max(1);
            let h: Vec<[f64; 2]> = self.history.iter().step_by(stride).map(|(t, v)| [float(*t), float(*v)]).collect();
            out["history"] = json!({"quantity": self.history_label, "x": self.history_x, "samples": h});
        }
        out
    }

    /// Sample `field` along a straight line from `from` to `to` (`samples` points in `2..=500`).
    pub fn path_plot(&self, mesh: &Mesh, field: Field, step: usize, cyl: CylindricalAxis, from: V3, to: V3, samples: usize) -> Result<PathPlot> {
        if !from.iter().all(|v| v.is_finite()) || !to.iter().all(|v| v.is_finite()) {
            bail!("path endpoints must be finite coordinates");
        }
        let d = sub(to, from);
        let len = norm(d);
        if !(len > 1e-12) {
            bail!("path start and end points must be distinct");
        }
        if !self.has(field) {
            bail!("field '{}' is not available in these results", field.label());
        }
        let n = samples.clamp(2, 500);
        let pts: Vec<V3> = (0..n)
            .map(|i| {
                let u = i as f64 / (n - 1) as f64;
                [from[0] + u * d[0], from[1] + u * d[1], from[2] + u * d[2]]
            })
            .collect();
        let sampled: Vec<Option<f64>> = if let Some(ci) = field.cyl_comp() {
            let st = self.stress.as_ref().ok_or_else(|| anyhow::anyhow!("stress tensor not available"))?;
            mesh.sample_path_tensor(st, &pts)
                .into_iter()
                .zip(&pts)
                .map(|(s, &p)| s.map(|ten| cylindrical_stress(p, &ten, cyl)[ci]))
                .collect()
        } else {
            let arr = self.scalar(field, step).ok_or_else(|| anyhow::anyhow!("field '{}' has no nodal values", field.label()))?;
            mesh.sample_path(&arr, &pts)
        };
        let mut points = Vec::with_capacity(n);
        let (mut mn, mut mx, mut sum, mut cnt) = (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0usize);
        for (i, (&pt, val)) in pts.iter().zip(sampled).enumerate() {
            let s = len * i as f64 / (n - 1) as f64;
            if let Some(v) = val {
                if v.is_finite() {
                    mn = mn.min(v);
                    mx = mx.max(v);
                    sum += v;
                    cnt += 1;
                }
            }
            points.push(PathPoint {
                distance_mm: float(s),
                point_mm: pt.map(float),
                value: val.filter(|v| v.is_finite()).map(float),
            });
        }
        Ok(PathPlot {
            field: field.key(),
            label: field.label().to_string(),
            unit: field.unit().to_string(),
            from_mm: from,
            to_mm: to,
            length_mm: float(len),
            valid_samples: cnt,
            points,
            min: (cnt > 0).then(|| float(mn)),
            max: (cnt > 0).then(|| float(mx)),
            mean: (cnt > 0).then(|| float(sum / cnt as f64)),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PathPoint {
    pub distance_mm: f64,
    pub point_mm: V3,
    pub value: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PathPlot {
    pub field: String,
    pub label: String,
    pub unit: String,
    pub from_mm: V3,
    pub to_mm: V3,
    pub length_mm: f64,
    pub valid_samples: usize,
    pub points: Vec<PathPoint>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
}

impl PathPlot {
    pub fn to_csv(&self) -> String {
        let mut out = if self.unit.is_empty() {
            format!("distance_mm,x_mm,y_mm,z_mm,{}\n", self.field)
        } else {
            format!("distance_mm,x_mm,y_mm,z_mm,{} [{}]\n", self.field, self.unit)
        };
        for pt in &self.points {
            let v = pt.value.map(|x| x.to_string()).unwrap_or_default();
            out.push_str(&format!("{},{},{},{},{}\n", pt.distance_mm, pt.point_mm[0], pt.point_mm[1], pt.point_mm[2], v));
        }
        out
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
