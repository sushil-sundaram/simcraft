//! STEP (ISO 10303-21 AP203 / AP214 / AP242) B-rep CAD importer.
//!
//! Parses STEP solids and assemblies in pure Rust via `monstertruck-io` +
//! `monstertruck-meshing`, detects STEP length units into SimCraft millimetres,
//! tessellates trimmed B-rep faces into a BVH-accelerated [`StlData`] for fast
//! voxelisation / rendering, and retains exact analytic or parametric surface
//! equations per B-rep face for exact boundary fitting.

use crate::geometry::{StlData, V3, add, closest_point_triangle, dot, norm, normalize, scale, sub};
use anyhow::{Context, Result, anyhow, bail};
use monstertruck_core::bounding_box::BoundingBox;
use monstertruck_core::cgmath64::{EuclideanSpace, InnerSpace, Matrix4, Point3, SquareMatrix, Transform, Vector3};
use monstertruck_geometry::prelude::{ParametricSurface, ParametricSurface3D, SearchNearestParameter};
use monstertruck_io::step::load::step_geometry::{ElementarySurface, Surface as StepSurface};
use monstertruck_io::step::load::step_p21::ast::{EntityInstance, Exchange, Name as P21Name, Parameter, SubSuperRecord};
use monstertruck_io::step::load::Table;
use monstertruck_meshing::prelude::*;
use monstertruck_topology::compress::{CompressedShell, CompressedSolid};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

const MAX_STEP_BYTES: u64 = 128 * 1024 * 1024;
const MAX_STEP_ENTITIES: usize = 500_000;
const MAX_STEP_BODIES: usize = 512;
const MAX_STEP_TRIS: usize = 5_000_000;

/// Kind of geometric surface backing a STEP B-rep face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepSurfaceKind {
    Plane,
    Cylinder,
    Sphere,
    Cone,
    Torus,
    Parametric,
    Mesh,
}

impl StepSurfaceKind {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Plane => "plane",
            Self::Cylinder => "cylinder",
            Self::Sphere => "sphere",
            Self::Cone => "cone",
            Self::Torus => "torus",
            Self::Parametric => "surface",
            Self::Mesh => "face",
        }
    }
}

/// Parametric STEP surface (NURBS / B-spline / Swept) with rigid+scale placement
/// into world millimetres for exact nearest-point projection.
#[derive(Debug)]
pub struct ParametricStepSurface {
    surface: StepSurface,
    /// Affine map from raw B-rep coordinates to world millimetres.
    to_world: Matrix4,
    /// Inverse affine map from world millimetres to raw B-rep coordinates.
    to_raw: Matrix4,
    outward_sign: f64,
}

impl ParametricStepSurface {
    fn translated(&self, delta: V3) -> Self {
        let t_delta = Matrix4::from_translation(Vector3::new(delta[0], delta[1], delta[2]));
        let to_world = t_delta * self.to_world;
        let to_raw = to_world.invert().unwrap_or(self.to_raw);
        Self {
            surface: self.surface.clone(),
            to_world,
            to_raw,
            outward_sign: self.outward_sign,
        }
    }

    fn project(&self, p: V3) -> Option<V3> {
        let pr = self.to_raw.transform_point(Point3::new(p[0], p[1], p[2]));
        let (u, v) = SearchNearestParameter::<monstertruck_geometry::prelude::D2>::search_nearest_parameter(&self.surface, pr, None, 64)?;
        let qr = self.surface.subs(u, v);
        let qw = self.to_world.transform_point(qr);
        let out = [qw.x, qw.y, qw.z];
        out.iter().all(|c| c.is_finite()).then_some(out)
    }

    fn normal_at(&self, p: V3) -> Option<V3> {
        let pr = self.to_raw.transform_point(Point3::new(p[0], p[1], p[2]));
        let (u, v) = SearchNearestParameter::<monstertruck_geometry::prelude::D2>::search_nearest_parameter(&self.surface, pr, None, 64)?;
        let nr = self.surface.normal(u, v);
        let nw = self.to_world.transform_vector(nr);
        let nv = [nw.x * self.outward_sign, nw.y * self.outward_sign, nw.z * self.outward_sign];
        let l = norm(nv);
        (l.is_finite() && l > 1e-12).then(|| scale(nv, 1.0 / l))
    }
}

/// Exact surface geometry attached to a STEP B-rep face.
#[derive(Clone, Debug)]
pub enum StepSurfaceAnalytic {
    Plane {
        origin: V3,
        normal: V3,
    },
    Cylinder {
        origin: V3,
        axis: V3,
        radius: f64,
        outward: f64,
    },
    Sphere {
        center: V3,
        radius: f64,
        outward: f64,
    },
    Cone {
        apex: V3,
        axis: V3,
        cos_a: f64,
        sin_a: f64,
        outward: f64,
    },
    Torus {
        center: V3,
        axis: V3,
        major_r: f64,
        minor_r: f64,
        outward: f64,
    },
    Parametric(Arc<ParametricStepSurface>),
    Mesh,
}

impl StepSurfaceAnalytic {
    /// Closest point on the unbounded surface (None when point is on an axis/center singularity).
    pub fn project(&self, p: V3) -> Option<V3> {
        match self {
            Self::Plane { origin, normal } => {
                let d = dot(sub(p, *origin), *normal);
                Some(sub(p, scale(*normal, d)))
            }
            Self::Cylinder { origin, axis, radius, .. } => {
                let d = sub(p, *origin);
                let ax = dot(d, *axis);
                let rad = sub(d, scale(*axis, ax));
                let r = norm(rad);
                (r > 1e-12).then(|| add(*origin, add(scale(*axis, ax), scale(rad, radius / r))))
            }
            Self::Sphere { center, radius, .. } => {
                let d = sub(p, *center);
                let r = norm(d);
                (r > 1e-12).then(|| add(*center, scale(d, radius / r)))
            }
            Self::Cone { apex, axis, cos_a, sin_a, .. } => {
                let d = sub(p, *apex);
                let z = dot(d, *axis);
                let rad = sub(d, scale(*axis, z));
                let rho = norm(rad);
                if rho < 1e-12 {
                    return None;
                }
                let s = z * cos_a + rho * sin_a;
                let rdir = scale(rad, 1.0 / rho);
                Some(add(*apex, add(scale(*axis, s * cos_a), scale(rdir, s * sin_a))))
            }
            Self::Torus { center, axis, major_r, minor_r, .. } => {
                let d = sub(p, *center);
                let perp = sub(d, scale(*axis, dot(d, *axis)));
                let rp = norm(perp);
                if rp < 1e-12 {
                    return None;
                }
                let ring = add(*center, scale(perp, major_r / rp));
                let v = sub(p, ring);
                let vl = norm(v);
                (vl > 1e-12).then(|| add(ring, scale(v, minor_r / vl)))
            }
            Self::Parametric(ps) => ps.project(p),
            Self::Mesh => None,
        }
    }

    /// Outward surface unit normal near `p` (None if degenerate or mesh-only).
    pub fn normal_at(&self, p: V3) -> Option<V3> {
        match self {
            Self::Plane { normal, .. } => Some(*normal),
            Self::Cylinder { origin, axis, outward, .. } => {
                let d = sub(p, *origin);
                let rad = sub(d, scale(*axis, dot(d, *axis)));
                let r = norm(rad);
                (r > 1e-12).then(|| scale(rad, outward / r))
            }
            Self::Sphere { center, outward, .. } => {
                let d = sub(p, *center);
                let r = norm(d);
                (r > 1e-12).then(|| scale(d, outward / r))
            }
            Self::Cone { apex, axis, cos_a, sin_a, outward } => {
                let d = sub(p, *apex);
                let rad = sub(d, scale(*axis, dot(d, *axis)));
                let rho = norm(rad);
                if rho < 1e-12 {
                    return None;
                }
                let rdir = scale(rad, 1.0 / rho);
                let n = sub(scale(rdir, *cos_a), scale(*axis, *sin_a));
                let l = norm(n);
                (l > 1e-12).then(|| scale(n, outward / l))
            }
            Self::Torus { center, axis, major_r, outward, .. } => {
                let d = sub(p, *center);
                let perp = sub(d, scale(*axis, dot(d, *axis)));
                let rp = norm(perp);
                if rp < 1e-12 {
                    return None;
                }
                let ring = add(*center, scale(perp, major_r / rp));
                let v = sub(p, ring);
                let vl = norm(v);
                (vl > 1e-12).then(|| scale(v, outward / vl))
            }
            Self::Parametric(ps) => ps.normal_at(p),
            Self::Mesh => None,
        }
    }

    fn translated(&self, delta: V3) -> Self {
        match self {
            Self::Plane { origin, normal } => Self::Plane { origin: add(*origin, delta), normal: *normal },
            Self::Cylinder { origin, axis, radius, outward } => Self::Cylinder {
                origin: add(*origin, delta),
                axis: *axis,
                radius: *radius,
                outward: *outward,
            },
            Self::Sphere { center, radius, outward } => Self::Sphere {
                center: add(*center, delta),
                radius: *radius,
                outward: *outward,
            },
            Self::Cone { apex, axis, cos_a, sin_a, outward } => Self::Cone {
                apex: add(*apex, delta),
                axis: *axis,
                cos_a: *cos_a,
                sin_a: *sin_a,
                outward: *outward,
            },
            Self::Torus { center, axis, major_r, minor_r, outward } => Self::Torus {
                center: add(*center, delta),
                axis: *axis,
                major_r: *major_r,
                minor_r: *minor_r,
                outward: *outward,
            },
            Self::Parametric(ps) => Self::Parametric(Arc::new(ps.translated(delta))),
            Self::Mesh => Self::Mesh,
        }
    }
}

/// One B-rep face within a [`StepData`] solid.
#[derive(Clone, Debug)]
pub struct StepFace {
    pub name: String,
    pub kind: StepSurfaceKind,
    pub analytic: StepSurfaceAnalytic,
}

/// Tessellated + exact-B-rep representation of one solid body from a STEP file.
#[derive(Debug)]
pub struct StepData {
    pub mesh: StlData,
    pub faces: Vec<StepFace>,
    pub part_name: String,
    pub part_index: usize,
    pub unit_name: String,
    pub unit_scale_mm: f64,
    pub volume_mm3: f64,
}

impl StepData {
    pub fn bounds(&self) -> (V3, V3) {
        (self.mesh.min, self.mesh.max)
    }

    pub fn patch_names(&self) -> Vec<String> {
        self.faces.iter().map(|f| f.name.clone()).collect()
    }

    pub fn translated(&self, delta: V3) -> Self {
        Self {
            mesh: self.mesh.translated(delta),
            faces: self
                .faces
                .iter()
                .map(|f| StepFace {
                    name: f.name.clone(),
                    kind: f.kind,
                    analytic: f.analytic.translated(delta),
                })
                .collect(),
            part_name: self.part_name.clone(),
            part_index: self.part_index,
            unit_name: self.unit_name.clone(),
            unit_scale_mm: self.unit_scale_mm,
            volume_mm3: self.volume_mm3,
        }
    }

    /// Closest point on B-rep face `patch`: uses exact analytic/parametric projection when
    /// available, falling back to the face's BVH triangle soup if singular or unsupported.
    pub fn project(&self, patch: usize, p: V3) -> Option<V3> {
        if let Some(q) = self.faces.get(patch).and_then(|f| f.analytic.project(p)) {
            return Some(q);
        }
        let mut best: Option<(f64, V3)> = None;
        self.mesh.visit_closest(p, |i, limit_d2| {
            if self.mesh.segment[i] as usize != patch {
                return;
            }
            let q = closest_point_triangle(p, &self.mesh.tris[i]);
            let dv = sub(p, q);
            let d2 = dot(dv, dv);
            if d2 < *limit_d2 {
                *limit_d2 = d2;
                best = Some((d2, q));
            }
        });
        best.map(|b| b.1)
    }

    /// Best matching B-rep face for a voxel boundary face at `p` with outward normal `n`.
    pub fn best_patch(&self, p: V3, n: V3, h: f64, subtract: bool) -> (usize, f64) {
        let sign = if subtract { -1.0 } else { 1.0 };
        let score = |dist: f64, pn: V3| {
            let d = (dot(pn, n) * sign).clamp(-1.0, 1.0);
            if d < -0.2 { f64::INFINITY } else { dist + h * (1.0 - d) }
        };
        let flat = |dist: f64, pn: V3| {
            let d = (dot(pn, n) * sign).clamp(-1.0, 1.0);
            if d < 0.5 { f64::INFINITY } else { dist + h * (1.0 - d) }
        };
        let mut best = (0usize, f64::INFINITY);
        self.mesh.visit_closest(p, |i, limit_d2| {
            let q = closest_point_triangle(p, &self.mesh.tris[i]);
            let dist = norm(sub(p, q));
            if dist > best.1 {
                return;
            }
            let seg = self.mesh.segment[i] as usize;
            let nn = self
                .faces
                .get(seg)
                .and_then(|f| f.analytic.normal_at(q))
                .unwrap_or(self.mesh.normals[i]);
            let is_axis_plane = self
                .faces
                .get(seg)
                .is_some_and(|f| f.kind == StepSurfaceKind::Plane && nn.iter().any(|c| c.abs() > 0.999))
                || nn.iter().any(|c| c.abs() > 0.999);
            let s = if is_axis_plane { flat(dist, nn) } else { score(dist, nn) };
            if s < best.1 {
                best = (seg, s);
                *limit_d2 = best.1 * best.1;
            }
        });
        best
    }
}

/// Detect the STEP length unit multiplier into millimetres (`mm_per_unit`) and its label.
pub fn detect_length_unit_mm(step_text: &str) -> Result<(f64, String, usize)> {
    let exchange = Exchange::from_str(step_text)
        .map_err(|e| anyhow!("invalid STEP Part 21 syntax: {e}"))?;
    let total_entities: usize = exchange.data.iter().map(|d| d.entities.len()).sum();
    if total_entities > MAX_STEP_ENTITIES {
        bail!(
            "STEP file has too many entities ({total_entities}; limit {MAX_STEP_ENTITIES})"
        );
    }
    let Some(data) = exchange.data.first() else {
        bail!("STEP file has no DATA section");
    };

    let si_scale = |prefix: &str, unit: &str| -> Option<(f64, String)> {
        if !unit.eq_ignore_ascii_case("METRE") {
            return None;
        }
        let (p, lbl) = match prefix.to_ascii_uppercase().as_str() {
            "MILLI" => (1e-3, "mm"),
            "CENTI" => (1e-2, "cm"),
            "DECI" => (1e-1, "dm"),
            "MICRO" => (1e-6, "µm"),
            "NANO" => (1e-9, "nm"),
            "KILO" => (1e3, "km"),
            "" | "$" => (1.0, "m"),
            other => return Some((1000.0, other.to_ascii_lowercase())),
        };
        Some((1000.0 * p, lbl.to_string()))
    };

    let push_unit_refs = |param: &Parameter, out: &mut Vec<u64>| {
        if let Parameter::List(params) = param
            && let Some(Parameter::List(refs)) = params.first()
        {
            for r in refs {
                if let Parameter::Ref(P21Name::Entity(eid)) = r {
                    out.push(*eid);
                }
            }
        }
    };
    let parse_si_param = |param: &Parameter| -> Option<(f64, String)> {
        let Parameter::List(params) = param else {
            return None;
        };
        let prefix = match params.first() {
            Some(Parameter::Enumeration(e)) => e.as_str(),
            _ => "",
        };
        let uname = match params.get(1) {
            Some(Parameter::Enumeration(e)) => e.as_str(),
            _ => "",
        };
        si_scale(prefix, uname)
    };

    let mut si_units = HashMap::<u64, (f64, String)>::new();
    let mut conv_units = Vec::<(u64, f64, u64)>::new();
    let mut length_ids = Vec::<u64>::new();
    let mut global_unit_refs = Vec::<u64>::new();

    for ent in &data.entities {
        match ent {
            EntityInstance::Simple { id, record } => {
                if record.name.eq_ignore_ascii_case("GLOBAL_UNIT_ASSIGNED_CONTEXT") {
                    push_unit_refs(&record.parameter, &mut global_unit_refs);
                } else if record.name.eq_ignore_ascii_case("LENGTH_MEASURE_WITH_UNIT")
                    && let Parameter::List(params) = &record.parameter
                    && params.len() >= 2
                {
                    let val = match &params[0] {
                        Parameter::Typed { parameter, .. } => match parameter.as_ref() {
                            Parameter::Real(x) => Some(*x),
                            Parameter::Integer(x) => Some(*x as f64),
                            _ => None,
                        },
                        Parameter::Real(x) => Some(*x),
                        Parameter::Integer(x) => Some(*x as f64),
                        _ => None,
                    };
                    let unit_ref = match &params[1] {
                        Parameter::Ref(P21Name::Entity(uid)) => Some(*uid),
                        _ => None,
                    };
                    if let (Some(v), Some(u)) = (val, unit_ref)
                        && v.is_finite()
                        && v > 0.0
                    {
                        conv_units.push((*id, v, u));
                    }
                } else if record.name.eq_ignore_ascii_case("SI_UNIT")
                    && let Some(sc) = parse_si_param(&record.parameter)
                {
                    si_units.insert(*id, sc);
                }
            }
            EntityInstance::Complex {
                id,
                subsuper: SubSuperRecord(recs),
            } => {
                let is_len = recs.iter().any(|r| r.name.eq_ignore_ascii_case("LENGTH_UNIT"));
                for r in recs {
                    if r.name.eq_ignore_ascii_case("SI_UNIT")
                        && let Some(sc) = parse_si_param(&r.parameter)
                    {
                        si_units.insert(*id, sc);
                        if is_len {
                            length_ids.push(*id);
                        }
                    } else if r.name.eq_ignore_ascii_case("GLOBAL_UNIT_ASSIGNED_CONTEXT") {
                        push_unit_refs(&r.parameter, &mut global_unit_refs);
                    }
                }
            }
        }
    }

    // Conversion-based length units (e.g. INCH = 25.4 * MILLIMETRE, FOOT, MIL, etc.)
    for ent in &data.entities {
        let EntityInstance::Complex {
            id,
            subsuper: SubSuperRecord(recs),
        } = ent
        else {
            continue;
        };
        if !recs.iter().any(|r| r.name.eq_ignore_ascii_case("LENGTH_UNIT")) {
            continue;
        }
        let Some(cbu) = recs.iter().find(|r| r.name.eq_ignore_ascii_case("CONVERSION_BASED_UNIT")) else {
            continue;
        };
        let Parameter::List(params) = &cbu.parameter else {
            continue;
        };
        let cbu_name = match params.first() {
            Some(Parameter::String(s)) => s.to_ascii_lowercase(),
            _ => "unit".to_string(),
        };
        if let Some(Parameter::Ref(P21Name::Entity(lmwu_id))) = params.get(1)
            && let Some((_, fac, target_si)) = conv_units.iter().find(|x| x.0 == *lmwu_id)
        {
            let base = si_units.get(target_si).map(|x| x.0).unwrap_or(1.0);
            let total = fac * base;
            if total.is_finite()
                && total > 0.0
                && (global_unit_refs.is_empty() || global_unit_refs.contains(id))
            {
                return Ok((total, cbu_name, total_entities));
            }
        }
    }

    for eid in &global_unit_refs {
        if let Some((sc, lbl)) = si_units.get(eid) {
            return Ok((*sc, lbl.clone(), total_entities));
        }
    }
    for eid in &length_ids {
        if let Some((sc, lbl)) = si_units.get(eid) {
            return Ok((*sc, lbl.clone(), total_entities));
        }
    }
    Ok((1.0, "mm".into(), total_entities))
}

fn check_cancel(cancel: Option<&AtomicBool>) -> Result<()> {
    if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
        bail!("STEP import cancelled");
    }
    Ok(())
}

fn is_generic_part_name(s: &str) -> bool {
    let t = s.trim();
    t.is_empty()
        || t.eq_ignore_ascii_case("NONE")
        || t.eq_ignore_ascii_case("PART")
        || t.eq_ignore_ascii_case("COMPOUND")
        || t.eq_ignore_ascii_case("SOLID")
        || t.contains("Open CASCADE STEP translator")
}

struct ExtractedSolid {
    name: String,
    xform: Matrix4,
    solid: CompressedSolid<Point3, monstertruck_io::step::load::step_geometry::Curve3D, StepSurface>,
}

fn extract_shell_as_solid(
    table: &Table,
    shell_idx: u64,
) -> Option<CompressedSolid<Point3, monstertruck_io::step::load::step_geometry::Curve3D, StepSurface>> {
    let sh = if let Some(s) = table.shell.get(&shell_idx) {
        table.to_compressed_shell(s).ok()?
    } else {
        let s = table.oriented_shell.get(&shell_idx)?;
        table.to_compressed_shell(s).ok()?
    };
    Some(CompressedSolid {
        boundaries: vec![sh],
        id_allocator: None,
        attributes: None,
    })
}

fn extract_item_solids(
    table: &Table,
    item_ids: &[u64],
    base_name: &str,
    xform: Matrix4,
    out: &mut Vec<ExtractedSolid>,
) -> Result<()> {
    for &id in item_ids {
        if let Some(brep) = table.manifold_solid_brep.get(&id) {
            if out.len() >= MAX_STEP_BODIES {
                bail!("STEP file has too many solid bodies (limit {MAX_STEP_BODIES})");
            }
            let solid = table
                .to_compressed_solid(brep)
                .map_err(|e| anyhow!("converting STEP solid #{id}: {e}"))?;
            out.push(ExtractedSolid {
                name: base_name.to_string(),
                xform,
                solid,
            });
        } else if let Some(sbsm) = table.shell_based_surface_model.get(&id) {
            let shells = table
                .to_compressed_shells(sbsm)
                .map_err(|e| anyhow!("converting STEP shell model #{id}: {e}"))?;
            if !shells.is_empty() {
                if out.len() >= MAX_STEP_BODIES {
                    bail!("STEP file has too many solid bodies (limit {MAX_STEP_BODIES})");
                }
                out.push(ExtractedSolid {
                    name: base_name.to_string(),
                    xform,
                    solid: CompressedSolid {
                        boundaries: shells,
                        id_allocator: None,
                        attributes: None,
                    },
                });
            }
        }
    }
    Ok(())
}

fn collect_step_solids(table: &Table, default_stem: &str, cancel: Option<&AtomicBool>) -> Result<Vec<ExtractedSolid>> {
    let mut out = Vec::new();
    let pick_name = |a: &str, b: &str| -> String {
        if !is_generic_part_name(a) {
            a.trim().to_string()
        } else if !is_generic_part_name(b) {
            b.trim().to_string()
        } else {
            default_stem.to_string()
        }
    };

    if let Ok(assy) = table.step_assy() {
        let mut top: Vec<_> = assy.top_nodes().map(|n| n.index()).collect();
        if top.is_empty() {
            top = assy.all_nodes().map(|n| n.index()).collect();
        }
        top.sort_by(|&a, &b| {
            let (na, nb) = (assy.node(a).attributes(), assy.node(b).attributes());
            na.name.cmp(&nb.name).then_with(|| na.id.cmp(&nb.id))
        });
        // Iterative DFS to collect node/instance transforms along the assembly DAG.
        let mut stack: Vec<(_, Matrix4, String, usize)> = top
            .into_iter()
            .rev()
            .map(|ni| (ni, Matrix4::identity(), String::new(), 0))
            .collect();
        while let Some((ni, mat, edge_name, depth)) = stack.pop() {
            check_cancel(cancel)?;
            if depth > 64 {
                bail!("STEP assembly nesting depth exceeded limit");
            }
            let node = assy.node(ni);
            let name = pick_name(&edge_name, &node.attributes().name);

            let mut item_ids: Vec<u64> = node.shape().clone();
            if let Some(rep_id) = table.shape_representation_of_node(node.entity()) {
                let (rel_ids, _) = table.solids_via_shape_relationship(rep_id);
                for rid in rel_ids {
                    if !item_ids.contains(&rid) {
                        item_ids.push(rid);
                    }
                }
            }
            item_ids.sort_unstable();
            extract_item_solids(table, &item_ids, &name, mat, &mut out)?;

            let mut edges: Vec<_> = node.edges().collect();
            edges.sort_by(|a, b| {
                let ca = pick_name(&a.attributes().name, &assy.node(a.nodes().1).attributes().name);
                let cb = pick_name(&b.attributes().name, &assy.node(b.nodes().1).attributes().name);
                let ia = &a.attributes().id;
                let ib = &b.attributes().id;
                ca.cmp(&cb).then_with(|| ia.cmp(ib))
            });
            for edge in edges.into_iter().rev() {
                let emat = Matrix4::try_from(&edge.entity().matrix).unwrap_or_else(|_| Matrix4::identity());
                let child = edge.nodes().1;
                let cname = pick_name(&edge.attributes().name, &assy.node(child).attributes().name);
                stack.push((child, mat * emat, cname, depth + 1));
            }
        }
    }

    if out.is_empty() {
        // Fallback for standalone or partial STEP files without complete product links.
        let mut ids: Vec<u64> = table.manifold_solid_brep.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            check_cancel(cancel)?;
            if out.len() >= MAX_STEP_BODIES {
                bail!("STEP file has too many solid bodies (limit {MAX_STEP_BODIES})");
            }
            let brep = &table.manifold_solid_brep[&id];
            let solid = table
                .to_compressed_solid(brep)
                .map_err(|e| anyhow!("converting STEP solid #{id}: {e}"))?;
            out.push(ExtractedSolid {
                name: default_stem.to_string(),
                xform: Matrix4::identity(),
                solid,
            });
        }
    }

    if out.is_empty() {
        let mut sids: Vec<u64> = table.shell_based_surface_model.keys().copied().collect();
        sids.sort_unstable();
        for sid in sids {
            check_cancel(cancel)?;
            extract_item_solids(table, &[sid], default_stem, Matrix4::identity(), &mut out)?;
        }
    }

    if out.is_empty() {
        let mut sh_ids: Vec<u64> = table.shell.keys().copied().collect();
        sh_ids.sort_unstable();
        for sh_id in sh_ids {
            check_cancel(cancel)?;
            if let Some(solid) = extract_shell_as_solid(table, sh_id) {
                if out.len() >= MAX_STEP_BODIES {
                    bail!("STEP file has too many solid bodies (limit {MAX_STEP_BODIES})");
                }
                out.push(ExtractedSolid {
                    name: default_stem.to_string(),
                    xform: Matrix4::identity(),
                    solid,
                });
            }
        }
    }

    if out.is_empty() {
        bail!("no solid bodies found in STEP file");
    }
    Ok(out)
}

fn transform_pt(m: &Matrix4, p: Point3, scale_mm: f64, offset: V3) -> Result<V3> {
    let q = m.transform_point(p);
    let out = [
        q.x * scale_mm + offset[0],
        q.y * scale_mm + offset[1],
        q.z * scale_mm + offset[2],
    ];
    if !out.iter().all(|c| c.is_finite()) {
        bail!("STEP geometry contains non-finite coordinates");
    }
    Ok(out)
}

fn transform_dir(m: &Matrix4, v: Vector3) -> V3 {
    let q = m.transform_vector(v);
    let out = [q.x, q.y, q.z];
    let l = norm(out);
    if l.is_finite() && l > 1e-12 {
        scale(out, 1.0 / l)
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn world_matrix(xform: &Matrix4, scale_mm: f64, offset: V3) -> Matrix4 {
    let t = Matrix4::from_translation(Vector3::new(offset[0], offset[1], offset[2]));
    let s = Matrix4::from_scale(scale_mm);
    t * s * (*xform)
}

fn signed_shell_volume(tris: &[([V3; 3], usize)]) -> f64 {
    let mut v = 0.0;
    for (t, _) in tris {
        let (a, b, c) = (t[0], t[1], t[2]);
        v += a[0] * (b[1] * c[2] - b[2] * c[1])
            + a[1] * (b[2] * c[0] - b[0] * c[2])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    v / 6.0
}

fn extract_face_analytic(
    surf: &StepSurface,
    xform: &Matrix4,
    scale_mm: f64,
    offset: V3,
) -> (StepSurfaceKind, StepSurfaceAnalytic) {
    let map_p = |p: Point3| -> V3 {
        let q = xform.transform_point(p);
        [q.x * scale_mm + offset[0], q.y * scale_mm + offset[1], q.z * scale_mm + offset[2]]
    };
    let map_v = |v: Vector3| transform_dir(xform, v);
    match surf {
        StepSurface::ElementarySurface(es) => match es {
            ElementarySurface::Plane(pl) => {
                let origin = map_p(pl.origin());
                let normal = map_v(pl.normal());
                (StepSurfaceKind::Plane, StepSurfaceAnalytic::Plane { origin, normal })
            }
            ElementarySurface::Sphere(sp) => {
                let m = sp.transform();
                let inner = &sp.entity().0;
                let center = map_p(m.transform_point(inner.center()));
                let axis_len = m.transform_vector(Vector3::unit_x()).magnitude();
                let radius = inner.radius() * axis_len * scale_mm;
                if !center.iter().all(|c| c.is_finite()) || !radius.is_finite() || radius <= 0.0 {
                    return (StepSurfaceKind::Mesh, StepSurfaceAnalytic::Mesh);
                }
                (
                    StepSurfaceKind::Sphere,
                    StepSurfaceAnalytic::Sphere { center, radius, outward: 1.0 },
                )
            }
            ElementarySurface::CylindricalSurface(cyl) => {
                let rev = cyl.entity();
                let mat = cyl.transform();
                let p0 = mat.transform_point(rev.subs(0.0, 0.0));
                let p0p = mat.transform_point(rev.subs(0.0, PI));
                let p1 = mat.transform_point(rev.subs(1.0, 0.0));
                let p1p = mat.transform_point(rev.subs(1.0, PI));
                let c0 = Point3::from_vec(0.5 * (p0.to_vec() + p0p.to_vec()));
                let c1 = Point3::from_vec(0.5 * (p1.to_vec() + p1p.to_vec()));
                let origin = map_p(c0);
                let radius = norm(sub(map_p(p0), origin));
                let axis = map_v(c1 - c0);
                if !origin.iter().all(|c| c.is_finite()) || !radius.is_finite() || radius <= 0.0 {
                    return (StepSurfaceKind::Mesh, StepSurfaceAnalytic::Mesh);
                }
                (
                    StepSurfaceKind::Cylinder,
                    StepSurfaceAnalytic::Cylinder { origin, axis, radius, outward: 1.0 },
                )
            }
            ElementarySurface::ConicalSurface(cone) => {
                let rev = cone.entity();
                let mat = cone.transform();
                let p00 = mat.transform_point(rev.subs(0.0, 0.0));
                let p0p = mat.transform_point(rev.subs(0.0, PI));
                let p10 = mat.transform_point(rev.subs(1.0, 0.0));
                let p1p = mat.transform_point(rev.subs(1.0, PI));
                let c0 = Point3::from_vec(0.5 * (p00.to_vec() + p0p.to_vec()));
                let c1 = Point3::from_vec(0.5 * (p10.to_vec() + p1p.to_vec()));
                let m0 = map_p(c0);
                let m1 = map_p(c1);
                let r0 = norm(sub(map_p(p00), m0));
                let r1 = norm(sub(map_p(p10), m1));
                let d_ax = sub(m1, m0);
                let dz = norm(d_ax);
                if dz < 1e-12 {
                    return (StepSurfaceKind::Mesh, StepSurfaceAnalytic::Mesh);
                }
                if (r1 - r0).abs() <= 1e-9 * dz {
                    let axis = scale(d_ax, 1.0 / dz);
                    return (
                        StepSurfaceKind::Cylinder,
                        StepSurfaceAnalytic::Cylinder { origin: m0, axis, radius: 0.5 * (r0 + r1), outward: 1.0 },
                    );
                }
                // Orient axis in widening direction (r_hi > r_lo) so apex sits below m_lo.
                let (p_lo, r_lo, p_hi, r_hi) = if r1 >= r0 { (m0, r0, m1, r1) } else { (m1, r1, m0, r0) };
                let axis = normalize(sub(p_hi, p_lo));
                let tan_a = (r_hi - r_lo) / dz;
                let apex = sub(p_lo, scale(axis, r_lo / tan_a));
                let cos_a = 1.0 / (1.0 + tan_a * tan_a).sqrt();
                let sin_a = tan_a * cos_a;
                (
                    StepSurfaceKind::Cone,
                    StepSurfaceAnalytic::Cone { apex, axis, cos_a, sin_a, outward: 1.0 },
                )
            }
            ElementarySurface::ToroidalSurface(tor) => {
                let m = tor.transform();
                let t = tor.entity();
                let center = map_p(m.transform_point(t.center()));
                let s = m.transform_vector(Vector3::unit_x()).magnitude() * scale_mm;
                let major_r = t.large_radius() * s;
                let minor_r = t.small_radius() * s;
                let axis = map_v(m.transform_vector(Vector3::unit_z()));
                if !major_r.is_finite() || !minor_r.is_finite() || major_r <= 0.0 || minor_r <= 0.0 {
                    return (StepSurfaceKind::Mesh, StepSurfaceAnalytic::Mesh);
                }
                (
                    StepSurfaceKind::Torus,
                    StepSurfaceAnalytic::Torus { center, axis, major_r, minor_r, outward: 1.0 },
                )
            }
        },
        StepSurface::SweepSurface(_) | StepSurface::BsplineSurface(_) | StepSurface::NurbsSurface(_) => {
            let to_world = world_matrix(xform, scale_mm, offset);
            let Some(to_raw) = to_world.invert() else {
                return (StepSurfaceKind::Mesh, StepSurfaceAnalytic::Mesh);
            };
            (
                StepSurfaceKind::Parametric,
                StepSurfaceAnalytic::Parametric(Arc::new(ParametricStepSurface {
                    surface: surf.clone(),
                    to_world,
                    to_raw,
                    outward_sign: 1.0,
                })),
            )
        }
    }
}

fn orient_analytic(analytic: &mut StepSurfaceAnalytic, tri_center: V3, mesh_normal: V3) {
    match analytic {
        StepSurfaceAnalytic::Plane { normal, .. } => {
            if dot(*normal, mesh_normal) < 0.0 {
                *normal = scale(*normal, -1.0);
            }
        }
        StepSurfaceAnalytic::Cylinder { .. }
        | StepSurfaceAnalytic::Sphere { .. }
        | StepSurfaceAnalytic::Cone { .. }
        | StepSurfaceAnalytic::Torus { .. } => {
            if let Some(an) = analytic.normal_at(tri_center)
                && dot(an, mesh_normal) < 0.0
            {
                match analytic {
                    StepSurfaceAnalytic::Cylinder { outward, .. }
                    | StepSurfaceAnalytic::Sphere { outward, .. }
                    | StepSurfaceAnalytic::Cone { outward, .. }
                    | StepSurfaceAnalytic::Torus { outward, .. } => *outward = -*outward,
                    _ => {}
                }
            }
        }
        StepSurfaceAnalytic::Parametric(ps) => {
            if let Some(an) = ps.normal_at(tri_center)
                && dot(an, mesh_normal) < 0.0
                && let Some(inner) = Arc::get_mut(ps)
            {
                inner.outward_sign = -inner.outward_sign;
            }
        }
        StepSurfaceAnalytic::Mesh => {}
    }
}

fn tessellation_tol_raw(sh: &CompressedShell<Point3, monstertruck_io::step::load::step_geometry::Curve3D, StepSurface>) -> f64 {
    let mut bdb = BoundingBox::<Point3>::new();
    for v in &sh.vertices {
        bdb.push(*v);
    }
    let diag = bdb.max() - bdb.min();
    let max_d = diag.x.max(diag.y).max(diag.z);
    if !max_d.is_finite() || max_d <= 0.0 {
        return 0.01;
    }
    let floor_d = max_d * 0.02;
    let min_d = diag.x.max(floor_d).min(diag.y.max(floor_d)).min(diag.z.max(floor_d));
    (0.002 * min_d).min(0.0005 * max_d).max(1e-7 * max_d)
}

fn build_step_part(
    part_index: usize,
    ext: &ExtractedSolid,
    scale_mm: f64,
    offset: V3,
    unit_name: &str,
    unit_scale_mm: f64,
    cancel: Option<&AtomicBool>,
) -> Result<StepData> {
    let mut all_tris = Vec::<[V3; 3]>::new();
    let mut all_normals = Vec::<V3>::new();
    let mut all_seg = Vec::<u32>::new();
    let mut faces = Vec::<StepFace>::new();
    let mut total_vol = 0.0;

    for (sh_idx, sh) in ext.solid.boundaries.iter().enumerate() {
        check_cancel(cancel)?;
        let tol = tessellation_tol_raw(sh);
        let poly = sh.robust_triangulation(tol);
        let mut sh_tris: Vec<([V3; 3], usize)> = Vec::new();
        let face_base = faces.len();

        for (fi, (f_orig, f_poly)) in sh.faces.iter().zip(poly.faces.iter()).enumerate() {
            let local_idx = face_base + fi;
            let (kind, analytic) = extract_face_analytic(&f_orig.surface, &ext.xform, scale_mm, offset);
            faces.push(StepFace {
                name: format!("face{local_idx}_{}", kind.tag()),
                kind,
                analytic,
            });
            if let Some(m) = &f_poly.surface {
                let pos = m.positions();
                for t in m.tri_faces() {
                    if all_tris.len() + sh_tris.len() >= MAX_STEP_TRIS {
                        bail!("STEP tessellation exceeded {MAX_STEP_TRIS} triangles");
                    }
                    let a = transform_pt(&ext.xform, pos[t[0].pos], scale_mm, offset)?;
                    let b = transform_pt(&ext.xform, pos[t[1].pos], scale_mm, offset)?;
                    let c = transform_pt(&ext.xform, pos[t[2].pos], scale_mm, offset)?;
                    let ab = sub(b, a);
                    let ac = sub(c, a);
                    let cr = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
                    if norm(cr) > 1e-18 {
                        sh_tris.push(([a, b, c], local_idx));
                    }
                }
            }
        }

        if sh_tris.is_empty() {
            continue;
        }
        let mut sv = signed_shell_volume(&sh_tris);
        // Shell 0 is the outer boundary (positive signed volume); subsequent shells are voids (negative).
        let want_positive = sh_idx == 0;
        if (want_positive && sv < 0.0) || (!want_positive && sv > 0.0) {
            for (tri, _) in &mut sh_tris {
                tri.swap(1, 2);
            }
            sv = -sv;
        }
        total_vol += sv;

        let mut face_nsum: HashMap<usize, (V3, V3)> = HashMap::new();
        for (tri, fidx) in sh_tris {
            let ab = sub(tri[1], tri[0]);
            let ac = sub(tri[2], tri[0]);
            let cr = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let l = norm(cr);
            let n = scale(cr, 1.0 / l);
            let ctr = scale(add(tri[0], add(tri[1], tri[2])), 1.0 / 3.0);
            face_nsum
                .entry(fidx)
                .and_modify(|(best_c, nsum)| {
                    *nsum = add(*nsum, cr);
                    let _ = best_c;
                })
                .or_insert((ctr, cr));
            all_tris.push(tri);
            all_normals.push(n);
            all_seg.push(fidx as u32);
        }
        let _ = face_nsum;
    }

    if all_tris.is_empty() || faces.is_empty() {
        bail!("STEP solid '{}' produced no triangles on tessellation", ext.name);
    }
    let mut mesh = StlData::from_segmented(all_tris, all_normals, all_seg, faces.len())?;

    // Orient every B-rep face's triangles & analytic surface strictly outward from the solid
    // using winding-independent ray-crossing parity (fixes faces whose kernel tessellation
    // kept an inverted planar loop, e.g. at z = 0).
    let span = sub(mesh.max, mesh.min);
    let max_d = span[0].max(span[1]).max(span[2]).max(1e-6);
    let min_d = span[0].max(max_d * 0.01).min(span[1].max(max_d * 0.01)).min(span[2].max(max_d * 0.01));
    let eps = (1e-3 * min_d).max(1e-5 * max_d);
    let inside_at = |p: V3| -> bool {
        let xs = mesh.x_crossings(p[1] + 1.37e-6 * max_d, p[2] + 0.91e-6 * max_d);
        xs.iter().filter(|&&q| q < p[0]).count() % 2 == 1
    };

    let mut tris_by_face: Vec<Vec<usize>> = vec![Vec::new(); faces.len()];
    for (ti, &seg) in mesh.segment.iter().enumerate() {
        if let Some(v) = tris_by_face.get_mut(seg as usize) {
            v.push(ti);
        }
    }
    let mut flip_face = vec![false; faces.len()];
    let mut best_tri_of_face = vec![None::<usize>; faces.len()];
    for (fidx, t_indices) in tris_by_face.iter().enumerate() {
        if t_indices.is_empty() {
            continue;
        }
        let mut ranked: Vec<(f64, usize)> = t_indices
            .iter()
            .map(|&ti| {
                let t = &mesh.tris[ti];
                let ab = sub(t[1], t[0]);
                let ac = sub(t[2], t[0]);
                let cr = [
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                ];
                (dot(cr, cr), ti)
            })
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        best_tri_of_face[fidx] = ranked.first().map(|x| x.1);
        let mut outward_votes = 0i32;
        for &(_, ti) in ranked.iter().take(8) {
            let t = &mesh.tris[ti];
            let ctr = scale(add(t[0], add(t[1], t[2])), 1.0 / 3.0);
            let n = mesh.normals[ti];
            let in_plus = inside_at(add(ctr, scale(n, eps)));
            let in_minus = inside_at(sub(ctr, scale(n, eps)));
            if in_minus && !in_plus {
                outward_votes += 1;
            } else if in_plus && !in_minus {
                outward_votes -= 1;
            }
        }
        if outward_votes < 0 {
            flip_face[fidx] = true;
        }
    }
    for (fidx, t_indices) in tris_by_face.iter().enumerate() {
        if flip_face[fidx] {
            for &ti in t_indices {
                mesh.tris[ti].swap(1, 2);
                mesh.normals[ti] = scale(mesh.normals[ti], -1.0);
            }
        }
        if let Some(best_ti) = best_tri_of_face[fidx] {
            let t = &mesh.tris[best_ti];
            let ctr = scale(add(t[0], add(t[1], t[2])), 1.0 / 3.0);
            let n_out = mesh.normals[best_ti];
            if let Some(face) = faces.get_mut(fidx) {
                orient_analytic(&mut face.analytic, ctr, n_out);
            }
        }
    }

    Ok(StepData {
        mesh,
        faces,
        part_name: ext.name.clone(),
        part_index,
        unit_name: unit_name.to_string(),
        unit_scale_mm,
        volume_mm3: total_vol.abs(),
    })
}

/// Load all solid bodies from a STEP file (`.step` / `.stp`), converting STEP length units
/// into millimetres and applying `(user_scale, offset)`.
pub fn load_step_bodies(
    path: &str,
    user_scale: f64,
    offset: V3,
    cancel: Option<&AtomicBool>,
) -> Result<Vec<StepData>> {
    if !user_scale.is_finite() || user_scale <= 0.0 || !offset.iter().all(|v| v.is_finite()) {
        bail!("STEP scale must be finite and positive, and offset must be finite");
    }
    let meta = std::fs::metadata(path).with_context(|| format!("reading STEP '{path}'"))?;
    if meta.len() > MAX_STEP_BYTES {
        bail!(
            "STEP '{path}' is too large ({} MB; limit {} MB)",
            meta.len() / (1024 * 1024),
            MAX_STEP_BYTES / (1024 * 1024)
        );
    }
    check_cancel(cancel)?;
    let text = std::fs::read_to_string(path).with_context(|| format!("reading STEP '{path}'"))?;
    check_cancel(cancel)?;
    let (unit_scale_mm, unit_name, _) = detect_length_unit_mm(&text)?;
    let total_scale = unit_scale_mm * user_scale;
    if !total_scale.is_finite() || total_scale <= 0.0 {
        bail!("invalid effective STEP scale ({total_scale})");
    }
    let table = Table::from_step(&text).map_err(|e| anyhow!("parsing STEP '{path}': {e}"))?;
    let stem = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Part");
    let raw_solids = collect_step_solids(&table, stem, cancel)?;
    let mut parts = Vec::with_capacity(raw_solids.len());
    let mut seen_names = HashSet::<String>::new();
    for (idx, mut ext) in raw_solids.into_iter().enumerate() {
        check_cancel(cancel)?;
        let base: String = ext
            .name
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect();
        let base = if base.is_empty() { format!("{stem}_{}", idx + 1) } else { base };
        let unique = if seen_names.insert(base.to_ascii_lowercase()) {
            base
        } else {
            let mut n = 2usize;
            loop {
                let cand = format!("{base}_{n}");
                if seen_names.insert(cand.to_ascii_lowercase()) {
                    break cand;
                }
                n += 1;
            }
        };
        ext.name = unique;
        parts.push(build_step_part(
            idx,
            &ext,
            total_scale,
            offset,
            &unit_name,
            unit_scale_mm,
            cancel,
        )?);
    }
    Ok(parts)
}

/// Load a specific part by index (`part_index`) from a STEP file.
pub fn load_step_part(path: &str, part_index: usize, user_scale: f64, offset: V3) -> Result<StepData> {
    let mut parts = load_step_bodies(path, user_scale, offset, None)?;
    if part_index >= parts.len() {
        bail!(
            "STEP '{path}' only has {} part(s) (requested part_index {part_index})",
            parts.len()
        );
    }
    Ok(parts.swap_remove(part_index))
}
