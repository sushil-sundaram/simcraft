//! Self-contained HTML simulation reports (Ansys Dynamic Reporting style): model, materials,
//! loads, mesh, rendered views, key results, pass/fail checks, studies and charts. Images are
//! embedded as base64 PNGs and charts as inline SVG, so the file works offline and prints to PDF.

use crate::assistant::Plan;
use crate::geometry::{Op, Role, Shape};
use crate::mesh::Mesh;
use crate::model::Model;
use crate::render::{self, Camera, Display};
use crate::results::{Field, Results};
use crate::study::{Constraint, Study};
use anyhow::Result;
use base64::Engine;
use std::collections::HashSet;
use std::fmt::Write;

pub struct ReportInput<'a> {
    pub title: String,
    pub model: &'a Model,
    pub mesh: &'a Mesh,
    pub results: Option<&'a Results>,
    pub display: &'a Display,
    pub study: Option<&'a Study>,
    pub plan: Option<&'a Plan>,
    pub checks: Vec<Constraint>,
    pub project: Option<String>,
    pub stale: bool,
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// Compact number: 4 significant figures, no padding zeros.
fn num(v: f64) -> String {
    if v == 0.0 || !v.is_finite() {
        return if v.is_finite() { "0".into() } else { "—".into() };
    }
    let a = v.abs();
    if !(1e-3..1e6).contains(&a) {
        return format!("{v:.3e}");
    }
    let decimals = (3 - a.log10().floor() as i32).max(0) as usize;
    let s = format!("{v:.decimals$}");
    let s = if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    esc(&s)
}

/// UTC date "YYYY-MM-DD HH:MM" from the system clock (no date crate needed).
fn now_utc() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} UTC", rem / 3600, (rem % 3600) / 60)
}

fn shape_desc(s: &Shape) -> String {
    let v = |a: &[f64; 3]| format!("{}, {}, {}", num(a[0]), num(a[1]), num(a[2]));
    match s {
        Shape::Box { min, size } => format!("Box — corner ({}) mm, size ({}) mm", v(min), v(size)),
        Shape::Cylinder { base, axis, radius, height } => format!("Cylinder — base ({}) mm, axis {:?}, r = {} mm, h = {} mm", v(base), axis, num(*radius), num(*height)),
        Shape::Sphere { center, radius } => format!("Sphere — centre ({}) mm, r = {} mm", v(center), num(*radius)),
        Shape::Stl { path, scale, .. } => format!("STL — {} (scale {})", esc(path), num(*scale)),
        Shape::Pcb { path, data, .. } => match data {
            Some(d) => format!("KiCad board — {} ({} × {} × {} mm, {} copper layers)", esc(path), num(d.size[0]), num(d.size[1]), num(d.thickness), d.layers.len()),
            None => format!("KiCad board — {}", esc(path)),
        },
    }
}

fn png_tag(png: &[u8], caption: &str) -> String {
    format!(
        "<figure><img alt=\"{c}\" src=\"data:image/png;base64,{}\"><figcaption>{c}</figcaption></figure>",
        base64::engine::general_purpose::STANDARD.encode(png),
        c = esc(caption)
    )
}

/// Minimal inline SVG line/scatter chart.
pub fn svg_chart(series: &[(f64, f64)], xlabel: &str, ylabel: &str, lines: bool) -> String {
    if series.len() < 2 {
        return String::new();
    }
    let (w, h, l, r, t, b) = (640.0, 300.0, 70.0, 20.0, 16.0, 46.0);
    let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for (x, y) in series {
        x0 = x0.min(*x);
        x1 = x1.max(*x);
        y0 = y0.min(*y);
        y1 = y1.max(*y);
    }
    if (x1 - x0).abs() < 1e-30 {
        x1 = x0 + 1.0;
    }
    if (y1 - y0).abs() < 1e-30 * y1.abs().max(1.0) {
        y0 -= 0.5;
        y1 += 0.5;
    }
    let pad = (y1 - y0) * 0.06;
    let (y0, y1) = (y0 - pad, y1 + pad);
    let px = |x: f64| l + (x - x0) / (x1 - x0) * (w - l - r);
    let py = |y: f64| t + (1.0 - (y - y0) / (y1 - y0)) * (h - t - b);
    let mut s = format!("<svg class=\"chart\" viewBox=\"0 0 {w} {h}\" role=\"img\" aria-label=\"{} vs {}\">", esc(ylabel), esc(xlabel));
    for i in 0..=4 {
        let fy = y0 + (y1 - y0) * i as f64 / 4.0;
        let yy = py(fy);
        let _ = write!(s, "<line x1=\"{l}\" y1=\"{yy:.1}\" x2=\"{}\" y2=\"{yy:.1}\" class=\"grid\"/><text x=\"{}\" y=\"{:.1}\" class=\"tick\" text-anchor=\"end\">{}</text>", w - r, l - 6.0, yy + 4.0, num(fy));
        let fx = x0 + (x1 - x0) * i as f64 / 4.0;
        let xx = px(fx);
        let _ = write!(s, "<text x=\"{xx:.1}\" y=\"{}\" class=\"tick\" text-anchor=\"middle\">{}</text>", h - b + 16.0, num(fx));
    }
    let _ = write!(s, "<line x1=\"{l}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"axis\"/><line x1=\"{l}\" y1=\"{t}\" x2=\"{l}\" y2=\"{}\" class=\"axis\"/>", h - b, w - r, h - b, h - b);
    let mut pts: Vec<(f64, f64)> = series.to_vec();
    if lines {
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let d: Vec<String> = pts.iter().map(|(x, y)| format!("{:.1},{:.1}", px(*x), py(*y))).collect();
        let _ = write!(s, "<polyline points=\"{}\" class=\"line\"/>", d.join(" "));
    }
    if pts.len() <= 60 || !lines {
        for (x, y) in &pts {
            let _ = write!(s, "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3.2\" class=\"dot\"/>", px(*x), py(*y));
        }
    }
    let _ = write!(s, "<text x=\"{}\" y=\"{}\" class=\"label\" text-anchor=\"middle\">{}</text>", (l + w - r) / 2.0, h - 8.0, esc(xlabel));
    let _ = write!(s, "<text x=\"14\" y=\"{}\" class=\"label\" text-anchor=\"middle\" transform=\"rotate(-90 14 {})\">{}</text>", (t + h - b) / 2.0, (t + h - b) / 2.0, esc(ylabel));
    s + "</svg>"
}

const CSS: &str = r#"
:root{--ink:#1d2230;--weak:#5d6577;--line:#e3e6ec;--accent:#3f6df2;--ok:#16834a;--bad:#c4302b;--warn:#b26a00;--bg:#ffffff;--card:#f7f8fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:32px 20px 60px}h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--line)}
.meta{color:var(--weak);font-size:13px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.card .k{color:var(--weak);font-size:12px;text-transform:uppercase;letter-spacing:.04em}.card .v{font-size:20px;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--weak);font-weight:600;background:var(--card)}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.pass{color:var(--ok);font-weight:700}.fail{color:var(--bad);font-weight:700}.warn{color:var(--warn)}
.best td{background:#eaf6ef;font-weight:600}figure{margin:0}.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px}
img{width:100%;border:1px solid var(--line);border-radius:8px}figcaption{color:var(--weak);font-size:12.5px;margin-top:4px}
.chart{width:100%;max-width:640px;height:auto}.chart .grid{stroke:#eef0f4}.chart .axis{stroke:#9aa1b0}.chart .tick{font-size:11px;fill:#5d6577}.chart .label{font-size:12px;fill:#1d2230}
.chart .line{fill:none;stroke:var(--accent);stroke-width:2}.chart .dot{fill:var(--accent)}footer{margin-top:40px;color:var(--weak);font-size:12px}
@media print{main{padding:0}h2{break-after:avoid}figure,table,.chart{break-inside:avoid}}
"#;

pub fn build_html(inp: &ReportInput) -> Result<String> {
    let m = inp.model;
    let mesh = inp.mesh;
    let mut h = String::new();
    let _ = write!(h, "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{}</title><style>{CSS}</style></head><body><main>", esc(&inp.title));
    let _ = write!(h, "<h1>{}</h1><div class=\"meta\">{} · {} · generated {} by SimCraft {}</div>", esc(&inp.title), esc(m.analysis.kind.label()), esc(inp.project.as_deref().unwrap_or("unsaved project")), now_utc(), env!("CARGO_PKG_VERSION"));
    if inp.stale {
        h += "<p class=\"warn\">⚠ The setup changed after these results were computed; solve again before relying on them.</p>";
    }

    // summary cards
    let summary = inp.results.map(|r| r.summary(mesh));
    h += "<div class=\"cards\">";
    let card = |h: &mut String, k: &str, v: String| {
        let _ = write!(h, "<div class=\"card\"><div class=\"k\">{}</div><div class=\"v\">{}</div></div>", esc(k), v);
    };
    card(&mut h, "Analysis", esc(m.analysis.kind.label()));
    card(&mut h, "Elements", format!("{}", mesh.elems.len()));
    if let Some(r) = inp.results {
        card(&mut h, "Mass", format!("{} kg", num(r.info.mass_kg)));
        card(&mut h, "Solve time", format!("{} s", num(r.info.seconds)));
    }
    h += "</div>";

    // checks
    let mut checks = inp.checks.clone();
    let mut default_check = false;
    if checks.is_empty() && inp.results.is_some_and(|r| r.has(Field::SafetyFactor)) {
        checks.push(Constraint { response: "min:safety_factor".into(), op: ">=".into(), value: 1.0 });
        default_check = true;
    }
    if let (Some(sum), Some(r), false) = (&summary, inp.results, checks.is_empty()) {
        h += "<h2>Checks</h2><table><tr><th>Check</th><th>Limit</th><th>Result</th><th>Status</th></tr>";
        let mut all = true;
        for c in &checks {
            let v = crate::study::response(sum, r.info.mass_kg, mesh.volume_mm3(), &c.response);
            let (val, ok) = match v {
                Ok(x) => (num(x), if c.op.starts_with('<') { x <= c.value } else { x >= c.value }),
                Err(_) => ("—".into(), false),
            };
            all &= ok;
            let _ = write!(h, "<tr><td>{}</td><td class=\"n\">{} {}</td><td class=\"n\">{val}</td><td class=\"{}\">{}</td></tr>", esc(&c.response), esc(&c.op.replace("<=", "≤").replace(">=", "≥")), num(c.value), if ok { "pass" } else { "fail" }, if ok { "PASS" } else { "FAIL" });
        }
        h += "</table>";
        let _ = write!(h, "<p class=\"{}\">{}</p>", if all { "pass" } else { "fail" }, if all { "All checks passed." } else { "One or more checks failed." });
        if default_check {
            h += "<p class=\"meta\">Default check (yield safety factor ≥ 1). Pass your own limits to make_report to replace it.</p>";
        }
    }

    // views
    h += "<h2>Views</h2><div class=\"figs\">";
    let hl: HashSet<u32> = HashSet::new();
    let mut views: Vec<(Display, &str, String)> = Vec::new();
    let mut setup = inp.display.clone();
    setup.show_results = false;
    setup.section = None;
    setup.dark = false;
    views.push((setup, "iso", "Model, supports and loads".into()));
    if let Some(r) = inp.results {
        let mut d = inp.display.clone();
        d.dark = false;
        d.show_results = true;
        // structural results lead with stress; otherwise use the field shown in the app
        let f = if r.has(Field::VonMises) && r.modes.is_empty() {
            Field::VonMises
        } else if r.has(d.field) {
            d.field
        } else {
            r.available_fields().first().copied().unwrap_or(Field::TotalDeformation)
        };
        d.field = f;
        let label = format!("{} [{}]", f.label(), f.unit());
        views.push((d.clone(), "iso", format!("{label} — isometric")));
        views.push((d.clone(), "front", format!("{label} — front")));
        if r.has(Field::TotalDeformation) && f != Field::TotalDeformation && r.modes.is_empty() {
            let mut dd = d.clone();
            dd.field = Field::TotalDeformation;
            views.push((dd, "iso", "Total Deformation [mm] — isometric".into()));
        } else {
            views.push((d, "top", format!("{label} — top")));
        }
    }
    for (d, view, caption) in views {
        let scene = render::build_scene(mesh, m, inp.results, &d, &hl, 1.0);
        let mut cam = Camera::default();
        cam.set_view(view);
        cam.fit(scene.lo, scene.hi, 900.0, 560.0);
        let unit = if d.show_results { d.field.unit() } else { "" };
        let png = render::render_png(&scene, &cam, 900, 560, d.show_mesh, unit, false)?;
        h += &png_tag(&png, &caption);
    }
    h += "</div>";

    // geometry & materials
    h += "<h2>Geometry</h2><table><tr><th>Body</th><th>Role</th><th>Shape</th><th>Material</th></tr>";
    for b in &m.bodies {
        let role = match (b.role, b.op) {
            (Role::Enclosure, _) => "enclosure",
            (_, Op::Subtract) => "cut",
            _ => "solid",
        };
        let _ = write!(h, "<tr><td>{}</td><td>{role}</td><td>{}</td><td>{}</td></tr>", esc(&b.name), shape_desc(&b.shape), if b.op == Op::Add { esc(&b.material) } else { "—".into() });
    }
    h += "</table>";
    let lib = m.materials();
    let used: Vec<&crate::material::Material> = lib.iter().filter(|mat| m.bodies.iter().any(|b| b.op == Op::Add && b.material == mat.name)).collect();
    if !used.is_empty() {
        h += "<h2>Materials</h2><table><tr><th>Material</th><th>E (GPa)</th><th>ν</th><th>ρ (kg/m³)</th><th>Yield (MPa)</th><th>k (W/m·K)</th><th>α (1/K)</th></tr>";
        for mat in used {
            let _ = write!(h, "<tr><td>{}</td><td class=\"n\">{}</td><td class=\"n\">{}</td><td class=\"n\">{}</td><td class=\"n\">{}</td><td class=\"n\">{}</td><td class=\"n\">{}</td></tr>", esc(&mat.name), num(mat.youngs_modulus_gpa), num(mat.poisson_ratio), num(mat.density), num(mat.yield_strength_mpa), num(mat.thermal_conductivity), num(mat.thermal_expansion));
        }
        h += "</table>";
    }

    // loads & mesh
    h += "<h2>Supports and loads</h2>";
    if m.loads.is_empty() {
        h += "<p class=\"meta\">None.</p>";
    } else {
        h += "<table><tr><th>#</th><th>Type</th><th>Definition</th><th>Used</th></tr>";
        for l in &m.loads {
            let used = l.load.applies_to(m.analysis.kind);
            let _ = write!(h, "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>", l.id, esc(l.load.label()), esc(&l.load.summary()), if used { "yes" } else { "<span class=\"warn\">ignored by this analysis</span>" });
        }
        h += "</table>";
    }
    let _ = write!(h, "<h2>Mesh</h2><table><tr><th>Element size</th><th>Elements</th><th>Nodes</th><th>Solid volume</th></tr><tr><td class=\"n\">{} mm</td><td class=\"n\">{}</td><td class=\"n\">{}</td><td class=\"n\">{} mm³</td></tr></table>", num(mesh.h), mesh.elems.len(), mesh.nodes.len(), num(mesh.volume_mm3()));
    if !mesh.warnings.is_empty() {
        let _ = write!(h, "<p class=\"warn\">{}</p>", esc(&mesh.warnings.join("; ")));
    }

    // results
    if let (Some(r), Some(sum)) = (inp.results, &summary) {
        h += "<h2>Results</h2>";
        if let Some(fields) = sum["fields"].as_object() {
            h += "<table><tr><th>Result</th><th>Min</th><th>Max</th><th>Unit</th><th>Max at (mm)</th></tr>";
            for (_, v) in fields {
                let at = v["max_at_mm"].as_array().map(|a| a.iter().map(|x| num(x.as_f64().unwrap_or(0.0))).collect::<Vec<_>>().join(", ")).unwrap_or_default();
                let _ = write!(h, "<tr><td>{}</td><td class=\"n\">{}</td><td class=\"n\">{}</td><td>{}</td><td class=\"n\">{at}</td></tr>", esc(v["label"].as_str().unwrap_or("")), num(v["min"].as_f64().unwrap_or(0.0)), num(v["max"].as_f64().unwrap_or(0.0)), esc(v["unit"].as_str().unwrap_or("")));
            }
            h += "</table>";
        }
        if !r.modes.is_empty() {
            h += "<h3>Modes</h3><table><tr><th>Mode</th><th>Value</th></tr>";
            for i in 0..r.modes.len() {
                let _ = write!(h, "<tr><td>{}</td><td>{}</td></tr>", i + 1, esc(&r.step_label(i)));
            }
            h += "</table>";
        }
        if !r.reactions.is_empty() {
            h += "<h3>Reactions</h3><table><tr><th>#</th><th>Support</th><th>Value</th></tr>";
            for rc in &r.reactions {
                let _ = write!(h, "<tr><td>{}</td><td>{}</td><td class=\"n\">({}, {}, {})</td></tr>", rc.load_id, esc(&rc.label), num(rc.value[0]), num(rc.value[1]), num(rc.value[2]));
            }
            h += "</table>";
        }
        if !r.quantities.is_empty() {
            h += "<h3>Quantities</h3><table><tr><th>Quantity</th><th>Value</th><th>Unit</th></tr>";
            for q in &r.quantities {
                let _ = write!(h, "<tr><td>{}</td><td class=\"n\">{}</td><td>{}</td></tr>", esc(&q.name), num(q.value), esc(&q.unit));
            }
            h += "</table>";
        }
        if r.history.len() > 1 {
            let _ = write!(h, "<h3>History</h3>{}", svg_chart(&r.history, if r.analysis == crate::model::AnalysisType::NonlinearStructural { "load factor" } else { "time [s]" }, &r.history_label, true));
        }
        if !r.warnings.is_empty() {
            let _ = write!(h, "<h3>Warnings</h3><ul>{}</ul>", r.warnings.iter().map(|w| format!("<li class=\"warn\">{}</li>", esc(w))).collect::<String>());
        }
        let _ = write!(h, "<p class=\"meta\">Solver: {} · {} DOF · linear solve {} s · total {} s</p>", esc(&r.info.solver), r.info.dofs, num(r.info.linear_solve_seconds), num(r.info.seconds));
    } else {
        h += "<h2>Results</h2><p class=\"meta\">Not solved yet.</p>";
    }

    // study
    if let Some(st) = inp.study {
        let _ = write!(h, "<h2>Study — {}</h2><p>{}</p>", esc(&st.kind.replace('_', " ")), esc(&st.message));
        if st.kind == "topology" {
            let comp: Vec<(f64, f64)> = st.history.iter().map(|(i, c, _)| (*i as f64, *c)).collect();
            h += &svg_chart(&comp, "iteration", "compliance [J]", true);
        } else if !st.runs.is_empty() {
            h += "<table><tr><th>#</th>";
            for p in &st.parameters {
                let _ = write!(h, "<th>{}</th>", esc(&p.target));
            }
            for r in &st.responses {
                let _ = write!(h, "<th>{}</th>", esc(r));
            }
            h += "</tr>";
            for (i, run) in st.runs.iter().enumerate() {
                let _ = write!(h, "<tr{}><td>{}{}</td>", if st.best == Some(i) { " class=\"best\"" } else { "" }, i + 1, if st.best == Some(i) { " ★" } else { "" });
                for v in &run.values {
                    let _ = write!(h, "<td class=\"n\">{}</td>", num(*v));
                }
                for v in &run.responses {
                    let _ = write!(h, "<td class=\"n\">{}</td>", v.map(num).unwrap_or_else(|| "—".into()));
                }
                h += "</tr>";
            }
            h += "</table>";
            if st.parameters.len() == 1 {
                for (k, name) in st.responses.iter().enumerate().take(3) {
                    let pts: Vec<(f64, f64)> = st.runs.iter().filter_map(|r| Some((r.values[0], r.responses[k]?))).collect();
                    h += &svg_chart(&pts, &st.parameters[0].target, name, st.kind == "parameter_study");
                }
            } else if st.kind == "optimization" {
                let pts: Vec<(f64, f64)> = st.runs.iter().enumerate().filter_map(|(i, r)| Some(((i + 1) as f64, r.responses[0]?))).collect();
                h += &svg_chart(&pts, "evaluation", &st.responses[0], false);
            }
        }
    }

    // plan
    if let Some(p) = inp.plan {
        let _ = write!(h, "<h2>Plan — {}</h2><ul>", esc(&p.title));
        for st in &p.steps {
            let _ = write!(h, "<li>{} <b>{}</b> <span class=\"meta\">({})</span> — {}</li>", if st.done { "☑" } else { "☐" }, esc(&st.task), esc(&st.status), esc(&st.detail));
        }
        h += "</ul>";
    }
    let _ = write!(h, "<footer>Units: mm, N, MPa, °C unless noted. Generated by SimCraft {}.</footer></main></body></html>", env!("CARGO_PKG_VERSION"));
    Ok(h)
}
