//! In-app help: the user guide (docs/USER_GUIDE.md) and the tutorials (docs/TUTORIALS.md) compiled
//! into the binary, split into sections, searchable, and shared by the Help window and the
//! `get_help` MCP tool.

use std::sync::OnceLock;

pub const GUIDE: &str = include_str!("../docs/USER_GUIDE.md");
pub const TUTORIALS: &str = include_str!("../docs/TUTORIALS.md");

#[derive(Debug, Clone)]
pub struct Section {
    /// 1 = document title, 2 = numbered chapter, 3 = subsection
    pub level: u8,
    pub title: String,
    /// markdown body up to the next heading
    pub body: String,
}

/// All sections of the guide followed by the tutorials, in order (headings inside code blocks are
/// ignored).
pub fn sections() -> &'static [Section] {
    static S: OnceLock<Vec<Section>> = OnceLock::new();
    S.get_or_init(|| {
        let mut v = parse(GUIDE);
        v.extend(parse(TUTORIALS));
        v
    })
}

/// Whether section `i` belongs to the tutorials (they follow the guide).
pub fn is_tutorial(i: usize) -> bool {
    static N: OnceLock<usize> = OnceLock::new();
    i >= *N.get_or_init(|| parse(GUIDE).len())
}

/// Tutorial sections: (index, short title without the "Tutorial n — " prefix).
pub fn tutorials() -> Vec<(usize, String)> {
    sections()
        .iter()
        .enumerate()
        .filter(|(_, s)| s.level == 2 && s.title.starts_with("Tutorial "))
        .map(|(i, s)| (i, s.title.split_once(" — ").map(|x| x.1.to_string()).unwrap_or_else(|| s.title.clone())))
        .collect()
}

pub fn parse(text: &str) -> Vec<Section> {
    let mut out: Vec<Section> = Vec::new();
    let mut fence = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            fence = !fence;
        }
        let level = if fence { 0 } else { line.chars().take_while(|c| *c == '#').count() };
        if (1..=3).contains(&level) && line.chars().nth(level) == Some(' ') {
            out.push(Section { level: level as u8, title: line[level + 1..].trim().to_string(), body: String::new() });
        } else if let Some(s) = out.last_mut() {
            s.body.push_str(line);
            s.body.push('\n');
        }
    }
    for s in out.iter_mut() {
        s.body = s.body.trim_matches('\n').to_string();
    }
    out
}

/// Section index plus its subsections (everything until the next heading at the same or a
/// higher level).
pub fn span(i: usize) -> std::ops::Range<usize> {
    let s = sections();
    let lvl = s[i].level;
    let end = (i + 1..s.len()).find(|&j| s[j].level <= lvl).unwrap_or(s.len());
    i..end
}

/// Markdown text of a section with its subsections.
pub fn section_text(i: usize) -> String {
    let s = sections();
    span(i)
        .map(|j| format!("{} {}\n\n{}", "#".repeat(s[j].level as usize), s[j].title, s[j].body))
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn norm(t: &str) -> String {
    t.to_lowercase()
}

/// Best section for a topic name ("meshing", "7", "cfd", "fatigue"…).
pub fn find(topic: &str) -> Option<usize> {
    let s = sections();
    let t = norm(topic.trim());
    if t.is_empty() {
        return None;
    }
    // chapter number
    if let Ok(n) = t.trim_end_matches('.').parse::<u32>() {
        return s.iter().position(|x| x.title.starts_with(&format!("{n}. ")));
    }
    // title match (shortest title containing the topic wins), then best search hit; the guide
    // answers before the tutorials unless a tutorial is asked for
    let title = |tut: bool| s.iter().enumerate().filter(|(i, x)| is_tutorial(*i) == tut && norm(&x.title).contains(&t)).min_by_key(|(_, x)| x.title.len()).map(|(i, _)| i);
    if t.contains("tutorial") {
        return title(true).or_else(|| title(false));
    }
    title(false).or_else(|| search(topic, 1).first().map(|h| h.0)).or_else(|| title(true))
}

/// Ranked search: (section index, score, snippet around the first body hit).
pub fn search(query: &str, max: usize) -> Vec<(usize, f64, String)> {
    let words: Vec<String> = query.split_whitespace().map(norm).filter(|w| w.len() >= 2).collect();
    if words.is_empty() {
        return vec![];
    }
    let mut hits: Vec<(usize, f64, String)> = sections()
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let title = norm(&s.title);
            let body = norm(&s.body);
            let mut score = 0.0;
            for w in &words {
                let tb = body.matches(w.as_str()).count();
                let tt = title.matches(w.as_str()).count();
                if tb + tt == 0 {
                    return None; // every word must appear somewhere in the section
                }
                score += 6.0 * tt as f64 + (tb as f64).sqrt();
            }
            if s.level == 1 {
                score *= 0.3;
            }
            let pos = body.find(words[0].as_str()).unwrap_or(0);
            Some((i, score, snippet(&s.body, pos)))
        })
        .collect();
    // reference first: tutorials follow the guide's hits unless a tutorial is asked for
    let want_tut = words.iter().any(|w| w.starts_with("tutorial"));
    hits.sort_by(|a, b| (is_tutorial(a.0) && !want_tut).cmp(&(is_tutorial(b.0) && !want_tut)).then(b.1.total_cmp(&a.1)));
    hits.truncate(max);
    hits
}

fn snippet(body: &str, byte_pos: usize) -> String {
    let start = body[..byte_pos.min(body.len())].char_indices().rev().nth(60).map(|(i, _)| i).unwrap_or(0);
    let tail: String = body[start..].chars().take(160).collect();
    let flat = tail.replace(['\n', '|', '*', '`', '#'], " ");
    let flat = flat.split_whitespace().collect::<Vec<_>>().join(" ");
    format!("{}{}…", if start > 0 { "…" } else { "" }, flat)
}

/// Help chapter for an analysis key (static_structural, fluid, …).
pub fn analysis_topic(key: &str) -> &'static str {
    match key {
        "steady_thermal" | "transient_thermal" | "thermal_stress" => "Thermal",
        "electrostatic" | "current_flow" | "thermal_electric" | "magnetostatic" => "Electromagnetic",
        "full_wave_em" | "microwave_heating" => "Full-wave electromagnetics",
        "fluid" | "conjugate_heat_transfer" => "Fluids",
        _ => "Structural",
    }
}
