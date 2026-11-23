//! Model Context Protocol server over stdio (JSON-RPC 2.0, newline-delimited).
//! When the SimCraft GUI is running, every tool call is forwarded to it so you can watch the
//! model being built; otherwise the server runs its own headless session.

use crate::bridge;
use crate::session::{self, Shared};
use base64::Engine;
use serde_json::{Value, json};
use std::io::{BufRead, Write};

const VERSIONS: [&str; 4] = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"];

const INSTRUCTIONS: &str = "SimCraft is a finite-element simulation app (structural, thermal, modal, thermal-stress). \
Units: mm, N, MPa, °C. Typical workflow: new_model -> add_box/add_cylinder/add_sphere/import_stl (operation=subtract makes holes) \
-> set_material -> set_analysis -> list_faces -> add_fixed_support + add_force/add_pressure/... -> solve -> get_results / render_image. \
Use launch_gui to open the desktop app so the user can watch; all later calls then drive the GUI live.";

pub fn run() {
    let local = session::new_core();
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    let mut last_backend = "";
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                send(&mut out, json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("parse error: {e}")}}));
                continue;
            }
        };
        let id = msg.get("id").cloned();
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(json!({}));
        let Some(id) = id else { continue }; // notification
        let result: Result<Value, (i64, String)> = match method {
            "initialize" => {
                let req = params.get("protocolVersion").and_then(|v| v.as_str()).unwrap_or("");
                let ver = if VERSIONS.contains(&req) { req } else { "2025-06-18" };
                Ok(json!({
                    "protocolVersion": ver,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": "simcraft", "title": "SimCraft FEA", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": INSTRUCTIONS,
                }))
            }
            "ping" => Ok(json!({})),
            "tools/list" => {
                let mut tools = session::tool_defs();
                tools.push(json!({"name": "launch_gui", "description": "Open the SimCraft desktop window (carrying over the current model) so the user can watch and interact. Subsequent tool calls drive the window live.", "inputSchema": {"type": "object", "properties": {}}}));
                Ok(json!({"tools": tools}))
            }
            "tools/call" => {
                let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                let (res, backend) = call_tool(&local, &name, &args);
                let mut res = res;
                if backend != last_backend && !last_backend.is_empty() {
                    if let Some(c) = res["content"].as_array_mut() {
                        c.insert(0, json!({"type": "text", "text": format!("(now running in SimCraft {backend})")}));
                    }
                }
                last_backend = backend;
                Ok(res)
            }
            "resources/list" => Ok(json!({"resources": []})),
            "prompts/list" => Ok(json!({"prompts": []})),
            _ => Err((-32601, format!("method not found: {method}"))),
        };
        let resp = match result {
            Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
            Err((c, m)) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": c, "message": m}}),
        };
        send(&mut out, resp);
    }
}

fn send(out: &mut std::io::Stdout, v: Value) {
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

fn content(ok: bool, text: String, png_b64: Option<String>) -> Value {
    let mut c = vec![json!({"type": "text", "text": text})];
    if let Some(b) = png_b64 {
        c.push(json!({"type": "image", "data": b, "mimeType": "image/png"}));
    }
    json!({"content": c, "isError": !ok})
}

fn call_tool(local: &Shared, name: &str, args: &Value) -> (Value, &'static str) {
    if name == "launch_gui" {
        return (launch_gui(local), "GUI");
    }
    if let Some(r) = bridge::remote_call(name, args) {
        return match r {
            Ok(v) => (
                content(
                    v["ok"].as_bool().unwrap_or(false),
                    v["text"].as_str().unwrap_or("").to_string(),
                    v["image_png_b64"].as_str().map(String::from),
                ),
                "GUI",
            ),
            Err(e) => (content(false, e, None), "GUI"),
        };
    }
    match local.call(name, args) {
        Ok(o) => (content(true, o.text, o.image_png.map(|p| base64::engine::general_purpose::STANDARD.encode(p))), "headless mode"),
        Err(e) => (content(false, format!("{e:#}"), None), "headless mode"),
    }
}

fn launch_gui(local: &Shared) -> Value {
    if bridge::gui_running() {
        return content(true, "SimCraft GUI is already running; tool calls are going to it.".into(), None);
    }
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => return content(false, format!("cannot locate simcraft executable: {e}"), None),
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("gui");
    let has_model = !local.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.bodies.is_empty();
    if has_model {
        let path = std::env::temp_dir().join(format!("simcraft-handoff-{}.simcraft", std::process::id()));
        let p = path.to_string_lossy().to_string();
        if let Err(e) = local.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).save(&p) {
            return content(false, format!("could not hand over model: {e}"), None);
        }
        cmd.arg(&p);
    }
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
    if let Err(e) = cmd.spawn() {
        return content(false, format!("failed to start GUI: {e}"), None);
    }
    for _ in 0..60 {
        std::thread::sleep(std::time::Duration::from_millis(250));
        if bridge::gui_running() {
            let note = if has_model { " The current model was carried over (re-run solve to see results)." } else { "" };
            return content(true, format!("SimCraft GUI started.{note} All further tool calls drive the window live."), None);
        }
    }
    content(false, "GUI process started but its bridge did not come up within 15 s (is a display available?).".into(), None)
}
