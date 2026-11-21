//! Local TCP bridge: the running GUI accepts newline-delimited JSON tool calls so the MCP
//! server (launched by Claude Code / Antigravity) can drive it live.
//! Request:  {"tool": "add_box", "args": {...}}
//! Response: {"ok": true, "text": "...", "image_png_b64": "..."}

use crate::session::Shared;
use base64::Engine;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};

pub fn port() -> u16 {
    std::env::var("SIMCRAFT_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(47800)
}

pub fn addr() -> String {
    format!("127.0.0.1:{}", port())
}

/// Start listening in a background thread. Returns an error string if the port is taken.
pub fn serve(core: Shared) -> Result<String, String> {
    let listener = TcpListener::bind(addr()).map_err(|e| format!("bridge disabled ({}: {e})", addr()))?;
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let core = core.clone();
            std::thread::spawn(move || handle(core, stream));
        }
    });
    Ok(format!("MCP bridge listening on {}", addr()))
}

fn handle(core: Shared, stream: TcpStream) {
    let Ok(read) = stream.try_clone() else { return };
    let mut w = stream;
    for line in BufReader::new(read).lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let resp = match serde_json::from_str::<Value>(&line) {
            Ok(req) => {
                let tool = req.get("tool").and_then(|t| t.as_str()).unwrap_or("");
                let args = req.get("args").cloned().unwrap_or(json!({}));
                match core.call(tool, &args) {
                    Ok(o) => json!({
                        "ok": true,
                        "text": o.text,
                        "image_png_b64": o.image_png.map(|p| base64::engine::general_purpose::STANDARD.encode(p)),
                    }),
                    Err(e) => json!({"ok": false, "text": format!("{e:#}")}),
                }
            }
            Err(e) => json!({"ok": false, "text": format!("bad request: {e}")}),
        };
        if writeln!(w, "{resp}").is_err() {
            break;
        }
    }
}

/// Client side: call a tool on a running GUI. Returns None if no GUI is listening.
pub fn remote_call(tool: &str, args: &Value) -> Option<Result<Value, String>> {
    let addr: std::net::SocketAddr = addr().parse().ok()?;
    let stream = TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(250)).ok()?;
    let mut w = stream.try_clone().ok()?;
    if writeln!(w, "{}", json!({"tool": tool, "args": args})).is_err() {
        return Some(Err("lost connection to SimCraft GUI".into()));
    }
    let mut line = String::new();
    match BufReader::new(stream).read_line(&mut line) {
        Ok(n) if n > 0 => Some(serde_json::from_str(&line).map_err(|e| e.to_string())),
        _ => Some(Err("SimCraft GUI closed the connection".into())),
    }
}

pub fn gui_running() -> bool {
    let Ok(addr) = addr().parse::<std::net::SocketAddr>() else { return false };
    TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(250)).is_ok()
}
