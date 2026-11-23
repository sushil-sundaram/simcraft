//! Local TCP bridge: the running GUI accepts newline-delimited JSON tool calls so the MCP
//! server (launched by Claude Code / Antigravity) and `pysimcraft` can drive it live.
//!
//! Protocol (one JSON object per line):
//!   1. client → `{"auth": "<token>"}`        server → `{"ok": true}` (or closes the connection)
//!   2. client → `{"tool": "add_box", "args": {...}}`
//!      server → `{"ok": true, "text": "...", "image_png_b64": "..."}`
//!
//! The token lives in `~/.config/simcraft/token` (created with mode 0600), so only processes of
//! the same user can drive SimCraft. Requests are limited to 4 MB per line.

use crate::session::Shared;
use base64::Engine;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

const MAX_LINE: u64 = 4 * 1024 * 1024;

pub fn port() -> u16 {
    std::env::var("SIMCRAFT_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(47800)
}

pub fn addr() -> String {
    format!("127.0.0.1:{}", port())
}

pub fn token_path() -> PathBuf {
    if let Some(p) = std::env::var_os("SIMCRAFT_TOKEN_FILE") {
        return PathBuf::from(p);
    }
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("simcraft").join("token")
}

fn random_token() -> String {
    let mut bytes = [0u8; 32];
    let ok = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut bytes)).is_ok();
    if !ok {
        // fallback: hash of time, pid and stack address
        let seed = format!("{:?}{}{:p}", std::time::SystemTime::now(), std::process::id(), &bytes);
        let mut h: u64 = 0xcbf29ce484222325;
        for (i, b) in seed.bytes().cycle().take(512).enumerate() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            bytes[i % 32] ^= (h >> 24) as u8;
        }
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Read the shared token, creating it (mode 0600) when missing.
pub fn token() -> std::io::Result<String> {
    let p = token_path();
    if let Ok(t) = std::fs::read_to_string(&p) {
        let t = t.trim().to_string();
        if t.len() >= 32 {
            return Ok(t);
        }
    }
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let t = random_token();
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(&p)?;
    f.write_all(t.as_bytes())?;
    Ok(t)
}

fn same(a: &str, b: &str) -> bool {
    // constant-time comparison
    if a.len() != b.len() {
        return false;
    }
    a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Start listening in a background thread. Returns an error string if the port is taken.
pub fn serve(core: Shared) -> Result<String, String> {
    let token = token().map_err(|e| format!("bridge disabled (cannot create {}: {e})", token_path().display()))?;
    let listener = TcpListener::bind(addr()).map_err(|e| format!("bridge disabled ({}: {e})", addr()))?;
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let (core, token) = (core.clone(), token.clone());
            std::thread::spawn(move || handle(core, stream, &token));
        }
    });
    Ok(format!("MCP bridge listening on {} (token-protected)", addr()))
}

/// Read one line of at most MAX_LINE bytes. Ok(None) = connection closed or line too long.
fn read_limited(r: &mut BufReader<TcpStream>) -> Option<String> {
    let mut buf = Vec::new();
    let n = r.by_ref().take(MAX_LINE + 1).read_until(b'\n', &mut buf).ok()?;
    if n == 0 || n as u64 > MAX_LINE {
        return None;
    }
    String::from_utf8(buf).ok()
}

fn handle(core: Shared, stream: TcpStream, token: &str) {
    let Ok(read) = stream.try_clone() else { return };
    let mut w = stream;
    let mut r = BufReader::new(read);
    // 1. authenticate
    let authed = read_limited(&mut r).and_then(|l| serde_json::from_str::<Value>(&l).ok()).and_then(|v| v.get("auth").and_then(|a| a.as_str()).map(|a| same(a, token))).unwrap_or(false);
    if !authed {
        let _ = writeln!(w, "{}", json!({"ok": false, "text": "unauthorized: send {\"auth\": <token from ~/.config/simcraft/token>} first"}));
        return;
    }
    if writeln!(w, "{}", json!({"ok": true})).is_err() {
        return;
    }
    // 2. tool calls
    while let Some(line) = read_limited(&mut r) {
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

/// Open an authenticated connection to a running SimCraft (None if nothing is listening).
fn connect() -> Option<Result<(TcpStream, BufReader<TcpStream>), String>> {
    let addr: std::net::SocketAddr = addr().parse().ok()?;
    let stream = TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(250)).ok()?;
    let token = match token() {
        Ok(t) => t,
        Err(e) => return Some(Err(format!("cannot read SimCraft token {}: {e}", token_path().display()))),
    };
    let mut w = match stream.try_clone() {
        Ok(w) => w,
        Err(e) => return Some(Err(e.to_string())),
    };
    let mut r = BufReader::new(stream);
    if writeln!(w, "{}", json!({"auth": token})).is_err() {
        return Some(Err("lost connection to SimCraft GUI".into()));
    }
    let mut line = String::new();
    match r.read_line(&mut line) {
        Ok(n) if n > 0 => match serde_json::from_str::<Value>(&line) {
            Ok(v) if v["ok"].as_bool() == Some(true) => Some(Ok((w, r))),
            Ok(v) => Some(Err(v["text"].as_str().unwrap_or("SimCraft refused the connection").to_string())),
            Err(e) => Some(Err(e.to_string())),
        },
        _ => Some(Err("SimCraft GUI closed the connection".into())),
    }
}

/// Client side: call a tool on a running GUI. Returns None if no GUI is listening.
pub fn remote_call(tool: &str, args: &Value) -> Option<Result<Value, String>> {
    let (mut w, mut r) = match connect()? {
        Ok(c) => c,
        Err(e) => return Some(Err(e)),
    };
    if writeln!(w, "{}", json!({"tool": tool, "args": args})).is_err() {
        return Some(Err("lost connection to SimCraft GUI".into()));
    }
    let mut line = String::new();
    match r.read_line(&mut line) {
        Ok(n) if n > 0 => Some(serde_json::from_str(&line).map_err(|e| e.to_string())),
        _ => Some(Err("SimCraft GUI closed the connection".into())),
    }
}

pub fn gui_running() -> bool {
    let Ok(addr) = addr().parse::<std::net::SocketAddr>() else { return false };
    TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(250)).is_ok()
}
