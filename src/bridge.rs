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
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

const MAX_LINE: u64 = 4 * 1024 * 1024;
const MAX_RESP: u64 = 64 * 1024 * 1024;
const MAX_CONNS: usize = 32;
const AUTH_TIMEOUT: Duration = Duration::from_secs(5);

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
    if let Some(base) = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .or_else(|| std::env::var_os("APPDATA").map(PathBuf::from))
        .or_else(|| std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join(".config")))
    {
        return base.join("simcraft").join("token");
    }
    PathBuf::new()
}

fn random_token() -> std::io::Result<String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|e| std::io::Error::other(format!("OS random generator failed: {e}")))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn check_existing_token(p: &std::path::Path) -> std::io::Result<Option<String>> {
    let meta = match std::fs::symlink_metadata(p) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    if meta.file_type().is_symlink() || !meta.is_file() {
        return Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "token path is not a regular file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o077 != 0 {
            std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600))?;
        }
    }
    let t = std::fs::read_to_string(p)?.trim().to_string();
    if t.len() >= 32 && t.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Ok(Some(t));
    }
    Ok(None)
}

/// Read the shared token, creating it (mode 0600) when missing.
pub fn token() -> std::io::Result<String> {
    let p = token_path();
    if p.as_os_str().is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no home/config directory for token"));
    }
    if let Some(t) = check_existing_token(&p)? {
        return Ok(t);
    }
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
        }
    }
    let t = random_token()?;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(&p)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = f.set_permissions(std::fs::Permissions::from_mode(0o600));
    }
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

struct ConnGuard(Arc<AtomicUsize>);
impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Start listening in a background thread. Returns an error string if the port is taken.
pub fn serve(core: Shared) -> Result<String, String> {
    let token = token().map_err(|e| format!("bridge disabled (cannot create {}: {e})", token_path().display()))?;
    let listener = TcpListener::bind(addr()).map_err(|e| format!("bridge disabled ({}: {e})", addr()))?;
    let active = Arc::new(AtomicUsize::new(0));
    std::thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            if active.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| (n < MAX_CONNS).then_some(n + 1)).is_err() {
                let _ = writeln!(stream, "{}", json!({"ok": false, "text": "too many bridge connections"}));
                continue;
            }
            let guard = ConnGuard(active.clone());
            let (core, token) = (core.clone(), token.clone());
            std::thread::spawn(move || {
                let _guard = guard;
                handle(core, stream, &token);
            });
        }
    });
    Ok(format!("MCP bridge listening on {} (token-protected)", addr()))
}

/// Read one line of at most `limit` bytes. None = connection closed, timed out, or line too long.
fn read_limited(r: &mut BufReader<TcpStream>, limit: u64) -> Option<String> {
    let mut buf = Vec::new();
    let n = r.by_ref().take(limit + 1).read_until(b'\n', &mut buf).ok()?;
    if n == 0 || n as u64 > limit {
        return None;
    }
    String::from_utf8(buf).ok()
}

fn handle(core: Shared, stream: TcpStream, token: &str) {
    let _ = stream.set_read_timeout(Some(AUTH_TIMEOUT));
    let Ok(read) = stream.try_clone() else { return };
    let mut w = stream;
    let mut r = BufReader::new(read);
    // 1. authenticate
    let authed = read_limited(&mut r, MAX_LINE)
        .and_then(|l| serde_json::from_str::<Value>(&l).ok())
        .and_then(|v| v.get("auth").and_then(|a| a.as_str()).map(|a| same(a, token)))
        .unwrap_or(false);
    if !authed {
        std::thread::sleep(Duration::from_millis(100));
        let _ = writeln!(w, "{}", json!({"ok": false, "text": "unauthorized: send {\"auth\": <token from ~/.config/simcraft/token>} first"}));
        return;
    }
    let _ = w.set_read_timeout(None);
    if writeln!(w, "{}", json!({"ok": true})).is_err() {
        return;
    }
    // 2. tool calls
    while let Some(line) = read_limited(&mut r, MAX_LINE) {
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
    let stream = TcpStream::connect_timeout(&addr, Duration::from_millis(250)).ok()?;
    let _ = stream.set_read_timeout(Some(AUTH_TIMEOUT));
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
    match read_limited(&mut r, MAX_LINE) {
        Some(line) => {
            let _ = w.set_read_timeout(None);
            match serde_json::from_str::<Value>(&line) {
                Ok(v) if v["ok"].as_bool() == Some(true) => Some(Ok((w, r))),
                Ok(v) => Some(Err(v["text"].as_str().unwrap_or("SimCraft refused the connection").to_string())),
                Err(e) => Some(Err(e.to_string())),
            }
        }
        None => Some(Err("SimCraft GUI closed the connection".into())),
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
    match read_limited(&mut r, MAX_RESP) {
        Some(line) => Some(serde_json::from_str(&line).map_err(|e| e.to_string())),
        None => Some(Err("SimCraft GUI closed the connection or sent an oversized response".into())),
    }
}

pub fn gui_running() -> bool {
    let Ok(addr) = addr().parse::<std::net::SocketAddr>() else { return false };
    TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(250)).is_ok()
}
