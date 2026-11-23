use simcraft::{bridge, gui, mcp, session};

const HELP: &str = "SimCraft — FEA simulation

USAGE:
  simcraft [gui] [project.simcraft]   open the desktop app (default)
  simcraft mcp                        run the MCP server on stdio (for Claude Code, Antigravity, …)
  simcraft serve [--port N]           headless server for scripts (pysimcraft), no window
  simcraft schema                     print every tool with its JSON schema
  simcraft --help

ENV:
  SIMCRAFT_PORT        local port of the GUI / headless server bridge (default 47800)
  SIMCRAFT_TOKEN_FILE  bridge token file (default ~/.config/simcraft/token)

Claude Code setup:
  claude mcp add simcraft -- /path/to/simcraft mcp";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("mcp") => mcp::run(),
        Some("schema") => println!("{}", serde_json::to_string_pretty(&session::tool_defs()).unwrap()),
        Some("serve") => {
            if let Some(i) = args.iter().position(|a| a == "--port") {
                let Some(port) = args.get(i + 1).filter(|p| p.parse::<u16>().is_ok()) else {
                    eprintln!("--port needs a number");
                    std::process::exit(2);
                };
                // SAFETY: start-up, single-threaded
                unsafe { std::env::set_var("SIMCRAFT_PORT", port) };
            }
            match bridge::serve(session::new_core()) {
                Ok(msg) => {
                    eprintln!("SimCraft headless server: {msg}. Press Ctrl+C to stop.");
                    loop {
                        std::thread::park();
                    }
                }
                Err(e) => {
                    eprintln!("cannot start server: {e}");
                    std::process::exit(1);
                }
            }
        }
        Some("-h") | Some("--help") | Some("help") => println!("{HELP}"),
        Some("--version") | Some("-V") => println!("simcraft {}", env!("CARGO_PKG_VERSION")),
        Some("gui") => run_gui(args.get(1).cloned()),
        Some(p) if !p.starts_with('-') => run_gui(Some(p.to_string())),
        None => run_gui(None),
        Some(o) => {
            eprintln!("unknown option {o}\n\n{HELP}");
            std::process::exit(2);
        }
    }
}

/// Keep the GUI on the integrated GPU. On hybrid laptops (e.g. AMD iGPU + NVIDIA dGPU) the
/// default GL vendor order loads the NVIDIA driver and wakes the discrete GPU, whose power-up can
/// disturb Wi-Fi/Bluetooth combo cards. SimCraft's viewport is software-rendered, so the iGPU is
/// plenty. Set SIMCRAFT_GPU=discrete to opt out.
fn prefer_integrated_gpu() {
    if std::env::var("SIMCRAFT_GPU").is_ok_and(|v| v == "discrete") || std::env::var_os("__EGL_VENDOR_LIBRARY_FILENAMES").is_some() {
        return;
    }
    let mesa = ["/usr/share/glvnd/egl_vendor.d/50_mesa.json", "/etc/glvnd/egl_vendor.d/50_mesa.json"].into_iter().find(|p| std::path::Path::new(p).exists());
    // SAFETY: called at start-up on the main thread before any other thread exists.
    unsafe {
        if let Some(p) = mesa {
            std::env::set_var("__EGL_VENDOR_LIBRARY_FILENAMES", p);
        }
        std::env::set_var("__GLX_VENDOR_LIBRARY_NAME", "mesa");
        std::env::set_var("DRI_PRIME", "0");
    }
}

fn run_gui(open: Option<String>) {
    prefer_integrated_gpu();
    if let Err(e) = gui::run(session::new_core(), open) {
        eprintln!("SimCraft GUI failed to start: {e}");
        std::process::exit(1);
    }
}
