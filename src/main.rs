use simcraft::{bench, bridge, gui, mcp, session};

const HELP: &str = "SimCraft — FEA simulation

USAGE:
  simcraft [gui] [project.simcraft]         open the desktop app (default)
  simcraft mcp [--workspace DIR]            run the MCP server on stdio (for Claude Code, Antigravity, …)
  simcraft serve [--port N] [--workspace DIR]
                                            headless server for scripts (pysimcraft), no window
  simcraft schema                           print every tool with its JSON schema
  simcraft bench [--quick] [--filter TEXT] [--out DIR] [--compare FILE] [--template]
                                            run the benchmark suite (bench.json + bench.html in DIR,
                                            default bench-out); --compare adds another solver's results
                                            (JSON from --template); exits 1 if a case fails
  simcraft --help

ENV:
  SIMCRAFT_PORT        local port of the GUI / headless server bridge (default 47800)
  SIMCRAFT_TOKEN_FILE  bridge token file (default ~/.config/simcraft/token)
  SIMCRAFT_WORKSPACE   restrict MCP / headless file tools to this directory

Claude Code setup:
  claude mcp add simcraft -- /path/to/simcraft mcp";

fn apply_workspace_arg(args: &[String]) {
    if let Some(i) = args.iter().position(|a| a == "--workspace") {
        let Some(dir) = args.get(i + 1) else {
            eprintln!("--workspace needs a directory path");
            std::process::exit(2);
        };
        if !std::path::Path::new(dir).is_dir() {
            eprintln!("--workspace '{dir}' is not an existing directory");
            std::process::exit(2);
        }
        // SAFETY: start-up, single-threaded
        unsafe { std::env::set_var("SIMCRAFT_WORKSPACE", dir) };
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("mcp") => {
            apply_workspace_arg(&args);
            mcp::run();
        }
        Some("schema") => println!("{}", serde_json::to_string_pretty(&session::tool_defs()).unwrap()),
        Some("serve") => {
            apply_workspace_arg(&args);
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
        Some("bench") => {
            if args.iter().any(|a| a == "--template") {
                println!("{}", serde_json::to_string_pretty(&bench::template()).unwrap());
                return;
            }
            let val = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
            let opts = bench::Options { quick: args.iter().any(|a| a == "--quick"), filter: val("--filter"), out_dir: val("--out").unwrap_or_else(|| "bench-out".into()), ansys: val("--compare") };
            match bench::run(&opts, &|m| eprintln!("  {m}")) {
                Ok(r) => {
                    for c in r["cases"].as_array().cloned().unwrap_or_default() {
                        for m in c["metrics"].as_array().cloned().unwrap_or_default() {
                            let err = m["error"].as_f64().map(|e| format!("{:6.2} %", 100.0 * e)).unwrap_or("  error ".into());
                            println!("{} {:<20} {:<34} {err}  (tol {:.1} %)", if m["pass"].as_bool() == Some(true) { "PASS" } else { "FAIL" }, c["id"].as_str().unwrap_or(""), m["title"].as_str().unwrap_or(""), 100.0 * m["tolerance"].as_f64().unwrap_or(0.0));
                            for a in m["anomalies"].as_array().cloned().unwrap_or_default() {
                                println!("       ⚠ {}", a.as_str().unwrap_or(""));
                            }
                        }
                    }
                    println!("{} passed, {} failed in {:.0} s — report: {}/bench.html", r["passed"], r["failed"], r["seconds"].as_f64().unwrap_or(0.0), opts.out_dir);
                    if r["failed"].as_u64().unwrap_or(0) > 0 {
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("bench: {e:#}");
                    std::process::exit(2);
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
