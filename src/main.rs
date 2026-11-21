use simcraft::{gui, mcp, session};

const HELP: &str = "SimCraft — FEA simulation

USAGE:
  simcraft [gui] [project.simcraft]   open the desktop app (default)
  simcraft mcp                        run the MCP server on stdio (for Claude Code, Antigravity, …)
  simcraft --help

ENV:
  SIMCRAFT_PORT   local port of the GUI <-> MCP bridge (default 47800)

Claude Code setup:
  claude mcp add simcraft -- /path/to/simcraft mcp";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("mcp") => mcp::run(),
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

fn run_gui(open: Option<String>) {
    if let Err(e) = gui::run(session::new_core(), open) {
        eprintln!("SimCraft GUI failed to start: {e}");
        std::process::exit(1);
    }
}
