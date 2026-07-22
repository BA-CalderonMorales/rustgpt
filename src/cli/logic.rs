use super::Mode;

fn usage() {
    println!("Usage: llm [--e2e <prompt>]");
    println!();
    println!("Examples:");
    println!("  llm");
    println!("  llm --e2e \"hello world\"");
}

fn try_parse_mode() -> Result<Mode, String> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None => Ok(Mode::Interactive),
        Some("--e2e") => {
            let prompt = args
                .next()
                .ok_or_else(|| "--e2e requires a prompt".to_string())?;
            if args.next().is_some() {
                return Err("--e2e accepts exactly one prompt".to_string());
            }
            Ok(Mode::E2e { prompt })
        }
        Some("--help") | Some("-h") => {
            usage();
            std::process::exit(0);
        }
        Some("--version") => {
            println!("llm {}", env!("CARGO_PKG_VERSION"));
            std::process::exit(0);
        }
        Some(argument) => Err(format!("unknown argument: {argument}")),
    }
}

pub(crate) fn parse_mode() -> Mode {
    try_parse_mode().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!("Try 'llm --help' for usage.");
        std::process::exit(2);
    })
}
