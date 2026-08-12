use super::{Invocation, Mode};

const DEFAULT_SEED: u64 = 42;

fn usage() {
    println!("Usage: llm [--seed <n>] [--e2e <prompt> | --eval]");
    println!();
    println!("Examples:");
    println!("  llm");
    println!("  llm --e2e \"hello world\"");
    println!("  llm --eval --seed 42");
}

fn try_parse() -> Result<Invocation, String> {
    let mut args = std::env::args().skip(1);
    let mut mode: Option<Mode> = None;
    let mut seed: Option<u64> = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            "--version" => {
                println!("llm {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--seed" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--seed requires a value".to_string())?;
                seed = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid seed: {value}"))?,
                );
            }
            "--e2e" => {
                if mode.is_some() {
                    return Err("--e2e and --eval are mutually exclusive".to_string());
                }
                let prompt = args
                    .next()
                    .ok_or_else(|| "--e2e requires a prompt".to_string())?;
                if args.next().is_some() {
                    return Err("--e2e accepts exactly one prompt".to_string());
                }
                mode = Some(Mode::E2e { prompt });
            }
            "--eval" => {
                if mode.is_some() {
                    return Err("--e2e and --eval are mutually exclusive".to_string());
                }
                mode = Some(Mode::Eval);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let mode = mode.unwrap_or(Mode::Interactive);
    let seed = seed.unwrap_or(match mode {
        Mode::Eval | Mode::E2e { .. } => DEFAULT_SEED,
        Mode::Interactive => rand::random::<u64>(),
    });

    Ok(Invocation { mode, seed })
}

pub(crate) fn parse_invocation() -> Invocation {
    try_parse().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!("Try 'llm --help' for usage.");
        std::process::exit(2);
    })
}
