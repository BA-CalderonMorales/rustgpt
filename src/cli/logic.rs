use super::{Invocation, Mode};

const DEFAULT_SEED: u64 = 42;
const DEFAULT_EPOCHS: usize = 100;

fn usage() {
    println!(
        "Usage: llm [--seed <n>] [--model <path>] [--epochs <n>] [--tiny] [--e2e <prompt> | --eval | --train <file.jsonl>]"
    );
    println!();
    println!("Examples:");
    println!("  llm");
    println!("  llm --e2e \"hello world\"");
    println!("  llm --eval --seed 42");
    println!("  llm --model models/mine.bin --eval --seed 42");
    println!(
        "  llm --tiny --train models/tinystories/train.jsonl --epochs 2 --model models/ts.bin"
    );
}

fn try_parse() -> Result<Invocation, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0usize;
    let mut mode: Option<Mode> = None;
    let mut seed: Option<u64> = None;
    let mut model: Option<String> = None;
    let mut epochs = DEFAULT_EPOCHS;
    let mut tiny = false;

    while index < args.len() {
        let argument = args[index].as_str();
        index += 1;
        // A bare token (not a flag) directly after a mode's value is that
        // mode's second positional; any other bare token is unknown.
        if !argument.starts_with('-') && mode.is_some() {
            return Err("mode argument accepts exactly one value".to_string());
        }
        match argument {
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
                    .get(index)
                    .ok_or_else(|| "--seed requires a value".to_string())?;
                index += 1;
                seed = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid seed: {value}"))?,
                );
            }
            "--model" => {
                let value = args
                    .get(index)
                    .ok_or_else(|| "--model requires a path".to_string())?;
                index += 1;
                model = Some(value.clone());
            }
            "--epochs" => {
                let value = args
                    .get(index)
                    .ok_or_else(|| "--epochs requires a value".to_string())?;
                index += 1;
                epochs = value
                    .parse()
                    .map_err(|_| format!("invalid epochs: {value}"))?;
            }
            "--tiny" => tiny = true,
            "--e2e" => {
                if mode.is_some() {
                    return Err("--e2e, --eval, and --train are mutually exclusive".to_string());
                }
                let prompt = args
                    .get(index)
                    .ok_or_else(|| "--e2e requires a prompt".to_string())?;
                index += 1;
                mode = Some(Mode::E2e {
                    prompt: prompt.clone(),
                });
            }
            "--eval" => {
                if mode.is_some() {
                    return Err("--e2e, --eval, and --train are mutually exclusive".to_string());
                }
                mode = Some(Mode::Eval);
            }
            "--train" => {
                if mode.is_some() {
                    return Err("--e2e, --eval, and --train are mutually exclusive".to_string());
                }
                let path = args
                    .get(index)
                    .ok_or_else(|| "--train requires a file path".to_string())?;
                index += 1;
                mode = Some(Mode::Train { path: path.clone() });
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let mode = mode.unwrap_or(Mode::Interactive);
    let seed = seed.unwrap_or(match mode {
        Mode::Eval | Mode::E2e { .. } | Mode::Train { .. } => DEFAULT_SEED,
        Mode::Interactive => rand::random::<u64>(),
    });

    Ok(Invocation {
        mode,
        seed,
        model,
        epochs,
        tiny,
    })
}

pub(crate) fn parse_invocation() -> Invocation {
    try_parse().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!("Try 'llm --help' for usage.");
        std::process::exit(2);
    })
}
