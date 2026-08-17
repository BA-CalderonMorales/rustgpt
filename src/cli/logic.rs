use super::{Invocation, Mode};

const DEFAULT_SEED: u64 = 42;
const DEFAULT_EPOCHS: usize = 100;
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_PRESENCE: f32 = 0.0;
const DEFAULT_REPETITION: f32 = 1.0;
const DEFAULT_TOP_P: f32 = 0.0;

fn usage() {
    // The one-line contract: flags first, then exactly one mode.
    println!(
        "Usage: llm [--seed <n>] [--model <path>] [--epochs <n>] [--tiny] [--trace] [--temperature <t>] [--fluency <n>] [--presence <c>] [--repetition <r>] [--top-p <p>] [--e2e <prompt> | --eval | --train <file.jsonl> | --probe | --models]"
    );
    println!();

    // One working command per surface.
    println!("Examples:");
    println!("  llm");
    println!("  llm --trace --seed 42");
    println!("  llm --models");
    println!("  llm --model models/watercycle-latest.bin");
    println!("  llm --e2e \"hello world\"");
    println!("  llm --eval --seed 42");
    println!("  llm --model models/mine.bin --eval --seed 42");
    println!(
        "  llm --tiny --train models/tinystories/train.jsonl --epochs 2 --model models/ts.bin"
    );
    println!("  llm --tiny --eval --model models/ts.bin --fluency 20");
    println!("  llm --tiny --eval --model models/ts.bin --temperature 0.7 --presence 1.5");
    println!(
        "  llm --tiny --eval --model models/ts.bin --temperature 0.7 --top-p 0.8 --presence 1.5"
    );
    println!("  llm --probe --model models/mine.bin --seed 42");
}

fn try_parse() -> Result<Invocation, String> {
    // Collect the argument vector and the mutable parse state.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0usize;
    let mut mode: Option<Mode> = None;
    let mut seed: Option<u64> = None;
    let mut model: Option<String> = None;
    let mut epochs = DEFAULT_EPOCHS;
    let mut tiny = false;
    let mut trace = false;
    let mut temperature = DEFAULT_TEMPERATURE;
    let mut fluency: Option<usize> = None;
    let mut presence = DEFAULT_PRESENCE;
    let mut repetition = DEFAULT_REPETITION;
    let mut top_p = DEFAULT_TOP_P;
    let mut top_p_given = false;

    // Consume every argument in order.
    while index < args.len() {
        // A bare token (not a flag) directly after a mode's value is that
        // mode's second positional; any other bare token is unknown. A
        // bare "--" is a no-op separator: this CLI has no positionals, so
        // it exists so `llm -- --model <path>` reads naturally.
        let argument = args[index].as_str();
        index += 1;
        if argument == "--" {
            continue;
        }
        if !argument.starts_with('-') && mode.is_some() {
            return Err("mode argument accepts exactly one value".to_string());
        }

        // Dispatch the flag (or bare token) to its parse action.
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
            "--trace" => trace = true,
            "--temperature" => {
                let value = args
                    .get(index)
                    .ok_or_else(|| "--temperature requires a value".to_string())?;
                index += 1;
                temperature = value
                    .parse()
                    .map_err(|_| format!("invalid temperature: {value}"))?;
            }
            "--fluency" => {
                let value = args
                    .get(index)
                    .ok_or_else(|| "--fluency requires a value".to_string())?;
                index += 1;
                fluency = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid fluency sample count: {value}"))?,
                );
            }
            "--presence" => {
                let value = args
                    .get(index)
                    .ok_or_else(|| "--presence requires a value".to_string())?;
                index += 1;
                presence = value
                    .parse()
                    .map_err(|_| format!("invalid presence: {value}"))?;
            }
            "--repetition" => {
                let value = args
                    .get(index)
                    .ok_or_else(|| "--repetition requires a value".to_string())?;
                index += 1;
                repetition = value
                    .parse()
                    .map_err(|_| format!("invalid repetition: {value}"))?;
            }
            "--top-p" => {
                let value = args
                    .get(index)
                    .ok_or_else(|| "--top-p requires a value".to_string())?;
                index += 1;
                top_p = value
                    .parse()
                    .map_err(|_| format!("invalid top-p: {value}"))?;
                top_p_given = true;
            }
            "--e2e" => {
                if mode.is_some() {
                    return Err(
                        "--e2e, --eval, --train, --probe, and --models are mutually exclusive"
                            .to_string(),
                    );
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
                    return Err(
                        "--e2e, --eval, --train, --probe, and --models are mutually exclusive"
                            .to_string(),
                    );
                }
                mode = Some(Mode::Eval);
            }
            "--train" => {
                if mode.is_some() {
                    return Err(
                        "--e2e, --eval, --train, --probe, and --models are mutually exclusive"
                            .to_string(),
                    );
                }
                let path = args
                    .get(index)
                    .ok_or_else(|| "--train requires a file path".to_string())?;
                index += 1;
                mode = Some(Mode::Train { path: path.clone() });
            }
            "--probe" => {
                if mode.is_some() {
                    return Err(
                        "--e2e, --eval, --train, --probe, and --models are mutually exclusive"
                            .to_string(),
                    );
                }
                mode = Some(Mode::Probe);
            }
            "--models" => {
                if mode.is_some() {
                    return Err(
                        "--e2e, --eval, --train, --probe, and --models are mutually exclusive"
                            .to_string(),
                    );
                }
                mode = Some(Mode::Models);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    // Resolve the mode default and the trace restriction.
    let mode = mode.unwrap_or(Mode::Interactive);
    if trace && !matches!(mode, Mode::Interactive) {
        return Err("--trace is only available in interactive mode".to_string());
    }

    // The temperature knob belongs to the tiny-lane eval formula alone.
    if temperature != DEFAULT_TEMPERATURE && !(tiny && matches!(mode, Mode::Eval)) {
        return Err("--temperature requires --tiny --eval".to_string());
    }
    if temperature <= 0.0 {
        return Err("--temperature must be positive".to_string());
    }

    // The fluency yardstick belongs to the tiny-lane eval formula alone.
    if fluency.is_some() && !(tiny && matches!(mode, Mode::Eval)) {
        return Err("--fluency requires --tiny --eval".to_string());
    }
    if fluency == Some(0) {
        return Err("--fluency needs at least one sample".to_string());
    }

    // The penalty and top-p knobs belong to the tiny-lane eval formula
    // alone.
    if (presence != DEFAULT_PRESENCE || repetition != DEFAULT_REPETITION)
        && !(tiny && matches!(mode, Mode::Eval))
    {
        return Err("--presence and --repetition require --tiny --eval".to_string());
    }
    if presence < 0.0 {
        return Err("--presence must be non-negative".to_string());
    }
    if repetition < 1.0 {
        return Err("--repetition must be >= 1.0".to_string());
    }
    if top_p_given && !(tiny && matches!(mode, Mode::Eval)) {
        return Err("--top-p requires --tiny --eval".to_string());
    }
    if top_p_given && (top_p <= 0.0 || top_p > 1.0) {
        return Err("--top-p must be in (0, 1]".to_string());
    }

    // Resolve the seed: machine modes default to 42, interactive draws
    // a random seed unless one was given.
    let seed = seed.unwrap_or(match mode {
        Mode::Eval | Mode::E2e { .. } | Mode::Train { .. } | Mode::Probe | Mode::Models => {
            DEFAULT_SEED
        }
        Mode::Interactive => rand::random::<u64>(),
    });

    // Assemble the invocation.
    Ok(Invocation {
        mode,
        seed,
        model,
        epochs,
        tiny,
        trace,
        temperature,
        fluency,
        presence,
        repetition,
        top_p,
    })
}

pub(crate) fn parse_invocation() -> Invocation {
    // A parse error is a usage error: exact message on stderr, exit 2.
    try_parse().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!("Try 'llm --help' for usage.");
        std::process::exit(2);
    })
}
