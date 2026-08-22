use std::{collections::BTreeSet, path::Path, process::Command};

fn run(arguments: &[&str], current_dir: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_llm"))
        .args(arguments)
        .current_dir(current_dir)
        .output()
        .expect("llm binary should start")
}

fn stdout(output: &std::process::Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("stdout should be UTF-8")
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8(output.stderr.clone()).expect("stderr should be UTF-8")
}

#[test]
fn help_flags_print_the_same_contract_without_loading_data() {
    let expected = concat!(
        "Usage: llm [--seed <n>] [--model <id-or-path>] [--epochs <n>] [--tiny] [--trace] [--temperature <t>] [--presence <c>] [--repetition <r>] [--top-p <p>] [--fluency <n>] [--eos] [--lr-decay <final_lr>] [--e2e <prompt> | --ask <prompt> | --eval | --train <file.jsonl> | --probe | --models | --demo]\n",
        "\n",
        "The operating path: start at the top, move down. Each step prepares\n",
        "the next; every number a step prints carries its seed.\n",
        "\n",
        "  1  llm --models                         pick an artifact from the trained catalog\n",
        "  2  llm --model <id> --ask \"<prompt>\"    one answer from it; decode knobs honored\n",
        "  3  llm --model <id>                     chat interactively (/help inside)\n",
        "  4  llm --demo                           watch raw text become a model, end to end\n",
        "  5  llm --tiny --train <corpus.jsonl>    teach your own model (add --eos --lr-decay)\n",
        "  6  llm --tiny --eval --model <path>     score it against held-out data\n",
        "  7  llm --eval --seed 42                 the micro arena oracle: fresh, seeded\n",
        "\n",
        "Decode knobs (usable with --tiny --eval, --ask, and chat):\n",
        "  --temperature <t>   sampling heat (> 0; 1.0 keeps the greedy pin)\n",
        "  --top-p <p>         nucleus mass cutoff ((0, 1]; 0 = off)\n",
        "  --presence <c>      flat penalty per seen word (>= 0; 0 = off)\n",
        "  --repetition <r>    count-scaled repeat penalty (>= 1; 1 = off)\n",
        "\n",
        "Training levers (with --tiny --train):\n",
        "  --eos               append </s> to every row: teach the model to stop\n",
        "  --lr-decay <lr>     linear per-epoch decay down to this final rate\n",
        "\n",
        "Reproducibility:\n",
        "  --seed <n>          default 42 on machine modes; same seed, same scores\n",
        "\n",
        "--e2e and --probe are contract probes: boundaries to test, not quality\n",
        "claims.\n",
        "\n",
        "Examples:\n",
        "  llm --models\n",
        "  llm --model stories-full --ask \"Once upon a time,\"\n",
        "  llm --model stories-full --ask \"Once upon a time,\" --temperature 0.7 --top-p 0.8 --presence 1.5 --repetition 1.1\n",
        "  llm --model watercycle-latest\n",
        "  llm --demo --seed 42\n",
        "  llm --tiny --train my-corpus.jsonl --epochs 6 --eos --lr-decay 5e-5 --model models/ts.bin\n",
        "  llm --tiny --eval --model models/ts.bin --fluency 20\n",
        "  llm --eval --seed 42\n",
        "  llm --probe --model stories-full --seed 42\n",
    );

    for flag in ["--help", "-h"] {
        let output = run(&[flag], &std::env::temp_dir());
        assert_eq!(output.status.code(), Some(0));
        assert_eq!(stdout(&output), expected);
        assert_eq!(stderr(&output), "");
    }
}

#[test]
fn version_prints_on_stdout_without_loading_data() {
    let output = run(&["--version"], &std::env::temp_dir());

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        stdout(&output),
        format!("llm {}\n", env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(stderr(&output), "");
}

#[test]
fn unknown_argument_fails_on_stderr() {
    let output = run(&["--unknown"], &std::env::temp_dir());

    assert_eq!(output.status.code(), Some(2));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        "error: unknown argument: --unknown\nTry 'llm --help' for usage.\n"
    );
}

#[test]
fn e2e_requires_exactly_one_prompt() {
    for (arguments, message) in [
        (
            vec!["--e2e"],
            "error: --e2e requires a prompt\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--e2e", "hello", "extra"],
            "error: mode argument accepts exactly one value\nTry 'llm --help' for usage.\n",
        ),
    ] {
        let output = run(&arguments, &std::env::temp_dir());
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(stdout(&output), "");
        assert_eq!(stderr(&output), message);
    }
}

#[test]
fn e2e_emits_one_json_line_with_the_public_schema() {
    let output = run(
        &["--e2e", "hello world"],
        Path::new(env!("CARGO_MANIFEST_DIR")),
    );

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(stderr(&output), "");
    let stdout = stdout(&output);
    assert_eq!(stdout.lines().count(), 1);

    let response: serde_json::Value =
        serde_json::from_str(stdout.trim_end()).expect("stdout should be one JSON object");
    let object = response.as_object().expect("response should be an object");
    let keys = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    assert_eq!(
        keys,
        BTreeSet::from(["output", "prompt", "status", "total_parameters"])
    );
    assert_eq!(response["status"].as_str(), Some("ok"));
    assert_eq!(response["prompt"].as_str(), Some("hello world"));
    assert!(
        response["output"]
            .as_str()
            .is_some_and(|output| !output.is_empty())
    );
    assert_eq!(response["total_parameters"].as_u64(), Some(385_776));
}

#[test]
fn e2e_oov_prompt_returns_explicit_fallback_not_silent_empty() {
    // "zzzzzz" is outside the water-cycle vocabulary; the contract demands
    // an explicit, non-empty answer instead of "" with status:ok.
    let output = run(&["--e2e", "zzzzzz"], Path::new(env!("CARGO_MANIFEST_DIR")));

    assert_eq!(output.status.code(), Some(0));
    let response: serde_json::Value =
        serde_json::from_str(stdout(&output).trim_end()).expect("one JSON object");
    assert_eq!(response["status"].as_str(), Some("ok"));
    assert_eq!(
        response["output"].as_str(),
        Some("Assistant : I do not know that word . </s>")
    );
}

#[test]
fn e2e_overlong_prompt_returns_explicit_truncation_report() {
    let long_prompt = "hello world ".repeat(200);
    let output = run(
        &["--e2e", long_prompt.trim_end()],
        Path::new(env!("CARGO_MANIFEST_DIR")),
    );

    assert_eq!(output.status.code(), Some(0));
    let response: serde_json::Value =
        serde_json::from_str(stdout(&output).trim_end()).expect("one JSON object");
    assert_eq!(response["status"].as_str(), Some("ok"));
    assert_eq!(
        response["output"].as_str(),
        Some("Assistant : The input is too long . </s>")
    );
}

#[test]
fn eval_rejects_bad_seed_arguments() {
    for (arguments, message) in [
        (
            vec!["--eval", "--seed"],
            "error: --seed requires a value\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--eval", "--seed", "abc"],
            "error: invalid seed: abc\nTry 'llm --help' for usage.\n",
        ),
    ] {
        let output = run(&arguments, &std::env::temp_dir());
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(stdout(&output), "");
        assert_eq!(stderr(&output), message);
    }
}

#[test]
fn tiny_eval_requires_a_checkpoint() {
    let output = run(&["--tiny", "--eval"], Path::new(env!("CARGO_MANIFEST_DIR")));
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        "error: --tiny requires --model <checkpoint> or --train <file.jsonl>\n"
    );
}

#[test]
fn missing_checkpoint_is_an_error_not_a_silent_fallback() {
    let output = run(
        &["--model", "does-not-exist.bin", "--e2e", "hello"],
        Path::new(env!("CARGO_MANIFEST_DIR")),
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        "error: checkpoint not found: does-not-exist.bin\n"
    );
}

#[test]
fn eval_and_e2e_are_mutually_exclusive() {
    let output = run(&["--eval", "--e2e", "hi"], &std::env::temp_dir());
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        "error: --ask, --demo, --e2e, --eval, --models, --probe, and --train are mutually exclusive\nTry 'llm --help' for usage.\n"
    );
}

#[test]
fn models_and_other_modes_are_mutually_exclusive() {
    let output = run(&["--models", "--eval"], &std::env::temp_dir());
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        "error: --ask, --demo, --e2e, --eval, --models, --probe, and --train are mutually exclusive\nTry 'llm --help' for usage.\n"
    );
}

#[test]
fn ask_demo_and_train_are_mutually_exclusive_with_every_mode() {
    for arguments in [
        vec!["--ask", "hi", "--eval"],
        vec!["--eval", "--ask", "hi"],
        vec!["--ask", "hi", "--demo"],
        vec!["--demo", "--train", "x.jsonl"],
        vec!["--demo", "--models"],
        vec!["--ask", "hi", "--probe"],
    ] {
        let output = run(&arguments, &std::env::temp_dir());
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(stdout(&output), "");
        assert_eq!(
            stderr(&output),
            "error: --ask, --demo, --e2e, --eval, --models, --probe, and --train are mutually exclusive\nTry 'llm --help' for usage.\n"
        );
    }
}

#[test]
fn models_emits_one_json_object_with_the_catalog() {
    let output = run(&["--models"], Path::new(env!("CARGO_MANIFEST_DIR")));
    assert_eq!(output.status.code(), Some(0));

    // stdout is the machine channel: exactly one JSON object.
    let stdout_text = stdout(&output);
    assert_eq!(stdout_text.lines().count(), 1);

    let response: serde_json::Value =
        serde_json::from_str(stdout_text.trim_end()).expect("stdout should be one JSON object");
    assert_eq!(response["status"].as_str(), Some("ok"));
    let catalog = response["catalog"]
        .as_array()
        .expect("catalog must be an array");
    assert!(!catalog.is_empty(), "catalog must list the trained models");
    assert!(
        catalog
            .iter()
            .all(|entry| entry["path"].is_string() && entry["seed"].is_u64())
    );

    // stderr carries the human-readable table.
    assert!(
        stderr(&output).contains("ID          Family      Parameters"),
        "the human table belongs on stderr"
    );
}

#[test]
fn double_dash_is_a_noop_separator() {
    // `llm -- --model <path>` must read exactly like `llm --model <path>`.
    let output = run(&["--", "--e2e"], &std::env::temp_dir());
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        "error: --e2e requires a prompt\nTry 'llm --help' for usage.\n"
    );
}

#[test]
fn trace_is_rejected_outside_interactive_mode() {
    for arguments in [
        vec!["--trace", "--eval"],
        vec!["--eval", "--trace"],
        vec!["--trace", "--e2e", "hi"],
        vec!["--trace", "--ask", "hi"],
        vec!["--trace", "--demo"],
        vec!["--trace", "--probe"],
        vec!["--trace", "--train", "data/pretraining_data.json"],
    ] {
        let output = run(&arguments, &std::env::temp_dir());
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(stdout(&output), "");
        assert_eq!(
            stderr(&output),
            "error: --trace is only available in interactive mode\nTry 'llm --help' for usage.\n"
        );
    }
}

#[test]
fn temperature_requires_a_decode_surface() {
    for (arguments, message) in [
        (
            vec!["--eval", "--temperature", "0.8"],
            "error: --temperature requires --tiny --eval, --ask, or interactive chat\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--train", "x.jsonl", "--temperature", "0.8"],
            "error: --temperature requires --tiny --eval, --ask, or interactive chat\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--demo", "--temperature", "0.8"],
            "error: --temperature requires --tiny --eval, --ask, or interactive chat\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--eval", "--temperature", "0"],
            "error: --temperature must be positive\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--ask", "hi", "--temperature", "0"],
            "error: --temperature must be positive\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--temperature", "abc"],
            "error: invalid temperature: abc\nTry 'llm --help' for usage.\n",
        ),
    ] {
        let output = run(&arguments, &std::env::temp_dir());
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(stdout(&output), "");
        assert_eq!(stderr(&output), message);
    }
}

#[test]
fn fluency_requires_tiny_eval() {
    for (arguments, message) in [
        (
            vec!["--fluency", "20"],
            "error: --fluency requires --tiny --eval\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--train", "x.jsonl", "--fluency", "20"],
            "error: --fluency requires --tiny --eval\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--eval", "--fluency", "0"],
            "error: --fluency needs at least one sample\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--eval", "--fluency", "abc"],
            "error: invalid fluency sample count: abc\nTry 'llm --help' for usage.\n",
        ),
    ] {
        let output = run(&arguments, &std::env::temp_dir());
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(stdout(&output), "");
        assert_eq!(stderr(&output), message);
    }
}

#[test]
fn penalties_require_a_decode_surface() {
    for (arguments, message) in [
        (
            vec!["--tiny", "--train", "x.jsonl", "--repetition", "1.1"],
            "error: --presence and --repetition require --tiny --eval, --ask, or interactive chat\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--eval", "--presence", "-1.0"],
            "error: --presence must be non-negative\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--eval", "--repetition", "0.9"],
            "error: --repetition must be >= 1.0\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--ask", "hi", "--presence", "abc"],
            "error: invalid presence: abc\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--ask", "hi", "--repetition", "xyz"],
            "error: invalid repetition: xyz\nTry 'llm --help' for usage.\n",
        ),
    ] {
        let output = run(&arguments, &std::env::temp_dir());
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(stdout(&output), "");
        assert_eq!(stderr(&output), message);
    }
}

#[test]
fn top_p_requires_a_decode_surface() {
    for (arguments, message) in [
        (
            vec!["--tiny", "--train", "x.jsonl", "--top-p", "0.8"],
            "error: --top-p requires --tiny --eval, --ask, or interactive chat\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--eval", "--top-p", "0"],
            "error: --top-p must be in (0, 1]\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--eval", "--top-p", "1.5"],
            "error: --top-p must be in (0, 1]\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--ask", "hi", "--top-p", "abc"],
            "error: invalid top-p: abc\nTry 'llm --help' for usage.\n",
        ),
    ] {
        let output = run(&arguments, &std::env::temp_dir());
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(stdout(&output), "");
        assert_eq!(stderr(&output), message);
    }
}

#[test]
fn probe_requires_a_checkpoint() {
    let output = run(&["--probe"], Path::new(env!("CARGO_MANIFEST_DIR")));
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        "error: --probe requires --model <checkpoint>\n"
    );
}

#[test]
fn ask_requires_a_checkpoint_flag() {
    let output = run(
        &["--ask", "Once upon a time,"],
        Path::new(env!("CARGO_MANIFEST_DIR")),
    );
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        "error: --ask requires --model <checkpoint>\n"
    );
}

#[test]
fn ask_with_missing_checkpoint_is_an_error_not_a_fallback() {
    let output = run(
        &["--model", "does-not-exist.bin", "--ask", "hi"],
        Path::new(env!("CARGO_MANIFEST_DIR")),
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        "error: checkpoint not found: does-not-exist.bin\n"
    );
}

#[test]
fn ask_needs_exactly_one_prompt() {
    for (arguments, message) in [
        (
            vec!["--model", "x.bin", "--ask"],
            "error: --ask requires a prompt\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--ask", "hi", "extra"],
            "error: mode argument accepts exactly one value\nTry 'llm --help' for usage.\n",
        ),
    ] {
        let output = run(&arguments, &std::env::temp_dir());
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(stdout(&output), "");
        assert_eq!(stderr(&output), message);
    }
}

#[test]
fn eos_and_lr_decay_require_tiny_train() {
    for (arguments, message) in [
        (
            vec!["--tiny", "--eval", "--eos"],
            "error: --eos requires --tiny --train\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--train", "x.jsonl", "--eos"],
            "error: --eos requires --tiny --train\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--eval", "--lr-decay", "5e-5"],
            "error: --lr-decay requires --tiny --train\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--train", "x.jsonl", "--lr-decay", "0"],
            "error: --lr-decay must be positive\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--train", "x.jsonl", "--lr-decay", "abc"],
            "error: invalid lr-decay: abc\nTry 'llm --help' for usage.\n",
        ),
    ] {
        let output = run(&arguments, &std::env::temp_dir());
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(stdout(&output), "");
        assert_eq!(stderr(&output), message);
    }
}

/// Hand-rolled FNV-1a 64: the artifact-immutability probe needs a stable
/// content fingerprint without adding a dependency.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[test]
fn ask_emits_one_json_object_and_never_touches_the_checkpoint() {
    // Build and save a small trained checkpoint, then ask it twice: the
    // happy path must print exactly one JSON object with the decode block,
    // and the artifact bytes must be identical before and after both runs
    // (--ask never trains, never saves).
    let checkpoint_dir = std::env::temp_dir().join("rustgpt-ask-checkpoint");
    std::fs::create_dir_all(&checkpoint_dir).unwrap();
    let path = checkpoint_dir.join("model.bin");

    let data = llm::Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        llm::DatasetType::JSON,
    );
    llm::set_seed(21);
    let mut vocab_set = std::collections::HashSet::new();
    llm::Vocab::process_text_for_vocab(&data.pretraining_data, &mut vocab_set);
    llm::Vocab::process_text_for_vocab(&data.chat_training_data, &mut vocab_set);
    let mut words: Vec<String> = vocab_set.into_iter().collect();
    words.sort();
    let refs: Vec<&str> = words.iter().map(String::as_str).collect();
    let vocab = llm::Vocab::new(refs);
    let network: Vec<Box<dyn llm::Layer>> = vec![
        Box::new(llm::Embeddings::new(vocab.clone())),
        Box::new(llm::transformer::TransformerBlock::new(
            llm::EMBEDDING_DIM,
            llm::HIDDEN_DIM,
        )),
        Box::new(llm::transformer::TransformerBlock::new(
            llm::EMBEDDING_DIM,
            llm::HIDDEN_DIM,
        )),
        Box::new(llm::transformer::TransformerBlock::new(
            llm::EMBEDDING_DIM,
            llm::HIDDEN_DIM,
        )),
        Box::new(llm::output_projection::OutputProjection::new(
            llm::EMBEDDING_DIM,
            vocab.words.len(),
        )),
    ];
    let examples: Vec<&str> = data.chat_training_data.iter().map(String::as_str).collect();
    let mut model = llm::LLM::new(vocab, network);
    model.train(examples, 2, 0.0005);
    llm::save(&model, path.to_str().unwrap()).expect("checkpoint save should succeed");
    let before = fnv1a(&std::fs::read(&path).unwrap());

    let output = run(
        &[
            "--model",
            path.to_str().unwrap(),
            "--ask",
            "Once upon a time,",
            "--temperature",
            "0.7",
            "--top-p",
            "0.8",
            "--presence",
            "1.5",
            "--repetition",
            "1.1",
        ],
        Path::new(env!("CARGO_MANIFEST_DIR")),
    );
    assert_eq!(output.status.code(), Some(0));
    let stdout_text = stdout(&output);
    assert_eq!(stdout_text.lines().count(), 1);

    let response: serde_json::Value =
        serde_json::from_str(stdout_text.trim_end()).expect("one JSON object");
    let object = response.as_object().expect("response should be an object");
    let keys = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    assert_eq!(
        keys,
        BTreeSet::from([
            "decode",
            "output",
            "prompt",
            "seed",
            "status",
            "total_parameters"
        ])
    );
    assert_eq!(response["status"].as_str(), Some("ok"));
    assert_eq!(response["prompt"].as_str(), Some("Once upon a time,"));
    assert_eq!(response["seed"].as_u64(), Some(42));
    assert!(
        response["output"]
            .as_str()
            .is_some_and(|output| !output.is_empty())
    );
    let decode = response["decode"].as_object().expect("decode block");
    let decode_keys = decode.keys().map(String::as_str).collect::<BTreeSet<_>>();
    assert_eq!(
        decode_keys,
        BTreeSet::from(["presence", "repetition", "temperature", "top_p"])
    );
    // f32 knobs widen through the JSON f64 channel; compare within f32
    // printing precision.
    let near = |key: &str, expected: f64| {
        assert!(
            decode[key]
                .as_f64()
                .is_some_and(|value| (value - expected).abs() < 1e-6),
            "decode.{key} should be ~{expected}"
        );
    };
    near("temperature", 0.7);
    near("top_p", 0.8);
    near("presence", 1.5);
    near("repetition", 1.1);

    let after = fnv1a(&std::fs::read(&path).unwrap());
    assert_eq!(before, after, "--ask must leave the checkpoint untouched");

    std::fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn model_requires_a_path() {
    let output = run(&["--model"], &std::env::temp_dir());
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(stdout(&output), "");
    assert_eq!(
        stderr(&output),
        "error: --model requires a path\nTry 'llm --help' for usage.\n"
    );
}

#[test]
fn e2e_with_checkpoint_loads_and_serves_the_saved_model() {
    let checkpoint_dir = std::env::temp_dir().join("rustgpt-cli-checkpoint");
    std::fs::create_dir_all(&checkpoint_dir).unwrap();
    let path = checkpoint_dir.join("model.bin");

    let data = llm::Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        llm::DatasetType::JSON,
    );
    llm::set_seed(21);
    let model = {
        let mut vocab_set = std::collections::HashSet::new();
        llm::Vocab::process_text_for_vocab(&data.pretraining_data, &mut vocab_set);
        llm::Vocab::process_text_for_vocab(&data.chat_training_data, &mut vocab_set);
        let mut words: Vec<String> = vocab_set.into_iter().collect();
        words.sort();
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let vocab = llm::Vocab::new(refs);
        let network: Vec<Box<dyn llm::Layer>> = vec![
            Box::new(llm::Embeddings::new(vocab.clone())),
            Box::new(llm::transformer::TransformerBlock::new(
                llm::EMBEDDING_DIM,
                llm::HIDDEN_DIM,
            )),
            Box::new(llm::transformer::TransformerBlock::new(
                llm::EMBEDDING_DIM,
                llm::HIDDEN_DIM,
            )),
            Box::new(llm::transformer::TransformerBlock::new(
                llm::EMBEDDING_DIM,
                llm::HIDDEN_DIM,
            )),
            Box::new(llm::output_projection::OutputProjection::new(
                llm::EMBEDDING_DIM,
                vocab.words.len(),
            )),
        ];
        let examples: Vec<&str> = data.chat_training_data.iter().map(String::as_str).collect();
        let mut model = llm::LLM::new(vocab, network);
        model.train(examples, 2, 0.0005);
        model
    };
    llm::save(&model, path.to_str().unwrap()).expect("checkpoint save should succeed");

    let output = run(
        &["--model", path.to_str().unwrap(), "--e2e", "hello world"],
        Path::new(env!("CARGO_MANIFEST_DIR")),
    );
    assert_eq!(output.status.code(), Some(0));
    let response: serde_json::Value =
        serde_json::from_str(stdout(&output).trim_end()).expect("one JSON object");
    assert_eq!(response["status"].as_str(), Some("ok"));
    assert_eq!(response["total_parameters"].as_u64(), Some(385_776));

    let oov = run(
        &["--model", path.to_str().unwrap(), "--e2e", "zzzzzz"],
        Path::new(env!("CARGO_MANIFEST_DIR")),
    );
    assert_eq!(oov.status.code(), Some(0));
    let response: serde_json::Value =
        serde_json::from_str(stdout(&oov).trim_end()).expect("one JSON object");
    assert_eq!(response["status"].as_str(), Some("ok"));
    assert!(
        response["output"]
            .as_str()
            .is_some_and(|output| !output.is_empty())
    );

    std::fs::remove_dir_all(checkpoint_dir).ok();
}

/// Save a small trained checkpoint and return its path; the shared setup
/// of the interactive-session probes.
fn trained_checkpoint(dir: &str) -> String {
    let checkpoint_dir = std::env::temp_dir().join(dir);
    std::fs::create_dir_all(&checkpoint_dir).unwrap();
    let path = checkpoint_dir.join("model.bin");

    let data = llm::Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        llm::DatasetType::JSON,
    );
    llm::set_seed(21);
    let mut vocab_set = std::collections::HashSet::new();
    llm::Vocab::process_text_for_vocab(&data.pretraining_data, &mut vocab_set);
    llm::Vocab::process_text_for_vocab(&data.chat_training_data, &mut vocab_set);
    let mut words: Vec<String> = vocab_set.into_iter().collect();
    words.sort();
    let refs: Vec<&str> = words.iter().map(String::as_str).collect();
    let vocab = llm::Vocab::new(refs);
    let network: Vec<Box<dyn llm::Layer>> = vec![
        Box::new(llm::Embeddings::new(vocab.clone())),
        Box::new(llm::transformer::TransformerBlock::new(
            llm::EMBEDDING_DIM,
            llm::HIDDEN_DIM,
        )),
        Box::new(llm::transformer::TransformerBlock::new(
            llm::EMBEDDING_DIM,
            llm::HIDDEN_DIM,
        )),
        Box::new(llm::transformer::TransformerBlock::new(
            llm::EMBEDDING_DIM,
            llm::HIDDEN_DIM,
        )),
        Box::new(llm::output_projection::OutputProjection::new(
            llm::EMBEDDING_DIM,
            vocab.words.len(),
        )),
    ];
    let examples: Vec<&str> = data.chat_training_data.iter().map(String::as_str).collect();
    let mut model = llm::LLM::new(vocab, network);
    model.train(examples, 2, 0.0005);
    llm::save(&model, path.to_str().unwrap()).expect("checkpoint save should succeed");
    path.to_str().unwrap().to_string()
}

#[test]
fn chat_slash_commands_mutate_only_valid_knobs() {
    let path = trained_checkpoint("rustgpt-chat-session");
    let script = "/help\n/config\n/temp abc\n/config\n/bogus\n/temp 0.7\n/top-p 0\n/config\n/reset\n/config\n/exit\n";
    let mut child = Command::new(env!("CARGO_BIN_EXE_llm"))
        .arg("--model")
        .arg(&path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("llm binary should start");
    use std::io::Write as _;
    child
        .stdin
        .as_mut()
        .expect("piped stdin")
        .write_all(script.as_bytes())
        .unwrap();
    let output = child.wait_with_output().expect("session should finish");

    assert_eq!(output.status.code(), Some(0));
    let out = String::from_utf8(output.stdout.clone()).unwrap();

    // /help lists the commands.
    assert!(out.contains("Commands:"), "/help must list commands");
    assert!(out.contains("/top-p <p>"));

    // A bad value is rejected and leaves the config unchanged.
    assert!(out.contains("error: 'abc' is not a number (config unchanged)"));
    // Unknown commands explain themselves.
    assert!(out.contains("Unknown command '/bogus'."));
    // Out-of-range knob values are rejected by the same validation.
    assert!(out.contains("error: top-p must be in (0, 1] (config unchanged)"));

    // Greedy default; sampling after a valid set and in every later
    // /config until reset; greedy again after reset.
    assert_eq!(out.matches("config: greedy").count(), 4);
    assert_eq!(out.matches("config: sampling").count(), 2);

    // The literal end marker never leaks into a rendered answer.
    for line in out.lines().filter(|l| l.starts_with("Model output:")) {
        assert!(!line.contains("</s>"), "marker must render clean: {line}");
    }

    std::fs::remove_dir_all(std::path::Path::new(&path).parent().unwrap()).ok();
}

#[test]
fn chat_session_ends_cleanly_at_end_of_input() {
    let path = trained_checkpoint("rustgpt-chat-eof");
    let output = run_with_stdin(&["--model", &path], b"What is rain?\n");
    assert_eq!(output.status.code(), Some(0));
    let out = stdout(&output);
    assert!(out.contains("Model output:"), "the prompt must be answered");
    assert_eq!(
        out.matches("Exiting interactive mode.").count(),
        1,
        "EOF ends the session exactly once"
    );

    std::fs::remove_dir_all(std::path::Path::new(&path).parent().unwrap()).ok();
}

fn run_with_stdin(arguments: &[&str], input: &[u8]) -> std::process::Output {
    use std::io::Write as _;
    let mut child = Command::new(env!("CARGO_BIN_EXE_llm"))
        .args(arguments)
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("llm binary should start");
    child
        .stdin
        .as_mut()
        .expect("piped stdin")
        .write_all(input)
        .unwrap();
    child.wait_with_output().expect("session should finish")
}

#[test]
fn chat_with_a_catalog_id_loads_it_and_never_creates_files() {
    // Regression pin: `--model <catalog-id>` must LOAD the resolved
    // artifact and chat with it. The old behavior re-checked the raw
    // argument, missed the file, treated the id as a first-run save
    // target, double-trained the loaded model, and wrote a stray
    // artifact named after the id into the working directory.
    // The pin needs the cataloged artifact on disk; artifacts are
    // gitignored, so a fresh clone skips this observation (same pattern
    // as the kv-cache throughput probe) instead of exercising the
    // documented first-run branch.
    let artifact = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/watercycle-latest.bin");
    if !artifact.exists() {
        eprintln!("catalog-id chat observation skipped: models/watercycle-latest.bin not present");
        return;
    }

    let stray = Path::new(env!("CARGO_MANIFEST_DIR")).join("watercycle-latest");
    let before = std::fs::read(&stray).ok();
    let _ = std::fs::remove_file(&stray);

    let output = run_with_stdin(&["--model", "watercycle-latest"], b"exit\n");
    assert_eq!(output.status.code(), Some(0));
    let out = stdout(&output);
    assert!(
        out.contains("LOADED MODEL"),
        "a catalog id must load its artifact: {out}"
    );
    assert!(
        !out.contains("BEFORE TRAINING"),
        "loading must not fall into the training branch: {out}"
    );
    assert!(
        !stray.exists(),
        "chat must never write an artifact into the working directory"
    );

    // Restore whatever the workspace had before the probe.
    match before {
        Some(bytes) => {
            let _ = std::fs::write(&stray, bytes);
        }
        None => {
            let _ = std::fs::remove_file(&stray);
        }
    }
}
