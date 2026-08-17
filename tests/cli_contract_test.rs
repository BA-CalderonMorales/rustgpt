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
        "Usage: llm [--seed <n>] [--model <path>] [--epochs <n>] [--tiny] [--trace] [--temperature <t>] [--fluency <n>] [--presence <c>] [--repetition <r>] [--top-p <p>] [--e2e <prompt> | --eval | --train <file.jsonl> | --probe]\n",
        "\n",
        "Examples:\n",
        "  llm\n",
        "  llm --trace --seed 42\n",
        "  llm --e2e \"hello world\"\n",
        "  llm --eval --seed 42\n",
        "  llm --model models/mine.bin --eval --seed 42\n",
        "  llm --tiny --train models/tinystories/train.jsonl --epochs 2 --model models/ts.bin\n",
        "  llm --tiny --eval --model models/ts.bin --fluency 20\n",
        "  llm --tiny --eval --model models/ts.bin --temperature 0.7 --presence 1.5\n",
        "  llm --tiny --eval --model models/ts.bin --temperature 0.7 --top-p 0.8 --presence 1.5\n",
        "  llm --probe --model models/mine.bin --seed 42\n",
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
        "error: --e2e, --eval, --train, and --probe are mutually exclusive\nTry 'llm --help' for usage.\n"
    );
}

#[test]
fn trace_is_rejected_outside_interactive_mode() {
    for arguments in [
        vec!["--trace", "--eval"],
        vec!["--eval", "--trace"],
        vec!["--trace", "--e2e", "hi"],
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
fn temperature_requires_tiny_eval() {
    for (arguments, message) in [
        (
            vec!["--temperature", "0.8"],
            "error: --temperature requires --tiny --eval\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--eval", "--temperature", "0.8"],
            "error: --temperature requires --tiny --eval\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--train", "x.jsonl", "--temperature", "0.8"],
            "error: --temperature requires --tiny --eval\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--eval", "--temperature", "0"],
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
fn penalties_require_tiny_eval() {
    for (arguments, message) in [
        (
            vec!["--presence", "1.5"],
            "error: --presence and --repetition require --tiny --eval\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--train", "x.jsonl", "--repetition", "1.1"],
            "error: --presence and --repetition require --tiny --eval\nTry 'llm --help' for usage.\n",
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
            vec!["--tiny", "--eval", "--presence", "abc"],
            "error: invalid presence: abc\nTry 'llm --help' for usage.\n",
        ),
        (
            vec!["--tiny", "--eval", "--repetition", "xyz"],
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
fn top_p_requires_tiny_eval() {
    for (arguments, message) in [
        (
            vec!["--top-p", "0.8"],
            "error: --top-p requires --tiny --eval\nTry 'llm --help' for usage.\n",
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
            vec!["--tiny", "--eval", "--top-p", "abc"],
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
