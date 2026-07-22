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
        "Usage: llm [--e2e <prompt>]\n",
        "\n",
        "Examples:\n",
        "  llm\n",
        "  llm --e2e \"hello world\"\n",
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
    assert_eq!(stdout(&output), "llm 0.1.0\n");
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
            "error: --e2e accepts exactly one prompt\nTry 'llm --help' for usage.\n",
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
    assert_eq!(response["total_parameters"].as_u64(), Some(380_893));
}
