# Running and Development

## Prerequisites

Install a current stable Rust toolchain with Cargo.

## Interactive Mode

```bash
git clone https://github.com/BA-CalderonMorales/rustgpt.git
cd rustgpt
cargo run
```

The default path builds a vocabulary, performs pre-training and instruction
tuning, prints a sample prediction, and then accepts prompts until `exit` is
entered.

```text
Enter prompt: How do mountains form?
Model output: ...
```

Because this is a small educational model, generated text should be treated as
an observation of the mechanics rather than a quality benchmark.

The original project demonstration is available in this
[GitHub attachment](https://github.com/user-attachments/assets/ec4a4100-b03a-4b3c-a7d6-806ea54ed4ed).

## Machine-Readable Mode

```bash
cargo run -- --e2e "hello world"
```

The `--e2e` path initializes the model and generates a response without
training or interactive input. It writes one JSON object:

```json
{"output":"...","prompt":"hello world","status":"ok","total_parameters":123}
```

This interface exists for smoke tests and public-contract evaluation. It does
not claim that the generated text is semantically correct.

## Development Commands

```bash
cargo fmt --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --all-targets
cargo build --release
```

To inspect output from a particular integration test:

```bash
cargo test --test llm_test -- --nocapture
```
