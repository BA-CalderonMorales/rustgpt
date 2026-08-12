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

## Evaluation Mode

```bash
cargo run -- --eval --seed 42
```

The `--eval` path trains both phases (progress on stderr), scores every
prompt/reference pair in `data/heldout.json`, and writes exactly one JSON
object to stdout with per-item and summary scores. The score formula is
defined in terms of greedy generation against each reference:

- `exact` -- generated tokens equal reference tokens,
- `prefix` -- generated tokens are a non-empty prefix of the reference,
- `accuracy` -- matching positions over the longer of the two sequences.

Summary fields are `exact_matches`, `prefix_matches`, and `mean_accuracy`.
Initialization is seeded; `--seed <n>` (default 42) makes every run
reproducible: same seed, same model, same scores. Every claimed result must
carry its seed. Debug builds are roughly 40x slower than release on a laptop;
use `cargo build --release` for real measurements.

## Checkpoints

```bash
cargo run --release -- --model models/mine.bin --eval --seed 42
cargo run --release -- --model models/mine.bin --e2e "hello world"
```

`--model <path>` loads the checkpoint when the file exists; training modes
(`--eval`, interactive) then re-save it after training, creating parent
directories as needed. Checkpoints are format v1 (`RGPT_V1` magic header):
the seed, the vocabulary, and each layer's learned weights. Optimizer state
and transient caches are never stored, so a loaded model starts with fresh
optimizers. Checkpoints live under `models/` (gitignored); a result is
reproducible as the pair (checkpoint or seed) plus its eval JSON.

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
