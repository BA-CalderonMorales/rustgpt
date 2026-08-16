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
directories as needed. A missing checkpoint is an error (exit 1) except as
the first-run save target of `--train`; it never silently falls back to a
fresh model. Checkpoints are format v2 (`RGPT_V2` magic header): the seed,
the vocabulary, and each layer's learned weights. Optimizer state and
transient caches are never stored, so a loaded model starts with fresh
optimizers. Checkpoints live under `models/` (gitignored); a result is
reproducible as the pair (checkpoint or seed) plus its eval JSON.

## Tiny-Lane Evaluation

```bash
cargo run --release -- --tiny --eval --model models/tinystories/ts-13m-s42.bin
```

The tiny lane's score formula (see `docs/dataset-curation.md`): per-item
teacher-forced CE over `models/tinystories/heldout.jsonl`, nearest-rank
p10/p50/p90 percentiles, vocabulary coverage, and the generation-collapse
gate. The same block ships as `eval` inside every `--tiny --train` output.
`--tiny --eval` requires `--model`; without a checkpoint there is nothing to
score. The held-out slice (split seed 20260816) is carved by
`scripts/slice_tinystories.py` and never appears in a training slice.

### The temperature knob

```bash
cargo run --release -- --tiny --eval --model models/tinystories/ts-13m-s42.bin --temperature 0.8
```

`--temperature <t>` (default 1.0, `--tiny --eval` only, exit 2 otherwise)
scales the output softmax logits by `1/t` in the gate's greedy decode. The
greedy argmax is temperature-invariant, so the gate number is a pin at
every T; the knob exists so the collapse gate can be measured under a
peaked softmax and gives the probability-weighted sampling probe its
baseline (see docs/learning-directions.md, W3 verdict).

### The fluency yardstick

```bash
cargo run --release -- --tiny --eval --model models/tinystories/ts-13m-s42.bin --fluency 20
```

`--fluency <n>` (default off, `--tiny --eval` only, exit 2 otherwise) adds
a `fluency` block to the eval JSON: distinct-1, distinct-2,
repetition-free rate, sentence-final punctuation count, and mean length
over `n` seeded completions of the gate's starter. This is the decode-
quality yardstick behind the collapse gate -- repetition-free is necessary
but not sufficient, and distinct-n measures lexical diversity (see
docs/learning-directions.md, W3 verdict for the calibrated pass floor).

## Development Commands

```bash
cargo fmt --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --all-targets
cargo build --release
```

## OneDrive-Backed Checkouts

A checkout inside a OneDrive-synced directory (for example under
`/mnt/c/Users/<user>/OneDrive/`) pays a per-file I/O tax on every build:
measured on this machine, an incremental release rebuild takes 14.1s on
the OneDrive mount versus 5.9s on an ext4 directory (2.4x), with the
sync client additionally uploading `target/` to the cloud. If builds
feel slow, point Cargo's output at an ext4 directory instead:

```bash
export CARGO_TARGET_DIR=~/projects/rustgpt-target
cargo build --release   # artifacts now live outside the mount
```

The binary behaves identically (same source, same compiler); the repo
keeps no `target/` artifacts when the variable is set. The workaround is
documented here because the mount's latency is invisible until measured.

To inspect output from a particular integration test:

```bash
cargo test --test llm_test -- --nocapture
```
