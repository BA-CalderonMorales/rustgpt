<div align="center">

# RustGPT

**A from-scratch transformer language model in pure Rust — inspectable mechanics, no external ML framework (fork - see [Attribution](https://github.com/BA-CalderonMorales/rustgpt#attribution))**

[![Crate](https://img.shields.io/badge/version-0.0.5-blue.svg?logo=rust&style=flat-square)](https://github.com/BA-CalderonMorales/rustgpt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Check](https://img.shields.io/github/actions/workflow/status/BA-CalderonMorales/rustgpt/check.yml?label=check&style=flat-square)](https://github.com/BA-CalderonMorales/rustgpt/actions/workflows/check.yml)
[![Test](https://img.shields.io/github/actions/workflow/status/BA-CalderonMorales/rustgpt/test.yml?label=test&style=flat-square)](https://github.com/BA-CalderonMorales/rustgpt/actions/workflows/test.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg?style=flat-square)](https://github.com/BA-CalderonMorales/rustgpt/blob/main/docs/architecture.md)

<img src="docs/demo-tui.gif" alt="rustgpt demo: contract probe, micro-arena eval, and the laptop lane training on real stories" width="100%">

</div>

A complete language-model implementation — tokenization, embeddings,
transformer blocks, optimization, training, and generation — built by hand
with `ndarray` tensors and no machine-learning framework. The goal is a mental
map of how a transformer works under the hood, not a competitive model:
every layer is meant to be read, traced, and tested.

## Quick Start

Build the release binary, then reproduce the v0.0.5 truth table and talk to
the trained artifact:

```bash
git clone https://github.com/BA-CalderonMorales/rustgpt.git
cd rustgpt
cargo build --release

target/release/llm --eval --seed 42                          # train + held-out score (exact 4/4, mean 1.0)
target/release/llm --model models/watercycle-latest.bin --e2e 'User: hi!'    # probe the artifact
target/release/llm --model models/watercycle-latest.bin      # chat with the trained artifact
```

`models/watercycle-latest.bin` is created by the eval command above (a
missing `--model` path is a first-run save target for training modes).
Running `target/release/llm` with no arguments enters interactive mode:
it initializes a fresh random-seed model, prints the untrained model's
noise, trains both phases (100 pretrain + 100 tuning epochs, live loss
bar), and then chats until `exit`. Use `--seed 42` there too for a
reproducible session. Because this is a small educational model, generated
text is an observation of the mechanics -- measured, seeded, and pinned by
the contract tests, not a quality benchmark.

## Commands

| Command | What it does |
|---|---|
| `target/release/llm` | Interactive: train from a fresh random seed, then chat until `exit` |
| `target/release/llm --e2e "..."` | Contract probe: generate once, print one JSON object (`status`, `output`, `total_parameters`) |
| `target/release/llm --eval --seed 42` | Train both phases, score the four held-out prompts, print the truth table (items, summary, CE trajectory) |
| `target/release/llm --model <path> --eval --seed 42` | First-run saves the trained checkpoint; re-runs load it, continue training, re-save |
| `target/release/llm --model <path> --e2e "..."` | Probe a trained artifact (unknown words answer `I do not know that word`) |
| `target/release/llm --probe --model <path> --seed 42` | Decode-time compute truth table: seeded top-k best-of-N vs greedy |
| `target/release/llm --tiny --eval --model models/tinystories/ts-13m-s42.bin` | Laptop lane score formula: held-out CE percentiles, coverage, collapse gate |
| `target/release/llm --tiny --train <file.jsonl> --epochs 1 --model <out.bin>` | Train the 14M-param laptop lane on a JSONL corpus, print trajectory + samples + eval |
| `cargo test --test output_properties_test -- --nocapture` | Property suite pass table against `models/watercycle-latest.bin` |
| `cargo test --test conversation_suite_test -- --nocapture` | Conversation-surface suite (greetings, OOV, junk probes) |
| `cargo fmt --check` / `cargo clippy --workspace --all-features --all-targets -- -D warnings` | Verify gates |
| `cargo test --all-targets` | Unit, property/invariant, integration, and contract tests |
| `cargo build --release` | Release binary (also produced by the release workflow) |

E2E JSON contract:

```json
{"output":"...","prompt":"hello world","status":"ok","total_parameters":385776}
```

## Layout

Each domain keeps a facade (`mod.rs`), its types and traits (`interfaces.rs`),
and its implementation (`logic.rs`) — one pattern to learn, then every module
reads the same way.

```text
src/
├── main.rs               Parse, load, build, and run
├── lib.rs                Domain declarations and compatibility re-exports
├── cli/                   CLI mode and argument behavior
├── application/           Dataset, model, training, and interaction orchestration
├── configuration/         Shared model constants
├── llm/                   Model API, composition, training, and generation
├── transformer/           Transformer block composition
├── self_attention/        Self-attention operation and private gradient test
├── feed_forward/          Position-wise feed-forward operation and optimizers
├── embeddings/            Token and positional embeddings
├── output_projection/     Vocabulary projection
├── layer_norm/            Layer normalization
├── vocab/                 Vocabulary and tokenization
├── dataset_loader/        JSON and CSV dataset loading
└── adam/                  Adam optimizer
tests/                     Integration, contract, and public-API checks
data/                      The compact water-cycle micro-domain
```

Correctness is separated by layer: unit tests validate operations,
mutation-resistant tests check optimizer invariants, integration tests
exercise layers together, and the companion
[rustgpt-evals](https://github.com/BA-CalderonMorales/rustgpt-evals) project
observes the compiled CLI as a black box.

## Docs

| Document | What |
|---|---|
| [Architecture](docs/architecture.md) | Model pipeline, source map, reading order |
| [Model and training](docs/model-and-training.md) | Current configuration and training phases |
| [Dataset curation](docs/dataset-curation.md) | Water-cycle micro-domain, budgets, held-out prompts |
| [Testing](docs/testing.md) | What each correctness boundary establishes |
| [Running and development](docs/running-and-development.md) | CLI surface and local commands |
| [Learning directions](docs/learning-directions.md) | The experiment backlog — no product promises |

## License

MIT — see [LICENSE.txt](LICENSE.txt).

## Attribution

RustGPT began as
[tekaratzas/RustGPT](https://github.com/tekaratzas/RustGPT), a gentle
introduction to building a transformer from scratch, created by
[Thomas Karatzas](https://github.com/tekaratzas). This repository is a
learning fork: the original implementation and the spirit of the project —
understand a language model by building one — belong to him. The original
copyright and license notice in [LICENSE.txt](LICENSE.txt) are preserved
unchanged; changes here focus on journey, not replacement.
