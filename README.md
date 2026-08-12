<div align="center">

# RustGPT

**A from-scratch transformer language model in pure Rust — inspectable mechanics, no external ML framework**

[![Crate](https://img.shields.io/badge/version-0.0.4-blue.svg?logo=rust&style=flat-square)](https://github.com/BA-CalderonMorales/rustgpt)
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

Run the interactive training-and-chat loop, or use the fast machine-readable
contract probe:

```bash
git clone https://github.com/BA-CalderonMorales/rustgpt.git
cd rustgpt
cargo run                          # train on the water-cycle micro-domain, then chat
cargo run -- --e2e "hello world"   # contract probe: one JSON object, no training
```

The default path pre-trains on 16 foundational statements, instruction-tunes on
28 QA pairs, prints a sample prediction, and accepts prompts until `exit`.
Because this is a small educational model, generated text is an observation of
the mechanics, not a quality benchmark.

## Commands

| Command | What it does |
|---|---|
| `cargo run` | Build vocab, pre-train + instruction-tune, chat interactively |
| `cargo run -- --e2e "..."` | Initialize model, generate once, print one JSON object (`status`, `output`, `total_parameters`) |
| `cargo run -- --eval --seed 42` | Train both phases, score the four held-out prompts, print one JSON object (`exact`/`prefix`/`accuracy`) |
| `cargo run -- --model models/mine.bin --eval --seed 42` | Save/load a trained checkpoint; eval reports the training trajectory |
| `cargo run -- --tiny --train models/tinystories/train.jsonl --epochs 1 --model models/ts.bin` | Train the 14M-param laptop lane on a single JSONL corpus, save a checkpoint, print trajectory + samples |
| `cargo fmt --check` | Format gate |
| `cargo clippy --workspace --all-features --all-targets -- -D warnings` | Lint gate |
| `cargo test --all-targets` | Unit, property/invariant, integration, and contract tests |
| `cargo build --release` | Release binary (also produced by the release workflow) |

E2E JSON contract:

```json
{"output":"...","prompt":"hello world","status":"ok","total_parameters":380893}
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
