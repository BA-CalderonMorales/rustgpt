<div align="center">

# RustGPT

**A from-scratch transformer language model in pure Rust — inspectable mechanics, no external ML framework (fork - see [Attribution](https://github.com/BA-CalderonMorales/rustgpt#attribution))**

[![Crate](https://img.shields.io/badge/version-0.0.9-blue.svg?logo=rust&style=flat-square)](https://github.com/BA-CalderonMorales/rustgpt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Check](https://img.shields.io/github/actions/workflow/status/BA-CalderonMorales/rustgpt/check.yml?label=check&style=flat-square)](https://github.com/BA-CalderonMorales/rustgpt/actions/workflows/check.yml)
[![Test](https://img.shields.io/github/actions/workflow/status/BA-CalderonMorales/rustgpt/test.yml?label=test&style=flat-square)](https://github.com/BA-CalderonMorales/rustgpt/actions/workflows/test.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg?style=flat-square)](https://github.com/BA-CalderonMorales/rustgpt/blob/main/docs/architecture.md)

<img src="docs/demo-tui.gif" alt="rustgpt demo: the operating path from --help to a model you trained yourself" width="100%">

</div>

A transformer language model written from scratch with `ndarray` tensors —
built to be read, traced, and tested, not to compete. Every layer, the
tokenizer, the optimizer, and the CLI are hand-rolled and documented in the
docs below. Generated text is an honest measurement of small-model mechanics,
never a quality benchmark.

## Start here

Build, then walk the operating path top to bottom — the same map
`target/release/llm --help` prints:

```bash
git clone https://github.com/BA-CalderonMorales/rustgpt.git
cd rustgpt
cargo build --release

target/release/llm --models                                  # 1. pick an artifact from the catalog
target/release/llm --model stories-full --ask "Once upon a time,"  # 2. one answer (greedy)
target/release/llm --model watercycle-latest                 # 3. chat (/help inside)
target/release/llm --demo --seed 42                          # 4. watch raw text become a model
target/release/llm --tiny --train my-corpus.jsonl \
  --epochs 6 --eos --lr-decay 5e-5 --seed 42 \
  --model models/ts.bin                                      # 5. teach your own model
target/release/llm --tiny --eval --model models/ts.bin --fluency 20  # 6. score it honestly
target/release/llm --eval --seed 42                          # 7. the micro arena oracle: 4/4 exact
```

Every step is seeded and reproducible; every score is measured against
held-out data the model never saw. What the numbers currently say —
including the us-vs-Qwen3-0.6B gap table — lives in the
[CHANGELOG](CHANGELOG.md).

## Docs

| Document | What |
|---|---|
| [Running and development](docs/running-and-development.md) | Every surface in detail: flags, knobs, checkpoints, exit codes |
| [Demo](docs/demo.md) | How the GIF is recorded and its non-destructive rules |
| [Architecture](docs/architecture.md) | Model pipeline, source map, reading order |
| [Model workflow](docs/model-workflow.md) | Create, score, record, use — and the artifact inventory |
| [Model and training](docs/model-and-training.md) | Current configuration and training phases |
| [Dataset curation](docs/dataset-curation.md) | The data: budgets, licenses, held-out scoring |
| [Testing](docs/testing.md) | What each correctness boundary establishes |
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
