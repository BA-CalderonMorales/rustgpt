# Model Workflow

How a trained model comes to exist and how you use it once it does. Every
step is a command from the repository root; every number carries its seed.

## 1. Create

```bash
# The micro lane: two phases (pretrain 100 epochs, tune 100 epochs, LR
# 5e-4 / 1e-4), min held-out CE promoted into the artifact.
cargo run --release -- --eval --seed 42 --model models/watercycle-latest.bin

# The tiny lane: real-corpus stories at the laptop config.
cargo run --release -- --tiny --train models/tinystories/train.jsonl \
  --epochs 1 --seed 42 --model models/tinystories/stories-full.bin
```

Training modes print exactly one JSON object to stdout: the trajectory
(per-epoch loss and held-out CE), and for the tiny lane the per-epoch
logit-regime profile, samples, and the collapse gate.

## 2. Score

```bash
cargo run --release -- --eval --seed 42                      # micro oracle: 4/4/1.0
cargo run --release -- --tiny --eval --model models/tinystories/stories-full.bin
cargo run --release -- --tiny --eval --model models/tinystories/stories-full.bin --fluency 20
```

The eval JSON is the checkpoint's score sheet: held-out exact/prefix/
per-position accuracy for the micro lane; CE percentiles, coverage, the
collapse gate, and the fluency yardstick for the tiny lane. The checkpoint
plus its eval JSON is the unit of evidence.

## 3. Tune the decode recipe (tiny lane)

The 0.0.7 decode knobs -- sampling temperature, presence/repetition
penalties, top-p -- are probed per config, seeded and reproducible:

```bash
cargo run --release -- --tiny --eval --model models/tinystories/stories-full.bin \
  --temperature 0.7 --top-p 0.8 --presence 1.5 --repetition 1.1 --fluency 20
```

The Qwen-honoring stack (T=0.7, top-p 0.80, presence 1.5) plus the
count-scaled repetition 1.1 is the measured winner: gate 0.021 vs the 1.0
greedy pin, repetition-free 0.65, distinct-1 0.70 (see
docs/learning-directions.md, W4-W6 verdicts).

## 4. Record

Every artifact earns a `models/catalog.json` entry -- path, family,
parameters, seed, recipe, eval, decode numbers, quality labels. The catalog
is the record of what was made and how; `llm --models` serves it as one
JSON object.

## 5. Use

```bash
cargo run --release -- --model models/tinystories/stories-full.bin   # chat, no retrain
cargo run --release -- --model models/watercycle-latest.bin --e2e "hello world"
cargo run --release -- --model models/watercycle-latest.bin --trace  # step-by-step decode
```

A loaded checkpoint is never re-saved by the use surfaces: the artifact
stays the artifact.

## 6. Ship

The release cadence: bump `version` in `Cargo.toml`, push to `main` (the
release workflow tags and publishes), and the CHANGELOG top section is the
release body with measured evidence and artifact pointers.

## The Artifact Inventory

Every `models/*.bin` is regenerable evidence, gitignored: its recipe, its
seed, and its eval JSON reproduce the same artifact. Treat the table as an
inventory of the exploration so far, not a product catalog.

| Artifact | Size | Recipe / experiment | What it demonstrates | Regenerate |
|---|---|---|---|---|
| `watercycle-latest.bin` | 1.5 MB | v0.0.5 eval recipe (seed 42, replay + min-CE promotion), E10 lineage | The release winner: greets and hedges; held-out 4/4 exact, mean 1.0 | `rm models/watercycle-latest.bin && target/release/llm --eval --seed 42 --model models/watercycle-latest.bin` |
| `watercycle-e10.bin` | 1.5 MB | E10 social-register era | Social register landed: "hi!" -> "Assistant : Hello !" | same recipe as latest (superseded by it) |
| `watercycle-e8.bin` | 1.5 MB | E8 hedge-stabilization era | Stable hedging across probe prompts | same-era recipe (superseded) |
| `watercycle-e7.bin` | 1.5 MB | E7 hedge era | "How do mountains form?" -> full hedge; OOV prompts never confidently hallucinate | same-era recipe (superseded) |
| `watercycle-e6.bin` | 1.5 MB | E6 targeted paraphrase expansion | Chain statements and paraphrase pairs: exact 2/4, mean 0.6534 era | same-era recipe (superseded) |
| `watercycle-e2.bin`, `watercycle-e1.bin` | 1.5 MB | Early recipe era | Water-cycle Q/A recital; no social register yet | same-era recipe (superseded) |
| `watercycle-0.0.3.bin` | 1.5 MB | v0.0.3 era, checkpoint format v1 | Legacy: no longer loads in the current CLI ("not a rustgpt checkpoint"); kept for format archaeology | not regenerable in this format |
| `tinystories/stories-full.bin` | 57 MB | 1 epoch over 40k TinyStories stories (seed 42, 1.5M tokens) | The laptop lane at full-corpus scale; greedy decode collapses (gate 1.0), the 0.0.7 decode stack defeats it (gate 0.021, repetition-free 0.65) | `python scripts/slice_tinystories.py && target/release/llm --tiny --train models/tinystories/train.jsonl --epochs 1 --seed 42 --model models/tinystories/stories-full.bin` (~1.5 h on a 14-thread laptop) |
| `tinystories/stories-demo.bin` | 29 MB | 6 epochs over the 300-story demo slice, seed 42, `--eos --lr-decay 5e-5` (v0.0.8 recipe) | The demo lane: termination-trained (completions end before the cap under sampling), monotone loss, best CE of the four E11/W8 runs (p50 6.40) -- also the artifact `--demo` trains from scratch in miniature | `target/release/llm --tiny --train models/tinystories/demo.jsonl --epochs 6 --seed 42 --eos --lr-decay 5e-5 --model models/tinystories/stories-demo.bin` (~2 min) |

`models/tinystories/train.jsonl` (40k stories) and
`models/tinystories/heldout.jsonl` (256, split seed 20260816) are rebuilt
by `scripts/slice_tinystories.py`; the held-out slice never touches a
training slice.
