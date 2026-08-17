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
  --epochs 1 --seed 42 --model models/tinystories/ts-13m-s42.bin
```

Training modes print exactly one JSON object to stdout: the trajectory
(per-epoch loss and held-out CE), and for the tiny lane the per-epoch
logit-regime profile, samples, and the collapse gate.

## 2. Score

```bash
cargo run --release -- --eval --seed 42                      # micro oracle: 4/4/1.0
cargo run --release -- --tiny --eval --model models/tinystories/ts-13m-s42.bin
cargo run --release -- --tiny --eval --model models/tinystories/ts-13m-s42.bin --fluency 20
```

The eval JSON is the checkpoint's score sheet: held-out exact/prefix/
per-position accuracy for the micro lane; CE percentiles, coverage, the
collapse gate, and the fluency yardstick for the tiny lane. The checkpoint
plus its eval JSON is the unit of evidence.

## 3. Tune the decode recipe (tiny lane)

The 0.0.7 decode knobs -- sampling temperature, presence/repetition
penalties, top-p -- are probed per config, seeded and reproducible:

```bash
cargo run --release -- --tiny --eval --model models/tinystories/ts-13m-s42.bin \
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
cargo run --release -- --model models/tinystories/ts-13m-s42.bin   # chat, no retrain
cargo run --release -- --model models/watercycle-latest.bin --e2e "hello world"
cargo run --release -- --model models/watercycle-latest.bin --trace  # step-by-step decode
```

A loaded checkpoint is never re-saved by the use surfaces: the artifact
stays the artifact.

## 6. Ship

The release cadence: bump `version` in `Cargo.toml`, push to `main` (the
release workflow tags and publishes), and the CHANGELOG top section is the
release body with measured evidence and artifact pointers.
