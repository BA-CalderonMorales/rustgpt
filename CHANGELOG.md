# Changelog

Every release entry records what changed and the measured evidence it
produced: score, trajectory, and artifacts. The top section becomes the
GitHub release body (see `.github/workflows/release.yml`), so the public
record and the repo history are the same document.

## [0.0.4] - 2026-08-12

Learning: `--train <file.jsonl>` opens the single-corpus laptop lane at
`--tiny` scale (Config presets; 14.2M params, 384/768/128, 6 blocks); JSONL
dataset loading with window truncation; `--epochs`; checkpoint format v2
carries the full model shape; TinyStories adopted (CDLA-Sharing-1.0) with a
reproducible slicing script.

Score (water-cycle held-out, seed 42): exact 1/4, prefix 1/4, mean accuracy
0.3125 -- micro arena unchanged.

Tiny-lane first truth table (seed 42, 1 epoch, 40k stories, 1.53M tokens,
laptop CPU): loss 5.84 from ~8.1 random start at 14.2M params; greedy
generation collapsed to the dominant token; throughput measured at ~283
tokens/s (~10M tokens per 9.8h). The lane works end to end; convergence
tier and corpus-scale budgets are the v0.0.5 problem, with batching and a
GPU racecar (ADR candidate) documented as the options.

Artifacts: release binary + checksums; local
`models/tinystories/ts-13m-s42.bin` (57 MB, gitignored), slice
reproducible via `scripts/slice_tinystories.py`.

## [0.0.3] - 2026-08-12

Learning: checkpoint save/load (`--model <path>`, format v1), training
trajectory in eval JSON (per-epoch loss, held-out CE samples), accuracy
percentiles, `llm::sequence_loss` teacher-forced probe.

Score (seed 42, water-cycle held-out): exact 1/4, prefix 1/4, mean accuracy
0.3125.

Trajectory finding: pretrain loss 4.73 -> 0.08, tuning 4.86 -> 0.21, but
held-out CE ran 5.30 -> 8.79 (pretraining hurt chat-format CE) -> 1.95
(tuning rescued) -> 2.11 -> 2.17 (drift up). Endpoints would have hidden
all of it; the curve is the claim.

Artifacts: release binary + checksums; local `models/watercycle-0.0.3.bin`
(1.5 MB, gitignored).

## [0.0.2] - 2026-08-12

Learning: seeded determinism by default (seed 42), `--eval` score formula
(exact / prefix / per-position accuracy over the four held-out prompts,
`data/heldout.json`), gorvernance contract in AGENTS.md.

Score (seed 42): exact 1/4, prefix 1/4, mean accuracy 0.3125. The first
truth table: item 2 exact match ("Condensation changes water vapor into
droplets"), item 3 a degenerate repetition loop -- now measured, not
anecdotal.

## [0.0.1] - 2026-08-12

Baseline: from-scratch transformer LLM in pure Rust, water-cycle micro
domain (16 statements + 28 QA pairs), interactive chat, `--e2e` contract
probe (totals validation: 380,893 parameters), release pipeline on push to
main.
