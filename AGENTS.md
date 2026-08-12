# AGENTS.md - rustgpt

## Current Shape

- From-scratch transformer LLM in pure Rust: `ndarray` tensors, **no external
  ML framework**. Intent is inspectable mechanics, not scale or quality
  (`docs/model-and-training.md`).
- Pipeline: tokenization -> embeddings -> 3 transformer blocks -> output
  projection (`docs/architecture.md`). Every domain follows facade + interface
  + logic: `mod.rs` is the facade, `interfaces.rs` types/traits, `logic.rs`
  implementation.
- Data is a compact water-cycle micro-domain: 16 pre-training statements
  (100 epochs, LR 5e-4) + 28 instruction QA pairs (100 epochs, LR 1e-4).
  `data/*.json` is the whole economy (`docs/dataset-curation.md`).
- Correctness layers: unit/component tests, mutation-resistant optimizer
  tests, integration tests, then a **separate black-box project** (rustgpt-evals)
  that only touches the compiled CLI (`docs/testing.md`).
- `scratch/`-style experiment ledger belongs in `docs/learning-directions.md`
  until it earns placement; the repo keeps zero dead code, so every change
  must explain, test, and reverse (ADR 0001, learning-directions).

## Key Sections

| To understand... | Read |
|---|---|
| Why this fork exists | `README.md` (Learning Project, Attribution) |
| Model pipeline and reading order | `docs/architecture.md` |
| Current model config and training phases | `docs/model-and-training.md` |
| The dataset: budgets, held-out prompts, E2E boundary | `docs/dataset-curation.md` |
| What each correctness layer establishes | `docs/testing.md` |
| The experiment backlog (no product promises) | `docs/learning-directions.md` |
| House style of this workspace's research method | root workspace `AGENTS.md`; `competitions/observations/` |

Lost in the woods? Read in this order: `README.md`, `docs/architecture.md`
reading order, then `docs/model-and-training.md` before touching code.

## Research Method (first principles)

This repo is a brain-lane learning build. Research here follows the same
meta-method that competitions run on, translated to a codebase where the
"game" is the math itself:

1. **Contract archaeology before code.** The CLI and public API are the
   contract: `tests/cli_contract_test.rs` and `tests/public_api_test.rs` pin
   it. `cargo run -- --e2e "..."` is a **contract probe, not evidence of
   learning** — never cite it as a result. Read the boundary tests before
   writing a line of source.
2. **Formula-first reasoning.** The objective is the game, like a leaderboard
   formula: cross-entropy loss, gradient clipping, LR schedules, 100 epochs,
   held-out prompts. Derive what a change does to the *objective* on paper
   before implementing it. If you cannot state the mechanism, you cannot run
   the experiment.
3. **Oracle-first engineering.** The oracle is finite-difference gradient
   checking and hand-computed forward/backward passes (see the private
   finite-difference attention gradient test in
   `src/self_attention/tests.rs`). Before changing a layer, verify your
   mental model against truth on a 1-token example: compute by hand, then
   compare to the code's tensors. Derive first, port last — and when porting
   upstream ideas, calibrate honestly against this repo's layer semantics.
4. **Falsify in one run.** One hypothesis per change, tested against a
   baseline. Name the claim ("top-k sampling changes the answer distribution
   on held-out prompts X"), run it, record the verdict. Likely outcomes for
   every claim live in `docs/learning-directions.md`; a falsification is a
   discovery, not a failure — the diagnosis IS the finding.
5. **Cite your ground truth.** Every claim must carry its seed, epochs, LR,
   data subset, and comparison target — the repos' determinism discipline
   (`docs/learning-directions.md` "explicit seeds"). A number without its
   run recipe is noise; quote the recipe with every result.

## Workflow Discipline

- One experiment per change. No bundled hypotheses; an ambiguous run proves
  nothing.
- Small, reversible, explainable experiments only (repository rule in
  `docs/learning-directions.md`). If it cannot be explained to a reader in
  three sentences, split it.
- Record surprises as observations in the repo docs before moving on; the
  mechanism is the artifact, code is only its embodiment.
- The `--e2e` path never trains and never loads weights — keep it as a
  contract probe, not a result.
- Machine-local experiment notes go to the workspace `scratch/` goal ledger,
  never committed here.

## Verify

```bash
cargo fmt --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --all-targets
cargo run -- --e2e "hello world"   # contract probe only
```

Before committing in this repo: run all four, then diff only the intended
experiment's files. This checkout is owned by the brain lane; run git in the
repo, never from the workspace root.

## CI and Release Cadence

- `check.yml` (fmt, clippy, typos) and `test.yml` run on push to `main` and on
  PRs.
- `dispatch-e2e.yml` pings rustgpt-evals when model-affecting paths change.
- Release cadence: bump `version` in `Cargo.toml`, push to `main`, and the
  release workflow tags `v<version>` and publishes a GitHub release with the
  release binary, checksums, and auto-generated notes (tag-once semantics;
  `workflow_dispatch` forces a re-cut). Version lives only in `Cargo.toml`.
