# AGENTS.md - rustgpt

## Current Shape

- From-scratch transformer LLM in pure Rust: `ndarray` tensors, **no external
  ML framework**. Intent is inspectable mechanics, not scale or quality
  (`docs/model-and-training.md`).
- Pipeline: tokenization -> embeddings -> 3 transformer blocks -> output
  projection (`docs/architecture.md`). Every domain is a folder under `src/`
  with `mod.rs` as the public face, `interfaces.rs` for types/traits, and
  `logic.rs` for implementation; specialized files (`constants.rs`, `tests.rs`,
  `seed.rs`) exist inside a domain only when that responsibility is present.
- CLI is hand-rolled and std-only by design: `src/cli/` owns every argument
  and exit code; machine modes print exactly one JSON object to stdout.
- Data is a compact water-cycle micro-domain: `data/pretraining_data.json`
  (16 statements, 100 epochs, LR 5e-4), `data/chat_training_data.json` (28 QA
  pairs, 100 epochs, LR 1e-4), and `data/heldout.json` (evaluation
  observations) (`docs/dataset-curation.md`).
- Correctness layers: unit/component tests, mutation-resistant optimizer
  tests, integration tests, then a **separate black-box project** (rustgpt-evals)
  that only touches the compiled CLI (`docs/testing.md`).
- The experiment backlog lives in `docs/learning-directions.md` until it earns
  placement; the repo keeps zero dead code (ADR 0001).

## Key Sections

| To understand... | Read |
|---|---|
| Why this fork exists | `README.md` (Learning Project, Attribution) |
| Model pipeline and reading order | `docs/architecture.md` |
| Current model config and training phases | `docs/model-and-training.md` |
| The dataset: budgets, held-out scoring, E2E boundary | `docs/dataset-curation.md`, `data/heldout.json` |
| What each correctness layer establishes | `docs/testing.md` |
| The experiment backlog (no product promises) | `docs/learning-directions.md` |
| Architecture decisions for this repo | `docs/architecture/decisions/` |
| CLI surface and local commands | `docs/running-and-development.md` |
| House style of the workspace's research method | root workspace `AGENTS.md`; `competitions/observations/` |

Lost in the woods? Read in this order: `README.md`, `docs/architecture.md`
reading order, then `docs/model-and-training.md` before touching code.

## Research Method (first principles)

This repo is a brain-lane learning build. Research here follows the
meta-method of the competitions observations, translated to a codebase where
the "game" is the math itself:

1. **Contract archaeology before code.** The CLI and public API are the
   contract: `tests/cli_contract_test.rs` and `tests/public_api_test.rs` pin
   it. `cargo run -- --e2e "..."` is a **contract probe, not evidence of
   learning** — never cite it as a result. Read the boundary tests before
   writing a line of source.
2. **Formula-first reasoning.** The objective is the game, like a leaderboard
   formula: cross-entropy loss, gradient clipping, LR schedules, held-out
   scores. The eval formula is
   `exact / prefix / per-position accuracy` against `data/heldout.json`
   (`llm::answer_score`), seeded and reproducible. Derive what a change does
   to the *objective* on paper before implementing it. If you cannot state
   the mechanism, you cannot run the experiment.
3. **Oracle-first engineering.** The oracle is finite-difference gradient
   checking and hand-computed forward/backward passes (see the private
   finite-difference attention gradient test in `src/self_attention/tests.rs`).
   Before changing a layer, verify your mental model against truth on a
   1-token example: compute by hand, then compare to the code's tensors.
   Derive first, port last — and when porting upstream ideas, calibrate
   honestly against this repo's layer semantics.
4. **Falsify in one run.** One hypothesis per change, tested against a
   baseline. Name the claim ("top-k sampling changes the answer distribution
   on held-out prompts X"), run it, record the verdict in
   `docs/learning-directions.md`; a falsification is a discovery, not a
   failure — the diagnosis IS the finding.
5. **Cite your ground truth.** Every claim must carry its seed, epochs, LR,
   data subset, and comparison target. A number without its run recipe is
   noise; quote the recipe with every result.

## Rules

- **Slim Rust.** The smallest amount of Rust that stays readable. One concept
  per file; facades as the only public surface; callers import through domain
  paths (`crate::llm::*`), never loose files at the `src/` root. New files
  target 120 lines or fewer; growing past is a signal to split, never a
  reason to keep writing. Splits are pure moves: no logic changes in a split.
- **No early abstraction.** Duplicate plainly before abstracting prematurely.
  Boring beats novel. If a reader needs a comment to understand the
  structure, split the file. Delete before adding.
- **No new dependencies without a documented tradeoff.** Hand-rolled
  mechanics (tokenizer, optimizer, CLI, attention) are the subject matter of
  this repo — do not replace them with crates. Additions require an ADR in
  `docs/architecture/decisions/`.
- **Deterministic by default.** Initialization is seeded via
  `llm::set_seed` (default 42, owned by `configuration/seed.rs`); every score
  must reproduce exactly with the same seed. Runs without a seed are
  observations; runs with a seed are evidence.
- **Free open data only.** New data derives from free-to-use sources (Kaggle,
  HuggingFace) under permissive licenses (MIT, Apache-2.0, CC-BY, OpenMDW-1.1).
  Record source and license in `docs/dataset-curation.md` before use;
  paywalled and non-commercial-licensed data never enters the corpus. The
  dataset budgets in that doc are ceilings, not targets.
- **Tests mirror sources.** In-domain private tests live in
  `src/<domain>/tests.rs`; cross-domain and contract tests in `tests/*_test.rs`.
  Public API and CLI contract output strings are pinned by their tests —
  changing them requires updating the contract tests in the same commit.
- **One experiment per change.** No bundled hypotheses; an ambiguous run
  proves nothing. Small, reversible, explainable experiments only (if it
  cannot be explained in three sentences, split it). The `--e2e` path never
  trains and never loads weights — keep it a contract probe, not a result.
- **Durable names.** Name files, fixtures, CI jobs, and artifacts by durable
  purpose, never by ephemeral phase counters.
- **Version lives only in `Cargo.toml`.** `Cargo.lock` and the README badge
  mirror it; the CLI version contract derives from `CARGO_PKG_VERSION`, never
  from a literal. Bump, push, release in one motion.
- **Never merge dev-local ledgers.** Workspace `scratch/` goal ledgers and
  local experiment notes are developer-local; they never commit here. Use
  `rg` for content search when available, `fzf` for interactive selection.
- **Anti-patterns ledger.** When you observe a new recurring mistake in
  committed work (yours or an agent's), record it in the workspace
  `docs/architecture/anti-patterns/` (single authoritative home) before
  carrying it forward.
- **Zero emojis** in root-level and docs-facing files; lowercase filenames
  (except README.md, AGENTS.md).

## Design Principles

- SRP — one responsibility per domain; `mod.rs` facade is its public face.
- OCP — extend by adding a domain or a file, not by widening an existing
  facade.
- ISP/DIP — domains depend on the facade, never on `logic/` internals; the
  `Layer` trait (`forward`/`backward`/`parameters`) is the seam that makes
  new layers pluggable without touching the model loop.
- Composition over inheritance — the model is a `Vec<Box<dyn Layer>>`;
  layers are composed, not inherited.
- DRY — one authoritative home per piece of knowledge: version in
  `Cargo.toml`, seed state in `configuration/`, dataset contracts in `data/`
  + `docs/dataset-curation.md`, scoring logic only in `llm::answer_score`.
- KISS — hand-rolled mechanics are the product; no premature abstraction, no
  dependencies without tradeoff, delete before adding.
- CQS — read-only contract probes (`--version`, `--help`, `--e2e`) never
  train or mutate; training happens only in training modes.
- POLA — behavior must not astonish: durable names, exact usage errors, exit
  code 2 for bad arguments, 1 JSON object on stdout for machine modes,
  training progress on stderr in `--eval`.
- Single-Choice — exhaustive alternatives live in one module (decoding,
  softmax, tokenization in `llm/logic.rs`; constants in `configuration/`).
- Self-Documentation — durable-purpose names; the research method above is
  part of every change's information.
- Evidence-first — seeded runs are evidence, unseeded runs are observations;
  oracle-truth (finite differences, hand computation) beats infrastructure
  opinion.

## CI and Release Cadence

- `check.yml` (fmt, clippy, typos) and `test.yml` run on push to `main` and
  on PRs; a red pipeline blocks nothing locally but is the first opponent —
  fix the harness before touching the model.
- `dispatch-e2e.yml` pings rustgpt-evals when model-affecting paths change.
- Release cadence: bump `version` in `Cargo.toml`, push to `main`, and the
  release workflow tags `v<version>` and publishes a GitHub release with the
  release binary, checksums, and auto-generated notes (tag-once semantics;
  `workflow_dispatch` forces a re-cut).
- `main` is the only branch. Commits to `main` must pass the Verify gates
  first and diff only the intended experiment's files. This checkout is
  owned by the workspace brain lane; run git in the repo, never from the
  workspace root.

## Verify

```bash
cargo fmt --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --all-targets
cargo run -- --e2e "hello world"   # contract probe only
cargo run -- --eval --seed 42      # score formula: held-out eval
```
