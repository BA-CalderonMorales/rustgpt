# Testing

rustgpt separates correctness checks according to what they can establish.

## Unit and Component Tests

Focused Rust tests exercise vocabulary behavior, embeddings, attention,
feed-forward operations, transformer composition, output projection, dataset
loading, and optimizer behavior.

## Test Source Map

| Boundary | Location | Purpose |
|---|---|---|
| Private implementation | `src/self_attention/tests.rs` | Finite-difference attention gradient invariant with private weights. |
| Public component | `tests/*_test.rs` | Domain behavior through public facades. |
| Public API | `tests/public_api_test.rs` | Existing module paths, re-exports, constants, and public fields compile. |
| Public CLI | `tests/cli_contract_test.rs` | Arguments, streams, exit codes, and one-line E2E JSON. |
| Cross-domain model | `tests/llm_test.rs` | Tokenization, construction, prediction, training, and parameter totals. |

Private implementation-sensitive tests live with their domain. Cross-domain,
CLI, and public-consumer tests stay under top-level `tests/`.

## Mutation-Resistant Tests

The optimizer tests assert invariants that should fail when meaningful logic is
mutated. They check the supplied learning rate, update direction, optimizer
history, moment updates, determinism, and compatibility with the feed-forward
layer.

These tests are described as mutation-resistant because they target likely
faults. They do not imply that every possible mutation has been measured or
killed by a mutation-testing tool.

## Integration Tests

Integration cases combine real components to cover tokenization, model
construction, prediction, training, and multi-layer behavior.

Run all Rust checks with:

```bash
cargo fmt --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --all-targets
```

## Black-Box Evaluation

[rustgpt-evals](https://github.com/BA-CalderonMorales/rustgpt-evals) is a
separate project that invokes the compiled CLI through its public process
boundary. It records each case as `passed`, `failed`, or `skipped` with a
reason.

Keeping that harness separate prevents the evaluator from importing RustGPT
internals. The target can change internally while the evaluator continues to
observe the same external contract.

The `.github/workflows/dispatch-e2e.yml` workflow can notify that companion
repository when relevant `main.rs` behavior changes, provided the repository
dispatch credential has been configured.
