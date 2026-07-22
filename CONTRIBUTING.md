# Contributing to rustgpt

Thank you for helping make this learning fork useful to other curious builders.
The goal is to expose how an LLM works through small, inspectable changes—not
to turn this repository into a production model framework.

## Before You Start

- Read the README and search existing issues.
- Open an issue before structural, cross-module, or difficult-to-reverse work.
- Keep each pull request focused on one concept or behavior.
- Preserve attribution to the upstream RustGPT project.

Good contributions include clearer explanations, focused experiments, stronger
tests, and small changes that make model mechanics easier to observe.

## Choose the Right Test Layer

- Unit tests validate one layer or mathematical operation.
- Mutation-resistant tests assert meaningful invariants such as update direction,
  optimizer state, determinism, or tensor shape.
- Integration tests exercise multiple real model layers together.
- Black-box CLI behavior belongs in the separate
  [rustgpt-evals](https://github.com/BA-CalderonMorales/rustgpt-evals)
  repository.

If a change affects `src/main.rs` or the `--e2e` JSON contract, explain that in
the pull request and link any corresponding evaluator change.

## Development Workflow

1. Fork the repository and create a focused branch.
2. Make the smallest change that demonstrates the idea.
3. Add or update tests at the appropriate layer.
4. Run the verification commands below.
5. Open a pull request using the repository template.

```bash
cargo fmt --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --all-targets
```

## Pull Request Expectations

Explain:

- what changed and why;
- what concept the change helps make visible;
- which test layer provides evidence;
- any known limitation or intentionally deferred work.

Generated or assisted code is welcome, but contributors remain responsible for
understanding, testing, and explaining what they submit.
