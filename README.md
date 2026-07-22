# rustgpt

A from-scratch language-model implementation in pure Rust that uses `ndarray`
for tensor operations and no external machine-learning framework.

## Learning Project

This fork is my space to explore, learn, poke, observe, and shape a small
language-model implementation in ways that make its design easier to
understand. It favors inspectable Rust and explicit model mechanics over
scale, output quality, or feature breadth.

Correctness is separated by layer:

- Rust unit tests validate individual operations and components.
- Mutation-resistant tests check meaningful optimizer invariants.
- Integration tests exercise real model layers together.
- The separate [rustgpt-evals](https://github.com/BA-CalderonMorales/rustgpt-evals)
  project observes the public CLI contract as a black-box process.

The point is not to present a production LLM. The point is to build a mental
map of tokenization, embeddings, transformer blocks, optimization, training,
and generation by working with a complete implementation.

## Run It

```bash
git clone https://github.com/BA-CalderonMorales/rustgpt.git
cd rustgpt
cargo run
```

For a fast, machine-readable smoke check that does not train the model or enter
interactive mode:

```bash
cargo run -- --e2e "hello world"
```

## Explore the Project

- [Architecture](docs/architecture.md) maps the model pipeline and source tree.
- [Running and development](docs/running-and-development.md) covers the CLI and
  local commands.
- [Model and training](docs/model-and-training.md) records the current model
  configuration and training phases.
- [Testing](docs/testing.md) explains the unit, mutation-resistant, integration,
  and black-box evaluation boundaries.
- [Learning directions](docs/learning-directions.md) collects possible
  experiments without turning them into product promises.

Start with [`src/main.rs`](src/main.rs) for orchestration and
[`src/llm.rs`](src/llm.rs) for forward passes, backward passes, training, and
generation.

## Contributing

Focused contributions that improve understanding are welcome. Read the
[contribution guide](CONTRIBUTING.md) and follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

This fork remains available under the original [MIT License](LICENSE.txt). The
copyright and license notice from Thomas Karatzas are preserved unchanged.

## Attribution

This repository is a learning fork of
[tekaratzas/RustGPT](https://github.com/tekaratzas/RustGPT), created by
[Thomas Karatzas](https://github.com/tekaratzas). The upstream project is the
source of the original implementation; changes here focus on my learning
journey.
