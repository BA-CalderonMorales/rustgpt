# Architecture

rustgpt is a transformer-based language-model implementation written in Rust.
It uses `ndarray` for tensor operations instead of an external machine-learning
framework.

## Model Pipeline

```text
Input text -> tokenization -> embeddings -> transformer blocks -> output projection -> predictions
```

The executable builds a vocabulary from the included datasets, creates three
transformer blocks, adds an output projection, and assembles those layers into
an `LLM`.

## Source Map

```text
src/
|-- main.rs               Parse, load, build, and run
|-- lib.rs                Domain declarations and compatibility re-exports
|-- cli/                   CLI mode and argument behavior
|-- application/           Dataset, model, training, and interaction orchestration
|-- configuration/         Shared model constants
|-- llm/                   Model API, composition, training, and generation
|-- transformer/           Transformer block composition
|-- self_attention/        Self-attention operation and private gradient test
|-- feed_forward/          Position-wise feed-forward operation and optimizers
|-- embeddings/            Token and positional embeddings
|-- output_projection/     Vocabulary projection
|-- layer_norm/            Layer normalization
|-- vocab/                 Vocabulary and tokenization
|-- dataset_loader/        JSON and CSV dataset loading
`-- adam/                  Adam optimizer
```

Each domain's `mod.rs` is its facade. `interfaces.rs` holds types and traits,
`logic.rs` holds implementations, and specialized files such as `constants.rs`
or `tests.rs` exist only when that responsibility is present. Category modules
stay private; callers use the facade. See
[`ADR 0001`](architecture/decisions/0001-domain-module-layout.md) for the
structural decision and [Testing](testing.md) for test boundaries.

## Reading Order

1. `src/main.rs` shows the four executable steps.
2. `src/application/logic.rs` connects data, model construction, training, and
   interaction.
3. `src/llm/logic.rs` shows how layers participate in prediction and training.
4. `src/transformer/logic.rs` narrows the view to attention and feed-forward
   work.
5. Enter each domain through its `mod.rs` facade, then read its interface and
   logic alongside the matching tests.

## Scope

This implementation is intentionally small. It is useful for tracing model
mechanics, but it is not designed for production workloads, large datasets, or
competitive language generation.
