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
|-- main.rs               CLI, dataset setup, training, and interactive mode
|-- lib.rs                Module exports and model constants
|-- llm.rs                Model composition, training, and generation
|-- transformer.rs        Transformer block composition
|-- self_attention.rs     Self-attention operation
|-- feed_forward.rs       Position-wise feed-forward operation
|-- embeddings.rs         Token and positional embeddings
|-- output_projection.rs  Vocabulary projection
|-- layer_norm.rs         Layer normalization
|-- vocab.rs              Vocabulary and tokenization
|-- dataset_loader.rs     JSON and CSV dataset loading
`-- adam.rs               Optimizer implementations
```

Tests mirror the major components under `tests/`. See [Testing](testing.md) for
the purpose of each test layer.

## Reading Order

1. `src/main.rs` shows how data, model layers, training phases, and CLI modes
   connect.
2. `src/llm.rs` shows how layers participate in prediction and training.
3. `src/transformer.rs` narrows the view to attention and feed-forward work.
4. Read the individual layer modules alongside their matching tests.

## Scope

This implementation is intentionally small. It is useful for tracing model
mechanics, but it is not designed for production workloads, large datasets, or
competitive language generation.
