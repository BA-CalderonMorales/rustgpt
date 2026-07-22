# Model and Training

This page records the current implementation rather than prescribing a final
architecture.

## Configuration

- Vocabulary size: derived from the bundled training data
- Embedding dimension: 128
- Hidden dimension: 256
- Maximum sequence length: 80 tokens
- Model body: embeddings, three transformer blocks, and an output projection
- Decoding: greedy token selection

The shared dimensions and sequence limit are defined in
`src/configuration/constants.rs` and compatibility-re-exported from the crate
root.

## Training Phases

The default executable performs two phases:

1. Pre-training uses the factual examples in `data/pretraining_data.json` for
   100 epochs with a learning rate of `0.0005`.
2. Instruction tuning uses the conversational examples in
   `data/chat_training_data.json` for 100 epochs with a learning rate of
   `0.0001`.

The model uses cross-entropy loss, backpropagation through its layers, and
gradient clipping. The code is intended to make those mechanics traceable, not
to provide a scalable training system.

## Dependencies

- `ndarray` supplies multidimensional arrays and matrix operations.
- `rand` and `rand_distr` support parameter initialization.
- `serde`, `serde_json`, and `csv` support structured data and output.
- `bincode` supports binary encoding.

There is no PyTorch, TensorFlow, Candle, or other model framework in the
implementation.
