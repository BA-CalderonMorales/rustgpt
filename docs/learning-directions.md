# Learning Directions

These are possible experiments, not a product roadmap. Each one should remain
small enough to explain, test, and reverse.

## Model State

- Save and load trained parameters.
- Explore checkpoint structure and compatibility.
- Make initialization and experiments reproducible with explicit seeds.

## Generation

- Compare greedy decoding with temperature, top-k, or top-p sampling.
- Define measurable expectations before comparing generated text.
- Keep generation experiments behind a clear interface.

## Architecture

- Study positional encoding alternatives.
- Compare attention configurations.
- Add observability for attention, gradients, or intermediate tensor shapes.

## Training and Data

- Compare optimizers and learning-rate schedules.
- Explore regularization and gradient behavior.
- Improve tokenization or dataset streaming without hiding the mechanics.

## Performance

- Measure before changing implementation details.
- Explore parallelism, allocation behavior, or SIMD as isolated experiments.
- Record the readability cost of each optimization alongside its benchmark.
