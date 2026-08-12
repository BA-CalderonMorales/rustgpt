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

## Continual and Test-Time Learning

- **Replay for the micro lane.** The sequential schedule (100-epoch pretrain,
  then 100-epoch chat tuning on disjoint data, `application/logic.rs`) is the
  textbook catastrophic-forgetting setup (arXiv 2403.05175); the v0.0.3
  held-out CE curve 5.30 -> 8.79 (pretrain spike) -> 1.95 -> 2.11 -> 2.17
  (tuning drift) is its measured signature. Claim: interleaving the 16
  pretraining statements into the tuning blocks at roughly 1:2 ratio keeps
  held-out CE below 2.0 across every phase at the same recipe (100 epochs,
  LR 0.0005/0.0001, seed 42), holding exact/prefix on `data/heldout.json`.
  Either verdict is the finding.
- **Save the checkpoint at min held-out CE.** `--eval` already samples the
  tuning trajectory (`TUNING_CE_SAMPLES`); promote the min-CE sample into
  the saved artifact so the drift tail (2.17) cannot ship over the 1.95
  point. One change, no recipe change.
- **OOV prompts return empty output.** `"hello"` produces `""` with
  `status:ok` because the vocabulary lacks the word and decode stops at the
  first token. Add a fallback or an explicit report line before the micro
  lane claims any quality.
- **MoE as the later alternative.** arXiv 2406.16437 (ICLR 2025 Spotlight):
  expert specialization mitigates forgetting, but the gating network must
  stop updating for convergence and added experts cost additional rounds.
  Bigger than replay; defer until the replay verdict is recorded.
- **Test-time training thread.** arXiv 1909.13231 and arXiv 2407.04620
  update the model on the test input itself (self-supervised loss, or the
  hidden state as a model). The 2407.04620 state-is-a-model view is the same
  online-learning mechanism as DeltaNet/KDA (doubleword.ai article); CQS
  means it needs an explicit opt-in mode with the eval formula untouched.
- **Test-time compute decode probe (micro lane, leveragable now).** arXiv
  2408.03314 applies here: test-time compute pays off when the base model
  has non-trivial success, and the micro lane does (exact 1/4, mean acc
  0.3125, seed 42). Probe: seeded top-k sampling (needs a small hand-rolled
  seeded PRNG, e.g. xorshift), K candidates per held-out prompt, rank by
  self-consistency, greedy contract untouched. Claim: sampled candidates
  recover more exact/prefix matches than greedy on `data/heldout.json`.
  The observed greedy failure (wrong-attractor recitation, mid-sentence
  fragments) is exactly the case sampling can recover.
- **Test-time compute scaling.** arXiv 2408.03314: compute-optimal per-prompt
  allocation beats best-of-N 4x and outperforms a 14x larger model in a
  FLOPs-matched comparison. Assessment for this lane: the decode probe above
  is the actionable slice; allocation by prompt difficulty uses the per-item
  scores already reported by `--eval`.
- **Diagnosis only, no lever today.** arXiv 2512.24695 frames Adam and
  momentum SGD as associative memories that compress gradient information
  into attractor states; that names the dot-collapse and memorization
  mechanisms but changes nothing planned. Its fix (deep-memory optimizers)
  is the experiment filed below.
- **Common thread across all five.** Every paper adds a learning moment
  where the naive schedule has none: test time, the state transition, the
  optimizer step, the expert update. The sequential pretrain-then-tune
  schedule is the only point not learning; replay is the cheapest first
  answer.
- **Expressive optimizers.** arXiv 2512.24695 (NeurIPS 2025) frames Adam and
  momentum SGD as associative memories compressing gradient information --
  the same step-on-reconstruction-loss derivation behind DeltaNet. A
  deep-memory optimizer at micro scale is a self-contained experiment:
  same-or-better loss at equal steps on the water-cycle recipe.

## Performance

- Measure before changing implementation details.
- Explore parallelism, allocation behavior, or SIMD as isolated experiments.
- Record the readability cost of each optimization alongside its benchmark.
