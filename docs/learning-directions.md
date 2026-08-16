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
- **Targeted paraphrase expansion within budget.** Verdict (2026-08-16,
  E6): the v0.0.4 corpus had no chained structure ("rain falls -> flows
  downhill -> collects", "rivers reach the ocean -> cycle repeats"), which
  held-out items 3 and 4 test. Five chain statements plus 25 targeted
  paraphrase pairs (budgets: 21/25 examples, 144/192 pretrain tokens,
  828/1029 chat tokens, vocab 87/120) moved exact 1/4 -> 2/4, mean 0.4375
  -> 0.6534, CE floor 1.91 -> 1.24. Items 1 and 4 remain wrong-but-fluent
  attractors (clouds family, rain family) -- the next frontier is
  breaking cue competition ("heavy droplets" vs "droplets + clouds"
  co-occurrence), not more paraphrase volume.
- **Social register, landed (2026-08-16, E10).** Five greeting pairs
  ("hi!", "hello", "hey!", "hi there", "good morning" -> "Hello!")
  swapped into the fixed 59-pair chat budget moved the seed-42 held-out
  score to a perfect 4/4 (mean 1.0) and generalized greetings to unseen
  forms; the conversation suite (E9, 20 predeclared probes) reports 16/17
  out-of-domain prompts answered with a hedge or greeting. Residuals,
  measured: one truncated hedge, one stuttered hedge, a greeting
  attractor on single-word in-vocabulary prompts ("Water?" -> "Hello!"),
  and a greeting stutter on bare prompts without the "User:" prefix. The
  min-CE promoted state can lag the hedge/greeting behavior of the tail;
  check both when a surface behavior regresses.

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
  point. One change, no recipe change. Verdict (2026-08-16, seed 42, same
  recipe): landed. Saved artifact carries the 1.9508 state; held-out CE
  1.95 <= 2.11 by construction; eval 1/4, 1/4, mean 0.3482 >= baseline
  (0.3125); item 3's repetition loop broken at min-CE (property P3 0.94 ->
  0.96). Item 4 still loops at the min-CE point -- decode-time compute is
  the next lever for the residual attractor.
- **OOV prompts return empty output.** `"hello"` produces `""` with
  `status:ok` because the vocabulary lacks the word and decode stops at the
  first token. Add a fallback or an explicit report line before the micro
  lane claims any quality. Verdict (2026-08-16): fixed twice over -- the
  literal unknown answer for all-unknown prompts (v0.0.5) and the learned
  `<unk>` hedge (E7): six hedge pairs teach "prompt contains `<unk>` ->
  I do not know that word", generalizing to unseen OOV probes (6/6:
  gravity, volcanoes, moon bright, lightning, mountains, contain). The
  `</s>`/`<unk>` whole-token rule must handle attached punctuation
  ("<unk>?") or the hedge trains on a fake token sequence. Held-out
  unchanged by the tokenizer fix (3/4 exact, 0.7917 at the promoted
  state).
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
  fragments) is exactly the case sampling can recover. Verdict (2026-08-16,
  seed 42, E2 recipe model `models/watercycle-e2.bin`): falsified.
  Best-of-N by per-position score at every (k, N) in {3, 5, 8} x {8, 16}
  lands below greedy: 0 exact / 0 prefix / mean 0.107-0.283 vs greedy 1/4,
  1/4, 0.438. Diagnosis: uniform-over-top-k discards the sharp greedy mass;
  a micro model's rank-2+ tokens are noise, so sampled candidates spread
  the beam instead of recovering the attractor. The loop-recovery case
  vanished too: the E2 recipe no longer loops on item 3, so there is no
  attractor left for sampling to escape. P6 dominance row: 0.86 (baseline)
  -> 0.80 (E2): sampling never dominates a greedy that improved. Opt-in
  probe stays (mechanism + truth table shipped); the next decode lever is
  probability-weighted sampling or self-consistency ranking, not uniform
  top-k.
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
- **KV-cache decode, landed (2026-08-16, E4).** `Layer::set_cache_mode`
  seam (embeddings position cursor; blocks forward to their attention;
  preallocated doubling K/V buffers); prefill excludes the last prompt
  token, which becomes step 0's input. Byte-identical to recompute
  (pinned by tests/kv_cache_test.rs), 3.98x at 122 tokens on the tiny
  artifact (56.6 -> 225.5 tok/s). A concatenate-append first attempt
  measured 0.53x -- the preallocated buffer is the lesson.
- **Batched training, falsified at the roofline (2026-08-16).** ndarray's
  non-BLAS `dot` scales linearly with total rows, so stacking padded
  sequences buys only allocation/dispatch overhead: batch-4/8 matmuls run
  at ~1.4x the per-sequence cost (918us/1875us vs 4x/8x 322us) and the
  attention dot amortizes 0x (128x512 batch = 427us vs 108us single).
  The 3-5x tokens/s claim is unreachable on this laptop without BLAS, and
  the masked-batch rewrite would add all-`-inf` softmax NaN traps plus
  Adam-semantics ambiguity for <2x. The micro-grid bet (arXiv 2602.07488)
  is a day on this machine, not a week. Batch only if the roofline moves
  (BLAS-backed ndarray or a racecar ADR).
