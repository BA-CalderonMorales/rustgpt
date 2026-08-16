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
- **The Qwen decode-knob family (observation, 2026-08-16).** Qwen3.8-27B's
  model card (open weights, apache-2.0) documents repetition control as a
  decode-time recipe: instruct mode `temperature=0.7, top_p=0.80,
  top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
  (thinking mode: `temperature=1.0, top_p=0.95, min_p=0.0, penalties
  off`). The card names presence_penalty as the "reduce endless
  repetition" knob, recommendable 0-2, warning that values near 2 can mix
  languages and cost quality -- exactly our collapse regime. We have
  temperature and uniform top-k; we lack presence/repetition penalties,
  probability-weighted temperature sampling, and top-p -- the untried
  decode-time levers for the tiny-lane repetition gate. min_p is disabled
  in both official recipes, so it is out of scope. The full instruct
  stack (T=0.7 + top-p 0.80 + presence 1.5) is the shipped combination;
  single-knob probes test each member, and the combined stack is the
  highest-fidelity port for the final verdict.
- **Temperature-scaled greedy, falsified (2026-08-16, W3, seed 42).** Claim:
  "logits / T before the output softmax moves the tiny-lane collapse-gate
  repetition rate below 0.5 from the T=1 pin of 1.0, without retraining."
  Table on models/tinystories/ts-13m-s42.bin (96-token gate, no retrain,
  recipe: --tiny --eval --seed 42 --temperature <T>):

  | T | repetition rate | collapsed |
  |---|---|-----------|
  | 0.7 | 1.0 | true |
  | 0.8 | 1.0 | true |
  | 0.9 | 1.0 | true |
  | 1.0 | 1.0 | true |
  | 1.1 | 1.0 | true |
  | 1.2 | 1.0 | true |

  Null confirmed. The mechanism is mathematical, not empirical: scaling
  logits by a positive constant preserves the softmax argmax, so greedy
  decode is byte-identical at every T (pinned by the
  temperature_scaled_greedy_is_argmax_invariant test). Diagnosis: the
  collapse is NOT a softmax-sharpness artifact and NOT a data-volume
  problem (demo-slice probe); the remaining decode-time lever is
  probability-weighted temperature SAMPLING (the mechanism paragraph's
  intended knob, and Qwen's actual usage); if that fails, the collapse is
  in the weights (label smoothing / LR decay / weight tying). The
  `--temperature` knob stays as the gate's instrument: greedy invariance
  makes any T a pin, which the sampling probe needs as its baseline.

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

## Training and Data (the collapse probes)

- **The continuous collapse profile, landed with a named falsification
  (2026-08-16, W2, seed 42).** Claim: "per-epoch top-1 margin, top-2 gap,
  logit norm, and softmax output entropy reveal the attractor's onset
  before repetition saturates to 1.0." `--tiny --train` now samples a
  `profile` block per epoch (mean top-1 margin, top-2 logit gap, logit
  norm, softmax entropy over the held-out token stream, teacher-forced;
  `--tiny --eval` JSON unchanged). Evidence run on the 300-story demo
  slice (6 epochs, constant LR 5e-4): loss 6.146 -> 5.231, collapse
  repetition 1.0, entropy 5.159 -> 4.802 (knee at epochs 3-4, where the
  prior 3-epoch run measured 0.9684), logit norm 89.6 -> 141.2, top-1
  margin flat-low 0.017-0.043 (0.017 at collapse), top-2 gap noisy
  0.15-0.47. Verdict: the instrument works -- the regime shift is visible
  as a trajectory (entropy falls, logit scale rises) before saturation --
  but the margin half of the claim is FALSIFIED: the attractor is a
  MOVING frequency head (p1-p2 stays small while the rank-1 identity
  drifts), not a confidence corner, so instantaneous margin does not
  foreshadow. The successor quantity is the free-running profile (logit
  stats collected during the gate's 96-token sample itself -- the regime
  where repetition actually happens); ready when a lever verdict needs it.
- **Training budget at the recipe level, falsified (2026-08-16, seed 42).**
  Claim: "more epochs on the 300-story demo slice breaks the collapse
  gate." Table (--tiny --train models/tinystories/demo.jsonl --seed 42,
  constant LR 5e-4):

  | epochs | repetition rate | final loss |
  |--------|-----------------|------------|
  | 1 | 1.0 (recites ".") | 5.97 |
  | 3 | 0.9684 (recites "the") | 5.39 |
  | 6 | 1.0 | 5.23 |
  | 12 | 1.0 | 5.37 |

  The collapse is a MOVING frequency-head attractor: at 1 epoch the model
  latches "." (the punctuation head), at 3-6 epochs it latches "the" (the
  most common story word; the gate's 3 non-repeating pairs are the
  "time a a a , , ," transitions). Loss instability at 12 epochs (5.23 ->
  5.37) with a constant LR shows the recipe cannot simply be trained
  longer. Three falsified hypotheses now triangulate the collapse: data
  volume (demo vs 40k, same gate), decode sharpness (temperature
  invariance), and epoch count at this recipe. CE stays 5.7-7.1 with
  coverage 1.0 while generation is fully degenerate: teacher-forced CE is
  blind to the collapse. The untested levers, in order: LR decay (the
  recipe is provably unstable without it), probability-weighted sampling
  and a repetition penalty (decode-time), label smoothing (calibration),
  and a continuous logit profile (top-1 margin / output entropy per
  epoch) as the instrument that makes collapse onset visible -- the
  boolean gate cannot see the regime structure.

## Training and Data (free-compute lane)

- **The pristine-data probe on a clean slice (2026-08-16, seed 42).** The
  300-story demo slice (`models/tinystories/demo.bin`, 7.3M params,
  regenerated by scripts/demo/make_demo_slice.py) scores BETTER held-out
  CE than the full-corpus 14M artifact (p10/p50/p90 = 5.44/5.83/6.25 vs
  5.49/5.87/6.31) yet collapses IDENTICALLY: repetition rate 1.0, sample
  length 96. Data cleanliness moves the loss curve, not the gate — the
  collapse is not a data-volume problem, so a pristine dataset is a strict
  second move behind a decode/weights diagnosis (W3's temperature sweep).
- Free compute verdict: see docs/free-compute.md — Colab/Kaggle free tiers
  verified 2026-08-16 (Colab FAQ; Kaggle docs + cross-checks): ~30 GPU
  h/week (Kaggle, T4/P100 auto-assigned, wall-clock quota), ~12h sessions.
  No CUDA path in this stack, so free GPUs are currently useless; free
  CPUs ~= this laptop. Cloud buys offload + reproducibility, not speed;
  scripts/cloud-train.sh is the offload lane.

## Performance

- Measure before changing implementation details.
- Explore parallelism, allocation behavior, or SIMD as isolated experiments.
- Record the readability cost of each optimization alongside its benchmark.
- **BLAS probe, blocked by environment (2026-08-16, W5).** Attempted the
  README-of-record change: optional `blas` feature (openblas-src). This
  WSL2 image has no gfortran (source build impossible) and
  `libopenblas-dev` (0.3.32 available in apt) requires interactive sudo,
  unavailable headless. Cargo.toml default features untouched; no
  half-wired feature flag merged. The clean next attempt is the W4 cloud
  path: Colab/Kaggle shell sessions have passwordless sudo, so
  `apt-get install -y libopenblas-dev` + `openblas-src` with the `system`
  feature (no gfortran) is the documented first move there; a measured
  >=2x matmul win earns the ADR. Until the roofline moves, the
  batched-training falsification stands.
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
