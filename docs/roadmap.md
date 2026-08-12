# Roadmap

The working map for the brain lane, aligned with everything observed so far.
Ponder, amend, then claim a line. Measured numbers carry their run recipe;
unclaimed arrows are open bets.

North star, restated: a bounded learning project, not an arms race. The hand-
rolled rig stays thin and legible (the verification instrument); anything
faster or bigger is the racecar, reached only through a documented decision.

## Where we are (v0.0.1 -> v0.0.4)

| Version | What landed | Measured evidence |
|---|---|---|
| 0.0.1 | Baseline micro model, release pipeline on push-to-main | 380,893 params, E2E contract |
| 0.0.2 | Seeded determinism, `--eval` score formula, governance contract | exact 1/4, prefix 1/4, mean acc 0.3125 (seed 42) |
| 0.0.3 | Checkpoints (v1), training trajectory, percentiles | held-out CE 5.30 -> 8.79 -> 1.95 -> 2.17: endpoints lie |
| 0.0.4 | `--tiny` laptop lane, JSONL + TinyStories (CDLA-Sharing-1.0), demo proof | 14.2M params, ~283 tok/s, one-epoch loss 5.84 |

Discipline now fixed: seeded runs are evidence, unseeded are observations;
a checkpoint plus its eval JSON is the unit of proof; every release ships
its CHANGELOG entry; free-open-data rule (license recorded before use);
zero secrets committed, ever.

## Definition of done for this project

1. Checkpoints -- landed (v0.0.3).
2. One named experiment run to a recorded verdict -- OPEN (claim one below).
3. The rig can score any open-weight model we want to interrogate -- OPEN.

After that, decide: graduate the lane or declare victory. All of it is
reversible.

## Open experiments (falsifiable claims, one per run)

- **Batch training.** Biggest win per engineering hour at the laptop lane:
  current per-example SGD is why 14.2M costs ~283 tok/s. Claim: batching
  (4-8 sequences) lifts throughput 3-5x at equal loss trajectory. Measure
  with the same seed 42 recipe.
- **MTP heads at micro scale.** Nemotron Lightning and DeepSeek V4 both
  train multi-token prediction heads. Port the mechanism to the 380K arena.
  Claim: MTP as auxiliary loss improves held-out CE trajectory on the
  water-cycle corpus, or it does not -- either verdict is the finding.
- **Converged micro-grid.** Converged loss vs params/tokens at 1M/3M/10M/
  13M on a TinyStories slice, seeds 42/7/11. Check against Cagnetta et al.
  exponent formula (arXiv 2602.07488): the no-free-parameter prediction is
  falsifiable from a laptop in about a day of CPU.
- **Trajectory formalization.** Promote the v0.0.3 non-monotonicity into a
  contract: report train-loss, held-out CE, and percentile summaries in
  every eval JSON; median-CE contest (arXiv 2605.24667) is a candidate
  probe for the tiny lane.
- **TinyStories eval clause.** The tiny lane currently reports trajectory +
  samples only. Design a story-arena score (held-out story continuation
  fidelity, e.g. next-token CE on a fixed 256-story held-out slice) before
  the lane claims any quality.
- **Parameter Golf entry.** 16MB artifact budget, BPB scoring, 84-technique
  taxonomy published (arXiv 2607.01517). A rustgpt-derived artifact is the
  finish-line event that externalizes the whole thesis. Cheap to attempt
  once the tiny lane converges.

## Lanes and their gates

### Laptop lane (current arena)
Ceiling is real but honest: converged runs at 1-13M params, overnight
slices at ~10M tokens per 9.8h. Next claims: batching, then the micro-grid.
Kaggle accelerates the *next* run, not this one.

### Kaggle lane (the arena beyond the laptop)
Gates agreed, in order:
1. Checkpoint contract -- landed (v0.0.2-v0.0.3).
2. A named, pre-registered hypothesis with its eval formula -- OPEN.
3. Toolchain-cache dataset (prebuilt CARGO_HOME) to kill session tax.
4. One warm-up session measuring real tokens/s on T4 before any training.
5. Then: 3 sessions/week, one hypothesis per session, checkpoints pulled
   home and scored on the laptop before believing.
Quota is a subscription, not capital: 29-30h/week, 9h/session, cannot
bank; shared 20GB/week internet with competition work. Opportunities
worth the tax: 30-124M converged grid, TinyStories-scale pretraining,
or the first replicate-the-paper run.

### Observatory lane (open weights)
Both beloved models are open: DeepSeek V4 Flash 0731 (MIT, 304B) and
Nemotron 3.5 Lightning 30B-A3B (OpenMDW-1.1; Q4 ~18-20GB -- laptop
runnable). The cheapest high-value first observation: run Lightning
locally, reproduce one published claim from its NeMo Gym recipes, and
publish the truth table. This is the score-formula economics move applied
to vendor claims, and it needs no GPU at all beyond the laptop's RAM.

## Decision points (write an ADR when crossed)

- **GPU racecar ADR.** Trigger: a corpus-scale run whose hours exceed the
  laptop's honest budget (batching first, since it may move the wall). The
  racecar stack would be `tch` (libtorch bindings) or candle/ort for
  serving; documented cost: legibility. The rig and the racecar never
  merge.
- **Kaggle integration ADR.** Written after the warm-up measurement, with
  the number in hand.
- **TinyStories eval ADR.** When the story-arena score formula is
  pre-registered.
- **Parameter Golf entry ADR.** When a 16MB artifact is within reach.

## Risks ledger

- Compute wall: honest at ~283 tok/s; batching and Kaggle are the levers,
  convergence is the binding constraint, not ambition.
- Degeneracy: the v0.0.4 demo shows early-training collapse (`. . .` / "the
  the the"). Expected for one epoch; the trajectory contract exists
  precisely to catch it before any claim is made.
- Scope creep: every new lane costs a claim on the definition of done.
  The finish line is three items; nothing is added to it without debate.
- The frontier is downloadable: any artifact we make loses to a free MIT
  checkpoint. The moat is understanding, recorded as evidence -- not the
  weights.

One rule worth re-reading each morning: one hypothesis per run, verdicts in
the changelog, falsification is a discovery.
