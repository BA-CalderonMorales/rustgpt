# Compact Water-Cycle Teaching Set

This dataset is intentionally a micro-domain, not a general knowledge corpus.
Its purpose is to let the current small, word-level model repeatedly encounter
one compatible set of facts and one stable question-and-answer structure.

## Capability

After the normal interactive training flow, the model should recognize a
`User:` prompt about the basic water cycle and attempt a short,
corpus-grounded `Assistant:` answer. The supported ideas are evaporation,
condensation, cloud formation, precipitation, collection, and repetition of
the cycle. Broad science knowledge and open-ended reasoning are out of scope.

## Format

- Pretraining examples are short declarative sentences ending in `. </s>`.
- Instruction examples use exactly
  `User: <question>? Assistant: <answer>. </s>`.
- Content uses sentence case, ASCII punctuation, and the same terms for the
  same concepts.
- Answers should be short enough to finish well inside the 80-token model
  limit.

## Budgets

The curated data must stay within all of these limits:

| Measure | Budget |
|---|---:|
| Pretraining examples | 25 |
| Pretraining whitespace tokens per epoch | 192 |
| Chat examples | 53 |
| Chat whitespace tokens per epoch | 1,029 |
| Combined model vocabulary | 120 |
| Maximum whitespace tokens in one example | 25 |
| Maximum model tokens in one example | 29 |

The example and token budgets are ceilings, not targets. Purposeful
paraphrases may repeat a fact, but one-off facts and synonyms do not belong in
the corpus. The vocabulary budget is deliberately much lower than the
530-token baseline because each word expands both embeddings and the output
projection.

## Relationship Between the Files

`pretraining_data.json` teaches only the foundational declarative relations.
`chat_training_data.json` reuses those relations in the runtime's exact role
format. Several controlled question paraphrases point to concise canonical
answers. Neither file should introduce an unrelated topic merely for factual
variety.

## Held-Out Prompts

These prompts are evaluation observations and must not appear verbatim in
either training file:

1. `User: Why do heavy droplets fall from clouds?`
2. `User: How does cooling change water vapor?`
3. `User: Where does rainwater collect after rainwater flows downhill?`
4. `User: What happens after rivers carry water to the ocean?`

The same four prompts ship as prompt/reference pairs in `data/heldout.json`
with canonical corpus-anchored answers. `cargo run -- --eval --seed 42` scores
them (see `docs/running-and-development.md`); per-item `exact`/`prefix` and
per-position `accuracy` are computed by `llm::answer_score`.

The measured v0.0.2 baseline (seed 42): `exact_matches: 1/4`,
`prefix_matches: 1/4`, `mean_accuracy: 0.3125`, with item 2 an exact match and
item 3 a degenerate repetition loop. Behavior is recognizable when at least
three of the four outputs begin with `Assistant :`; since seeding, runs are
reproducible evidence rather than uncontrolled observations.

## Baseline and Inference Boundary

Before curation, the two files contain 78 examples and 1,221 whitespace tokens
but create a 530-token model vocabulary; 368 vocabulary entries occur only
once. There are no exact duplicates or repeated prompts, yet the examples span
unrelated geography, biology, technology, history, greetings, and general
science. Case variants, contractions split by punctuation, numerical facts,
and one-off names consume capacity without reinforcing one behavior.

The `--e2e` path builds a newly randomized model from the dataset-derived
vocabulary and immediately calls prediction. It does not train or load learned
weights. It therefore remains a fast CLI contract probe; learned-behavior
observations require the interactive training flow. This dataset update does
not add a checkpoint format or startup training.

## Measured Result

The curated files contain 16 pretraining examples and 28 chat examples. They
use 562 whitespace tokens per epoch, have an 89-token model vocabulary, and
contain no exact duplicate examples. The longest example is 19 whitespace
tokens (23 model tokens).

Three independently initialized training runs evaluated all four held-out
prompts. All 12 observations generated `</s>`, and 10 began with the exact
`Assistant :` token sequence. Outputs used short water-cycle phrases. One run
satisfied the stricter predeclared requirement for at least three matched
prompt-answer relations, while two did not. The demonstrated capability is
therefore consistent role/termination structure plus occasional controlled
paraphrase matching, not reliable question-specific semantic generalization.

The median complete trained run, including four held-out predictions, took
11.47 seconds and 9,552 KiB peak resident memory on the measurement machine.
The baseline did not finish its 200 training epochs within a 30.62-second
bounded run; it reached instruction epoch 90 after completing pretraining.
These timings show a large practical improvement, but ordinary run-to-run and
cache noise still applies.

## TinyStories Lane (v0.0.4)

Second corpus arena for the `--tiny` preset. Source:
[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(Eldan & Li, 2023), license **CDLA-Sharing-1.0** (open data license, no
paywall; recorded 2026-08-12). Free-open-data rule satisfied; cite the source
in any artifact.

Rebuild the local slice (gitignored under `models/tinystories/`):

```bash
curl -L -o models/tinystories/TinyStories-train.txt \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
python3 scripts/slice_tinystories.py  # 40k stories, <=120 words each -> train.jsonl
```

Measured baseline (2026-08-12, laptop CPU, `--tiny`, seed 42, 1 epoch over
40k stories / ~1.5M tokens): single-epoch loss 5.84 from a ln(vocab) ~ 8.1
random start; greedy generation collapsed to the dominant token (`.`).
Throughput measured at ~283 tokens/s -> ~10M tokens per 9.8 hours. The tiny
preset at 14.2M params is therefore an overnight-slice tier, not a
converged tier; batching or a GPU racecar (documented ADR candidate) is
required before corpus-scale runs earn their hours.
