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

For one trained model, behavior is recognizable when at least three of the
four outputs begin with `Assistant :`, express the expected water-cycle
relation using corpus terms, and generate `</s>` before the sequence limit.
Initialization is uncontrolled, so individual runs are observations rather
than deterministic quality guarantees or causal proof.

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
