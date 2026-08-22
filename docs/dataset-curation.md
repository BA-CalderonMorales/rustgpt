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
| Chat examples | 59 |
| Chat whitespace tokens per epoch | 1,029 |
| Combined model vocabulary | 120 |
| Maximum whitespace tokens in one example | 25 |
| Maximum model tokens in one example | 29 |

The chat-example ceiling rose from 53 to 59 in v0.0.5 (E7): six hedge
pairs teach the model to answer `I do not know that word.` when a prompt
contains `<unk>` -- an epistemic capability, not paraphrase volume. The
hedge pairs are written with the literal `<unk>` placeholder so the topic
words never enter the vocabulary.

The example and token budgets are ceilings, not targets. Purposeful
paraphrases may repeat a fact, but one-off facts and synonyms do not belong in
the corpus. The vocabulary budget is deliberately much lower than the
530-token baseline because each word expands both embeddings and the output
projection.

## Paraphrase Expansion (v0.0.5, E6)

The v0.0.4 corpus had no chained structure: "rain falls -> flows downhill ->
collects" and "rivers reach the ocean -> cycle repeats" existed only as
isolated facts, which the held-out items 3 and 4 test. The corpus was
expanded within the budgets above (16 -> 21 pretrain statements, 28 -> 53
chat pairs; 144/192 pretrain tokens, 828/1029 chat tokens, vocab 85 -> 88):

- Five chain statements link the fall -> downhill -> collection -> ocean ->
  cycle arc.
- Twenty-five targeted paraphrase pairs balance the four concepts, with
  the collection and cycle families getting several wording variants each.
- No held-out prompt appears verbatim anywhere in either file (checked by
  script before commit), and the three new vocabulary words (Flowing,
  flowing, rise) stay inside the 120-token budget.

Verdict (2026-08-16, seed 42, E2 recipe + min-CE promotion): exact 2/4,
prefix 2/4, mean accuracy 0.6534 (E2: 1/4, 1/4, 0.4375); held-out CE
trajectory [5.13, 1.54, 1.24, 1.28, 1.30] with a 1.24 floor (E2: 1.91).
Item 3 became an exact match; items 1 and 4 remain wrong-but-fluent
attractors (clouds family, rain family). Property suite: P2-P5 1.00, P6
0.88.

## Learned Unknown-Word Hedging (v0.0.5, E7)

Every dataset-derived vocabulary now contains `<unk>`, and the tokenizer
maps out-of-vocabulary words to it instead of silently dropping them (an
all-`<unk>` prompt still decodes to the literal unknown answer). Six hedge
pairs teach the model: a prompt containing `<unk>` deserves
`I do not know that word.` rather than a confident in-domain answer.
Tokenizer care: `<unk>` is a whole token even with attached punctuation
("<unk>?"), otherwise the hedge trains on a fake `<` `unk` `>` sequence.
Verdict (2026-08-16, seed 42): held-out exact 3/4, prefix 3/4, mean
0.7917 (items 1-3 exact; item 4's condensation attractor is the residual);
the hedge fires on unseen OOV probes ("What is gravity?", "How do
volcanoes work?", "Why is the moon bright?", "How do mountains form?",
"What is lightning?", "What does the ocean contain?") -- 6/6 hedged, no
confident wrong answers.

Follow-ups in the same release:

- E8 (hedge stabilization): four more hedge pairs with varied `<unk>`
  positions (swapped for redundant paraphrase pairs; the corpus stays at
  59 chat examples; one exact duplicate pair was removed per the corpus
  rule). Held-out exact 3/4, mean 0.8409; the full hedge fires on 9/10
  unseen OOV probes.
- E9 (conversation suite): a predeclared 20-probe score formula for the
  interactive surface -- five classes (greeting, OOV, mixed, casual junk,
  in-vocabulary single words) with per-prompt tokenization and OOV counts.
  The property suite's prompt pool was frozen to the v0.0.4-era 32 prompts
  so pass-table deltas stay comparable as the corpus grows.
- E10 (social register): five greeting pairs ("hi!", "hello", "hey!",
  "hi there", "good morning" -> "Hello!") swapped in at the same budget.
  Held-out exact 4/4, prefix 4/4, mean 1.0 -- a perfect held-out score on
  a clean corpus; greetings generalize to unseen forms ("good morning");
  the conversation suite reports 16/17 out-of-domain prompts answered
  with a hedge or greeting, never a confident water-cycle sentence or
  fragment. Residuals, measured not hidden: one truncated hedge ("I do
  not know that ."), one stuttered hedge, and the greeting attractor can
  steal a single-word in-vocabulary prompt ("Water?" -> "Hello!").

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

## Free-Open-Data Shortlist (v0.0.8, verified 2026-08-22)

Candidate corpora for future lanes. Licenses were read from each dataset
card on the date above; nothing here has been downloaded or merged. The
allow-list (MIT, Apache-2.0, CC-BY, OpenMDW-1.1) is the gate: share-alike
and non-commercial candidates are recorded with their blocker, never
merged silently.

| Candidate | Source / license | Lane it would feed | Why not yet merged |
|---|---|---|---|
| [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | HF; **CDLA-Sharing-1.0** | stories (MERGED -- the `--tiny` lane) | already in use; recorded above |
| [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) (wikitext-2/103) | HF card: CC BY-SA (tags 3.0 + GFDL; card text cites 4.0) | long-context prose lane | **Share-Alike is outside the allow-list**; would obligate derivative corpus licensing. Blocked until the allow-list itself changes by ADR |
| [karpathy/tiny_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) | HF card records **no license** ("More Information Needed") | dialogue/prose demo lane (small: 1.1 MB, ideal for `--demo`) | license unrecorded = unmergeable under the free-open-data rule even though the underlying plays are public domain. Unblock path: re-source from a public-domain provider that states terms (e.g. Project Gutenberg texts + our own slicing script), then record it |
| [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) (10k human-written chat pairs) | HF card: **CC-BY-NC-4.0** | chat lane upgrade (real multi-turn QA) | **Non-commercial = rejected outright** by rule; recorded so nobody re-litigates it |
| Local open-weights teacher over TinyStories prefixes (distillation input set) | teacher: Qwen3 family GGUF (**apache-2.0 weights**, run locally via ollama/llama.cpp); inputs: already-licensed TinyStories rows | distillation experiments (teacher completions -> student training rows) | not merged because it does not exist yet: outputs are generated locally at experiment time, carry no third-party data, and only need their own recipe + seed record. This is the cheapest next data move that needs NO new license |

Budget note: the dataset budgets earlier on this page remain ceilings for
the water-cycle micro-domain; any new lane defines its own budget table in
this file before its first training row enters `models/tinystories/` or
`data/`.

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
python3 scripts/slice_tinystories.py  # 40k train stories + 256 held-out stories
```

The script writes `train.jsonl` (first 40,000 qualifying stories, exactly the
v0.0.4 slice) and `heldout.jsonl` (256 stories drawn with a seeded shuffle,
split seed **20260816** in the `python` random module, from qualifying
stories that follow the train boundary, deduplicated verbatim against the
train slice). The held-out slice never enters any training file; the split
seed must not silently redefine the training slice.

## Tiny-Lane Score Formula (v0.0.5)

The tiny lane cannot cite quality without its score formula. `--tiny --eval
--model <checkpoint>` (and the `eval` block of every `--tiny --train` JSON)
reports, against `models/tinystories/heldout.jsonl`:

- `per_item_ce`: teacher-forced cross-entropy per held-out story
  (`llm::sequence_loss`), the exact signal training optimizes.
- `ce_percentiles`: nearest-rank p10 / p50 / p90 across items; median tracks
  task quality better than the mean (arXiv 2605.24667), so the mean is
  never the trajectory claim.
- `coverage`: fraction of held-out tokens inside the model vocabulary
  (`tokenize` drops OOV words, which would otherwise bias CE downward).
- `collapse`: a generation-collapse gate. Greedy sample from the fixed
  starter `Once upon a time,` capped at 96 tokens; `repetition_rate` is the
  fraction of adjacent token pairs that are identical, and `collapsed` is
  true above 0.5. A lane whose greedy output degenerates is reported, not
  hidden.

Measured on the v0.0.4 artifact (`models/tinystories/stories-full.bin`, seed
42): coverage 0.9976; p10/p50/p90 = 5.49 / 5.87 / 6.31; mean CE 5.88;
collapse gate `collapsed: true` with repetition rate 1.0 (the known dot
degeneracy, now a number on the formula).

Measured baseline (2026-08-12, laptop CPU, `--tiny`, seed 42, 1 epoch over
40k stories / ~1.5M tokens): single-epoch loss 5.84 from a ln(vocab) ~ 8.1
random start; greedy generation collapsed to the dominant token (`.`).
Throughput measured at ~283 tokens/s -> ~10M tokens per 9.8 hours. The tiny
preset at 14.2M params is therefore an overnight-slice tier, not a
converged tier; batching or a GPU racecar (documented ADR candidate) is
required before corpus-scale runs earn their hours.
