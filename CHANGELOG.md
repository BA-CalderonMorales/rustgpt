# Changelog

Every release entry records what changed and the measured evidence it
produced: score, trajectory, and artifacts. The top section becomes the
GitHub release body (see `.github/workflows/release.yml`), so the public
record and the repo history are the same document.

## [0.0.12] - 2026-08-22

`--models` renders one record per audience: in a terminal, the human
table alone -- no more raw JSON wall drowning it; piped, exactly one
JSON object with a silent stderr (the machine contract, now clean on
both channels). Same record, one rendering per reader; the terminal
audience can still have the JSON by redirecting (`llm --models | cat`).

The showcase GIF drove this: its terminal showed table plus JSON dump
interleaved, which is what a real user sees. Contract pin moved in the
same change (piped --models now asserts empty stderr); docs updated.
Gates: fmt, clippy -D warnings, 104 tests passed. GIF re-recorded.

## [0.0.11] - 2026-08-22

The legibility pass: every human-facing surface that predates the help
and demo redesigns now speaks the same visual grammar -- numbered steps,
aligned columns, thousands-separated numbers, and values that read as
words when they mean "off".

`--models` rebuilt: dynamic-width columns (long ids like
watercycle-latest no longer collapse into their neighbors), right-aligned
PARAMS with separators, artifact filenames, quality verdicts verbatim.
The machine JSON contract on stdout is untouched; the table stays on
stderr. Interactive chat rebuilt: the boot block is an aligned Model
section (network, dimensions, parameters, seed), loaded checkpoints and
before/after-training phases read as titled sections instead of banner
dumps, the prompt is `you>`, answers print as `model>`, and `/config`
renders a knob table where neutral values say "off". The tiny-lane
training narration (`--tiny --train`, stderr) joined the demo's numbered
step grammar with DONE bars; its stdout JSON is untouched.

New homes per the facade rules: application/catalog_table.rs renders the
catalog table, application/format.rs owns number rendering (with a unit
test), narrate.rs gained the stderr twins of the demo's step primitives;
the stage-banner constants are gone. Contract pins moved in the same
change (models header, config verdicts, answer marker, loaded-checkpoint
regression pin). One behavior bug caught during the pass: the restyle
initially dropped the train-lane save_checkpoint call -- restored before
commit, covered by existing checkpoint round-trip tests.

Seeded evidence unchanged by design: micro lane exact 4/4, prefix 4/4,
mean 1.0 at seed 42; stories-full eval CE p10/p50/p90 5.49/5.87/6.31.
Gates: fmt, clippy -D warnings, 104 tests passed (one new). GIF/PNG
re-recorded so the showcase matches the shipped surfaces.

## [0.0.10] - 2026-08-22

The demo, retold for humans: `--demo` is now a seven-step numbered
pipeline walkthrough -- raw text in, working model out -- built to be
followed, not just watched.

Each step prints what it is doing, closes a dot-bar DONE marker in the
trainer's visual grammar, then says what happened in plain language:
(1) pull the dataset (path, story count, first excerpt); (2) clean the
text into the whole-word vocabulary (<unk> stranger, </s> end); (3) the
new configuration table -- every applied setting beside what it decides
and where to tweak it (`src/configuration/constants.rs` for model shape,
`--epochs` / `--lr-decay` / `--seed` flags on your own run, the corpus
itself for vocabulary), with values read from the same sources the run
uses; (4) build (parameter count and arrangement); (5) train on the live
loss bar; (6) score held-out stories honestly; (7) decode greedy vs
tuned side by side and hand over the keyboard inside interactive chat.

Layout: the tour split at its natural seam into `application/demo.rs`
(steps 1-5) and `application/demo_use.rs` (score, use, handover), with
the table renderer in `application/settings_table.rs`; the stdout lane
of `narrate.rs` swapped stage banners for numbered-step primitives while
the stderr lane keeps serving `--tiny --train`. docs/demo.md documents
the tour's shape ("Inside --demo"). Nothing is saved by the tour;
training stays in memory.

Seeded evidence (--demo --seed 42, release binary): 300 stories,
1,379-word vocabulary, 7,318,883 parameters; loss fell 6.15 -> 5.39
over 3 epochs at lr 5e-4; held-out CE p50 6.45 (teacher-forced);
collapse gate 0.97 at greedy (measured, not hidden) while the tuned
stack (T=0.7, top-p 0.80, presence 1.5, repetition 1.1) samples
repetition-free. Gates: fmt, clippy -D warnings, 103 tests passed.
GIF/PNG re-record deferred: the tape still closes pointing at the tour.

## [0.0.9] - 2026-08-22

The guided-path patch: `--help` becomes the map, the demo walks it, and a
state-mutation bug in the flagship use surface is fixed.

Help as the operating path (the temperwright pattern): `--help` now opens
with seven numbered steps -- models, ask, chat, demo, train your own,
score, oracle -- each with a one-line utility statement, then the decode
knobs / training levers / reproducibility flags grouped in plain language,
then one working example per step. The contract pin moved with it in the
same change.

The demo rebuilt around that path, non-destructive by contract. The old
tape silently corrupted what it showcased: its training step pointed at
the cataloged stories-demo artifact (continue-training it with a
different recipe) and its eval step re-trained and re-saved the flagship.
The new tape obeys three written rules (docs/demo.md): cataloged
artifacts are only ever LOADED; training writes to a scratch artifact
(models/tinystories/showcase.bin) behind an explicit rm -f; machine JSON
passes through verdict formatters (scripts/demo/show_eval.py,
show_gate.py) so viewers read meaning, not dumps. The session opens on
`--help`, walks all seven steps including train-your-own on a tracked
12-story fixture (scripts/demo/my-first-corpus.jsonl), and closes on the
map. Competing homes deleted: scripts/demo/demo.tape (stale v0.0.5) and
demo_session.sh (v0.0.7); tui.tape is the one home the makefile records.
GIF and PNG re-recorded against this release binary.

Bug fix: `llm --model <catalog-id>` no longer mutates state. The old
code resolved a catalog id for loading but let interactive mode re-check
the RAW argument for existence; missing as a file, the id fell into the
first-run branch -- double-training the already-loaded model for 200
epochs and writing a stray artifact named after the id into the working
directory. Reproduced before the fix (stray file + md5 mismatch vs the
catalog artifact), pinned after: chat_with_a_catalog_id_loads_it_and_
never_creates_files asserts the LOADED MODEL branch fires, the training
branch does not, and the working directory gains nothing. Resolution now
happens exactly once at the parse boundary (application::
resolve_model_arg), so loads, interactive's loaded-model check, and save
targets share one real path.

Home page slimmed 189 -> 75 lines: the operating path quick start, two
sentences of honesty, the docs table. The artifact inventory moved to
docs/model-workflow.md; the command reference lives where it belongs --
in `--help` and docs/running-and-development.md.

Gates green: fmt, clippy -D warnings, 103 tests passed (one new
regression pin). Micro lane 4/4/1.0 unchanged; every number in the 0.0.8
gap table stands.

## [0.0.8] - 2026-08-22

The use-surface release, aimed at Qwen3-0.6B. The 0.0.7 decode win was
unreachable by humans -- every knob was welded to `--tiny --eval`, so
loading the flagship and typing a prompt still printed 80 periods. This
release moves the knobs to where users are, teaches the model to stop,
gives training a schedule, and narrates the whole pipeline for a curious
beginner. One contract drift is also repaired: `--models` prints its
machine JSON on stdout again (the human table moved to stderr), matching
the pinned one-object rule.

Surfaces. `--ask <prompt>`: single-shot raw continuation against a loaded
checkpoint (prompt sent verbatim; never trains, never saves -- the
checkpoint's FNV-1a hash is asserted unchanged by the contract test); one
JSON object carrying `status`, `seed`, `total_parameters`, `prompt`,
`output`, and a `decode` block echoing `{temperature, top_p, presence,
repetition}`. Interactive chat accepts the decode knobs at launch and in
session via slash commands (`/help`, `/temp`, `/top-p`, `/presence`,
`/repetition`, `/config`, `/reset`, `/exit`); bad values mutate nothing;
a trailing `</s>` renders as a clean end of answer instead of leaking the
marker; closed stdin ends the session instead of spinning. The chat
surface split into `src/application/chat.rs`. `--demo`: a guided six-stage
novice tour (data -> vocabulary -> model -> training -> evaluation ->
use) on the 300-story slice, ending greedy-vs-tuned side by side on the
same starter, then chat; seeded, reproducible, saves nothing.
`--tiny --train` now narrates the same six stages on stderr while stdout
stays the single machine JSON.

Decode engine. `llm::generate_with_steps` unifies every decode leg behind
one dispatch (greedy pin at T=1.0 without knobs; penalties before the
softmax; nucleus or probability-weighted sampling otherwise) with an
optional per-step capture (`DecodeStep{token, prob}`). Parity is pinned:
captured streams equal the recompute `predict_*` family token-for-token,
greedy argmax stays temperature-invariant over a seeded prompt pool, and
the full tiny-lane eval JSON is byte-identical to the pre-release binary
on both legs (verified old-vs-new on stories-full). New property tests
(hand-rolled seeded generators): capture parity across configs, knob
state machine (valid sets move exactly their own field; invalid move
nothing), and black-box chat-session probes.

Training levers (seed 42, demo slice, 6 epochs each, fresh init):

| run | loss | held-out CE p50 | greedy gate |
|---|---|---|---|
| control (constant 5e-4) | 6.15 -> 5.23, U-shape | 6.66 | 1.000 |
| W8 decay -> 5e-5 | 6.15 -> 5.06, monotone | 6.47 | 0.979 |
| E11 `--eos` | 6.12 -> 5.23 | 6.80 (+0.14) | 1.000 |
| combo (eos + decay) = new stories-demo | 6.12 -> 5.02, monotone | **6.40** | 0.990 |

W8 verdict: SPLIT -- stability half LANDED (monotone curve where constant
LR bends up; CE improves 0.18), repetition half FALSIFIED (0.979 is
nowhere near the gate). E11 verdict: LANDED -- termination becomes a
learned outcome for the first time (mean sampled completion length 3.0
vs the 123-token cap; CE inside the predeclared +0.25 margin); greedy
still ignores `</s>` (gate 1.0), confirming the collapse is argmax-level,
not labelable from below; side effect measured: termination overshoot
(3-token stubs), which decay moderates (combo: 19.4-token completions,
repetition-free 0.45, distinct-1 0.823 under the stack). Catalog refreshed
for stories-demo under the combo recipe.

Us-vs-Qwen3-0.6B gap table (same yardsticks; Qwen3-0.6B GGUF via
llama.cpp on this laptop; our numbers seed 42):

| yardstick | rustgpt stories-full (14.2M) | Qwen3-0.6B (GGUF) |
|---|---|---|
| held-out CE, 256 stories | 5.88 nats/token (PPL ~358) | 2.61 nats/token (PPL 13.5) |
| greedy repetition rate (96 tok) | 1.000 (collapsed) | 0.000 |
| stack distinct-1 / rep-free | 0.701 / 0.65 | 0.709 / 0.70 |
| stack completion len / sentences | 123-cap / 7.05 | 121.5 / 7.55 |

Reading the scoreboard honestly: at the DECODE layer we are at near-parity
with Qwen3-0.6B (distinct-1 0.70 vs 0.71, repetition-free 0.65 vs 0.70);
at the WEIGHTS layer the chasm is intact -- Qwen's GREEDY decode is clean
(0.000) while ours collapses (1.000), and it holds 2.6 nats/token against
our 5.88. The knobs route around the attractor; they do not remove it --
that is what 0.4B extra parameters plus subword tokenization buys. Metric
humility note: even Qwen's "clean" greedy soft-loops at n-gram level
("He had a son named Mew" x4); adjacent-pair metrics score it perfect,
which is exactly why `is_degenerate`'s n-gram window exists.

Perspective (perception evidence only, never merged into scores): the
codex-cli human-eyes loop was quota-blocked ("You've hit your usage limit
... try again at Sep 10th, 2026"), so the cold-viewer critique ran through
the local open-weights oracle (qwen3.8:27b via ollama), same prompts,
quoted verbatim. BEFORE (worst three): "--models leaks a raw internal-QA
JSON blob"; "--help is an undifferentiated wall of 20+ flags"; "The model's
output leaks a raw special token ... `Assistant : Heavy water droplets ...
rain . </s>`". AFTER: named wins -- "Each knob gets a one-line
plain-English gloss", "I always know which setting produced the text I'm
staring at", "`/reset` removes ambiguity"; remaining worst: "--models is
unusable for model selection" (catalog jargon stays deliberate: it is the
evidence record; stderr keeps the human table), "1.0 = greedy inverts the
standard convention" (fixed: help now reads "unscaled -- greedy while no
other knob moves"), and "no A/B surface" (answered by `--demo` stage 6).
Demo comprehension check, played as a novice who never studied ML: token
("a whole chunk from a fixed menu"), epoch ("one full read of every
story"), loss ("how surprised ... on average") all answered correctly; the
dual-instrument point landed ("skipping one would have made the model look
way better than it is"); two copy defects found and fixed in this release:
"teacher-forced" was used undefined, and "p50" dropped unexplained.

## [0.0.7] - 2026-08-16

The quality release: from boolean gate to measurable fluency. Greedy decode
collapses (repetition rate 1.0, a 123-period loop); the Qwen-honoring
decode stack -- probability-weighted temperature sampling at 0.7 with
top-p 0.80, presence 1.5, and the count-scaled repetition penalty at 1.1 --
lands the gate at 0.021 with 65% of completions fully repetition-free,
distinct-1 0.70, distinct-2 0.99, and ~7 sentences per completion, on the
same 14M-param artifact, no retraining.

Foundation (W1): one-batch overfit audit. A micro-config model drives a
single story to teacher-forced loss 0.0008 over 200 epochs (seed 42, LR
5e-4) and re-emits it greedily from a mid-story prefix: reproduction 1.0,
repetition-free. The mechanics (token/label alignment, optimizer, loss
masking) are sound -- every downstream lever is trustworthy.

Instruments (W2, W3): `--tiny --train` now samples a per-epoch logit
profile (mean top-1 margin, top-2 gap, logit norm, softmax entropy) over
the held-out stream; `--tiny --eval --fluency <n>` adds the decode-quality
yardstick (distinct-1/2, repetition-free rate, completion probe). W2's
margin half is falsified: the attractor is a MOVING frequency-correlated
head (p1-p2 stays flat while the rank-1 identity drifts), and the regime
shift is visible as entropy falling 5.16 -> 4.80 and logit norm rising
89.6 -> 141.2 across 6 epochs. W3 calibrated the pass floor
(repetition-free >= 0.5 AND distinct-1 >= 0.1) on a fully collapsed
artifact (distinct-1 0.008, 123 periods).

Decode levers (W4 landed, W5 landed, W6 falsified): W4 -- sampling from
the temperature-scaled softmax (seeded) breaks the gate at every T
(0.7: 0.095, 0.8: 0.021, 0.9: 0.021, 1.1: 0.011, 1.2: 0.000 vs the 1.0
greedy pin); the attractor is a deterministic-argmax phenomenon -- the
falsified uniform top-k threw away exactly the rank-2+ mass sampling
restores. W5 -- the count-scaled repetition penalty is a TOTAL
deterministic loop-breaker at greedy (gate 0.000 at coefficient 1.1, every
presence); flat presence alone only shifts the attractor (1.000 -> 0.832).
W6 -- top-p is falsified as an improvement on the W4 winner: it trims the
very low-mass tail that supplies diversity (distinct-1 0.820 -> 0.737 at
p=0.80), but the full Qwen stack probe earns the release recipe. The
recipe honors Qwen3.8-27B's documented instruct decode family
(temperature 0.7, top_p 0.80, presence 1.5; unsloth GGUF card, apache-2.0)
with one principled deviation: repetition 1.1, the coefficient our W5 grid
measured as the count-scaled winner on this artifact.

Surfaces: `--models` serves the model catalog (models/catalog.json --
path, family, parameters, seed, recipe, eval, quality per artifact);
interactive `--model <path>` is now the use surface -- load a trained
checkpoint and chat, no training, no re-save; a bare `--` is a no-op
separator; `--temperature`, `--presence`, `--repetition`, `--top-p`, and
`--fluency` are tiny-eval decode knobs, seeded and reproducible, with the
greedy leg pinned. The demo tape (re-recorded, 4.0 MB) walks: catalog,
contract probe, the domain-labeled trace on the loaded artifact, micro
arena, laptop lane, and the headline gate defeat via
`scripts/demo/show_gate.py`. A root `makefile` surfaces the gates and
lanes (`make verify`, `make build`, `make demo`, ...). The OneDrive build
tax is measured (14.1s vs 5.9s incremental release rebuild, 2.4x) and the
`CARGO_TARGET_DIR` workaround is documented. docs/model-workflow.md walks
create -> score -> decode recipe -> record -> use -> ship.

Scores and pins (seed 42 throughout): micro lane held-out 4/4 exact, 4/4
prefix, mean 1.0, trajectory [5.67, 1.53, 1.52, 1.55, 1.68] -- unchanged
by design; tiny lane CE p10/p50/p90 = 5.49/5.87/6.31, coverage 0.9976;
collapse gate 1.0 (greedy pin) -> 0.021 (release stack). W7 (label
smoothing) and W8 (LR decay) are predeclared in the backlog with recipes
and falsification criteria; the decode-side win made them optional for
this release.

Perspective (three oracles, quoted; perception evidence, never merged into
score numbers): codex cold-viewer -- "the release's strongest evidence is
the W4/W5/W6 decode study... a clean intervention ladder on one fixed
artifact"; weakest is "the model's claimed language quality... sampling
and penalties route around that failure; they do not establish learned
storytelling"; next moves ranked: train longer with checkpoints, a
repetition-aware training loss, a richer continuation-quality corpus.
codex ML-scientist advisory -- the claim should stay narrow:
"decode-time controls defeat greedy collapse on ts-13m-s42.bin under the
specified yardstick"; regime-sensitive risks named: single-seed
sensitivity, teacher-forced/free-running mismatch, the T=1-greedy vs
T!=1-sampled decoder-mode discontinuity, n=20 fluency sample size,
distinct-n gaming by incoherent tails, single-starter prompt dependence,
and Qwen transfer risk (tokenizer/scale/calibration differ). DeepSeek V4
Pro 0813 (headless opencode) -- "the evidence supports 'greedy
deterministic-argmax over a count-correlated logit bias causes repetition
collapse on this artifact' -- that claim is tight and reproducible";
labeling it a 'frequency head' without internals evidence overreaches
("an inference, not a measurement"), and teacher-forced CE blindness is a
measurement gap, not a discovery about the model. Bounded claim: sound.

## [0.0.6] - 2026-08-16

The observability release: make the pipeline traceable first, spend the
freed credibility on the collapse gate, and state the free-compute truth.
No new score to celebrate -- instead, three honest falsifications that
triangulate the frontier, plus the instrument that points at the code.

Observability (W1): `--trace` turns the interactive lane into a
domain-labeled event stream on stderr -- `[cli]`, `[configuration]`,
`[dataset]`, `[checkpoint]`, `[llm]`, `[vocab]`, `[decode]` -- where the
label IS the file to open when output misbehaves. Every generated token
prints its domain, its string, and its softmax probability via the new
`DecodeStep` capture (`predict_with_steps`), so the greeting-attractor
story on "water?" is visible as `"Hello" p=0.9992` followed by the
water-cycle chain reasserting at `p=0.6822`. Greedy output is
byte-identical: the shared empty-output fallback now has one authoritative
home (`answer_string`), and main.rs is the two-view dispatcher
(`run_interactive` / `run_headless`). `--trace` with any machine mode
errors out (exit 2); machine JSON contracts are untouched.

Artifacts (W2): the README now inventories every `models/*.bin` -- size,
recipe, what it demonstrates, how to regenerate -- including the honest
verdict that `watercycle-0.0.3.bin` is pre-format-v2 ("not a rustgpt
checkpoint") and kept for format archaeology only. Every artifact is
regenerable evidence, gitignored, reproduced from its seed.

The collapse attack (W3), verdict: falsified, and the falsification is the
finding. Temperature-scaled greedy decode (logits / T before the output
softmax) at T in {0.7, 0.8, 0.9, 1.0, 1.1, 1.2} holds the tiny lane's
collapse-gate repetition rate at 1.0 -- mathematically pinned, because
softmax argmax is invariant to positive logit scaling, and test-verified.
A new probe then attacked the "undertrained" hypothesis: 3 / 6 / 12 epochs
on the 300-story demo slice (constant LR 5e-4) moves the attractor from
"." to "the" but never below the gate (0.9684 / 1.0 / 1.0), and the recipe
goes unstable at 12 epochs (loss 5.23 -> 5.37). Three hypotheses are now
falsified with evidence: data volume, decode sharpness, epoch count at
this recipe. The gate is a moving frequency-head attractor; teacher-forced
CE (5.7-7.1, coverage 1.0) is blind to it. Untested levers, ranked by the
advisory: LR decay, probability-weighted sampling and a repetition
penalty, label smoothing, and a continuous logit profile (top-1 margin /
output entropy) as the instrument the boolean gate cannot be.

Free compute (W4): Colab/Kaggle quotas verified by webfetch (Colab: GPU
"heavily restricted", ~12h sessions, dynamic limits; Kaggle: ~30 GPU
h/week, T4/P100, wall-clock quota). The honest verdict: this stack has no
CUDA path, so free GPUs are useless today and free CPUs ~= this laptop;
the cloud buys offload and reproducibility, not speed. `scripts/cloud-train.sh`
and `docs/free-compute.md` ship the lane. BLAS (W5) was attempted and
blocked by the environment (no gfortran; system OpenBLAS needs interactive
sudo); Cargo.toml default features untouched, and the cloud path is the
documented first move for the next attempt.

Score (water-cycle held-out, seed 42): exact 4/4, prefix 4/4, mean 1.0,
trajectory [5.67, 1.53, 1.52, 1.55, 1.68] -- unchanged by design; this
release changes surfaces, not weights. Tiny-lane pin (ts-13m-s42.bin):
CE p10/p50/p90 = 5.49/5.87/6.31, coverage 0.9976, collapse gate 1.0 -- the
number this release names, measures, and hands to the next.

Perspective (three oracles, quoted; perception evidence, never merged
into score numbers): codex cold-viewer -- "the strongest artifact is the
seeded micro-eval plus trace... small and likely overfit, but reproducible,
inspectable evidence tied to an objective, not a cherry-picked chat
transcript"; weakest is "the tiny lane: CE around 5-6 and repetition rate
1.0 indicate a model or training/evaluation failure, not merely a decoding
preference". codex ML-scientist advisory -- "temperature is falsified as
the cause... one epoch at ~1.5M tokens for a 14M model is still a very
undertrained regime; the collapse may be an optimization/data-budget
failure rather than an architectural failure"; the honest next run is "one
properly controlled longer-training/LR-decay baseline and one
probability-weighted sampling diagnostic". DeepSeek V4 Pro 0813
(headless opencode) -- "your own data already proves two things people
usually take three months to learn: CE and collapse are decoupled, and
temperature can't rescue greedy"; the repo's value proposition: "the
collapse is not a bug to paper over -- it's the best teaching artifact you
have. Make the collapse the pin, not the problem."

## [0.0.5] - 2026-08-16

The no-framework rig stops apologizing: contracts first, then three
measured wins and one honest falsification, all seeded and reproducible on
the same laptop.

Learning: OOV prompts emit an explicit fallback answer instead of a silent
empty output with status:ok; the tiny lane gains a pinned score formula
(per-item CE, p10/p50/p90, vocab coverage, generation-collapse gate)
against a 256-story held-out slice (split seed 20260816) that never enters
training; a hand-rolled seeded output-property suite (P1-P6) pins the
baseline pass table; `--probe` ships the opt-in decode-time compute truth
table; a KV-cache decode path (Layer::set_cache_mode seam, per-block K/V
buffers) is byte-identical to recompute; `--eval` promotes the min-CE
tuning state into the saved artifact; every epoch interleaves chat and
pretrain statements at ~2:1 (arXiv 2403.05175 rehearsal); the corpus
expands within its documented budgets (16 -> 21 statements, 28 -> 53 chat
pairs, vocab 85 -> 88) with five chain statements and 25 targeted
paraphrase pairs, no held-out wording verbatim.

Score (water-cycle held-out, seed 42, replay recipe + min-CE promotion):
exact 4/4, prefix 4/4, mean accuracy 1.0 -- up from the v0.0.2-0.0.4 pin
of 1/4, 1/4, 0.3125 (a 4x exact-match jump to a perfect held-out score).
Held-out CE trajectory [5.67, 1.53, 1.52, 1.55, 1.68]: no pretrain spike,
drift capped, shipped artifact at min-CE 1.52.

Epistemic honesty (E7): every dataset vocabulary now carries `<unk>`, the
tokenizer maps unknown words to it instead of silently dropping them, and
six hedge pairs teach "a prompt containing <unk> deserves 'I do not know
that word.'". Unseen OOV probes hedge 6/6 ("What is gravity?", "How do
volcanoes work?", "Why is the moon bright?", "How do mountains form?",
"What is lightning?", "What does the ocean contain?") -- the model no
longer answers "How do mountains form?" with a confident water-cycle
sentence. Two tokenizer traps were found and fixed along the way: the
marker must be a whole token even with attached punctuation ("<unk>?"),
and the corpus rule against exact duplicates applies to hedge pairs too.

Property suite (seed 20260816, 50 draws, same generator seed as the
baseline): P2 termination 50/50, P3 format 50/50, P4 budget 50/50, P5
non-degeneracy 50/50 (baseline: 0.94 / 0.92), P6 best-of-N dominance
0.92. The item-3 repetition loop and item-4 fragment loop of v0.0.4 are
gone from the draw distribution.

Perspective (codex exec cold-viewer reports, quoted): v0.0.4 -- "mediocre,
bordering on broken... 'Assistant : lakes' repeated three times". v0.0.5
(E7) -- "the honest 'I do not know' responses to mountains and gravity
are the most impressive artifact: the model does not confidently
hallucinate water-cycle content for unfamiliar prompts"; the weakest
artifact is item 4's condensation attractor, "brittle retrieval rather
than robust question understanding". The loops are gone, the hedge is
real, and the residual attractor is the next frontier -- none of it
hidden.

Experiments and verdicts:
- E1 min-CE promotion: landed (saved artifact CE 1.24 <= 2.11; eval >=
  baseline).
- E2 replay scheduling: landed with a diagnosed tail (spike 8.79 -> 1.88,
  drift 2.17 -> 2.05, final sample 0.05 above the 2.0 claim line).
- E3 best-of-N probe: falsified with diagnosis (uniform top-k discards
  the sharp greedy mass; 0 exact/prefix at every (k, N) vs greedy 1/4;
  the loop-recovery case vanished with E2).
- E4 KV-cache decode: landed (3.98x: 56.6 -> 225.5 tok/s, byte-identical,
  pinned by tests/kv_cache_test.rs).
- E5 batched training: falsified at the roofline (ndarray non-BLAS dots
  scale linearly; stacked batches buy ~1.4x, attention 0x; the 3-5x
  claim is unreachable on this laptop without BLAS -- measured, not
  assumed).
- E6 paraphrase expansion: landed (exact 2/4, mean 0.6534, CE floor 1.24).
- E7 learned `<unk>` hedging: landed (exact 3/4, mean 0.7917; unseen OOV
  probes hedge 6/6; no confident wrong answers on unknown prompts).
- E8 hedge stabilization: landed (9/10 full hedges on unseen probes,
  mean 0.8409; a duplicate hedge pair was removed per the corpus rule).
- E9 conversation suite: the interactive surface gains a predeclared
  20-probe score formula (five classes, per-prompt tokenization and OOV
  counts); the property suite's prompt pool is frozen for comparable
  deltas.
- E10 social register: landed with measured residuals (held-out 4/4,
  mean 1.0; greetings generalize to unseen forms; conversation suite
  16/17 out-of-domain prompts answered with a hedge or greeting; residual
  truncated/stuttered hedges and a greeting attractor on single-word
  in-vocabulary prompts are recorded, not hidden).

Training progress on a terminal now renders as a live ASCII bar
(`Epoch 42/100 | Loss = 0.1234 | [##########..........]`) -- pipes keep
the per-epoch line format, so captures and CI stay byte-stable. The demo
GIF re-records on the winning recipe with the bar in motion.

Tiny lane (seed 42, artifact ts-13m-s42.bin): coverage 0.9976, CE
p10/p50/p90 = 5.49 / 5.87 / 6.31, collapse gate collapsed with repetition
rate 1.0 -- the lane's dot degeneracy is now a number on its formula, and
the decode path that will eventually fix it is 3.98x faster.

Artifacts: release binary + checksums; docs/demo-tui.gif re-recorded on
the winning recipe (vhs, scripts/demo/demo.tape); local
`models/watercycle-latest.bin` and `models/watercycle-e6.bin` (gitignored)
reproducible as `--eval --seed 42 --model <path>` from a missing file.

## [0.0.4] - 2026-08-12

Learning: `--train <file.jsonl>` opens the single-corpus laptop lane at
`--tiny` scale (Config presets; 14.2M params, 384/768/128, 6 blocks); JSONL
dataset loading with window truncation; `--epochs`; checkpoint format v2
carries the full model shape; TinyStories adopted (CDLA-Sharing-1.0) with a
reproducible slicing script.

Score (water-cycle held-out, seed 42): exact 1/4, prefix 1/4, mean accuracy
0.3125 -- micro arena unchanged.

Tiny-lane first truth table (seed 42, 1 epoch, 40k stories, 1.53M tokens,
laptop CPU): loss 5.84 from ~8.1 random start at 14.2M params; greedy
generation collapsed to the dominant token; throughput measured at ~283
tokens/s (~10M tokens per 9.8h). The lane works end to end; convergence
tier and corpus-scale budgets are the v0.0.5 problem, with batching and a
GPU racecar (ADR candidate) documented as the options.

Artifacts: release binary + checksums; local
`models/tinystories/ts-13m-s42.bin` (57 MB, gitignored), slice
reproducible via `scripts/slice_tinystories.py`.

## [0.0.3] - 2026-08-12

Learning: checkpoint save/load (`--model <path>`, format v1), training
trajectory in eval JSON (per-epoch loss, held-out CE samples), accuracy
percentiles, `llm::sequence_loss` teacher-forced probe.

Score (seed 42, water-cycle held-out): exact 1/4, prefix 1/4, mean accuracy
0.3125.

Trajectory finding: pretrain loss 4.73 -> 0.08, tuning 4.86 -> 0.21, but
held-out CE ran 5.30 -> 8.79 (pretraining hurt chat-format CE) -> 1.95
(tuning rescued) -> 2.11 -> 2.17 (drift up). Endpoints would have hidden
all of it; the curve is the claim.

Artifacts: release binary + checksums; local `models/watercycle-0.0.3.bin`
(1.5 MB, gitignored).

## [0.0.2] - 2026-08-12

Learning: seeded determinism by default (seed 42), `--eval` score formula
(exact / prefix / per-position accuracy over the four held-out prompts,
`data/heldout.json`), gorvernance contract in AGENTS.md.

Score (seed 42): exact 1/4, prefix 1/4, mean accuracy 0.3125. The first
truth table: item 2 exact match ("Condensation changes water vapor into
droplets"), item 3 a degenerate repetition loop -- now measured, not
anecdotal.

## [0.0.1] - 2026-08-12

Baseline: from-scratch transformer LLM in pure Rust, water-cycle micro
domain (16 statements + 28 QA pairs), interactive chat, `--e2e` contract
probe (totals validation: 380,893 parameters), release pipeline on push to
main.
