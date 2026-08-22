# Demo

The README's `docs/demo-tui.gif` is the proof loop, not a mockup: it walks
`llm --help`'s operating path top to bottom on the release binary, against
real artifacts, seeded and reproducible.

## The operating path

`llm --help` prints the map the demo follows. Start at the top, move down;
each step prepares the next:

```text
 1  llm --models                         pick an artifact from the trained catalog
 2  llm --model <id> --ask "<prompt>"    one answer from it; decode knobs honored
 3  llm --model <id>                     chat interactively (/help inside)
 4  llm --demo                           watch raw text become a model, end to end
 5  llm --tiny --train <corpus.jsonl>    teach your own model (add --eos --lr-decay)
 6  llm --tiny --eval --model <path>     score it against held-out data
 7  llm --eval --seed 42                 the micro arena oracle: fresh, seeded
```

## Inside --demo

The guided tour is the human-readable pipeline: seven numbered steps from
raw text to a working model, each closing a `.......... DONE` bar and
saying what just happened in plain language:

```text
1) Pulling the training dataset        path, story count, first excerpt
2) Cleaning the text                   whole-word vocabulary, <unk> / </s>
3) Configuration for pretraining       the settings table (see below)
4) Building the model                  parameter count and arrangement
5) Training                            the live loss bar, start -> end loss
6) Scoring held-out stories            CE p50 (teacher-forced), collapse gate
7) Using your model                    greedy vs tuned decode, chat handover
```

Step 3 is the configuration table: every applied value next to what it
decides and where to tweak it (`src/configuration/constants.rs` for model
shape, `--epochs` / `--lr-decay` / `--seed` flags on your own run, or the
corpus itself for vocabulary). Values come from the same sources the run
uses; the table never paraphrases a second truth.

Nothing is saved by the tour: training happens in memory only, and the
tour ends inside interactive chat on the freshly trained model.

## Regenerate the GIF

Requires `vhs` (dev-only tool, never a cargo dependency):

```bash
cargo build --release
vhs scripts/demo/tui.tape   # -> docs/demo-tui.gif, deterministic
```

`tui.tape` is the single scripted session -- identical key presses,
identical sleeps, identical output on every re-run. The recording is
evidence: it re-proves what current main really does, so re-record after
any model-affecting change and treat the GIF as such.

## Non-destructive by contract

A showcase must never corrupt what it showcases. The tape obeys three
rules, and any new step must too:

- Cataloged artifacts (`stories-full`, `stories-demo`, `watercycle-*`) are
  only ever LOADED. No `--train` or `--eval` step points `--model` at one:
  those modes continue-train and re-save an existing checkpoint, which
  would silently overwrite the recorded recipe.
- Training writes to a scratch artifact (`models/tinystories/showcase.bin`)
  after an explicit `rm -f`, so every recording starts fresh.
- Machine JSON passes through the demo formatters (`scripts/demo/
  show_eval.py`, `scripts/demo/show_gate.py`) so the viewer reads verdicts
  with meaning, not raw dumps.

## Fixtures and helpers

- `scripts/demo/my-first-corpus.jsonl`: twelve hand-written stories, the
  train-your-own corpus used on screen. Anyone can swap in their own.
- `scripts/demo/show_eval.py`: micro-eval and tiny-train summaries
  (trajectory, recipe, honest gate).
- `scripts/demo/show_gate.py`: tiny-eval + fluency gate verdicts at a
  decode config.
- `scripts/demo/make_demo_slice.py`: rebuilds the 300-story demo slice
  the `--demo` mode trains on.

## Regenerate the still (PNG)

`docs/demo-tui.png` backs registry pages where GIFs are unsupported. It is
captured from the interactive boot using the std-only ANSI renderer:

```bash
cargo build --release
scripts/demo/capture_frame.sh tui     # -> docs/demo-tui.png
```

Heads-up for contributors: a full recording runs a few minutes of real
compute (the oracle eval alone trains both phases); re-recording re-proves
what current main really does, so the GIF is evidence -- treat it as such.
