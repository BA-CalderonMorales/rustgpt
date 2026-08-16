# Demo

The README's `docs/demo-tui.gif` is the proof loop, not a mockup: every
frame runs the release binary against real artifacts -- version contract,
the OOV fallback contract, the micro-arena truth table, the seeded
output-property suite, and the tiny lane's score formula.

## Regenerate the GIF (vhs, primary path since v0.0.5)

Requires `vhs` (dev-only tool, never a cargo dependency):

```bash
cargo build --release
vhs scripts/demo/demo.tape   # -> docs/demo-tui.gif, deterministic
```

The session is scripted (`scripts/demo/demo.tape`): identical key presses,
identical sleeps, identical output on every re-run. The recording is
evidence: it re-proves what the current main really does, so re-record
after any model-affecting change and treat the GIF as such.

## Legacy path (asciinema + agg)

`scripts/demo/tui.tape` is the v0.0.4 showcase archive, and
`scripts/demo/demo_session.sh` remains the fallback recorder:

```bash
asciinema rec -q --cols 100 --rows 24 -c "bash scripts/demo/demo_session.sh" scripts/demo/demo.jsonl.cast
agg --speed 2 --font-size 15 scripts/demo/demo.jsonl.cast docs/demo-tui.gif
```

## Regenerate the still (PNG)

`docs/demo-tui.png` backs registry pages where GIFs are unsupported. It is
captured from the interactive boot (trained model answering a probe, prompt
cursor live) using the std-only ANSI renderer:

```bash
cargo build --release
scripts/demo/capture_frame.sh tui     # -> docs/demo-tui.png
```

Heads-up for contributors: the recorded session is ~90 seconds of real
compute; re-recording re-proves what the current main really does, so the
GIF is evidence -- treat it as such.
