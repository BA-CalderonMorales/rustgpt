# Demo

The README's `docs/demo-tui.gif` is the proof loop, not a mockup: every
frame runs the release binary against real artifacts -- version contract,
checkpoint-backed E2E probe, the micro-arena truth table, and the laptop
lane training on real TinyStories text.

## Regenerate the GIF

Requires `asciinema` (recorder) and `agg` (gif renderer):

```bash
cargo build --release
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
