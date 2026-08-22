# rustgpt showcase: the operating path, live on the release binary.
# Mirrors `llm --help`: start at the top, move down. Every step runs the
# real binary; every step is non-destructive -- the cataloged artifacts
# are only ever LOADED, and training writes to a scratch artifact that is
# removed first, so a re-record always starts fresh. From the repo root:
#   vhs scripts/demo/tui.tape

Output docs/demo-tui.gif
Set FontSize 16
Set Width 1100
Set Height 700
Set Padding 24
Set Shell bash
Set TypingSpeed 20ms
Set Framerate 20

Type "./target/release/llm --help"
Enter
Sleep 6s
# 0. the map: the operating path, the knob groups, one example per step.
#    Everything the demo does next is written here first.

Type "./target/release/llm --version"
Enter
Sleep 2s
# 1. the release checkpoint: v0.0.8, derived from Cargo.toml, pinned by tests

Type "./target/release/llm --models"
Enter
Sleep 4s
# 2. PATH STEP 1 -- the catalog: human table on stderr, machine JSON on
#    stdout; every artifact carries its recipe, seed, and scores

Type "./target/release/llm --model stories-full --ask 'Once upon a time,'"
Enter
Sleep 5s
# 3. PATH STEP 2 (greedy) -- the honest collapse: at T=1.0 with no knobs
#    the decode is the argmax pin, and the attractor wins: 80 periods.
#    Measured, not hidden.

Type "./target/release/llm --model stories-full --ask 'Once upon a time,' --temperature 0.7 --top-p 0.8 --presence 1.5 --repetition 1.1"
Enter
Sleep 6s
# 4. PATH STEP 2 (tuned) -- same model, same starter, four knobs: the
#    Qwen-honoring stack samples past the frequency head. The knobs are
#    the difference; nothing was retrained.

Type "printf '/help\nWhat is rain?\n/temp 0.7\nhi!\n/config\n/reset\nexit\n' | ./target/release/llm --model watercycle-latest 2>&1 | tail -22"
Enter
Sleep 6s
# 5. PATH STEP 3 -- chat interactively: slash commands move the knobs
#    mid-session (/config echoes them), /reset returns to greedy, and a
#    trailing </s> renders as a clean end of answer

Type "./target/release/llm --eval --seed 42 | python3 scripts/demo/show_eval.py"
Enter
Sleep 15s
# 6. PATH STEP 7 -- the micro arena oracle: trains fresh from seed 42,
#    scores the four held-out prompts, saves NOTHING (no --model flag),
#    so re-runs never touch an artifact

Type "rm -f models/tinystories/showcase.bin"
Enter
Sleep 1s
# 7. clear the scratch artifact: train-your-own must start fresh every time

Type "./target/release/llm --tiny --train scripts/demo/my-first-corpus.jsonl --epochs 2 --eos --lr-decay 1e-4 --seed 42 --model models/tinystories/showcase.bin | python3 scripts/demo/show_eval.py"
Enter
Sleep 10s
# 8. PATH STEP 5 -- teach your own model: a 12-story corpus you could
#    have written, the v0.0.8 recipe (--eos teaches stopping, --lr-decay
#    stabilizes), scored against held-out stories it NEVER saw

Type "./target/release/llm --tiny --eval --model models/tinystories/showcase.bin --fluency 20 | python3 scripts/demo/show_gate.py"
Enter
Sleep 5s
# 9. PATH STEP 6 (greedy leg) -- score it honestly: your first model
#    collapses at greedy (gate 1.0). The gate exists exactly so this is
#    a measurement, not a surprise.

Type "./target/release/llm --tiny --eval --model models/tinystories/showcase.bin --temperature 0.7 --top-p 0.8 --presence 1.5 --repetition 1.1 --fluency 20 | python3 scripts/demo/show_gate.py"
Enter
Sleep 5s
# 10. PATH STEP 6 (stack leg) -- same fresh model, four knobs: repetition-
#     free completions again. Collapse is a decode phenomenon you can
#     measure your way out of; the weights take longer.

Type "echo 'you just walked llm --help, top to bottom. go deeper: ./target/release/llm --demo --seed 42'"
Enter
Sleep 4s
# 11. close on the map: the long-form guided tour is one command away
