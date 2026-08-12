# rustgpt showcase: proof the loop is real, not a gimmick.
# Every command below runs the actual release binary against real artifacts:
# version, contract probe, micro-arena eval, and the laptop lane on a small
# real slice of TinyStories. From the repository root:
#   vhs scripts/demo/tui.tape

Output docs/demo-tui.gif
Set FontSize 16
Set Width 1100
Set Height 700
Set Padding 24
Set Shell bash
Set TypingSpeed 20ms
Set Framerate 20

Type "python3 scripts/demo/make_demo_slice.py"
Enter
Sleep 2s
# 0. stage the tiny demo slice (300 real stories from the 40k train set)

Type "./target/release/llm --version"
Enter
Sleep 2s
# 1. the release checkpoint: v0.0.4, seeded, version contract pinned by tests

Type "./target/release/llm --e2e 'hello world'"
Enter
Sleep 3s
# 2. the contract probe: one JSON object, deterministic across processes

Type "./target/release/llm --model models/watercycle-0.0.3.bin --eval --seed 42 | python3 -m json.tool | head -40"
Enter
Sleep 14s
# 3. the micro arena truth table: train both phases, score the four held-out
#    prompts, ship the trajectory -- the same 380K-param model, end to end

Type "./target/release/llm --tiny --train models/tinystories/demo.jsonl --epochs 1 --seed 42 --model models/tinystories/demo.bin | python3 -m json.tool | head -60"
Enter
Sleep 50s
# 4. the laptop lane: 14.2M params trained on real story text, checkpoint
#    saved, trajectory and samples printed -- made on this machine, by hand

Sleep 3s
# 5. freeze on the close-up: loss curve and the three generation samples
