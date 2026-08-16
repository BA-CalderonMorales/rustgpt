# rustgpt showcase: proof the loop is real, not a gimmick.
# Every command below runs the actual release binary against real artifacts:
# version, contract probe, the domain-labeled trace, micro-arena eval, and
# the laptop lane on a small real slice of TinyStories. From the repo root:
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
# 1. the release checkpoint: v0.0.6, seeded, version contract pinned by tests

Type "./target/release/llm --e2e 'hello world'"
Enter
Sleep 3s
# 2. the contract probe: one JSON object, deterministic across processes

Type "cp models/watercycle-latest.bin /tmp/opencode/trace-demo.bin && printf 'yo!\nhello!\nwater?\nexit\n' | ./target/release/llm --trace --model /tmp/opencode/trace-demo.bin --seed 42 2>&1 | tail -30"
Enter
Sleep 16s
# 3. the domain-labeled trace: every generated token carries its domain
#    (vocab / llm / decode) and softmax probability -- the 0.0.6 surface

Type "./target/release/llm --model models/watercycle-latest.bin --eval --seed 42 | python3 -m json.tool | head -40"
Enter
Sleep 14s
# 4. the micro arena truth table: train both phases, score the four held-out
#    prompts, ship the trajectory -- the same 385K-param model, end to end

Type "./target/release/llm --tiny --train models/tinystories/demo.jsonl --epochs 1 --seed 42 --model models/tinystories/demo.bin | python3 -m json.tool | head -60"
Enter
Sleep 50s
# 5. the laptop lane: 7.3M params trained on real story text, checkpoint
#    saved, trajectory, samples, and the honest collapse gate printed

Sleep 3s
# 6. freeze on the close-up: the collapse gate -- the number 0.0.6 attacks
