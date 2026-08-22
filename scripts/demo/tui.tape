# rustgpt showcase: proof the loop is real, not a gimmick.
# Every command below runs the actual release binary against real artifacts:
# version, the model catalog, the contract probe, the domain-labeled trace,
# micro-arena eval, the laptop lane, and the headline: the collapse gate
# defeated by the Qwen-honoring decode stack. From the repo root:
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
# 1. the release checkpoint: v0.0.8, seeded, version contract pinned by tests

Type "./target/release/llm --models | python3 -m json.tool | head -30"
Enter
Sleep 3s
# 2. the model catalog: every trained artifact carries its path, family,
#    parameters, seed, recipe, eval, and quality labels -- one JSON object

Type "./target/release/llm --e2e 'hello world'"
Enter
Sleep 3s
# 3. the contract probe: one JSON object, deterministic across processes

Type "./target/release/llm --model stories-full --ask 'Once upon a time,' --temperature 0.7 --top-p 0.8 --presence 1.5 --repetition 1.1"
Enter
Sleep 8s
# 3b. the v0.0.8 use surface: single-shot ask against a loaded artifact at
#     the Qwen-stack decode recipe -- knobs finally reachable by humans

Type "printf 'yo!\nhello!\nwater?\nexit\n' | ./target/release/llm --trace --model models/watercycle-latest.bin --seed 42 2>&1 | tail -30"
Enter
Sleep 12s
# 4. the domain-labeled trace on the loaded artifact: the use surface --
#    no training, no re-save (provenance: catalog entry watercycle-latest,
#    seed 42, min held-out CE promoted)

Type "./target/release/llm --model models/watercycle-latest.bin --eval --seed 42 | python3 -m json.tool | head -40"
Enter
Sleep 14s
# 5. the micro arena truth table: train both phases, score the four held-out
#    prompts, ship the trajectory -- the same 385K-param model, end to end

Type "./target/release/llm --tiny --train models/tinystories/demo.jsonl --epochs 1 --seed 42 --model models/tinystories/stories-demo.bin | python3 -m json.tool | head -60"
Enter
Sleep 50s
# 6. the laptop lane: 7.3M params trained on real story text, checkpoint
#    saved, trajectory, the per-epoch logit profile, and the collapse gate

Type "./target/release/llm --tiny --eval --model models/tinystories/stories-full.bin --temperature 0.7 --top-p 0.8 --presence 1.5 --repetition 1.1 --fluency 20 | python3 scripts/demo/show_gate.py"
Enter
Sleep 60s
# 7. freeze on the headline: the collapse gate, defeated. Greedy decode is
#    pinned at repetition 1.0 (the "." frequency head); the Qwen-honoring
#    stack (temperature 0.7, top-p 0.80, presence 1.5, repetition 1.1) lands
#    the gate at 0.021 with 65% of completions fully repetition-free and
#    distinct-1 0.70 -- a 14M from-scratch model, no retraining

Sleep 3s
