# rustgpt 0.0.5 showcase: a six-step story, live on the release binary.
# Every step runs the real binary against real artifacts; the echo lines
# are part of the recording, so the GIF narrates itself. From the repo
# root:  vhs scripts/demo/demo.tape

Output docs/demo-tui.gif
Set FontSize 16
Set Width 1100
Set Height 700
Set Padding 24
Set Shell bash
Set TypingSpeed 20ms
Set Framerate 20
Set PlaybackSpeed 2

Type "echo 'rustgpt 0.0.5 -- a transformer written from scratch in Rust'"
Enter
Sleep 1s
Type "echo 'no PyTorch. no frameworks. 386K parameters, hand-rolled.'"
Enter
Sleep 2s
Type "./target/release/llm --version"
Enter
Sleep 3s
# 1. the rig: version contract, derived from Cargo.toml, pinned by tests

Type "echo '-- step 1: the contract. unknown words get an honest answer, never a silent empty output'"
Enter
Sleep 2s
Type "./target/release/llm --model models/watercycle-latest.bin --e2e 'zzzzzz'"
Enter
Sleep 4s
# 2. the OOV contract in action

Type "echo '-- step 2: watch it learn. 100 pretrain + 100 tuning epochs, live loss bar'"
Enter
Sleep 2s
Type "./target/release/llm --eval --seed 42 | python3 -m json.tool | head -60"
Enter
Sleep 26s
# 3. the learning moment: the bar moves, then the truth table prints

Type "echo '-- step 3: the held-out score -- exact 4/4, mean accuracy 1.0'"
Enter
Sleep 3s
Type "echo '-- step 4: talk to it. greetings, questions, and out-of-domain prompts'"
Enter
Sleep 2s
Type "./target/release/llm --model models/watercycle-latest.bin --e2e 'User: hi!'"
Enter
Sleep 3s
Type "./target/release/llm --model models/watercycle-latest.bin --e2e 'User: Where does rainwater collect after rainwater flows downhill?'"
Enter
Sleep 3s
Type "./target/release/llm --model models/watercycle-latest.bin --e2e 'User: What is gravity?'"
Enter
Sleep 3s
Type "./target/release/llm --model models/watercycle-latest.bin --e2e 'User: How do mountains form?'"
Enter
Sleep 4s
# 4. the conversation: greeting, exact answer, and two honest hedges

Type "echo '-- step 5: the property harness -- 50 seeded draws, P2-P5 at 1.00'"
Enter
Sleep 2s
Type "cargo test --test output_properties_test -- --nocapture 2>&1 | grep -A6 'property pass table'"
Enter
Sleep 18s
# 5. the property lens: determinism, termination, format, budget, non-degeneracy

Type "echo '-- step 6: the laptop lane -- 14M params on TinyStories, its own score formula'"
Enter
Sleep 2s
Type "./target/release/llm --tiny --eval --model models/tinystories/ts-13m-s42.bin | python3 -m json.tool | head -30"
Enter
Sleep 10s
# 6. the second lane: held-out CE percentiles, coverage, collapse gate

Type "echo 'seeded, reproducible, measured -- re-run any number above on your own machine'"
Enter
Sleep 3s
