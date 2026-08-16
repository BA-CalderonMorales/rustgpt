# rustgpt 0.0.5 showcase: the three lenses, live on the release binary.
# Every command runs the real binary against real artifacts: version
# contract, OOV fallback contract, micro-arena truth table (fresh seed-42
# run), the seeded output-property suite, and the tiny lane's score
# formula. From the repository root:
#   vhs scripts/demo/demo.tape
# The recording is evidence: it re-proves what the current main really does.

Output docs/demo-tui.gif
Set FontSize 16
Set Width 1100
Set Height 700
Set Padding 24
Set Shell bash
Set TypingSpeed 20ms
Set Framerate 20
Set PlaybackSpeed 2

Type "./target/release/llm --version"
Enter
Sleep 3s
# 1. the version contract: 0.0.5, derived from Cargo.toml, pinned by tests

Type "./target/release/llm --model models/watercycle-latest.bin --e2e 'hello'"
Enter
Sleep 4s
# 2. the OOV contract: an unknown word gets an explicit answer, never a
#    silent empty output with status:ok

Type "./target/release/llm --eval --seed 42 2>&1 | python3 -m json.tool | head -70"
Enter
Sleep 12s
# 3. the micro arena truth table: train both phases from seed 42, score the
#    four held-out prompts, ship the trajectory -- 380K params, by hand

Type "cargo test --test output_properties_test -- --nocapture 2>&1 | grep -A6 'property pass table'"
Enter
Sleep 18s
# 4. the property lens: seeded output-property suite against the artifact;
#    improvement is the pass-table delta, same seed, same draws

Type "./target/release/llm --tiny --eval --model models/tinystories/ts-13m-s42.bin | python3 -m json.tool | head -45"
Enter
Sleep 10s
# 5. the laptop lane formula: held-out CE percentiles, coverage, and the
#    generation-collapse gate

Sleep 3s
# 6. freeze: formula, properties, and what a human sees -- all three lenses
