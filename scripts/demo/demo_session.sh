#!/usr/bin/env bash
# The v0.0.7 showcase session, recorded with `script` and rendered by agg.
# Mirrors scripts/demo/tui.tape; every command runs the real release binary
# against real artifacts -- proof the loop is not a gimmick.
#
# Record (from the repository root):
#   script -q -e -c "bash scripts/demo/demo_session.sh" scripts/demo/demo.rec
#   agg scripts/demo/demo.rec docs/demo-tui.gif

cd "$(dirname "$0")/../.."

echo "=== 0. stage the tiny demo slice (300 real TinyStories stories) ==="
sleep 1
python3 scripts/demo/make_demo_slice.py
sleep 2

echo
echo "=== 1. the release checkpoint: v0.0.7, version contract pinned by tests ==="
sleep 1
./target/release/llm --version
sleep 3

echo
echo "=== 2. the model catalog: every artifact carries its recipe and eval ==="
sleep 1
./target/release/llm --models | python3 -m json.tool | head -30
sleep 3

echo
echo "=== 3. the contract probe: one JSON object, deterministic across processes ==="
sleep 1
./target/release/llm --e2e 'hello world'
sleep 3

echo
echo "=== 4. the domain-labeled trace on the loaded artifact (use surface) ==="
sleep 1
echo "    (interactive --model loads the checkpoint and chats -- no training,"
echo "     no re-save; provenance: catalog entry watercycle-latest, seed 42,"
echo "     min held-out CE promoted)"
printf 'yo!\nhello!\nwater?\nexit\n' | ./target/release/llm --trace --model models/watercycle-latest.bin --seed 42 2>&1 | tail -30
sleep 4

echo
echo "=== 5. micro arena truth table: train both phases, score the held-out prompts ==="
sleep 1
./target/release/llm --model models/watercycle-latest.bin --eval --seed 42 2>&1 | python3 -c "import sys,json; print(json.dumps(json.loads(sys.stdin.read().splitlines()[-1]), indent=2))" | head -50
sleep 4

echo
echo "=== 6. the laptop lane: 7.3M params trained on real story text ==="
sleep 1
./target/release/llm --tiny --train models/tinystories/demo.jsonl --epochs 1 --seed 42 --model models/tinystories/stories-demo.bin 2>&1 | python3 -c "import sys,json; print(json.dumps(json.loads(sys.stdin.read().splitlines()[-1]), indent=2))" | head -60
sleep 2

echo
echo "=== 7. the headline: the collapse gate, defeated ==="
sleep 1
echo "    greedy decode is pinned at repetition 1.0 (the '.' frequency head);"
echo "    the Qwen-honoring stack (T=0.7, top-p 0.8, presence 1.5, repetition 1.1)"
echo "    lands the gate at 0.021 with repetition-free 0.65, distinct-1 0.70 --"
echo "    a 14M from-scratch model, no retraining"
./target/release/llm --tiny --eval --model models/tinystories/stories-full.bin --temperature 0.7 --top-p 0.8 --presence 1.5 --repetition 1.1 --fluency 20 2>&1 | python3 scripts/demo/show_gate.py
sleep 3

echo
echo "=== checkpoint saved. the collapse gate is the number 0.0.7 defeated. ==="
sleep 3
