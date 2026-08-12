#!/usr/bin/env bash
# The v0.0.4 showcase session, recorded with `script` and rendered by agg.
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
echo "=== 1. the release checkpoint: v0.0.4, version contract pinned by tests ==="
sleep 1
./target/release/llm --version
sleep 3

echo
echo "=== 2. the contract probe: one JSON object, deterministic across processes ==="
sleep 1
./target/release/llm --model models/watercycle-latest.bin --e2e 'hello world'
sleep 4

echo
echo "=== 3. micro arena truth table: train both phases, score the held-out prompts ==="
sleep 1
./target/release/llm --model models/watercycle-latest.bin --eval --seed 42 2>&1 | python3 -c "import sys,json; print(json.dumps(json.loads(sys.stdin.read().splitlines()[-1]), indent=2))" | head -50
sleep 4

echo
echo "=== 4. the laptop lane: 14.2M params trained on real story text ==="
sleep 1
./target/release/llm --tiny --train models/tinystories/demo.jsonl --epochs 1 --seed 42 --model models/tinystories/demo.bin 2>&1 | python3 -c "import sys,json; print(json.dumps(json.loads(sys.stdin.read().splitlines()[-1]), indent=2))" | head -50
sleep 2

echo
echo "=== checkpoint saved. the model was made on this machine, by hand. ==="
sleep 3
