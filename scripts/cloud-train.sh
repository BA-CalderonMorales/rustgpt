#!/usr/bin/env bash
# cloud-train.sh — offload a rustgpt tiny-lane run to a free cloud shell.
#
# Colab (cell) or Kaggle (notebook cell): Ubuntu shell, internet enabled.
# Paste this file's contents into a shell cell, or run:
#     bash cloud-train.sh --corpus <url> --epochs 1 --seed 42 --out ts-cloud.bin
#
# What it does: rustup install -> clone -> release build -> --tiny --train
# on the given corpus -> save the checkpoint AND its eval JSON. The eval
# JSON is the unit of evidence next to the checkpoint.
#
# Honest expectations (docs/free-compute.md): these platforms give FREE
# CPU only, or free GPUs this stack cannot use (pure Rust + ndarray, no
# BLAS, no CUDA path). This script buys offload and reproducibility, not
# speed. Cloud CPUs are ~2-4 vCPU, on par with or below a 14-thread
# laptop; expect single-threaded ndarray throughput either way.
set -euo pipefail

SEED=42
EPOCHS=1
CORPUS=""
OUT="ts-cloud.bin"

while [ $# -gt 0 ]; do
    case "$1" in
        --corpus) CORPUS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed)   SEED="$2";   shift 2 ;;
        --out)    OUT="$2";    shift 2 ;;
        *) echo "usage: cloud-train.sh --corpus <url> [--epochs 1] [--seed 42] [--out ts-cloud.bin]" >&2; exit 2 ;;
    esac
done

[ -n "$CORPUS" ] || { echo "error: --corpus <url> is required (any .jsonl, free license)" >&2; exit 2; }

# Rust toolchain. Colab/Kaggle images change; this is idempotent either way.
if ! command -v rustc >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    . "$HOME/.cargo/env"
fi

# The corpus, fetched fresh each run.
curl -sSL "$CORPUS" -o train.jsonl
wc -l train.jsonl

# Clone and build. The binary is the release artifact; the git tree stays
# untouched so the run is attributable to a pinned commit.
if [ ! -d rustgpt ]; then
    git clone --depth 1 https://github.com/BA-CalderonMorales/rustgpt.git
fi
cd rustgpt
git pull --ff-only || true
cargo build --release

# The tiny lane trains from the corpus and prints one JSON object carrying
# trajectory + samples + the eval block (held-out CE, collapse gate).
./target/release/llm --tiny --train "../train.jsonl" --epochs "$EPOCHS" \
    --seed "$SEED" --model "$OUT" 2>train.log | tee eval.json

# Evidence pair: checkpoint + eval JSON. Download both before the session
# ends (cloud VMs are ephemeral).
echo "done: $OUT and eval.json; save both, they are the evidence pair."
