#!/usr/bin/env bash
# Resolve a model choice to a runnable engine and start it.
#
#   scripts/run_model.sh <model>
#
# <model> is a catalog id (ts-13m-s42, watercycle-latest, ...), a
# checkpoint path (models/x.bin), or an external name (qwen). Catalog ids
# and paths run the release binary, which owns the authoritative
# "not found" story; external names run a local GGUF engine (ollama or
# llama.cpp) when one is installed, and explain how to get one otherwise.

set -euo pipefail
cd "$(dirname "$0")/.."

model="${1:?usage: scripts/run_model.sh <model>}"

# Probe the release binary: it resolves catalog ids and checkpoint paths.
# A resolved model exits 0 on 'exit'; a missing one errors before stdin.
if printf 'exit\n' | ./target/release/llm --model "$model" >/dev/null 2>&1; then
    exec ./target/release/llm --model "$model"
fi

# External engine fallback for names the catalog does not know.
case "$model" in
    qwen*)
        if command -v ollama >/dev/null 2>&1 && ollama list 2>/dev/null | grep -qi qwen; then
            name="$(ollama list | awk 'NR > 1 && tolower($1) ~ /qwen/ {print $1; exit}')"
            exec ollama run "$name"
        fi
        if command -v llama-cli >/dev/null 2>&1; then
            echo "llama.cpp found; pulling the Qwen3.8-27B GGUF on first run (large download)."
            exec llama-cli -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
        fi
        echo "error: '$model' needs a local engine. Install ollama and 'ollama pull qwen3.8'," >&2
        echo "       or install llama.cpp (the Qwen3.8-27B GGUF is the reference open-weights model)." >&2
        exit 2
        ;;
esac

echo "error: '$model' is not a catalog id or checkpoint path, and no external engine matches." >&2
exit 2
