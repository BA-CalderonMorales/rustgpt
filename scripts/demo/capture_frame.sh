#!/usr/bin/env bash
# Capture a live rustgpt frame and render it to a PNG.
# Mirrors the terminal-jarvis demo pattern; the README keeps the animated
# demo-tui.gif recorded with vhs from tui.tape, the PNG backs registry pages.
#
# Usage (from the repository root):
#   scripts/demo/capture_frame.sh tui     # boot frame -> docs/demo-tui.png
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
name=${1:?usage: capture_frame.sh <frame-name>}
bin=${RELEASE_BIN:-"$repo/target/release/llm"}

[[ -x "$bin" ]] || { echo "missing binary; build first: cargo build --release" >&2; exit 1; }

font=${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf}
raw=$(mktemp --suffix=.raw)
png="$repo/docs/demo-$name.png"

tmux -f /dev/null new-session -d -x 96 -y 16 -s llm-shot \
    "export PATH=$(dirname "$bin"):\$PATH; cd $repo; exec llm"
tmux set-option -t llm-shot status off
sleep 9
tmux capture-pane -t llm-shot -p -e > "$raw"
tmux kill-session -t llm-shot 2>/dev/null || true

cd "$repo/scripts/demo"
go run . "$raw" "$font" "$png"
rm -f "$raw"
echo "wrote $png"
