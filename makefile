# rustgpt: quality-of-life command surface for the standard lanes and gates.
# Every recipe runs the exact commands documented in
# docs/running-and-development.md and pinned by the Verify gates in AGENTS.md.
# Run from the repository root.
#
# OneDrive-backed checkout? Export CARGO_TARGET_DIR to an ext4 directory
# first (e.g. ~/projects/rustgpt-target) -- measured 2.4x faster builds;
# see the OneDrive note in docs/running-and-development.md.

# The verification gates, in order (the e2e contract probe is its own target).
.PHONY: fmt clippy test verify

fmt:
	cargo fmt --check

clippy:
	cargo clippy --workspace --all-features --all-targets -- -D warnings

test:
	cargo test --all-targets

verify: fmt clippy test

# Release build: the measured lanes always run the release binary.
.PHONY: build

build:
	cargo build --release

# Contract probe only: never trains, never loads weights, never evidence.
.PHONY: e2e

e2e:
	cargo run -- --e2e "hello world"

# Micro-lane oracle: must stay 4/4/1.0 at seed 42 (fresh init).
.PHONY: eval

eval:
	cargo run --release -- --eval --seed 42

# Checkpoint round-trip: the eval recipe against a saved artifact
# (override the path with MODEL=<path>).
.PHONY: eval-model

MODEL ?= models/mine.bin

eval-model:
	cargo run --release -- --model $(MODEL) --eval --seed 42

# Tiny-lane eval: held-out CE percentiles, coverage, collapse gate.
.PHONY: tiny-eval

tiny-eval:
	cargo run --release -- --tiny --eval --model models/tinystories/ts-13m-s42.bin

# Tiny-lane training (override with FILE=<jsonl> EPOCHS=<n> MODEL=<path>;
# the default corpus is the 40k TinyStories lane, 1 epoch).
.PHONY: tiny-train

FILE ?= models/tinystories/train.jsonl
EPOCHS ?= 1
TINY_MODEL ?= models/tinystories/ts.bin

tiny-train:
	cargo run --release -- --tiny --train $(FILE) --epochs $(EPOCHS) --seed 42 --model $(TINY_MODEL)

# Re-record the demo gif: vhs renders scripts/demo/tui.tape to
# docs/demo-tui.gif (a behavior-affecting change demands a re-record).
.PHONY: demo

demo:
	vhs scripts/demo/tui.tape
