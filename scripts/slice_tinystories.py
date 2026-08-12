#!/usr/bin/env python3
"""Slice TinyStories-train.txt into the tl project's train.jsonl.

Reproduces the v0.0.4 dataset: first 40,000 stories, 20-120 words each,
written as one {"text": ...} JSON object per line. Run from the repo root.
"""

import json

SRC = "models/tinystories/TinyStories-train.txt"
DEST = "models/tinystories/train.jsonl"
LIMIT = 40000
MIN_WORDS = 20
MAX_WORDS = 120


def main() -> None:
    stories = 0
    tokens = 0
    with open(SRC, encoding="utf-8", errors="replace") as source, open(
        DEST, "w", encoding="utf-8"
    ) as dest:
        for line in source:
            if stories >= LIMIT:
                break
            text = line.strip()
            if not text:
                continue
            words = text.split()
            if len(words) < MIN_WORDS:
                continue
            words = words[:MAX_WORDS]
            dest.write(json.dumps({"text": " ".join(words)}) + "\n")
            stories += 1
            tokens += len(words)
    print(f"stories: {stories}, tokens: {tokens}")


if __name__ == "__main__":
    main()
