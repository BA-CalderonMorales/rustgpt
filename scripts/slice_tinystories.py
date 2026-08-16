#!/usr/bin/env python3
"""Slice TinyStories-train.txt into the tl project's train.jsonl.

Reproduces the v0.0.4 dataset: first 40,000 stories, 20-120 words each,
written as one {"text": ...} JSON object per line. Additionally carves a
fixed 256-story held-out slice (split seed 20260816, python random module)
from the qualifying stories that follow the train boundary, excluding any
train text verbatim. Run from the repo root.
"""

import json
import random

SRC = "models/tinystories/TinyStories-train.txt"
DEST = "models/tinystories/train.jsonl"
HELDOUT_DEST = "models/tinystories/heldout.jsonl"
LIMIT = 40000
MIN_WORDS = 20
MAX_WORDS = 120
HELDOUT_SIZE = 256
SPLIT_SEED = 20260816


def slice_story(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None
    words = text.split()
    if len(words) < MIN_WORDS:
        return None
    return " ".join(words[:MAX_WORDS])


def main() -> None:
    stories = 0
    tokens = 0
    train_texts: set[str] = set()
    candidates: list[str] = []
    with open(SRC, encoding="utf-8", errors="replace") as source, open(
        DEST, "w", encoding="utf-8"
    ) as dest:
        for line in source:
            if stories >= LIMIT and len(candidates) >= HELDOUT_SIZE * 16:
                break
            story = slice_story(line)
            if story is None:
                continue
            if stories < LIMIT:
                dest.write(json.dumps({"text": story}) + "\n")
                train_texts.add(story)
                stories += 1
                tokens += len(story.split())
            else:
                if story not in train_texts:
                    candidates.append(story)
    print(f"train stories: {stories}, tokens: {tokens}")

    rng = random.Random(SPLIT_SEED)
    rng.shuffle(candidates)
    heldout = candidates[:HELDOUT_SIZE]
    with open(HELDOUT_DEST, "w", encoding="utf-8") as dest:
        for story in heldout:
            dest.write(json.dumps({"text": story}) + "\n")
    print(
        f"heldout stories: {len(heldout)} (split seed {SPLIT_SEED}), "
        f"candidates considered: {len(candidates)}"
    )


if __name__ == "__main__":
    main()
