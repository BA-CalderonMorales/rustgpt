#!/usr/bin/env python3
"""Build the tiny demo corpus for the v0.0.4 showcase (first 300 stories)."""

import json

SRC = "models/tinystories/train.jsonl"
DEST = "models/tinystories/demo.jsonl"
LIMIT = 300


def main() -> None:
    count = 0
    with open(SRC, encoding="utf-8") as source, open(DEST, "w", encoding="utf-8") as dest:
        for line in source:
            if count >= LIMIT:
                break
            dest.write(line)
            count += 1
    print(f"demo slices {count} stories -> {DEST}")


if __name__ == "__main__":
    main()
