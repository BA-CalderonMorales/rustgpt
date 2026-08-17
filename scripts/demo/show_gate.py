#!/usr/bin/env python3
"""Read the tiny-eval JSON from stdin and print the 0.0.7 headline numbers.

Used by the demo (scripts/demo/tui.tape, scripts/demo/demo_session.sh) to
show the collapse-gate verdict of a decode stack without burying it in the
full eval JSON.
"""

import json
import sys

data = json.loads(sys.stdin.read().splitlines()[-1])
fluency = data["fluency"]
collapse = data["eval"]["collapse"]
print("decode stack: T={} top-p={} presence={} repetition={}".format(
    data["temperature"], data["top_p"], data["presence"], data["repetition"]))
print("gate repetition rate:", collapse["repetition_rate"], "(greedy pin was 1.0)")
print("repetition-free rate:", fluency["repetition_free_rate"],
      "| distinct-1:", round(fluency["distinct_1"], 3))
print("distinct-2:", round(fluency["distinct_2"], 3),
      "| sentences per completion:", fluency["completion_sentences"])
