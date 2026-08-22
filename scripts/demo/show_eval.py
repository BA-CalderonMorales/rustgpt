#!/usr/bin/env python3
"""Summarize a rustgpt machine-JSON object into a few lines with meaning.

Used by the demo (scripts/demo/tui.tape) so the showcase shows verdicts,
not raw JSON. Handles the two shapes the path produces: the micro arena
eval (has "summary") and the tiny-lane training run (has trajectory.loss).
Anything else falls through to a truncated dump so the demo never lies.
"""

import json
import sys

data = json.loads(sys.stdin.read().splitlines()[-1])

if "summary" in data:
    # The micro arena: train both phases, score the four held-out prompts.
    summary = data["summary"]
    print(
        "micro arena (seed {}): exact {}/4  prefix {}/4  mean accuracy {:.4f}".format(
            data["seed"],
            summary["exact_matches"],
            summary["prefix_matches"],
            summary["mean_accuracy"],
        )
    )
    heldout = data["trajectory"]["heldout_ce"]
    print("held-out CE trajectory: " + " -> ".join(f"{v:.2f}" for v in heldout))
elif "trajectory" in data and "loss" in data.get("trajectory", {}):
    # A tiny-lane training run: trajectory, recipe, and the honest gate.
    loss = data["trajectory"]["loss"]
    print(
        "trained on {} stories, {} epochs (seed {}): loss {} -> {}".format(
            data["stories"],
            data["epochs"],
            data["seed"],
            f"{loss[0]:.2f}",
            f"{loss[-1]:.2f}",
        )
    )
    training = data.get("training", {})
    if training:
        print(
            "recipe: eos={} lr {:.1e} -> {:.1e}".format(
                training.get("eos_appended"),
                training.get("lr_start"),
                training.get("lr_final"),
            )
        )
    gate = data["eval"]["collapse"]
    percentiles = data["eval"]["ce_percentiles"]
    verdict = "collapsed" if gate["collapsed"] else "not collapsed"
    print(
        "held-out CE p50: {:.2f} | collapse gate: {} ({})".format(
            percentiles["p50"], gate["repetition_rate"], verdict
        )
    )
else:
    # Unknown shape: show the head honestly instead of guessing.
    print(json.dumps(data, indent=2)[:400])
