# Free Compute: the Verified Verdict

Status: measured 2026-08-16. Question: can free cloud compute (Colab,
Kaggle) train this stack faster or better, at zero cost?

## Verified quotas (webfetch, cited)

### Google Colab free tier

- Free of charge; GPU/TPU runtimes exist but access to "expensive resources
  like GPUs is heavily restricted" in the free tier; limits are dynamic and
  unadvertised by design ("Colab does not publish these limits, in part
  because they can vary over time").
- Notebooks run at most ~12 hours; runtimes time out when idle; VMs are
  deleted when idle for a while.
- GPU types vary over time; nothing is guaranteed.
- Source: https://research.google.com/colaboratory/faq.html (Resource
  Limits, How long can notebooks run, What types of GPU/TPU).

### Kaggle notebooks

- Free tier: up to ~30 GPU-hours per week (quota resets weekly; "30 hours
  or sometimes higher depending on demand and resources"). Quota counts
  wall-clock notebook time with GPU enabled, not utilization.
- GPU hardware is auto-assigned (NVIDIA T4 or P100 class); you cannot
  choose. CPU sessions are ~2-4 vCPU.
- Session length: ~9 hours GPU / ~12 hours CPU, then the session ends.
- Internet is enabled by default on modern notebook sessions (verify at
  session start; it was a toggle in the past).
- Sources: https://www.kaggle.com/docs/efficient-gpu-usage,
  https://www.kaggle.com/docs/notebooks (quota section; the page is
  JS-rendered, quota numbers cross-checked against the docs search index).

### rustup + cargo on both

- Both platforms run Ubuntu VMs with curl/bash; `curl https://sh.rustup.rs
  | sh` and `cargo build --release` work on plain-CPU notebooks. Plain-CPU
  notebooks exist on both platforms (Colab: Hardware accelerator = None;
  Kaggle: GPU = None).

## The honest math

This stack has **no CUDA path**: pure Rust + ndarray, no BLAS, no GPU
kernels. Free GPU hours are therefore **useless to rustgpt today** —
nothing in the pipeline can address a GPU.

Free cloud CPUs (~2-4 vCPU) are on par with or below the 14-thread laptop
this repo develops on, and ndarray without BLAS is single-threaded anyway:
expect roughly the same token throughput as the laptop lane, not more.

**Verdict: free compute buys offload and reproducibility — overnight runs
that do not tie up the owner's machine, and a re-runnable notebook with a
pinned commit and seed — not speed.** A claim made on the cloud is the
same evidence unit as a claim made locally: checkpoint + eval JSON + seed
+ recipe.

The levers that change this verdict, in order:

1. BLAS (optional `blas` feature, W5): a matmul substrate makes the CPU
   path several times faster everywhere, including the cloud.
2. A racecar ADR or explicit CUDA kernels (a big ADR, not this release):
   only then do free GPU hours become addressable compute.

Do NOT promise a GPU speedup anywhere: there is no code path that uses one.

## Using the cloud lane

`scripts/cloud-train.sh` runs a full tiny-lane train on either platform's
shell/notebook cell:

```bash
curl -sSL https://raw.githubusercontent.com/BA-CalderonMorales/rustgpt/main/scripts/cloud-train.sh | bash -s -- \
    --corpus https://example.com/train.jsonl --epochs 1 --seed 42 --out ts-cloud.bin
```

- `--corpus <url>`: any .jsonl under a free license (Kaggle/HuggingFace,
  per docs/dataset-curation.md). Required.
- `--epochs <n>`, `--seed <n>` (default 42), `--out <path>` (default
  ts-cloud.bin).
- The script installs rustup if missing, clones the repo, builds release,
  trains, and writes the evidence pair: the checkpoint plus `eval.json`
  (one JSON object with trajectory, samples, and the lane's eval block).
- Download both before the session ends; cloud VMs are ephemeral.
