use llm::{FluencyScore, LLM};

/// Held-out slice carved by scripts/slice_tinystories.py (split seed
/// 20260816); never touches any training slice.
const TINY_HELDOUT: &str = "models/tinystories/heldout.jsonl";
/// Fixed greedy sample length for the generation-collapse gate.
const TINY_SAMPLE_LEN: usize = 96;
const COLLAPSE_THRESHOLD: f32 = 0.5;
/// Fixed starter for the generation-collapse gate and the fluency probe.
const TINY_STARTER: &str = "Once upon a time,";

/// The held-out slice's stories: the fixed probe stream for the collapse
/// profile, so training-mode instruments sample exactly what `--tiny
/// --eval` scores.
pub(crate) fn tiny_heldout_stories() -> Vec<String> {
    llm::load_jsonl(TINY_HELDOUT)
}

/// Evaluate the tiny lane's score formula: per-item CE on the held-out
/// slice, its p10/p50/p90 percentiles, vocab coverage, and a
/// generation-collapse gate over a fixed-length greedy sample.
pub(crate) fn tiny_eval(llm: &mut LLM, temperature: f32) -> serde_json::Value {
    // Per-item teacher-forced CE and vocabulary coverage over the held-out
    // slice (never a training slice).
    let stories = tiny_heldout_stories();
    let mut per_item_ce = Vec::with_capacity(stories.len());
    let mut in_vocab = 0usize;
    let mut raw_total = 0usize;
    for story in &stories {
        per_item_ce.push(llm.sequence_loss(story));
        in_vocab += llm.tokenize(story).len();
        raw_total += llm.raw_token_count(story);
    }

    // Nearest-rank CE percentiles.
    let mut sorted = per_item_ce.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let percentile = |q: usize| sorted[(sorted.len() * q / 100).min(sorted.len() - 1)];

    // The generation-collapse gate over a fixed-length sample decoded at
    // the given temperature (greedy argmax is temperature-invariant, so
    // the gate number is a pin at every T).
    let (repetition_rate, sample_len) = collapse_gate(llm, temperature);
    serde_json::json!({
        "source": TINY_HELDOUT,
        "items": stories.len(),
        "coverage": in_vocab as f32 / raw_total.max(1) as f32,
        "per_item_ce": per_item_ce,
        "ce_percentiles": {
            "p10": percentile(10),
            "p50": percentile(50),
            "p90": percentile(90),
        },
        "collapse": {
            "sample_len": sample_len,
            "repetition_rate": repetition_rate,
            "collapsed": repetition_rate > COLLAPSE_THRESHOLD,
        },
    })
}

/// Greedy sample from a fixed starter; rate is the fraction of adjacent
/// token pairs that are identical (a collapsed model approaches 1.0).
/// The decode-quality yardstick over `samples` seeded generations from the
/// fixed starter, scored by the llm::fluency_score instrument. The sampler
/// is the lane's decoder at the run's temperature; greedy today, the
/// probability-weighted temperature sampler from W4 when that lands.
pub(crate) fn fluency_probe(llm: &mut LLM, temperature: f32, samples: usize) -> FluencyScore {
    let generated: Vec<String> = (0..samples)
        .map(|_| llm.predict_scaled(TINY_STARTER, temperature))
        .collect();
    llm.fluency_score(&generated)
}

fn collapse_gate(llm: &mut LLM, temperature: f32) -> (f32, usize) {
    // Generate the fixed-length sample.
    let generated = llm.predict_scaled(TINY_STARTER, temperature);
    let tokens = llm.tokenize(&generated);
    let sample = &tokens[..tokens.len().min(TINY_SAMPLE_LEN)];

    // Rate identical adjacent pairs over the sampled length.
    let repeats = sample
        .windows(2)
        .filter(|window| window[0] == window[1])
        .count();
    let rate = repeats as f32 / sample.len().saturating_sub(1).max(1) as f32;
    (rate, sample.len())
}

/// `--tiny --eval` prints exactly one JSON object carrying the score
/// formula; the lane never claims quality without it. `--fluency <n>` adds
/// the decode-quality yardstick block (additive key, greedy leg pinned).
pub(crate) fn run_tiny_eval(llm: &mut LLM, temperature: f32, fluency: Option<usize>) {
    // Exactly one JSON object carrying the lane's score formula, plus the
    // fluency yardstick when the caller asked for it.
    let mut output = serde_json::json!({
        "status": "ok",
        "seed": llm::seed(),
        "total_parameters": llm.total_parameters(),
        "temperature": temperature,
        "eval": tiny_eval(llm, temperature),
    });
    if let Some(samples) = fluency {
        let score = fluency_probe(llm, temperature, samples);
        output["fluency"] = serde_json::json!({
            "samples": samples,
            "distinct_1": score.distinct_1,
            "distinct_2": score.distinct_2,
            "repetition_free_rate": score.repetition_free_rate,
            "completion_sentences": score.completion_sentences,
            "mean_completion_len": score.mean_completion_len,
        });
    }
    println!("{output}");
}
