use llm::LLM;

/// Held-out slice carved by scripts/slice_tinystories.py (split seed
/// 20260816); never touches any training slice.
const TINY_HELDOUT: &str = "models/tinystories/heldout.jsonl";
/// Fixed greedy sample length for the generation-collapse gate.
const TINY_SAMPLE_LEN: usize = 96;
const COLLAPSE_THRESHOLD: f32 = 0.5;

/// Evaluate the tiny lane's score formula: per-item CE on the held-out
/// slice, its p10/p50/p90 percentiles, vocab coverage, and a
/// generation-collapse gate over a fixed-length greedy sample.
pub(crate) fn tiny_eval(llm: &mut LLM) -> serde_json::Value {
    let stories = llm::load_jsonl(TINY_HELDOUT);
    let mut per_item_ce = Vec::with_capacity(stories.len());
    let mut in_vocab = 0usize;
    let mut raw_total = 0usize;
    for story in &stories {
        per_item_ce.push(llm.sequence_loss(story));
        in_vocab += llm.tokenize(story).len();
        raw_total += llm.raw_token_count(story);
    }

    let mut sorted = per_item_ce.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let percentile = |q: usize| sorted[(sorted.len() * q / 100).min(sorted.len() - 1)];

    let (repetition_rate, sample_len) = collapse_gate(llm);
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
fn collapse_gate(llm: &mut LLM) -> (f32, usize) {
    let generated = llm.predict_cached("Once upon a time,");
    let tokens = llm.tokenize(&generated);
    let sample = &tokens[..tokens.len().min(TINY_SAMPLE_LEN)];
    let repeats = sample
        .windows(2)
        .filter(|window| window[0] == window[1])
        .count();
    let rate = repeats as f32 / sample.len().saturating_sub(1).max(1) as f32;
    (rate, sample.len())
}

/// `--tiny --eval` prints exactly one JSON object carrying the score
/// formula; the lane never claims quality without it.
pub(crate) fn run_tiny_eval(llm: &mut LLM) {
    println!(
        "{}",
        serde_json::json!({
            "status": "ok",
            "seed": llm::seed(),
            "total_parameters": llm.total_parameters(),
            "eval": tiny_eval(llm),
        })
    );
}
