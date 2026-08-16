use llm::{AnswerScore, LLM, Xorshift};

const K_VALUES: [usize; 3] = [3, 5, 8];
const N_VALUES: [usize; 2] = [8, 16];
/// Probe rng seed offset: the sampler never shares a draw chain with the
/// property suite's prompt generator.
const PROBE_SEED_OFFSET: u64 = 0xE3;

fn heldout() -> Vec<(String, String)> {
    let text = std::fs::read_to_string("data/heldout.json")
        .expect("data/heldout.json must exist for --probe");
    serde_json::from_str(&text).expect("data/heldout.json must list prompt/reference pairs")
}

fn greedy_summary(llm: &mut LLM) -> serde_json::Value {
    let mut exact_total = 0usize;
    let mut prefix_total = 0usize;
    let mut accuracies = Vec::new();
    for (prompt, reference) in heldout() {
        let score = llm.answer_score(&format!("User: {prompt}"), &reference);
        exact_total += usize::from(score.exact);
        prefix_total += usize::from(score.prefix);
        accuracies.push(score.accuracy);
    }
    serde_json::json!({
        "exact_matches": exact_total,
        "prefix_matches": prefix_total,
        "mean_accuracy": accuracies.iter().sum::<f32>() / accuracies.len().max(1) as f32,
    })
}

/// One (k, N) cell: N seeded top-k candidates per held-out item, the
/// best-scoring candidate wins per position (arXiv 2408.03314). Item 3's
/// loop-free candidate count is the degeneracy recovery report.
fn best_of_n_cell(llm: &mut LLM, k: usize, n: usize, rng: &mut Xorshift) -> serde_json::Value {
    let mut exact_total = 0usize;
    let mut prefix_total = 0usize;
    let mut accuracies = Vec::new();
    let mut item3_loop_free = 0usize;
    for (index, (prompt, reference)) in heldout().iter().enumerate() {
        let prompt = format!("User: {prompt}");
        let mut best: Option<AnswerScore> = None;
        for _ in 0..n {
            let candidate = llm.predict_sampled(&prompt, k, rng);
            if index == 2 && !llm.is_degenerate(&candidate) {
                item3_loop_free += 1;
            }
            let score = llm.score_generated(&candidate, reference);
            if best.as_ref().is_none_or(|b| score.accuracy > b.accuracy) {
                best = Some(score);
            }
        }
        let best = best.expect("n candidates must be > 0");
        exact_total += usize::from(best.exact);
        prefix_total += usize::from(best.prefix);
        accuracies.push(best.accuracy);
    }
    serde_json::json!({
        "exact_matches": exact_total,
        "prefix_matches": prefix_total,
        "mean_accuracy": accuracies.iter().sum::<f32>() / accuracies.len().max(1) as f32,
        "item3_loop_free_candidates": item3_loop_free,
    })
}

/// `--probe --model <checkpoint>`: the decode-time compute truth table
/// (k x N grid against greedy), exactly one JSON object on stdout.
pub(crate) fn run_probe(llm: &mut LLM) {
    let mut rng = Xorshift::new(llm::seed() ^ PROBE_SEED_OFFSET);
    let mut grid = serde_json::Map::new();
    for k in K_VALUES {
        for n in N_VALUES {
            grid.insert(
                format!("k{k}_n{n}"),
                best_of_n_cell(llm, k, n, &mut rng),
            );
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "status": "ok",
            "seed": llm::seed(),
            "greedy": greedy_summary(llm),
            "best_of_n": grid,
        })
    );
}
