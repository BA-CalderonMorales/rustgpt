use llm::{DecodeKnobs, LLM, Xorshift};

use super::{Trace, chat_loop, done_stdout, note_stdout, render_answer, step_stdout, tiny_eval};

/// The shared starter every decode comparison uses.
const STARTER: &str = "Once upon a time,";

/// The tour's back half: score against held-out stories, then show the
/// same starter under two decode recipes and hand the keyboard over.
/// A pure move from demo.rs -- no behavior change.
pub(crate) fn score_and_use(model: &mut LLM) {
    // STEP 6 EVALUATION: held-out data, honest scores.
    step_stdout(6, "Scoring against held-out stories it NEVER saw");
    let eval = tiny_eval(model, 1.0, 0.0, 1.0, 0.0);
    done_stdout();
    let percentiles = &eval["ce_percentiles"];
    note_stdout(&format!(
        "Cross-entropy p50 = {:.2} (median surprise; TEACHER-FORCED: true previous words shown).",
        percentiles["p50"].as_f64().unwrap_or(0.0)
    ));
    let gate = &eval["collapse"];
    note_stdout(&format!(
        "Collapse gate: free-running repetition rate {:.2} ({}) -- generation is measured, not assumed.",
        gate["repetition_rate"].as_f64().unwrap_or(0.0),
        if gate["collapsed"].as_bool().unwrap_or(false) {
            "collapsed"
        } else {
            "not collapsed"
        }
    ));

    // STEP 7 USE: same starter, two decode recipes, side by side.
    step_stdout(7, "Using your model");
    let mut rng = Xorshift::new(llm::seed());
    let greedy = model.predict_cached(STARTER);
    let tuned = model.generate(STARTER, 0.7, 0.80, 1.5, 1.1, &mut rng);
    println!("   Greedy decode (always the single most likely word):");
    println!("     {}", render_answer(&greedy));
    println!("   Tuned sampling (temperature 0.7, top-p 0.80, presence 1.5, repetition 1.1):");
    println!("     {}", render_answer(&tuned));
    note_stdout(
        "Same model, same starter: the knobs decide whether repetition wins. Now you drive:",
    );

    // Hand the keyboard over: the tour ends inside the chat surface.
    let mut knobs = DecodeKnobs::greedy();
    chat_loop(model, &Trace::new(false), &mut knobs);
}
