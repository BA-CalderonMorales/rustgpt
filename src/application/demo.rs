use llm::{DecodeKnobs, LLM, Vocab, Xorshift};

use super::{
    Trace, build_tiny_llm, chat_loop, note_stdout, render_answer, stage_stdout, tiny_eval,
};

/// The fast demo slice: 300 TinyStories stories, the tour's data stage.
const DEMO_SLICE: &str = "models/tinystories/demo.jsonl";
/// Short training leg: enough epochs for the loss bar to visibly fall.
const DEMO_EPOCHS: usize = 3;
const DEMO_LR: f32 = 0.0005;
/// The shared starter every decode comparison uses.
const STARTER: &str = "Once upon a time,";

/// `--demo`: the novice pipeline tour, data to use surface, on the fast
/// slice. Seeded and reproducible; stdout carries the tour, stderr the
/// training bar; nothing is saved (the tour never touches artifacts).
pub(crate) fn run_demo() {
    // STAGE 1 DATA: the examples the model learns from.
    stage_stdout(1);
    if !std::path::Path::new(DEMO_SLICE).exists() {
        eprintln!(
            "error: {DEMO_SLICE} not found; rebuild it with: python scripts/demo/make_demo_slice.py"
        );
        std::process::exit(1);
    }
    let texts = llm::load_jsonl(DEMO_SLICE);
    note_stdout(&format!(
        "{DEMO_SLICE} holds {} tiny stories. Each line is one example. The first one:",
        texts.len()
    ));
    println!("    {}", first_words(&texts[0], 30));

    // STAGE 2 VOCABULARY: every whole word the model will ever see.
    stage_stdout(2);
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(&texts, &mut vocab_set);
    note_stdout(&format!(
        "The model only sees whole words from this list: {} words, plus <unk> (a stranger) and </s> (the end). Nothing outside the list exists for it -- that is why small data means a small world.",
        vocab_set.len()
    ));

    // STAGE 3 MODEL: the dials, arranged.
    stage_stdout(3);
    let mut model = build_tiny_llm(&texts);
    note_stdout(&format!(
        "This model is {} adjustable dials (parameters): a word lookup (embeddings), {} thinking blocks (transformers), and a word guesser (output projection). Training turns the dials; using the model reads them.",
        model.total_parameters(),
        block_count(&model),
    ));

    // STAGE 4 TRAINING: guess, measure surprise, adjust, repeat.
    stage_stdout(4);
    note_stdout(&format!(
        "Training: {DEMO_EPOCHS} epochs (an epoch = one full read of every story) at learning rate {DEMO_LR}. Loss = average surprise. Lower = wrong less often."
    ));
    let examples: Vec<&str> = texts.iter().map(String::as_str).collect();
    let losses = model.train_with_progress(examples, DEMO_EPOCHS, DEMO_LR, true);
    note_stdout(&format!(
        "Loss fell {:.2} -> {:.2}: the guesses got closer to the real words.",
        losses.first().copied().unwrap_or(0.0),
        losses.last().copied().unwrap_or(0.0),
    ));

    // STAGE 5 EVALUATION: held-out data, the score formula, the gate.
    stage_stdout(5);
    let eval = tiny_eval(&mut model, 1.0, 0.0, 1.0, 0.0);
    let percentiles = &eval["ce_percentiles"];
    let gate = &eval["collapse"];
    note_stdout(&format!(
        "Held-out stories the model NEVER saw. Cross-entropy p50 (the middle score) = {:.2} (still often surprised -- honest small-model math). This is a TEACHER-FORCED score: the model always sees the true previous words and only guesses the next one.",
        percentiles["p50"].as_f64().unwrap_or(0.0)
    ));
    note_stdout(&format!(
        "Collapse gate: repetition rate {:.2} ({}) -- the teacher-forced score can look fine while free-running generation (the model picking every word itself) loops. Measuring both is the rule here.",
        gate["repetition_rate"].as_f64().unwrap_or(0.0),
        if gate["collapsed"].as_bool().unwrap_or(false) {
            "collapsed: it repeats itself"
        } else {
            "not collapsed"
        }
    ));

    // STAGE 6 USE: the same starter, two decode recipes, side by side.
    stage_stdout(6);
    let mut rng = Xorshift::new(llm::seed());
    let greedy = model.predict_cached(STARTER);
    let tuned = model.generate(STARTER, 0.7, 0.80, 1.5, 1.1, &mut rng);
    println!("\n  Greedy decode (always the single most likely word):");
    println!("    {}", render_answer(&greedy));
    println!("  Tuned sampling (temperature 0.7, top-p 0.80, presence 1.5, repetition 1.1):");
    println!("    {}", render_answer(&tuned));
    note_stdout(
        "Same model, same starter: the knobs decide whether repetition wins. Now you drive:",
    );

    // Hand the keyboard over: the tour ends inside the chat surface.
    let mut knobs = DecodeKnobs::greedy();
    chat_loop(&mut model, &Trace::new(false), &mut knobs);
}

/// The opening words of a story, for the data-stage excerpt.
fn first_words(text: &str, words: usize) -> String {
    text.split_whitespace()
        .take(words)
        .collect::<Vec<_>>()
        .join(" ")
}

/// The number of transformer blocks between embeddings and projection.
fn block_count(llm: &LLM) -> usize {
    llm.network_description()
        .split(", ")
        .filter(|layer| layer.starts_with("Transformer"))
        .count()
}
