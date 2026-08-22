use std::collections::HashSet;

use llm::{LLM, Vocab};

use super::{
    build_tiny_llm, done_stdout, note_stdout, print_pretraining, score_and_use, step_stdout,
};

/// The fast demo slice: 300 TinyStories stories, the tour's data stage.
const DEMO_SLICE: &str = "models/tinystories/demo.jsonl";
/// Short training leg: enough epochs for the loss bar to visibly fall.
const DEMO_EPOCHS: usize = 3;
const DEMO_LR: f32 = 0.0005;

/// `--demo`: the pipeline tour for humans, raw text to a working model in
/// seven numbered steps -- pull the data, clean it, show the configuration
/// table, build, train (the live loss bar), then score held-out and hand
/// over the keyboard. Seeded and reproducible; nothing is saved; stdout
/// carries the tour, stderr the training bar.
pub(crate) fn run_demo() {
    // The tour's contract, up front.
    println!("Building a tiny language model, step by step: raw text in, working model out.");
    println!("Nothing is saved; every number carries its seed.");

    // STEP 1 DATA: where the examples come from.
    step_stdout(1, "Pulling the training dataset");
    if !std::path::Path::new(DEMO_SLICE).exists() {
        eprintln!(
            "error: {DEMO_SLICE} not found; rebuild it with: python scripts/demo/make_demo_slice.py"
        );
        std::process::exit(1);
    }
    let texts = llm::load_jsonl(DEMO_SLICE);
    done_stdout();
    note_stdout(&format!(
        "{DEMO_SLICE} holds {} stories; each line is one example the model reads.",
        texts.len()
    ));
    note_stdout(&format!("First one: \"{}\"", first_words(&texts[0], 24)));

    // STEP 2 CLEANING: whole-word vocabulary harvest.
    step_stdout(
        2,
        "Cleaning the text for pretraining (quick when the corpus is tidy)",
    );
    let mut vocab_set = HashSet::new();
    Vocab::process_text_for_vocab(&texts, &mut vocab_set);
    done_stdout();
    note_stdout(&format!(
        "{} unique whole words harvested, plus <unk> (a stranger) and </s> (the end).",
        vocab_set.len()
    ));
    note_stdout("Anything outside this list is invisible to the model: small data, small world.");

    // STEP 3 CONFIGURATION: what will steer pretraining, and its levers.
    step_stdout(3, "Configuration for pretraining");
    print_pretraining(vocab_set.len(), llm::seed(), DEMO_EPOCHS, DEMO_LR);

    // STEP 4 BUILD: dials before they are turned.
    step_stdout(4, "Building the model");
    let mut model = build_tiny_llm(&texts);
    done_stdout();
    note_stdout(&format!(
        "{} adjustable dials (parameters): word lookup -> {} thinking blocks -> word guesser.",
        model.total_parameters(),
        block_count(&model),
    ));

    // STEP 5 TRAINING: guess, measure surprise, adjust, repeat.
    step_stdout(5, "Training (watch the bar fall)");
    note_stdout(&format!(
        "{DEMO_EPOCHS} epochs at learning rate {DEMO_LR}; an epoch = one full read of every story."
    ));
    let examples: Vec<&str> = texts.iter().map(String::as_str).collect();
    let losses = model.train_with_progress(examples, DEMO_EPOCHS, DEMO_LR, true);
    done_stdout();
    note_stdout(&format!(
        "Loss fell {:.2} -> {:.2}: the guesses got closer to the real words.",
        losses.first().copied().unwrap_or(0.0),
        losses.last().copied().unwrap_or(0.0),
    ));

    // The back half: held-out score, decode recipes, keyboard handover.
    score_and_use(&mut model);
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
