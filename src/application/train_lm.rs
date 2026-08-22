use llm::LLM;

use super::{note, save_checkpoint, stage, tiny_eval, tiny_heldout_stories};
use crate::cli::Invocation;

const TRAINING_LR: f32 = 0.0005;

/// The tiny-lane training run: narrated stages on stderr, exactly one JSON
/// object on stdout. `--eos` (E11) appends " </s>" to every row; a
/// `--lr-decay <final_lr>` (W8) rides the linear per-epoch schedule.
pub(crate) fn run_training_lm(
    path: &str,
    llm: &mut LLM,
    model_path: Option<&str>,
    invocation: &Invocation,
) {
    // Load the corpus; a degenerate file is a hard error.
    let mut texts = llm::load_jsonl(path);
    if texts.len() < 2 {
        eprintln!("error: --train needs at least 2 lines in {path}");
        std::process::exit(1);
    }
    let epochs = invocation.epochs;

    // STAGE 1 DATA: what the model will read, one example per line.
    stage(1);
    note(&format!(
        "{} examples loaded from {path}; each line is one example the model reads during training.",
        texts.len()
    ));
    note(&format!(
        "The first example: {:?}...",
        first_words(&texts[0], 14)
    ));

    // E11 lever: append " </s>" to every row so every story ends.
    if invocation.eos {
        for text in &mut texts {
            text.push_str(" </s>");
        }
        note("Every example now ends with </s>, the model's word for 'the end'.");
    }

    // STAGE 2 VOCABULARY and STAGE 3 MODEL: what the model can see and how
    // many dials it has.
    stage(2);
    note(&format!(
        "The vocabulary holds {} whole words, plus <unk> for strangers and </s> for 'the end'. Words outside this list are invisible to the model.",
        llm.vocab.words.len()
    ));
    stage(3);
    note(&format!(
        "{} adjustable dials (parameters), arranged as: word lookup -> {} thinking blocks -> word guesser.",
        llm.total_parameters(),
        block_count(llm),
    ));

    // STAGE 4 TRAINING: epochs at learning rate, live loss bar; the W8
    // decay target rides the linear per-epoch schedule when given.
    stage(4);
    note(&format!(
        "Training for {epochs} epochs (an epoch is one full read of every example) at learning rate {TRAINING_LR}{}.",
        match invocation.lr_decay {
            Some(final_lr) => format!(" decaying to {final_lr}"),
            None => String::new(),
        }
    ));
    note("Loss is the average surprise: lower means wrong less often. Watch the bar fall.");
    eprintln!(
        "=== LANGUAGE MODEL TRAINING === {} stories, {} epochs, lr {}",
        texts.len(),
        epochs,
        TRAINING_LR
    );

    // Train, sampling the held-out logit-regime profile every epoch: the
    // continuous instrument that makes collapse onset visible. Then persist
    // the checkpoint.
    let examples: Vec<&str> = texts.iter().map(String::as_str).collect();
    let profile_stories = tiny_heldout_stories();
    let profile_texts: Vec<&str> = profile_stories.iter().map(String::as_str).collect();
    let (losses, profile) = llm.train_with_schedule(
        examples,
        epochs,
        TRAINING_LR,
        invocation.lr_decay,
        true,
        &profile_texts,
    );
    save_checkpoint(llm, model_path);
    note(&format!(
        "Loss went {:.2} -> {:.2}: the guesses moved closer to the real words. Past the lowest point, memorizing replaces learning -- so we stop at the budget and keep the curve as evidence.",
        losses.first().copied().unwrap_or(0.0),
        losses.last().copied().unwrap_or(0.0),
    ));

    // Sample a few fixed starters from the trained lane.
    let starters = ["Once upon a time,", "The sun", "Why did the"];
    let samples: Vec<serde_json::Value> = starters
        .iter()
        .map(|starter| {
            serde_json::json!({
                "prompt": starter,
                "generated": llm.predict_cached(starter),
            })
        })
        .collect();

    // STAGE 5 EVALUATION: held-out stories the model never saw.
    let eval = tiny_eval(llm, 1.0, 0.0, 1.0, 0.0);
    stage(5);
    narrate_eval(&eval);

    // STAGE 6 USE: how to talk to the trained artifact.
    stage(6);
    note("Talk to your model with: llm --model <checkpoint path> --ask \"Once upon a time,\"");
    note("Or load it for a chat: llm --model <checkpoint path>");

    // Exactly one JSON object: trajectory, the per-epoch logit profile,
    // samples, the training recipe, and the lane's eval.
    println!(
        "{}",
        serde_json::json!({
            "status": "ok",
            "seed": llm::seed(),
            "total_parameters": llm.total_parameters(),
            "stories": texts.len(),
            "epochs": epochs,
            "training": {
                "eos_appended": invocation.eos,
                "lr_start": TRAINING_LR,
                "lr_final": invocation.lr_decay.unwrap_or(TRAINING_LR),
            },
            "trajectory": { "loss": losses },
            "profile": llm::profile_json(&profile),
            "samples": samples,
            "eval": eval,
        })
    );
}

/// The beginner evaluation narration: percentiles with meaning, then the
/// collapse gate with meaning.
fn narrate_eval(eval: &serde_json::Value) {
    let percentiles = &eval["ce_percentiles"];
    let gate = &eval["collapse"];
    note(&format!(
        "Scored against held-out stories the model NEVER saw. Cross-entropy p10/p50/p90 = {:.2}/{:.2}/{:.2} (the middle score is the median): even the best case still surprises the model -- that is honest small-model math. This is a TEACHER-FORCED score: the model always sees the true previous words and only guesses the next one.",
        percentiles["p10"].as_f64().unwrap_or(0.0),
        percentiles["p50"].as_f64().unwrap_or(0.0),
        percentiles["p90"].as_f64().unwrap_or(0.0),
    ));
    let verdict = if gate["collapsed"].as_bool().unwrap_or(false) {
        "collapsed: the model repeats itself"
    } else {
        "not collapsed"
    };
    note(&format!(
        "Collapse gate: repetition rate {:.2} over a fixed sample -- {verdict}.",
        gate["repetition_rate"].as_f64().unwrap_or(0.0),
    ));
}

/// The opening words of an example, for the data-stage excerpt.
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
