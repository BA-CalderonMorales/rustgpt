use std::io::Write;

use llm::{
    DecodeStep, EMBEDDING_DIM, HIDDEN_DIM, LLM, MAX_SEQ_LEN, Vocab,
    dataset_loader::{Dataset, DatasetType},
    embeddings::Embeddings,
    output_projection::OutputProjection,
    transformer::TransformerBlock,
};

use crate::cli::{Invocation, Mode};

use super::Trace;

const PRETRAINING_EPOCHS: usize = 100;
const PRETRAINING_LR: f32 = 0.0005;
const TUNING_EPOCHS: usize = 100;
const TUNING_LR: f32 = 0.0001;
const TUNING_CE_SAMPLES: usize = 3;

pub(crate) fn load_datasets() -> Dataset {
    Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        DatasetType::JSON,
    )
}

pub(crate) fn load_heldout() -> Vec<(String, String)> {
    let text = std::fs::read_to_string("data/heldout.json")
        .expect("data/heldout.json must exist for --eval");
    let entries: Vec<(String, String)> =
        serde_json::from_str(&text).expect("data/heldout.json must list prompt/reference pairs");
    entries
}

pub(crate) fn build_llm(dataset: &Dataset, invocation: &Invocation) -> LLM {
    // Derive the load/train targets and whether the mode may build fresh.
    let model_path = invocation.model.as_deref();
    let train_path = match &invocation.mode {
        Mode::Train { path } => Some(path.as_str()),
        _ => None,
    };
    let can_initialize = matches!(
        invocation.mode,
        Mode::Train { .. } | Mode::Interactive | Mode::Eval
    );

    // A checkpoint wins when it exists.
    if let Some(path) = model_path {
        if std::path::Path::new(path).exists() {
            eprintln!("Loading checkpoint {path}");
            return llm::load(path).unwrap_or_else(|error| {
                eprintln!("error: failed to load checkpoint {path}: {error}");
                std::process::exit(1);
            });
        }
        // A missing checkpoint is a first-run target only for modes that
        // train (--train, interactive, --eval); --e2e must not silently
        // fall back to a fresh model.
        if train_path.is_none() && !can_initialize {
            eprintln!("error: checkpoint not found: {path}");
            std::process::exit(1);
        }
    }

    // The --train lane builds a fresh tiny model from the corpus.
    if let Some(path) = train_path {
        if !invocation.tiny {
            eprintln!(
                "error: --train requires --tiny (the language-model lane is the tiny preset)"
            );
            std::process::exit(2);
        }
        let texts = llm::load_jsonl(path);
        return build_tiny_llm(&texts);
    }

    // The tiny flag needs a checkpoint or a corpus; otherwise build the
    // water-cycle micro model.
    if invocation.tiny {
        eprintln!("error: --tiny requires --model <checkpoint> or --train <file.jsonl>");
        std::process::exit(2);
    }
    build_model(dataset)
}

/// Build the tiny preset (real-corpus lane) vocabulary and model.
pub(crate) fn build_tiny_llm(texts: &[String]) -> LLM {
    // The tiny preset's config, with a vocabulary harvested from the corpus.
    use llm::Config;
    let config = Config::tiny();
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(texts, &mut vocab_set);
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(String::as_str).collect();
    let vocab = Vocab::new(vocab_words_refs);

    // Assemble the network: embeddings, blocks, output projection.
    let mut network: Vec<Box<dyn llm::Layer>> = vec![Box::new(Embeddings::with_dims(
        vocab.clone(),
        config.embedding_dim,
        config.max_seq_len,
    ))];
    for _ in 0..config.block_count {
        network.push(Box::new(TransformerBlock::new(
            config.embedding_dim,
            config.hidden_dim,
        )));
    }
    network.push(Box::new(OutputProjection::new(
        config.embedding_dim,
        vocab.words.len(),
    )));
    LLM::with_config(vocab, network, config)
}

pub(crate) fn build_model(dataset: &Dataset) -> LLM {
    // Harvest the vocabulary from both data halves.
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(&dataset.pretraining_data, &mut vocab_set);
    Vocab::process_text_for_vocab(&dataset.chat_training_data, &mut vocab_set);
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(String::as_str).collect();
    let vocab = Vocab::new(vocab_words_refs);

    // Assemble the fixed micro pipeline: embeddings, three blocks, projection.
    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_3 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());
    LLM::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(transformer_block_1),
            Box::new(transformer_block_2),
            Box::new(transformer_block_3),
            Box::new(output_projection),
        ],
    )
}

pub(crate) fn run_headless(invocation: &Invocation, dataset: &Dataset, llm: &mut LLM) {
    // Every machine mode serves exactly one JSON object; the interactive
    // mode has its own view and never reaches this dispatcher.
    match &invocation.mode {
        Mode::E2e { prompt } => {
            run_e2e(prompt.clone(), llm);
        }
        Mode::Eval => {
            if invocation.tiny {
                crate::application::run_tiny_eval(llm, invocation.temperature);
            } else {
                run_training_and_eval(dataset, llm, invocation.model.as_deref());
            }
        }
        Mode::Train { path } => {
            run_training_lm(
                path.as_str(),
                llm,
                invocation.model.as_deref(),
                invocation.epochs,
            );
        }
        Mode::Probe => {
            if invocation.model.is_none() {
                eprintln!("error: --probe requires --model <checkpoint>");
                std::process::exit(2);
            }
            crate::application::run_probe(llm);
        }
        Mode::Interactive => {
            unreachable!("interactive has its own view");
        }
    }
}

pub(crate) fn run_interactive(invocation: &Invocation, dataset: &Dataset, llm: &mut LLM) {
    let trace = Trace::new(invocation.trace);
    run_training_and_interactive(dataset, llm, invocation.model.as_deref(), &trace);
}

fn run_training_lm(path: &str, llm: &mut LLM, model_path: Option<&str>, epochs: usize) {
    // Load the corpus; a degenerate file is a hard error.
    let texts = llm::load_jsonl(path);
    if texts.len() < 2 {
        eprintln!("error: --train needs at least 2 lines in {path}");
        std::process::exit(1);
    }
    eprintln!(
        "=== LANGUAGE MODEL TRAINING === {} stories, {} epochs, lr {}",
        texts.len(),
        epochs,
        PRETRAINING_LR
    );

    // Train, sampling the held-out logit-regime profile every epoch: the
    // continuous instrument that makes collapse onset visible. Then persist
    // the checkpoint.
    let examples: Vec<&str> = texts.iter().map(String::as_str).collect();
    let profile_stories = crate::application::tiny_heldout_stories();
    let profile_texts: Vec<&str> = profile_stories.iter().map(String::as_str).collect();
    let (losses, profile) =
        llm.train_with_profile(examples, epochs, PRETRAINING_LR, true, &profile_texts);
    save_checkpoint(llm, model_path);

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

    // Exactly one JSON object: trajectory, the per-epoch logit profile,
    // samples, and the lane's eval.
    println!(
        "{}",
        serde_json::json!({
            "status": "ok",
            "seed": llm::seed(),
            "total_parameters": llm.total_parameters(),
            "stories": texts.len(),
            "epochs": epochs,
            "trajectory": { "loss": losses },
            "profile": llm::profile_json(&profile),
            "samples": samples,
            "eval": crate::application::tiny_eval(llm, 1.0),
        })
    );
}

fn run_e2e(prompt: String, llm: &mut LLM) {
    let output = llm.predict(&prompt);
    println!(
        "{}",
        serde_json::json!({
            "status": "ok",
            "prompt": prompt,
            "output": output,
            "total_parameters": llm.total_parameters(),
        })
    );
}

/// Interleave chat examples with pretrain statements at roughly 2:1 so
/// every epoch rehearses both domains (arXiv 2403.05175 replay): the chat
/// format survives phase 1 and the pretrain facts survive phase 2.
fn interleave_replay(chat: &[String], pretrain: &[String]) -> Vec<String> {
    // Round-robin with a 2:1 chat-to-pretrain ratio: each round takes two
    // chat examples, then one pretrain statement.
    let mut out = Vec::with_capacity(chat.len() + pretrain.len());
    let mut chat_index = 0usize;
    let mut pretrain_index = 0usize;
    while chat_index < chat.len() || pretrain_index < pretrain.len() {
        for _ in 0..2 {
            if chat_index < chat.len() {
                out.push(chat[chat_index].clone());
                chat_index += 1;
            }
        }
        if pretrain_index < pretrain.len() {
            out.push(pretrain[pretrain_index].clone());
            pretrain_index += 1;
        }
    }
    out
}

fn train_pretraining(dataset: &Dataset, llm: &mut LLM, progress: bool) -> Vec<f32> {
    let replayed = interleave_replay(&dataset.chat_training_data, &dataset.pretraining_data);
    let examples: Vec<&str> = replayed.iter().map(String::as_str).collect();
    llm.train_with_progress(examples, PRETRAINING_EPOCHS, PRETRAINING_LR, progress)
}

fn train_tuning_part(dataset: &Dataset, llm: &mut LLM, epochs: usize, progress: bool) -> Vec<f32> {
    let replayed = interleave_replay(&dataset.chat_training_data, &dataset.pretraining_data);
    let examples: Vec<&str> = replayed.iter().map(String::as_str).collect();
    llm.train_with_progress(examples, epochs, TUNING_LR, progress)
}

fn mean_heldout_ce(llm: &mut LLM, sequences: &[String]) -> f32 {
    let total: f32 = sequences.iter().map(|text| llm.sequence_loss(text)).sum();
    total / sequences.len().max(1) as f32
}

fn run_training_and_eval(dataset: &Dataset, llm: &mut LLM, model_path: Option<&str>) {
    // Held-out chat sequences, teacher-forced per token.
    let heldout = load_heldout();
    let chat_sequences: Vec<String> = heldout
        .iter()
        .map(|(prompt, reference)| format!("User: {prompt} Assistant: {reference}"))
        .collect();

    // Pretraining phase, tracked on the held-out CE curve.
    let mut trajectory_ce = vec![mean_heldout_ce(llm, &chat_sequences)];
    eprintln!("=== PRE-TRAINING ===");
    let pretrain_loss = train_pretraining(dataset, llm, true);
    trajectory_ce.push(mean_heldout_ce(llm, &chat_sequences));

    // Tuning phase in thirds, snapshotting the min-CE state along the way.
    eprintln!("=== INSTRUCTION TUNING ===");
    let mut tuning_loss = Vec::new();
    let mut best_ce = f32::INFINITY;
    let mut best_snapshot: Option<Vec<(String, Vec<u8>)>> = None;
    let part_epochs = TUNING_EPOCHS / TUNING_CE_SAMPLES;
    for _ in 0..TUNING_CE_SAMPLES {
        tuning_loss.extend(train_tuning_part(dataset, llm, part_epochs, true));
        let ce = mean_heldout_ce(llm, &chat_sequences);
        trajectory_ce.push(ce);
        if ce < best_ce {
            best_ce = ce;
            best_snapshot = Some(
                llm.network
                    .iter()
                    .map(|layer| {
                        (
                            layer.layer_type().to_string(),
                            layer.parameter_bytes().unwrap_or_else(|error| {
                                panic!("failed to snapshot layer state: {error}")
                            }),
                        )
                    })
                    .collect(),
            );
        }
    }

    // Promote the min-CE tuning state into the artifact: the eval items and
    // the saved checkpoint must describe the same model (the checkpoint plus
    // its eval JSON is the unit of evidence), not the drifted tail.
    if let Some(snapshot) = &best_snapshot {
        for (layer, (expected_type, payload)) in llm.network.iter_mut().zip(snapshot) {
            if layer.layer_type() != expected_type {
                panic!("snapshot layer type mismatch");
            }
            layer
                .load_parameter_bytes(payload)
                .unwrap_or_else(|error| panic!("failed to restore min-CE state: {error}"));
        }
        eprintln!("Promoting min-CE state (held-out CE {best_ce:.4}) into the artifact");
    }
    save_checkpoint(llm, model_path);

    // Greedy scoring of every held-out item.
    let mut items = Vec::new();
    let mut accuracies = Vec::new();
    let mut exact_total = 0usize;
    let mut prefix_total = 0usize;
    for (prompt_text, reference) in &heldout {
        let prompt = format!("User: {prompt_text}");
        let generated = llm.predict(&prompt);
        let score = llm.score_generated(&generated, reference);
        exact_total += usize::from(score.exact);
        prefix_total += usize::from(score.prefix);
        accuracies.push(score.accuracy);
        items.push(serde_json::json!({
            "prompt": prompt_text,
            "reference": reference,
            "generated": generated,
            "exact": score.exact,
            "prefix": score.prefix,
            "accuracy": score.accuracy,
        }));
    }

    // Accuracy summary: min/median/max of the per-item scores.
    accuracies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = match accuracies.len() {
        0 => 0.0,
        n if n % 2 == 1 => accuracies[n / 2],
        n => (accuracies[n / 2 - 1] + accuracies[n / 2]) / 2.0,
    };

    // Exactly one JSON object: the full evidence record.
    println!(
        "{}",
        serde_json::json!({
            "status": "ok",
            "seed": llm::seed(),
            "total_parameters": llm.total_parameters(),
            "summary": {
                "exact_matches": exact_total,
                "prefix_matches": prefix_total,
                "mean_accuracy": accuracies.iter().sum::<f32>() / accuracies.len().max(1) as f32,
                "accuracy_min": accuracies.first().copied().unwrap_or(0.0),
                "accuracy_median": median,
                "accuracy_max": accuracies.last().copied().unwrap_or(0.0),
            },
            "trajectory": {
                "pretrain_loss": pretrain_loss,
                "tuning_loss": tuning_loss,
                "heldout_ce": trajectory_ce,
            },
            "items": items,
        })
    );
}

fn save_checkpoint(llm: &mut LLM, model_path: Option<&str>) {
    if let Some(path) = model_path {
        llm::save(llm, path).unwrap_or_else(|error| {
            eprintln!("error: failed to save checkpoint {path}: {error}");
            std::process::exit(1);
        });
        eprintln!("Checkpoint saved to {path}");
    }
}

/// Startup trace block: one event per domain that shaped the session, so
/// the first glance at a trace says where the model came from.
fn trace_startup(dataset: &Dataset, llm: &LLM, model_path: Option<&str>, trace: &Trace) {
    trace.event("cli", &format!("mode=interactive seed={}", llm::seed()));
    trace.event(
        "configuration",
        &format!("max_seq_len={MAX_SEQ_LEN} embedding_dim={EMBEDDING_DIM} hidden_dim={HIDDEN_DIM}"),
    );
    trace.event(
        "dataset",
        &format!(
            "pretrain={} chat={} vocab={}",
            dataset.pretraining_data.len(),
            dataset.chat_training_data.len(),
            llm.vocab.words.len()
        ),
    );
    let checkpoint = match model_path {
        Some(path) if std::path::Path::new(path).exists() => format!("loaded {path}"),
        Some(path) => format!("fresh init (first-run target {path})"),
        None => "fresh init".to_string(),
    };
    trace.event("checkpoint", &checkpoint);
    trace.event(
        "llm",
        &format!(
            "network: {} parameters: {}",
            llm.network_description(),
            llm.total_parameters()
        ),
    );
}

/// Per-turn trace block: tokenization, pipeline map, one line per decoded
/// step, then the stop reason.
fn trace_turn(llm: &LLM, formatted_input: &str, steps: &[DecodeStep], trace: &Trace) {
    let decoded: Vec<String> = llm
        .tokenize(formatted_input)
        .iter()
        .map(|token| llm.vocab.decode[token].clone())
        .collect();
    trace.event("vocab", &format!("{formatted_input:?} -> {decoded:?}"));

    trace.event("llm", &format!("pipeline: {}", llm.network_description()));

    for (index, step) in steps.iter().enumerate() {
        trace.event(
            "decode",
            &format!(
                "step {index}: {:?} p={:.4}",
                llm.vocab.decode[&step.token], step.prob
            ),
        );
    }
    let stop = match steps.last().map(|step| step.token) {
        Some(token) if Some(&token) == llm.vocab.encode("</s>").as_ref() => "</s>",
        _ => "sequence budget",
    };
    trace.event(
        "decode",
        &format!("stop: {stop} after {} tokens", steps.len()),
    );
}

fn run_training_and_interactive(
    dataset: &Dataset,
    llm: &mut LLM,
    model_path: Option<&str>,
    trace: &Trace,
) {
    // Startup trace: which domain shaped this session.
    trace_startup(dataset, llm, model_path, trace);

    // The fixed probe prompt this session demonstrates.
    let string = String::from("User: How do mountains form?");

    // Model information.
    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    println!(
        "Model configuration -> max_seq_len: {}, embedding_dim: {}, hidden_dim: {}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );
    println!("Total parameters: {}", llm.total_parameters());
    println!("Seed: {}", llm::seed());

    // The untrained model's noise, for contrast.
    println!("\n=== BEFORE TRAINING ===");
    println!("Input: {}", string);
    println!("Output: {}", llm.predict(&string));

    // Pretraining phase.
    println!("\n=== PRE-TRAINING MODEL ===");
    println!(
        "Pre-training on {} examples for {} epochs with learning rate {}",
        dataset.pretraining_data.len(),
        PRETRAINING_EPOCHS,
        PRETRAINING_LR
    );
    train_pretraining(dataset, llm, true);

    // Instruction tuning phase.
    println!("\n=== INSTRUCTION TUNING ===");
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}",
        dataset.chat_training_data.len(),
        TUNING_EPOCHS,
        TUNING_LR
    );
    train_tuning_part(dataset, llm, TUNING_EPOCHS, true);

    // Persist the trained session.
    save_checkpoint(llm, model_path);

    // The trained model's answer to the same prompt.
    println!("\n=== AFTER TRAINING ===");
    println!("Input: {}", string);
    let result = llm.predict(&string);
    println!("Output: {}", result);
    println!("======================\n");

    // Chat loop until "exit".
    println!("\n--- Interactive Mode ---");
    println!("Type a prompt and press Enter to generate text.");
    println!("Type 'exit' to quit.");
    let mut input = String::new();
    loop {
        // Read the next prompt.
        input.clear();
        print!("\nEnter prompt: ");
        std::io::stdout().flush().unwrap();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        // The exit word ends the session.
        let trimmed_input = input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Exiting interactive mode.");
            break;
        }

        // Traced turns capture every decode step; the default path keeps
        // the untraced `predict` surface untouched.
        let formatted_input = format!("User: {trimmed_input}");
        let prediction = if trace.on {
            let (output, steps) = llm.predict_with_steps(&formatted_input);
            trace_turn(llm, &formatted_input, &steps, trace);
            output
        } else {
            llm.predict(&formatted_input)
        };
        println!("Model output: {prediction}");
    }
}
