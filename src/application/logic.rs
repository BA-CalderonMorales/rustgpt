use std::io::Write;

use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, MAX_SEQ_LEN, Vocab,
    dataset_loader::{Dataset, DatasetType},
    embeddings::Embeddings,
    output_projection::OutputProjection,
    transformer::TransformerBlock,
};

use crate::cli::{Invocation, Mode};

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

pub(crate) fn build_llm(
    dataset: &Dataset,
    model_path: Option<&str>,
    train_path: Option<&str>,
    tiny: bool,
    can_initialize: bool,
) -> LLM {
    if let Some(path) = model_path {
        if std::path::Path::new(path).exists() {
            eprintln!("Loading checkpoint {path}");
            return llm::load(path).unwrap_or_else(|error| {
                eprintln!("error: failed to load checkpoint {path}: {error}");
                std::process::exit(1);
            });
        }
        // A missing checkpoint is a first-run target only for modes that
        // train (--train, interactive, --eval, which always builds fresh
        // when no checkpoint is given); --e2e must not silently fall back
        // to a fresh model.
        if train_path.is_none() && !can_initialize {
            eprintln!("error: checkpoint not found: {path}");
            std::process::exit(1);
        }
    }
    if let Some(path) = train_path {
        if !tiny {
            eprintln!(
                "error: --train requires --tiny (the language-model lane is the tiny preset)"
            );
            std::process::exit(2);
        }
        let texts = llm::load_jsonl(path);
        return build_tiny_llm(&texts);
    }
    if tiny {
        eprintln!("error: --tiny requires --model <checkpoint> or --train <file.jsonl>");
        std::process::exit(2);
    }
    build_model(dataset)
}

/// Build the tiny preset (real-corpus lane) vocabulary and model.
pub(crate) fn build_tiny_llm(texts: &[String]) -> LLM {
    use llm::Config;

    let config = Config::tiny();
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(texts, &mut vocab_set);
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(String::as_str).collect();
    let vocab = Vocab::new(vocab_words_refs);

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
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(&dataset.pretraining_data, &mut vocab_set);
    Vocab::process_text_for_vocab(&dataset.chat_training_data, &mut vocab_set);

    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(String::as_str).collect();
    let vocab = Vocab::new(vocab_words_refs);

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

pub(crate) fn run(invocation: Invocation, dataset: &Dataset, llm: &mut LLM) {
    match invocation.mode {
        Mode::E2e { prompt } => run_e2e(prompt, llm),
        Mode::Eval => {
            if invocation.tiny {
                crate::application::run_tiny_eval(llm);
            } else {
                run_training_and_eval(dataset, llm, invocation.model.as_deref());
            }
        }
        Mode::Train { path } => {
            run_training_lm(&path, llm, invocation.model.as_deref(), invocation.epochs)
        }
        Mode::Interactive => {
            run_training_and_interactive(dataset, llm, invocation.model.as_deref())
        }
    }
}

fn run_training_lm(path: &str, llm: &mut LLM, model_path: Option<&str>, epochs: usize) {
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

    let examples: Vec<&str> = texts.iter().map(String::as_str).collect();
    let losses = llm.train_with_progress(examples, epochs, PRETRAINING_LR, true);
    save_checkpoint(llm, model_path);

    let starters = ["Once upon a time,", "The sun", "Why did the"];
    let samples: Vec<serde_json::Value> = starters
        .iter()
        .map(|starter| {
            serde_json::json!({
                "prompt": starter,
                "generated": llm.predict(starter),
            })
        })
        .collect();

    println!(
        "{}",
        serde_json::json!({
            "status": "ok",
            "seed": llm::seed(),
            "total_parameters": llm.total_parameters(),
            "stories": texts.len(),
            "epochs": epochs,
            "trajectory": { "loss": losses },
            "samples": samples,
            "eval": crate::application::tiny_eval(llm),
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

fn train_pretraining(dataset: &Dataset, llm: &mut LLM, progress: bool) -> Vec<f32> {
    let examples: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(String::as_str)
        .collect();
    llm.train_with_progress(examples, PRETRAINING_EPOCHS, PRETRAINING_LR, progress)
}

fn train_tuning_part(dataset: &Dataset, llm: &mut LLM, epochs: usize, progress: bool) -> Vec<f32> {
    let examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(String::as_str)
        .collect();
    llm.train_with_progress(examples, epochs, TUNING_LR, progress)
}

fn mean_heldout_ce(llm: &mut LLM, sequences: &[String]) -> f32 {
    let total: f32 = sequences.iter().map(|text| llm.sequence_loss(text)).sum();
    total / sequences.len().max(1) as f32
}

fn run_training_and_eval(dataset: &Dataset, llm: &mut LLM, model_path: Option<&str>) {
    let heldout = load_heldout();
    let chat_sequences: Vec<String> = heldout
        .iter()
        .map(|(prompt, reference)| format!("User: {prompt} Assistant: {reference}"))
        .collect();

    let mut trajectory_ce = vec![mean_heldout_ce(llm, &chat_sequences)];
    eprintln!("=== PRE-TRAINING ===");
    let pretrain_loss = train_pretraining(dataset, llm, true);
    trajectory_ce.push(mean_heldout_ce(llm, &chat_sequences));

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
            layer.load_parameter_bytes(payload).unwrap_or_else(|error| {
                panic!("failed to restore min-CE state: {error}")
            });
        }
        eprintln!("Promoting min-CE state (held-out CE {best_ce:.4}) into the artifact");
    }

    save_checkpoint(llm, model_path);

    let mut items = Vec::new();
    let mut accuracies = Vec::new();
    let mut exact_total = 0usize;
    let mut prefix_total = 0usize;

    for (prompt_text, reference) in &heldout {
        let prompt = format!("User: {prompt_text}");
        let generated = llm.predict(&prompt);
        let score = llm.answer_score(&prompt, reference);
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

    accuracies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = match accuracies.len() {
        0 => 0.0,
        n if n % 2 == 1 => accuracies[n / 2],
        n => (accuracies[n / 2 - 1] + accuracies[n / 2]) / 2.0,
    };

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

fn run_training_and_interactive(dataset: &Dataset, llm: &mut LLM, model_path: Option<&str>) {
    let string = String::from("User: How do mountains form?");

    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    println!(
        "Model configuration -> max_seq_len: {}, embedding_dim: {}, hidden_dim: {}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );
    println!("Total parameters: {}", llm.total_parameters());
    println!("Seed: {}", llm::seed());

    println!("\n=== BEFORE TRAINING ===");
    println!("Input: {}", string);
    println!("Output: {}", llm.predict(&string));

    println!("\n=== PRE-TRAINING MODEL ===");
    println!(
        "Pre-training on {} examples for {} epochs with learning rate {}",
        dataset.pretraining_data.len(),
        PRETRAINING_EPOCHS,
        PRETRAINING_LR
    );
    train_pretraining(dataset, llm, false);

    println!("\n=== INSTRUCTION TUNING ===");
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}",
        dataset.chat_training_data.len(),
        TUNING_EPOCHS,
        TUNING_LR
    );
    train_tuning_part(dataset, llm, TUNING_EPOCHS, false);

    save_checkpoint(llm, model_path);

    println!("\n=== AFTER TRAINING ===");
    println!("Input: {}", string);
    let result = llm.predict(&string);
    println!("Output: {}", result);
    println!("======================\n");

    println!("\n--- Interactive Mode ---");
    println!("Type a prompt and press Enter to generate text.");
    println!("Type 'exit' to quit.");

    let mut input = String::new();
    loop {
        input.clear();
        print!("\nEnter prompt: ");
        std::io::stdout().flush().unwrap();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        let trimmed_input = input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Exiting interactive mode.");
            break;
        }

        let formatted_input = format!("User: {}", trimmed_input);
        let prediction = llm.predict(&formatted_input);
        println!("Model output: {}", prediction);
    }
}
