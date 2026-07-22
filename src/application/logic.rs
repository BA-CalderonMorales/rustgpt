use std::io::Write;

use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, MAX_SEQ_LEN, Vocab,
    dataset_loader::{Dataset, DatasetType},
    embeddings::Embeddings,
    output_projection::OutputProjection,
    transformer::TransformerBlock,
};

use crate::cli::Mode;

pub(crate) fn load_datasets() -> Dataset {
    Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        DatasetType::JSON,
    )
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

pub(crate) fn run(mode: Mode, dataset: &Dataset, llm: &mut LLM) {
    match mode {
        Mode::E2e { prompt } => run_e2e(prompt, llm),
        Mode::Interactive => run_training_and_interactive(dataset, llm),
    }
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

fn run_training_and_interactive(dataset: &Dataset, llm: &mut LLM) {
    let string = String::from("User: How do mountains form?");

    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    println!(
        "Model configuration -> max_seq_len: {}, embedding_dim: {}, hidden_dim: {}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );
    println!("Total parameters: {}", llm.total_parameters());

    println!("\n=== BEFORE TRAINING ===");
    println!("Input: {}", string);
    println!("Output: {}", llm.predict(&string));

    println!("\n=== PRE-TRAINING MODEL ===");
    println!(
        "Pre-training on {} examples for {} epochs with learning rate {}",
        dataset.pretraining_data.len(),
        100,
        0.0005
    );

    let pretraining_examples: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(String::as_str)
        .collect();
    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(String::as_str)
        .collect();

    llm.train(pretraining_examples, 100, 0.0005);

    println!("\n=== INSTRUCTION TUNING ===");
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}",
        dataset.chat_training_data.len(),
        100,
        0.0001
    );

    llm.train(chat_training_examples, 100, 0.0001);

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
