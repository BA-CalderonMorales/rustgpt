use std::io::Write;

use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, MAX_SEQ_LEN, Vocab,
    dataset_loader::{Dataset, DatasetType},
    embeddings::Embeddings,
    output_projection::OutputProjection,
    transformer::TransformerBlock,
};

enum Mode {
    Interactive,
    E2e { prompt: String },
}

fn usage() {
    println!("Usage: llm [--e2e <prompt>]");
    println!();
    println!("Examples:");
    println!("  llm");
    println!("  llm --e2e \"hello world\"");
}

fn parse_mode() -> Result<Mode, String> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None => Ok(Mode::Interactive),
        Some("--e2e") => {
            let prompt = args
                .next()
                .ok_or_else(|| "--e2e requires a prompt".to_string())?;
            if args.next().is_some() {
                return Err("--e2e accepts exactly one prompt".to_string());
            }
            Ok(Mode::E2e { prompt })
        }
        Some("--help") | Some("-h") => {
            usage();
            std::process::exit(0);
        }
        Some("--version") => {
            println!("llm {}", env!("CARGO_PKG_VERSION"));
            std::process::exit(0);
        }
        Some(argument) => Err(format!("unknown argument: {argument}")),
    }
}

fn main() {
    let mode = parse_mode().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!("Try 'llm --help' for usage.");
        std::process::exit(2);
    });

    // Mock input - test conversational format
    let string = String::from("User: How do mountains form?");

    let dataset = Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        DatasetType::JSON,
    ); // Placeholder, not used in this example

    // Extract all unique words from training data to create vocabulary
    let mut vocab_set = std::collections::HashSet::new();

    // Process all training examples for vocabulary
    // First process pre-training data
    Vocab::process_text_for_vocab(&dataset.pretraining_data, &mut vocab_set);

    // Then process chat training data
    Vocab::process_text_for_vocab(&dataset.chat_training_data, &mut vocab_set);

    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort(); // Sort for deterministic ordering
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s: &String| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);

    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_3 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());
    let mut llm = LLM::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(transformer_block_1),
            Box::new(transformer_block_2),
            Box::new(transformer_block_3),
            Box::new(output_projection),
        ],
    );

    if let Mode::E2e { prompt } = mode {
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
        return;
    }

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
        .map(|s| s.as_str())
        .collect();

    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    llm.train(pretraining_examples, 100, 0.0005);

    println!("\n=== INSTRUCTION TUNING ===");
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}",
        dataset.chat_training_data.len(),
        100,
        0.0001
    );

    llm.train(chat_training_examples, 100, 0.0001); // Much lower learning rate for stability

    println!("\n=== AFTER TRAINING ===");
    println!("Input: {}", string);
    let result = llm.predict(&string);
    println!("Output: {}", result);
    println!("======================\n");

    // Interactive mode for user input
    println!("\n--- Interactive Mode ---");
    println!("Type a prompt and press Enter to generate text.");
    println!("Type 'exit' to quit.");

    let mut input = String::new();
    loop {
        // Clear the input string
        input.clear();

        // Prompt for user input
        print!("\nEnter prompt: ");
        std::io::stdout().flush().unwrap();

        // Read user input
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        // Trim whitespace and check for exit command
        let trimmed_input = input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Exiting interactive mode.");
            break;
        }

        // Generate prediction based on user input with "User:" prefix
        let formatted_input = format!("User: {}", trimmed_input);
        let prediction = llm.predict(&formatted_input);
        println!("Model output: {}", prediction);
    }
}
