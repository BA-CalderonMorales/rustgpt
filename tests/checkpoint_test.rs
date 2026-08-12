use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, Vocab, embeddings::Embeddings, load,
    output_projection::OutputProjection, save, set_seed, transformer::TransformerBlock,
};

fn build_model(training_data: &[String]) -> LLM {
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(training_data, &mut vocab_set);
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

fn training_data() -> Vec<String> {
    [
        "User: What is rain? Assistant: Rain falls from clouds. </s>",
        "User: How does rain fall? Assistant: Rain falls from clouds. </s>",
        "User: What is the cycle? Assistant: The water cycle repeats. </s>",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn train_tiny(model: &mut LLM, data: &[String]) {
    let examples: Vec<&str> = data.iter().map(String::as_str).collect();
    model.train(examples, 3, 0.0005);
}

#[test]
fn checkpoint_round_trip_preserves_predictions_and_scores() {
    let data = training_data();
    set_seed(5);
    let mut original = build_model(&data);
    train_tiny(&mut original, &data);

    let path = std::env::temp_dir().join("rustgpt-roundtrip.bin");
    save(&original, path.to_str().unwrap()).expect("save should succeed");

    set_seed(999);
    let mut restored = load(path.to_str().unwrap()).expect("load should succeed");

    let prompt = "User: What is rain?";
    let reference = "Assistant: Rain falls from clouds.";
    let original_score = original.answer_score(prompt, reference);
    let restored_score = restored.answer_score(prompt, reference);

    assert_eq!(original.predict(prompt), restored.predict(prompt));
    assert_eq!(original.total_parameters(), restored.total_parameters());
    assert_eq!(original_score.exact, restored_score.exact);
    assert_eq!(original_score.prefix, restored_score.prefix);
    assert_eq!(original_score.accuracy, restored_score.accuracy);

    std::fs::remove_file(path).ok();
}

#[test]
fn checkpoint_rejects_foreign_files() {
    let path = std::env::temp_dir().join("rustgpt-not-a-checkpoint.bin");
    std::fs::write(&path, b"definitely not a rustgpt checkpoint").unwrap();
    assert!(load(path.to_str().unwrap()).is_err());
    std::fs::remove_file(path).ok();
}

#[test]
fn sequence_loss_is_finite_and_stable() {
    let data = training_data();
    set_seed(13);
    let mut model = build_model(&data);

    let text = "User: What is rain? Assistant: Rain falls from clouds. </s>";
    let first = model.sequence_loss(text);
    let second = model.sequence_loss(text);

    assert!(first.is_finite());
    assert_eq!(first, second);
}

#[test]
fn training_reports_one_loss_per_epoch() {
    let data = training_data();
    set_seed(17);
    let mut model = build_model(&data);
    let examples: Vec<&str> = data.iter().map(String::as_str).collect();

    let losses = model.train_with_progress(examples, 4, 0.0005, false);

    assert_eq!(losses.len(), 4);
    assert!(losses.iter().all(|loss| loss.is_finite()));
}
