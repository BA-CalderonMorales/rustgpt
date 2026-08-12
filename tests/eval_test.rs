use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, Vocab, embeddings::Embeddings,
    output_projection::OutputProjection, set_seed, transformer::TransformerBlock,
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

fn train(model: &mut LLM, data: &[String]) {
    let examples: Vec<&str> = data.iter().map(String::as_str).collect();
    model.train(examples, 2, 0.0005);
}

#[test]
fn same_seed_produces_identical_scores() {
    let data = training_data();
    set_seed(7);
    let mut first = build_model(&data);
    set_seed(7);
    let mut second = build_model(&data);

    let prompt = "User: What is rain?";
    let reference = "Assistant: Rain falls from clouds.";

    let first_score = first.answer_score(prompt, reference);
    let second_score = second.answer_score(prompt, reference);

    assert_eq!(first_score.exact, second_score.exact);
    assert_eq!(first_score.prefix, second_score.prefix);
    assert_eq!(first_score.accuracy, second_score.accuracy);
    assert_eq!(first.predict(prompt), second.predict(prompt));
}

#[test]
fn accuracy_is_bounded_and_exact_implies_perfect() {
    let data = training_data();
    set_seed(11);
    let mut model = build_model(&data);

    let score = model.answer_score("User: how does rain fall?", "rain falls from clouds");

    assert!((0.0..=1.0).contains(&score.accuracy));
    if score.exact {
        assert_eq!(score.accuracy, 1.0);
    }
}

#[test]
fn training_is_reproducible_with_seed() {
    let data = training_data();
    set_seed(3);
    let mut first = build_model(&data);
    set_seed(3);
    let mut second = build_model(&data);

    train(&mut first, &data);
    train(&mut second, &data);

    let prompt = "User: What is rain?";
    assert_eq!(first.predict(prompt), second.predict(prompt));
}
