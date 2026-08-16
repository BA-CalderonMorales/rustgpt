use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, Vocab, embeddings::Embeddings,
    output_projection::OutputProjection, set_seed, transformer::TransformerBlock,
};

/// Rows without </s>, so free-running continues past the corpus end and
/// drifts into a repeat loop -- the micro shape of the tiny-lane collapse.
const STREAM: [&str; 3] = [
    "the duck the frog the pond the duck swam away .",
    "the frog the duck the pond the frog jumped away .",
    "the pond the duck the frog the pond dried up .",
];

fn build_micro_model(texts: &[&str]) -> LLM {
    // Harvest the vocabulary, exactly as the micro lane does from a corpus.
    let text_vecs: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(&text_vecs, &mut vocab_set);
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(String::as_str).collect();
    let vocab = Vocab::new(vocab_words_refs);

    // The fixed micro pipeline: embeddings, three blocks, projection.
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

/// Fraction of adjacent identical token pairs in a generation.
fn repetition_rate(model: &LLM, generated: &str) -> f32 {
    let tokens = model.tokenize(generated);
    let repeats = tokens
        .windows(2)
        .filter(|window| window[0] == window[1])
        .count();
    repeats as f32 / tokens.len().saturating_sub(1).max(1) as f32
}

/// W5 mechanism: a logit-level anti-repetition penalty moves the argmax
/// deterministically -- the greedy loop rate drops at some grid cell.
#[test]
fn penalties_break_the_greedy_loop_at_some_cell() {
    set_seed(42);
    let mut model = build_micro_model(&STREAM);
    let examples: Vec<&str> = STREAM.to_vec();
    model.train_with_progress(examples, 300, 0.0005, false);

    // The unpenalized greedy stream must actually be repeating.
    let greedy = model.predict_scaled("the duck", 1.0);
    let greedy_rate = repetition_rate(&model, &greedy);
    assert!(
        greedy_rate > 0.05,
        "fixture must repeat, got {greedy_rate:.2}"
    );

    // The two-axis grid: at least one cell strictly reduces the rate.
    let mut best_cell = greedy_rate;
    for presence in [0.5, 1.0, 1.5, 2.0] {
        for repetition in [1.1, 1.2, 1.3] {
            let penalized = model.predict_penalized("the duck", 1.0, presence, repetition);
            let rate = repetition_rate(&model, &penalized);
            best_cell = best_cell.min(rate);
            assert_ne!(
                penalized, greedy,
                "penalties must move the argmax (presence {presence}, repetition {repetition})"
            );
        }
    }
    assert!(
        best_cell < greedy_rate,
        "some grid cell must beat greedy: {best_cell:.2} vs {greedy_rate:.2}"
    );
}

/// The penalties are off at their defaults: presence 0.0, repetition 1.0
/// reproduces the exact greedy stream (determinism of the pinned leg).
#[test]
fn default_penalties_reproduce_the_greedy_stream() {
    set_seed(42);
    let mut model = build_micro_model(&STREAM);
    let examples: Vec<&str> = STREAM.to_vec();
    model.train_with_progress(examples, 300, 0.0005, false);

    let greedy = model.predict_scaled("the duck", 1.0);
    let penalized = model.predict_penalized("the duck", 1.0, 0.0, 1.0);
    assert_eq!(greedy, penalized, "defaults must be identity");
}
