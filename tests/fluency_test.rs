use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, Vocab, embeddings::Embeddings,
    output_projection::OutputProjection, set_seed, transformer::TransformerBlock,
};

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

/// The W3 yardstick must separate the two regimes the boolean gate
/// confuses: a collapsed sample ("the the the ...") and non-repeating
/// gibberish are both failed, but a diverse multi-sentence sample is not.
#[test]
fn fluency_score_separates_collapsed_from_fluent_samples() {
    set_seed(42);
    let model = build_micro_model(&["the duck and the frog . </s>"]);
    let fluent = vec![
        "the duck swam . the frog jumped . the pond dried up . the sun rose . </s>".to_string(),
    ];
    let vocab_model = build_micro_model(&[
        "the duck and the frog . </s>",
        "the duck swam . the frog jumped . the pond dried up . the sun rose . </s>",
    ]);

    // A collapsed sample: one token repeated, no sentence structure.
    let collapsed = vec!["the the the the the the the the the the the </s>".to_string()];
    let collapsed_score = model.fluency_score(&collapsed);
    assert_eq!(collapsed_score.repetition_free_rate, 0.0);
    assert!(
        collapsed_score.distinct_1 < 0.2,
        "distinct-1 must flag collapse"
    );
    assert!(collapsed_score.completion_sentences < 1.0);

    // A fluent sample: many distinct tokens, no adjacent repeats, several
    // sentence-final punctuation marks (vocab covers every word).
    let fluent_score = vocab_model.fluency_score(&fluent);
    assert_eq!(fluent_score.repetition_free_rate, 1.0);
    assert!(
        fluent_score.distinct_1 > 0.6,
        "distinct-1 must pass fluent text"
    );
    assert!(fluent_score.completion_sentences >= 4.0);
    assert!(fluent_score.mean_completion_len > 10.0);

    // The yardstick must be strictly ordered: fluent beats collapsed.
    assert!(fluent_score.distinct_1 > collapsed_score.distinct_1);
    assert!(fluent_score.distinct_2 > collapsed_score.distinct_2);
}

/// Distinct-2 catches what distinct-1 misses: a sample that reuses the
/// same words but never repeats a bigram is lexically poorer than one
/// that varies both.
#[test]
fn distinct_2_orders_varied_bigrams_higher() {
    set_seed(42);
    let model = build_micro_model(&["the duck and the frog . </s>"]);

    // Same word set, different bigram structure.
    let fixed = vec!["the duck the duck the duck the duck the duck the duck </s>".to_string()];
    let varied =
        vec!["the duck swam the frog the pond dried the duck met the frog </s>".to_string()];
    let fixed_score = model.fluency_score(&fixed);
    let varied_score = model.fluency_score(&varied);
    assert!(
        varied_score.distinct_2 > fixed_score.distinct_2,
        "distinct-2 must separate repeated bigrams from varied ones"
    );
}
