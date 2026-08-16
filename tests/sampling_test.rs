use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, Vocab, Xorshift, embeddings::Embeddings,
    output_projection::OutputProjection, set_seed, transformer::TransformerBlock,
};

const TRAINING_STREAM: [&str; 3] = [
    "the duck the frog the pond the duck swam away . </s>",
    "the frog the duck the pond the frog jumped away . </s>",
    "the pond the duck the frog the pond dried up . </s>",
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

/// The first generated token of a weighted decode (the prompt token is
/// not part of the output string).
fn first_generated_token(model: &mut LLM, output: &str) -> usize {
    model.tokenize(output)[0]
}

/// W4 mechanism: T=1 probability-weighted sampling with a fixed seed is
/// byte-reproducible -- one seed, one token stream.
#[test]
fn weighted_sampling_reproduces_from_a_fixed_seed() {
    set_seed(42);
    let mut first = build_micro_model(&TRAINING_STREAM);
    set_seed(42);
    let mut second = build_micro_model(&TRAINING_STREAM);

    let mut rng_a = Xorshift::new(42);
    let mut rng_b = Xorshift::new(42);
    let stream_a = first.predict_weighted("the duck", 1.0, &mut rng_a);
    let stream_b = second.predict_weighted("the duck", 1.0, &mut rng_b);
    assert_eq!(
        stream_a, stream_b,
        "one seed must reproduce one token stream"
    );
}

/// W4 discriminator vs the falsified uniform top-k: probability-weighted
/// draws follow the model's own distribution, so over a batch of draws
/// the first generated token reaches beyond the top-2 ranks; the uniform
/// k=2 sampler cannot leave {rank-1, rank-2}.
#[test]
fn weighted_draws_reach_beyond_the_top_two_ranks() {
    set_seed(42);
    let mut model = build_micro_model(&TRAINING_STREAM);

    // Weighted draws over a shared seeded rng: first generated token per
    // sample, distinct across the batch.
    let mut weighted_rng = Xorshift::new(42);
    let mut weighted_tokens = std::collections::HashSet::new();
    for _ in 0..60 {
        let output = model.predict_weighted("the duck", 1.0, &mut weighted_rng);
        weighted_tokens.insert(first_generated_token(&mut model, &output));
    }
    assert!(
        weighted_tokens.len() >= 3,
        "weighted draws must reach rank-3+ tokens, saw {} distinct",
        weighted_tokens.len()
    );

    // Uniform k=2 over the same batch size: confined to two ranks.
    let mut uniform_rng = Xorshift::new(42);
    let mut uniform_tokens = std::collections::HashSet::new();
    for _ in 0..60 {
        let output = model.predict_sampled("the duck", 2, &mut uniform_rng);
        uniform_tokens.insert(first_generated_token(&mut model, &output));
    }
    assert!(
        uniform_tokens.len() <= 2,
        "uniform top-2 must stay inside two ranks, saw {}",
        uniform_tokens.len()
    );
}
