//! Decode-capture parity (v0.0.8): the captured stream from
//! `generate_with_steps` must be token-identical to the uncached
//! `predict_*` family at the same config, and greedy argmax must stay
//! temperature-invariant over a seeded prompt pool. House style: seeded
//! xorshift generators, no proptest.

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
    let network: Vec<Box<dyn llm::Layer>> = vec![
        Box::new(Embeddings::new(vocab.clone())),
        Box::new(TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM)),
        Box::new(TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM)),
        Box::new(TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM)),
        Box::new(OutputProjection::new(EMBEDDING_DIM, vocab.words.len())),
    ];
    LLM::new(vocab, network)
}

/// The captured token ids of one generation, prompt tokens excluded.
fn captured_tokens(steps: &[llm::DecodeStep]) -> Vec<usize> {
    steps.iter().map(|step| step.token).collect()
}

#[test]
fn captured_greedy_stream_matches_the_recompute_path() {
    set_seed(42);
    let mut model = build_micro_model(&TRAINING_STREAM);
    let prompt = "the duck";

    let recomputed = model.predict(prompt);
    let mut rng = Xorshift::new(42);
    let (output, steps) = model.generate_with_steps(prompt, 1.0, 0.0, 0.0, 1.0, &mut rng);

    assert_eq!(
        output, recomputed,
        "greedy capture must not move the stream"
    );
    assert_eq!(
        captured_tokens(&steps),
        model.tokenize(&recomputed),
        "captured tokens must equal the recompute tokens"
    );
    assert!(
        steps.iter().all(|step| (0.0..=1.0).contains(&step.prob)),
        "every captured probability is a softmax mass"
    );
}

#[test]
fn captured_sampled_streams_match_the_predict_family() {
    set_seed(42);
    let mut model = build_micro_model(&TRAINING_STREAM);
    let prompt = "the duck";

    // Weighted sampling: captured stream == predict_weighted at the same
    // seed.
    let mut rng_a = Xorshift::new(7);
    let weighted = model.predict_weighted(prompt, 0.8, &mut rng_a);
    let mut rng_b = Xorshift::new(7);
    let (output, steps) = model.generate_with_steps(prompt, 0.8, 0.0, 0.0, 1.0, &mut rng_b);
    assert_eq!(output, weighted);
    assert_eq!(captured_tokens(&steps), model.tokenize(&weighted));

    // Nucleus + penalties (the Qwen stack): captured == predict_nucleus_penalized.
    let mut rng_c = Xorshift::new(9);
    let stacked = model.predict_nucleus_penalized(prompt, 0.7, 0.8, 1.5, 1.1, &mut rng_c);
    let mut rng_d = Xorshift::new(9);
    let (output, steps) = model.generate_with_steps(prompt, 0.7, 0.8, 1.5, 1.1, &mut rng_d);
    assert_eq!(output, stacked);
    assert_eq!(captured_tokens(&steps), model.tokenize(&stacked));

    // Penalized greedy at T=1: captured == predict_penalized (no rng).
    let penalized_greedy = model.predict_penalized(prompt, 1.0, 1.5, 1.1);
    let mut rng_unused = Xorshift::new(0);
    let (output, steps) = model.generate_with_steps(prompt, 1.0, 0.0, 1.5, 1.1, &mut rng_unused);
    assert_eq!(output, penalized_greedy);
    assert_eq!(captured_tokens(&steps), model.tokenize(&penalized_greedy));
}

#[test]
fn greedy_argmax_is_temperature_invariant_over_a_seeded_pool() {
    set_seed(42);
    let mut model = build_micro_model(&TRAINING_STREAM);
    let prompts = [
        "the duck",
        "the frog",
        "the pond",
        "the duck the frog",
        "the sun",
    ];
    for prompt in prompts {
        let greedy = model.predict(prompt);
        for temperature in [0.7, 0.9, 1.0, 1.3] {
            assert_eq!(
                model.predict_scaled(prompt, temperature),
                greedy,
                "argmax must be invariant at T={temperature} for {prompt:?}"
            );
        }
    }
}

#[test]
fn generate_matches_generate_with_steps_token_for_token() {
    set_seed(42);
    let mut model = build_micro_model(&TRAINING_STREAM);
    let prompt = "the pond";

    let mut rng_a = Xorshift::new(11);
    let plain = model.generate(prompt, 1.2, 0.0, 0.0, 1.0, &mut rng_a);
    let mut rng_b = Xorshift::new(11);
    let (with_steps, steps) = model.generate_with_steps(prompt, 1.2, 0.0, 0.0, 1.0, &mut rng_b);

    assert_eq!(plain, with_steps);
    assert_eq!(captured_tokens(&steps), model.tokenize(&plain));
}
