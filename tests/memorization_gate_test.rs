use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, Vocab, embeddings::Embeddings,
    output_projection::OutputProjection, set_seed, transformer::TransformerBlock,
};

/// One synthetic story, short enough that the greedy completion fits the
/// 80-token sequence budget with room to spare.
const STORY: &str = "Once a little duck swam across the pond . The duck met a frog and said hello . The frog \
     jumped into the water and swam away . </s>";

fn build_micro_model(story: &str) -> LLM {
    // Harvest the vocabulary from the single story, exactly as the micro
    // lane does from its corpus.
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(&[story.to_string()], &mut vocab_set);
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

/// W1 foundation gate: the training loop can memorize one story to
/// near-zero teacher-forced loss AND reproduce it greedily, repetition-free.
/// A mechanics bug (token/label alignment, optimizer, checkpoint fidelity)
/// would fail here, before any data/decode/regularization lever is trusted.
#[test]
fn one_story_overfits_and_reproduces_greedily() {
    set_seed(42);
    let mut model = build_micro_model(STORY);

    // The story must fit the sequence budget with a usable prefix.
    let story_tokens = model.tokenize(STORY);
    assert!(story_tokens.len() < 40, "story must fit the micro budget");
    let prefix_len = story_tokens.len() / 2;

    // Drive the teacher-forced loss to the floor on the single example.
    let losses = model.train_with_progress(vec![STORY], 200, 0.0005, false);
    let final_loss = *losses.last().unwrap();
    assert!(
        final_loss < 0.05,
        "one story must overfit below 0.05 teacher-forced loss, got {final_loss}"
    );

    // Greedily complete the story's prefix and compare token for token
    // against the held-back continuation (</s> excluded from both sides).
    let prefix: Vec<String> = story_tokens[..prefix_len]
        .iter()
        .map(|token| model.vocab.decode[token].clone())
        .collect();
    let generated = model.predict(&prefix.join(" "));
    let mut continuation: Vec<usize> = model.tokenize(&generated);
    if continuation.last() == model.vocab.encode("</s>").as_ref() {
        continuation.pop();
    }
    let mut expected: Vec<usize> = story_tokens[prefix_len..].to_vec();
    if expected.last() == model.vocab.encode("</s>").as_ref() {
        expected.pop();
    }
    assert!(
        !continuation.is_empty(),
        "greedy completion must not be empty"
    );

    // Reproduction: fraction of expected tokens matched in order.
    let matching = continuation
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a == b)
        .count();
    let reproduction = matching as f32 / expected.len() as f32;
    assert!(
        reproduction >= 0.9,
        "greedy completion must reproduce >= 0.9 of the story, got {reproduction:.3}"
    );

    // Repetition-free: no adjacent identical pair and no n-gram window
    // repeats anywhere in the completion.
    let adjacent_repeats = continuation
        .windows(2)
        .filter(|window| window[0] == window[1])
        .count();
    assert_eq!(adjacent_repeats, 0, "completion must be repetition-free");
    assert!(
        !model.is_degenerate(&generated),
        "completion must pass the degeneration gate"
    );
}
