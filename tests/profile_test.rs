use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, LogitProfile, Vocab, embeddings::Embeddings,
    output_projection::OutputProjection, set_seed, transformer::TransformerBlock,
};

/// A short training stream with the frequency structure the collapse gate
/// trips on: several statements that all share the same most-common word.
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

fn assert_sane(profile: &LogitProfile) {
    assert!(profile.top1_margin.is_finite() && (0.0..=1.0).contains(&profile.top1_margin));
    assert!(profile.top2_gap.is_finite() && profile.top2_gap > 0.0);
    assert!(profile.logit_norm.is_finite() && profile.logit_norm > 0.0);
    assert!(profile.entropy.is_finite() && profile.entropy > 0.0);
}

/// W2 instrument sanity: the four profile quantities are computable over a
/// token stream and live in their mathematical bounds.
#[test]
fn profile_quantities_are_bounded() {
    set_seed(42);
    let mut model = build_micro_model(&TRAINING_STREAM);
    let profile = model.eval_profile(&TRAINING_STREAM);
    assert_sane(&profile);
}

/// W2 mechanism: an overfitting model sharpens its logit regime -- the
/// top-1 margin widens and the output entropy collapses across epochs.
/// This is the exact signature the boolean collapse gate cannot see.
#[test]
fn profile_foreshadows_memorization() {
    set_seed(42);
    let mut model = build_micro_model(&TRAINING_STREAM);
    let before = model.eval_profile(&TRAINING_STREAM);

    // A memorizing run on the same stream, far short of the W1 audit.
    let examples: Vec<&str> = TRAINING_STREAM.to_vec();
    model.train_with_progress(examples, 30, 0.0005, false);
    let after = model.eval_profile(&TRAINING_STREAM);

    // The attractor announces itself as a widening margin and collapsing
    // entropy; both must move monotonically toward the corner.
    assert!(
        after.top1_margin > before.top1_margin,
        "margin must rise with memorization: {:.3} -> {:.3}",
        before.top1_margin,
        after.top1_margin
    );
    assert!(
        after.entropy < before.entropy,
        "entropy must fall with memorization: {:.3} -> {:.3}",
        before.entropy,
        after.entropy
    );
    assert!(after.top2_gap > before.top2_gap);
    assert_sane(&after);
}

/// The machine-JSON block: four epoch-aligned arrays, one entry per epoch.
#[test]
fn profile_json_emits_four_epoch_aligned_arrays() {
    let profiles = vec![
        LogitProfile {
            top1_margin: 0.1,
            top2_gap: 0.5,
            logit_norm: 3.0,
            entropy: 4.0,
        },
        LogitProfile {
            top1_margin: 0.9,
            top2_gap: 4.5,
            logit_norm: 9.0,
            entropy: 0.5,
        },
    ];
    let value = llm::profile_json(&profiles);
    let object = value.as_object().expect("profile block must be an object");
    for (key, expected) in [
        ("top1_margin", vec![0.1, 0.9]),
        ("top2_gap", vec![0.5, 4.5]),
        ("logit_norm", vec![3.0, 9.0]),
        ("entropy", vec![4.0, 0.5]),
    ] {
        let array = object[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} must be an array"));
        let floats: Vec<f32> = array.iter().map(|v| v.as_f64().unwrap() as f32).collect();
        assert_eq!(floats, expected, "{key} must carry one entry per epoch");
    }
}
