//! KV-cache decode (E4): byte-identical output to the recompute path, and
//! the throughput measurement on the tiny config. The correctness gate is
//! the identity assertion; the tokens/s ratio is reported evidence, not a
//! CI gate (measured on the same laptop, same seed, same binary).

use std::time::Instant;

use llm::{Config, LLM, Vocab, set_seed};

fn synthetic_stories() -> Vec<String> {
    let words: Vec<&str> = "the sun was warm and bright lily played in the garden with her red ball \
                            she saw a small bird on the fence the bird sang a happy song lily smiled \
                            and ran to show her mother the bird flew away into the blue sky"
        .split_whitespace()
        .collect();
    (0..40)
        .map(|i| {
            (0..100)
                .map(|j| words[(i * 7 + j * 3) % words.len()])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

fn build_tiny() -> LLM {
    let texts = synthetic_stories();
    let config = Config::tiny();
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(&texts, &mut vocab_set);
    let mut words: Vec<String> = vocab_set.into_iter().collect();
    words.sort();
    let refs: Vec<&str> = words.iter().map(String::as_str).collect();
    let vocab = Vocab::new(refs);
    let mut network: Vec<Box<dyn llm::Layer>> = vec![Box::new(llm::Embeddings::with_dims(
        vocab.clone(),
        config.embedding_dim,
        config.max_seq_len,
    ))];
    for _ in 0..config.block_count {
        network.push(Box::new(llm::transformer::TransformerBlock::new(
            config.embedding_dim,
            config.hidden_dim,
        )));
    }
    network.push(Box::new(llm::output_projection::OutputProjection::new(
        config.embedding_dim,
        vocab.words.len(),
    )));
    LLM::with_config(vocab, network, config)
}

#[test]
fn cached_decode_is_byte_identical_to_recompute() {
    set_seed(42);
    let mut model = build_tiny();
    let prompts = [
        "the sun was warm",
        "lily saw a small bird",
        "the bird sang",
        "once upon a time",
    ];
    for prompt in prompts {
        let recomputed = model.predict(prompt);
        let cached = model.predict_cached(prompt);
        assert_eq!(
            cached, recomputed,
            "cached decode must be byte-identical for {prompt:?}"
        );
    }
}

#[test]
fn tiny_lane_throughput_ratio_is_reported() {
    // The claim is measured on the real trained artifact when it exists
    // (a random tiny model stops at </s> immediately, which measures only
    // prefill); without the artifact the run reports the random-model
    // numbers as an observation, not evidence.
    let checkpoint = "models/tinystories/ts-13m-s42.bin";
    let Some(mut model) = std::path::Path::new(checkpoint)
        .exists()
        .then(|| llm::load(checkpoint).expect("checkpoint loads"))
    else {
        eprintln!("throughput observation skipped: {checkpoint} not present");
        return;
    };
    let prompt = "Once upon a time, Lily";

    let recompute_start = Instant::now();
    let recomputed = model.predict(prompt);
    let recompute_elapsed = recompute_start.elapsed();

    let cached_start = Instant::now();
    let cached = model.predict_cached(prompt);
    let cached_elapsed = cached_start.elapsed();

    assert_eq!(cached, recomputed, "cached and recompute must be identical");
    let tokens = cached.split_whitespace().count().max(1);
    let recompute_tps = tokens as f64 / recompute_elapsed.as_secs_f64().max(1e-9);
    let cached_tps = tokens as f64 / cached_elapsed.as_secs_f64().max(1e-9);
    println!(
        "tiny decode throughput (seed 42, artifact {checkpoint}, same laptop): recompute \
         {:.1} tok/s ({recompute_elapsed:?}), cached {:.1} tok/s ({cached_elapsed:?}), \
         speedup {:.2}x, output len {tokens} tokens",
        recompute_tps,
        cached_tps,
        cached_tps / recompute_tps,
    );
}
