use bincode::{config::standard, serde::{decode_from_slice, encode_to_vec}};
use serde::{Deserialize, Serialize};

use crate::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, Layer, Vocab, embeddings::Embeddings,
    output_projection::OutputProjection, seed, transformer::TransformerBlock,
};

/// Checkpoint format v1: magic header, then one bincode `CheckpointData`.
///
/// Trained weights only (no optimizer state, no transient caches):
/// loading resets optimizers, exactly as a fresh model would.
const MAGIC: &[u8] = b"RGPT_V1";

#[derive(Serialize, Deserialize)]
struct CheckpointData {
    seed: u64,
    vocab: Vec<String>,
    layers: Vec<(String, Vec<u8>)>,
}

/// Persist the model's trained weights to `path`.
pub fn save(llm: &LLM, path: &str) -> std::io::Result<()> {
    let mut layers = Vec::new();
    for layer in &llm.network {
        layers.push((
            layer.layer_type().to_string(),
            layer.parameter_bytes().map_err(std::io::Error::other)?,
        ));
    }

    let data = CheckpointData {
        seed: seed(),
        vocab: llm.vocab.words.clone(),
        layers,
    };

    let mut bytes = MAGIC.to_vec();
    bytes.extend(encode_to_vec(&data, standard()).map_err(std::io::Error::other)?);

    if let Some(parent) = std::path::Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)
}

/// Restore a model from a checkpoint saved with [`save`].
///
/// The vocab and layer shapes are rebuilt from the checkpoint itself; the
/// caller's current seed only shapes the discarded initial weights.
pub fn load(path: &str) -> std::io::Result<LLM> {
    let bytes = std::fs::read(path)?;
    if !bytes.starts_with(MAGIC) {
        return Err(std::io::Error::other("not a rustgpt checkpoint"));
    }

    let data: CheckpointData = decode_from_slice(&bytes[MAGIC.len()..], standard())
        .map_err(std::io::Error::other)?
        .0;

    let vocab_words_refs: Vec<&str> = data.vocab.iter().map(String::as_str).collect();
    let vocab = Vocab::new(vocab_words_refs);

    let mut network: Vec<Box<dyn Layer>> = vec![
        Box::new(Embeddings::new(vocab.clone())),
        Box::new(TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM)),
        Box::new(TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM)),
        Box::new(TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM)),
        Box::new(OutputProjection::new(EMBEDDING_DIM, vocab.words.len())),
    ];

    if network.len() != data.layers.len() {
        return Err(std::io::Error::other("checkpoint layer count mismatch"));
    }
    for (layer, (expected_type, payload)) in network.iter_mut().zip(&data.layers) {
        if layer.layer_type() != expected_type {
            return Err(std::io::Error::other(format!(
                "checkpoint layer type mismatch: expected {}, found {}",
                layer.layer_type(),
                expected_type
            )));
        }
        layer
            .load_parameter_bytes(payload)
            .map_err(std::io::Error::other)?;
    }

    Ok(LLM::new(vocab, network))
}
