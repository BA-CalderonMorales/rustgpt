use ndarray::Array2;

use crate::Vocab;

pub trait Layer {
    fn layer_type(&self) -> &str;

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;

    fn parameters(&self) -> usize;

    /// Encode the layer's learned weights for checkpointing.
    fn parameter_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError>;

    /// Restore learned weights from `parameter_bytes` output.
    fn load_parameter_bytes(&mut self, bytes: &[u8]) -> Result<(), bincode::error::DecodeError>;

    /// Turn per-position decode caching on or off. The default is a
    /// stateless pass-through; embeddings and attention hold the cache
    /// state. Outputs are byte-identical to the uncached forward path.
    fn set_cache_mode(&mut self, _active: bool) {}
}

#[allow(clippy::upper_case_acronyms)]
pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<Box<dyn Layer>>,
    pub max_seq_len: usize,
    pub config: crate::configuration::Config,
}

/// One scored answer against a reference: exact equality, prefix match, and
/// per-position token accuracy.
pub struct AnswerScore {
    pub exact: bool,
    pub prefix: bool,
    pub accuracy: f32,
}

/// One decoded step: the greedily chosen token id and the probability the
/// output softmax assigned to it. Prompt tokens are not steps.
pub struct DecodeStep {
    pub token: usize,
    pub prob: f32,
}
