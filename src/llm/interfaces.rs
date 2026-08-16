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

/// Per-position logit-regime statistics, averaged over a whole token
/// stream. The collapse gate is a boolean terminal readout; these means
/// make the attractor's formation visible as a trajectory before repetition
/// saturates to 1.0.
pub struct LogitProfile {
    /// Mean softmax probability of the top-1 token minus the top-2 token:
    /// a widening margin is the frequency-head attractor sharpening.
    pub top1_margin: f32,
    /// Mean raw-logit gap between the two most likely tokens.
    pub top2_gap: f32,
    /// Mean L2 norm of the raw logits row.
    pub logit_norm: f32,
    /// Mean softmax output entropy over the vocabulary.
    pub entropy: f32,
}

/// The decode-quality yardstick over N generated samples. Repetition-free
/// is necessary but not sufficient (a model can emit non-repeating
/// gibberish); distinct-n measures lexical diversity and the completion
/// probe counts sentence-final punctuation as the multi-sentence signal.
pub struct FluencyScore {
    /// Mean unique-token ratio per sample: distinct-1.
    pub distinct_1: f32,
    /// Mean unique-bigram ratio per sample: distinct-2.
    pub distinct_2: f32,
    /// Fraction of samples with no adjacent identical pair.
    pub repetition_free_rate: f32,
    /// Mean count of sentence-final punctuation tokens per sample.
    pub completion_sentences: f32,
    /// Mean non-</s> token count per sample.
    pub mean_completion_len: f32,
}
