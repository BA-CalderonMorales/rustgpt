use ndarray::Array2;

use crate::Vocab;

pub trait Layer {
    fn layer_type(&self) -> &str;

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;

    fn parameters(&self) -> usize;
}

#[allow(clippy::upper_case_acronyms)]
pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<Box<dyn Layer>>,
}

/// One scored answer against a reference: exact equality, prefix match, and
/// per-position token accuracy.
pub struct AnswerScore {
    pub exact: bool,
    pub prefix: bool,
    pub accuracy: f32,
}
