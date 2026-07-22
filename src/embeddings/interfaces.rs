use ndarray::Array2;

use crate::adam::Adam;

pub struct Embeddings {
    pub token_embeddings: Array2<f32>,
    pub positional_embeddings: Array2<f32>,
    pub cached_input: Option<Array2<f32>>,
    pub token_optimizer: Adam,
    pub positional_optimizer: Adam,
}
