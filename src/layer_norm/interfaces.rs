use ndarray::Array2;

use crate::adam::Adam;

pub struct LayerNorm {
    pub(super) epsilon: f32,
    pub(super) gamma: Array2<f32>,
    pub(super) beta: Array2<f32>,
    pub(super) cached_input: Option<Array2<f32>>,
    pub(super) cached_mean: Option<Array2<f32>>,
    pub(super) cached_std: Option<Array2<f32>>,
    pub(super) optimizer_gamma: Adam,
    pub(super) optimizer_beta: Adam,
}
