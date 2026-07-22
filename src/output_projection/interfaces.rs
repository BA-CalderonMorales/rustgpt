use ndarray::Array2;

use crate::adam::Adam;

pub struct OutputProjection {
    pub w_out: Array2<f32>,
    pub b_out: Array2<f32>,
    pub optimizer: Adam,
    pub cached_input: Option<Array2<f32>>,
}
