use ndarray::Array2;

pub struct Adam {
    pub(super) beta1: f32,
    pub(super) beta2: f32,
    pub(super) epsilon: f32,
    pub(super) timestep: usize,
    pub m: Array2<f32>,
    pub v: Array2<f32>,
}
