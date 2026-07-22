use ndarray::Array2;

use crate::adam::Adam;

/// Chooses the update rule used by each feed-forward parameter tensor.
#[derive(Clone, Copy, Debug)]
pub enum OptimizerKind {
    Adam,
    Sgd,
    RmsProp,
}

pub(super) enum Optimizer {
    Adam(Adam),
    Sgd(Sgd),
    RmsProp(RmsProp),
}

pub struct Sgd;

pub struct RmsProp {
    pub(super) alpha: f32,
    pub(super) epsilon: f32,
    pub squared_gradients: Array2<f32>,
}

pub struct FeedForward {
    pub(super) w1: Array2<f32>,
    pub(super) b1: Array2<f32>,
    pub(super) w2: Array2<f32>,
    pub(super) b2: Array2<f32>,
    pub(super) input: Option<Array2<f32>>,
    pub(super) hidden_pre_activation: Option<Array2<f32>>,
    pub(super) hidden_post_activation: Option<Array2<f32>>,
    pub(super) optimizer_w1: Optimizer,
    pub(super) optimizer_b1: Optimizer,
    pub(super) optimizer_w2: Optimizer,
    pub(super) optimizer_b2: Optimizer,
}
