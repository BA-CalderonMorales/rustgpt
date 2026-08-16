use ndarray::Array2;

use crate::adam::Adam;

pub struct SelfAttention {
    pub embedding_dim: usize,
    pub(super) w_q: Array2<f32>,
    pub(super) w_k: Array2<f32>,
    pub(super) w_v: Array2<f32>,
    pub(super) cached_input: Option<Array2<f32>>,
    pub(super) optimizer_w_q: Adam,
    pub(super) optimizer_w_k: Adam,
    pub(super) optimizer_w_v: Adam,
    /// Decode-time K/V cache: preallocated per-position keys and values per
    /// block plus the filled-row count. Never part of a checkpoint.
    pub(super) kv_cache: Option<(Array2<f32>, Array2<f32>, usize)>,
    pub(super) step_mode: bool,
}
