use bincode::{
    config::standard,
    error::{DecodeError, EncodeError},
    serde::{decode_from_slice, encode_to_vec},
};
use ndarray::{Array2, Axis};

use super::LayerNorm;
use crate::{adam::Adam, llm::Layer};

impl LayerNorm {
    /// Initialize LayerNorm with learnable parameters
    pub fn new(embedding_dim: usize) -> Self {
        LayerNorm {
            epsilon: 1e-5,
            gamma: Array2::ones((1, embedding_dim)), // Initialize gamma to 1
            beta: Array2::zeros((1, embedding_dim)), // Initialize beta to 0
            cached_input: None,
            cached_mean: None,
            cached_std: None,
            optimizer_gamma: Adam::new((1, embedding_dim)),
            optimizer_beta: Adam::new((1, embedding_dim)),
        }
    }

    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Per-token mean and standard deviation.
        let mean = input.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let std = input.std_axis(Axis(1), 0.0).insert_axis(Axis(1));

        // Cache the input and statistics for the backward pass.
        self.cached_input = Some(input.clone());
        self.cached_mean = Some(mean.clone());
        self.cached_std = Some(std.clone());

        // Normalize, then scale and shift by the learnable parameters.
        let normalized = (input - &mean) / (&std + self.epsilon);
        &self.gamma * &normalized + &self.beta
    }
}

impl Layer for LayerNorm {
    fn layer_type(&self) -> &str {
        "LayerNorm"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.normalize(input)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Restore the cached forward state.
        let input = self.cached_input.as_ref().unwrap();
        let mean = self.cached_mean.as_ref().unwrap();
        let std = self.cached_std.as_ref().unwrap();
        let normalized = (input - mean) / (std + self.epsilon);
        let n_features = input.shape()[1] as f32;

        // Gradients w.r.t. the learnable scale and shift.
        let grad_gamma = (&normalized * grads).sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_beta = grads.sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_normalized = &self.gamma * grads;

        // Full chain rule back through variance and mean.
        let grad_input = {
            let variance = std * std + self.epsilon;
            let grad_var = (&grad_normalized * &normalized)
                .sum_axis(Axis(1))
                .insert_axis(Axis(1))
                * (-0.5)
                / variance.mapv(|x| x * x.sqrt());
            let grad_mean = grad_normalized.sum_axis(Axis(1)).insert_axis(Axis(1)) * (-1.0)
                / (std + self.epsilon)
                + &grad_var * (input - mean).sum_axis(Axis(1)).insert_axis(Axis(1)) * (-2.0)
                    / n_features;

            &grad_normalized / (std + self.epsilon)
                + &grad_var * 2.0 * (input - mean) / n_features
                + &grad_mean / n_features
        };

        // Update the learnable parameters and propagate the gradient.
        self.optimizer_gamma.step(&mut self.gamma, &grad_gamma, lr);
        self.optimizer_beta.step(&mut self.beta, &grad_beta, lr);

        grad_input
    }

    fn parameters(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }

    fn parameter_bytes(&self) -> Result<Vec<u8>, EncodeError> {
        self.leaf_parameter_bytes()
    }

    fn load_parameter_bytes(&mut self, bytes: &[u8]) -> Result<(), DecodeError> {
        self.load_leaf_parameter_bytes(bytes)
    }
}

impl LayerNorm {
    pub(crate) fn leaf_parameter_bytes(&self) -> Result<Vec<u8>, EncodeError> {
        encode_to_vec((&self.gamma, &self.beta), standard())
    }

    pub(crate) fn load_leaf_parameter_bytes(&mut self, bytes: &[u8]) -> Result<(), DecodeError> {
        let (gamma, beta) =
            decode_from_slice::<(Array2<f32>, Array2<f32>), _>(bytes, standard())?.0;
        self.gamma = gamma;
        self.beta = beta;
        Ok(())
    }
}
