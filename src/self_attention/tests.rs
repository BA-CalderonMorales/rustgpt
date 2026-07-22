use ndarray::{Array2, array};

use super::SelfAttention;
use crate::llm::Layer;

fn scalar_loss(
    attention: &mut SelfAttention,
    input: &Array2<f32>,
    upstream_gradient: &Array2<f32>,
) -> f32 {
    attention
        .forward(input)
        .iter()
        .zip(upstream_gradient.iter())
        .map(|(output, gradient)| output * gradient)
        .sum()
}

#[test]
fn backward_input_gradient_matches_finite_difference() {
    let mut attention = SelfAttention::new(3);
    attention.w_q = array![[0.2, -0.3, 0.4], [0.5, 0.1, -0.2], [-0.4, 0.3, 0.6]];
    attention.w_k = array![[-0.1, 0.4, 0.2], [0.3, -0.5, 0.6], [0.2, 0.1, -0.3]];
    attention.w_v = array![[0.6, -0.2, 0.1], [-0.3, 0.5, 0.4], [0.2, -0.4, 0.3]];

    let input = array![[0.7, -0.2, 0.5], [0.1, 0.8, -0.4], [-0.6, 0.3, 0.9]];
    let upstream_gradient = array![[0.3, -0.7, 0.2], [-0.5, 0.4, 0.6], [0.8, -0.1, -0.3]];

    attention.forward(&input);
    let analytical_gradient = attention.backward(&upstream_gradient, 0.0);

    let epsilon = 1e-3;
    let mut numerical_gradient = Array2::zeros(input.dim());
    for row in 0..input.nrows() {
        for column in 0..input.ncols() {
            let mut input_plus = input.clone();
            input_plus[[row, column]] += epsilon;
            let loss_plus = scalar_loss(&mut attention, &input_plus, &upstream_gradient);

            let mut input_minus = input.clone();
            input_minus[[row, column]] -= epsilon;
            let loss_minus = scalar_loss(&mut attention, &input_minus, &upstream_gradient);

            numerical_gradient[[row, column]] = (loss_plus - loss_minus) / (2.0 * epsilon);
        }
    }

    for ((row, column), analytical) in analytical_gradient.indexed_iter() {
        let numerical = numerical_gradient[[row, column]];
        assert!(
            (analytical - numerical).abs() < 2e-3,
            "input gradient mismatch at [{row}, {column}]: analytical={analytical}, numerical={numerical}"
        );
    }
}
