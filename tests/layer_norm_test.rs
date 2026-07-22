use llm::{Layer, layer_norm::LayerNorm};
use ndarray::{Axis, array};

#[test]
fn layer_norm_forward_and_backward_preserve_shape_and_finite_values() {
    let mut layer_norm = LayerNorm::new(3);
    let input = array![[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]];

    let output = layer_norm.forward(&input);

    assert_eq!(output.dim(), input.dim());
    for mean in output.mean_axis(Axis(1)).expect("rows should have means") {
        assert!(mean.abs() < 1e-5);
    }
    assert!(output.iter().all(|value| value.is_finite()));

    let upstream_gradients = array![[0.5, -0.25, 0.75], [-0.5, 1.0, 0.25]];
    let input_gradients = layer_norm.backward(&upstream_gradients, 0.0);

    assert_eq!(input_gradients.dim(), input.dim());
    assert!(input_gradients.iter().all(|value| value.is_finite()));
    assert_eq!(layer_norm.parameters(), 6);
}
