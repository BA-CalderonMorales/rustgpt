use llm::{
    Layer,
    adam::Adam,
    feed_forward::{FeedForward, OptimizerKind, RmsProp, Sgd},
};
use ndarray::Array2;

#[test]
fn sgd_uses_the_supplied_learning_rate() {
    let mut optimizer = Sgd;
    let mut parameters = Array2::ones((2, 2));
    let gradients = Array2::from_elem((2, 2), 2.0);

    optimizer.step(&mut parameters, &gradients, 0.25);

    assert_eq!(parameters, Array2::from_elem((2, 2), 0.5));
}

#[test]
fn rms_prop_retains_squared_gradient_history() {
    let mut optimizer = RmsProp::new((1, 1));
    let mut parameters = Array2::ones((1, 1));
    let gradients = Array2::from_elem((1, 1), 2.0);

    optimizer.step(&mut parameters, &gradients, 0.1);
    let first_update = 1.0 - parameters[[0, 0]];
    optimizer.step(&mut parameters, &gradients, 0.1);
    let second_update = 1.0 - first_update - parameters[[0, 0]];

    assert!((optimizer.squared_gradients[[0, 0]] - 0.76).abs() < 1e-6);
    assert!(second_update < first_update);
}

#[test]
fn adam_updates_moments_and_is_deterministic() {
    let gradients = Array2::from_elem((2, 2), 2.0);
    let mut first = Adam::new((2, 2));
    let mut second = Adam::new((2, 2));
    let mut first_parameters = Array2::ones((2, 2));
    let mut second_parameters = Array2::ones((2, 2));

    first.step(&mut first_parameters, &gradients, 0.01);
    second.step(&mut second_parameters, &gradients, 0.01);

    assert_eq!(first_parameters, second_parameters);
    assert!(first.m.iter().all(|moment| (*moment - 0.2).abs() < 1e-6));
    assert!(
        first
            .v
            .iter()
            .all(|squared_moment| (*squared_moment - 0.004).abs() < 1e-6)
    );
    assert!(first_parameters.iter().all(|parameter| *parameter < 1.0));
}

#[test]
fn feed_forward_supports_each_optimizer_kind() {
    let input = Array2::ones((2, 4));
    let gradients = Array2::ones((2, 4));

    for optimizer_kind in [
        OptimizerKind::Adam,
        OptimizerKind::Sgd,
        OptimizerKind::RmsProp,
    ] {
        let mut layer = FeedForward::new_with_optimizer(4, 8, optimizer_kind);
        let output = layer.forward(&input);
        let input_gradients = layer.backward(&gradients, 0.01);

        assert_eq!(output.dim(), input.dim());
        assert_eq!(input_gradients.dim(), input.dim());
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(input_gradients.iter().all(|value| value.is_finite()));
    }
}
