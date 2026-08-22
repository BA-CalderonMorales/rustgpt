//! The decode-knob state machine (v0.0.8): seeded random command
//! sequences -- a valid set mutates exactly its own knob; an invalid one
//! mutates nothing. House style: hand-rolled xorshift generator, no
//! proptest.

use llm::{DecodeKnobs, Xorshift};

/// One operation in the generated sequence: knob choice plus a value that
/// may or may not be valid for it (the pool decides).
struct Op {
    knob: usize,
    value: f32,
}

fn generate_ops(seed: u64) -> Vec<Op> {
    let mut rng = Xorshift::new(seed);
    let valid_values: [&[f32]; 4] = [
        &[0.5, 0.7, 1.0, 1.3], // temperature > 0
        &[0.2, 0.8, 1.0],      // top-p in (0, 1]
        &[0.0, 0.5, 2.0],      // presence >= 0
        &[1.0, 1.1, 2.0],      // repetition >= 1
    ];
    let invalid_values: [&[f32]; 4] = [
        &[-1.0, 0.0],      // temperature <= 0
        &[-0.1, 0.0, 1.5], // top-p outside (0, 1]
        &[-2.0],           // presence < 0
        &[0.9, -3.0],      // repetition < 1
    ];
    (0..64)
        .map(|_| {
            let knob = rng.below(4) as usize;
            let valid = rng.below(2) == 0;
            let pool = if valid {
                valid_values[knob]
            } else {
                invalid_values[knob]
            };
            Op {
                knob,
                value: pool[rng.below(pool.len() as u64) as usize],
            }
        })
        .collect()
}

fn apply(knobs: &mut DecodeKnobs, op: &Op) -> Result<(), String> {
    match op.knob {
        0 => knobs.set_temperature(op.value),
        1 => knobs.set_top_p(op.value),
        2 => knobs.set_presence(op.value),
        _ => knobs.set_repetition(op.value),
    }
}

#[test]
fn valid_sets_move_only_their_own_knob_and_invalid_sets_move_nothing() {
    for seed in [1u64, 42, 2718] {
        let mut rng = Xorshift::new(seed);
        let mut knobs = DecodeKnobs::greedy();
        for op in generate_ops(seed ^ rng.next_u64()) {
            let before = knobs;
            let result = apply(&mut knobs, &op);

            // A rejected value leaves every knob untouched.
            if result.is_err() {
                assert_eq!(knobs, before, "invalid set must not mutate at all");
                continue;
            }

            // An accepted set moves exactly the targeted field.
            let expected = DecodeKnobs {
                temperature: if op.knob == 0 {
                    op.value
                } else {
                    before.temperature
                },
                top_p: if op.knob == 1 { op.value } else { before.top_p },
                presence: if op.knob == 2 {
                    op.value
                } else {
                    before.presence
                },
                repetition: if op.knob == 3 {
                    op.value
                } else {
                    before.repetition
                },
            };
            assert_eq!(knobs, expected);
        }
    }
}

#[test]
fn reset_restores_greedy_and_is_greedy_tracks_the_pin() {
    let mut knobs = DecodeKnobs::greedy();
    assert!(knobs.is_greedy());

    knobs.set_temperature(0.7).unwrap();
    knobs.set_top_p(0.8).unwrap();
    knobs.set_presence(1.5).unwrap();
    knobs.set_repetition(1.1).unwrap();
    assert!(!knobs.is_greedy());

    knobs.reset();
    assert!(knobs.is_greedy());
    assert_eq!(knobs, DecodeKnobs::greedy());
}
