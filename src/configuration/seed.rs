use std::cell::Cell;

use rand::{rngs::StdRng, SeedableRng};

const DEFAULT_SEED: u64 = 42;

thread_local! {
    static CURRENT_SEED: Cell<u64> = const { Cell::new(DEFAULT_SEED) };
}

/// Set the seed used for every random initialization in the current thread.
///
/// Runs initialized with the same seed reproduce the same model and the same
/// scores. Evidence is a run plus its seed; observations are runs without one.
pub fn set_seed(seed: u64) {
    CURRENT_SEED.with(|cell| cell.set(seed));
}

/// The seed currently driving random initialization.
pub fn seed() -> u64 {
    CURRENT_SEED.with(|cell| cell.get())
}

/// A deterministic random source derived from the current seed.
pub(crate) fn random_source() -> StdRng {
    StdRng::seed_from_u64(seed())
}
