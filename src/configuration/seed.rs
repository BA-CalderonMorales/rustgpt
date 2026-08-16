use std::cell::Cell;

use rand::{SeedableRng, rngs::StdRng};

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

/// A hand-rolled xorshift64* PRNG: the decode-time sampling probe and the
/// output-property suite's prompt generator ride the same source, so every
/// sampled run and every property draw reproduces exactly from one seed.
#[derive(Debug, Clone, Copy)]
pub struct Xorshift {
    state: u64,
}

impl Xorshift {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    /// Next 64-bit draw; xorshift64* core with the standard 0x2545F4914F6CDD1D
    /// multiplier.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_4914_F4CD_DD1D)
    }

    /// Uniform draw in `0..bound` (rejection-free modulo).
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}
