mod constants;
mod seed;

pub use constants::{Config, EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN};
pub(crate) use seed::random_source;
pub use seed::{seed, set_seed};
