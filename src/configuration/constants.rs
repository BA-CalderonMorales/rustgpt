pub const MAX_SEQ_LEN: usize = 80;
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;

/// Size and shape of a model. Named presets are the supported lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub max_seq_len: usize,
    pub block_count: usize,
}

impl Config {
    /// The curated water-cycle micro model (380K parameters).
    pub const fn micro() -> Config {
        Config {
            embedding_dim: EMBEDDING_DIM,
            hidden_dim: HIDDEN_DIM,
            max_seq_len: MAX_SEQ_LEN,
            block_count: 3,
        }
    }

    /// The laptop model lane (~13M parameters): real-corpus stories.
    pub const fn tiny() -> Config {
        Config {
            embedding_dim: 384,
            hidden_dim: 768,
            max_seq_len: 128,
            block_count: 6,
        }
    }
}
