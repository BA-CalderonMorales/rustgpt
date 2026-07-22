use crate::{feed_forward::FeedForward, layer_norm::LayerNorm, self_attention::SelfAttention};

pub struct TransformerBlock {
    pub(super) attention: SelfAttention,
    pub(super) feed_forward: FeedForward,
    pub(super) norm1: LayerNorm,
    pub(super) norm2: LayerNorm,
}
