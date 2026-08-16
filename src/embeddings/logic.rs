use bincode::{
    config::standard,
    error::{DecodeError, EncodeError},
    serde::{decode_from_slice, encode_to_vec},
};
use ndarray::{Array2, s};
use rand_distr::{Distribution, Normal};

use super::Embeddings;
use crate::{EMBEDDING_DIM, MAX_SEQ_LEN, adam::Adam, llm::Layer, vocab::Vocab};

impl Default for Embeddings {
    fn default() -> Self {
        Self {
            token_embeddings: Self::init_embeddings(Vocab::default_words().len(), EMBEDDING_DIM),
            positional_embeddings: Self::init_positional_embeddings(MAX_SEQ_LEN, EMBEDDING_DIM),
            cached_input: None,
            token_optimizer: Adam::new((Vocab::default_words().len(), EMBEDDING_DIM)),
            positional_optimizer: Adam::new((MAX_SEQ_LEN, EMBEDDING_DIM)),
            step_mode: false,
            position: 0,
        }
    }
}

impl Embeddings {
    pub fn new(vocab: Vocab) -> Self {
        Self::with_dims(vocab, EMBEDDING_DIM, MAX_SEQ_LEN)
    }

    pub fn with_dims(vocab: Vocab, embedding_dim: usize, max_seq_len: usize) -> Self {
        Self {
            token_embeddings: Self::init_embeddings(vocab.words.len(), embedding_dim),
            positional_embeddings: Self::init_positional_embeddings(max_seq_len, embedding_dim),
            cached_input: None,
            token_optimizer: Adam::new((vocab.words.len(), embedding_dim)),
            positional_optimizer: Adam::new((max_seq_len, embedding_dim)),
            step_mode: false,
            position: 0,
        }
    }

    fn init_embeddings(vocab_size: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = crate::configuration::random_source();
        let normal = Normal::new(0.0, 0.02).unwrap(); // Increased for better learning
        Array2::from_shape_fn((vocab_size, embedding_dim), |_| normal.sample(&mut rng))
    }

    fn init_positional_embeddings(max_seq_len: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = crate::configuration::random_source();
        let normal = Normal::new(0.0, 0.02).unwrap(); // Increased for better learning
        Array2::from_shape_fn((max_seq_len, embedding_dim), |_| normal.sample(&mut rng))
    }

    fn get_token_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let mut token_embeds = Array2::<f32>::zeros((token_ids.len(), embeddings.ncols()));
        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= embeddings.nrows() {
                panic!(
                    "Token ID {} out of bounds for vocab size {}",
                    token_id,
                    embeddings.nrows()
                );
            }
            token_embeds.row_mut(i).assign(&embeddings.row(token_id));
        }
        token_embeds
    }

    fn get_positional_embeddings(
        positional_encodings: &Array2<f32>,
        seq_len: usize,
    ) -> Array2<f32> {
        if seq_len > positional_encodings.nrows() {
            panic!(
                "Sequence length {} exceeds maximum {}",
                seq_len,
                positional_encodings.nrows()
            );
        }
        positional_encodings.slice(s![0..seq_len, ..]).to_owned()
    }

    pub fn embed_tokens(&self, token_ids: &[usize]) -> Array2<f32> {
        let token_embeds = Self::get_token_embeddings(&self.token_embeddings, token_ids);
        let position_embeds =
            Self::get_positional_embeddings(&self.positional_embeddings, token_ids.len());
        token_embeds + position_embeds // Element-wise sum
    }
}

impl Layer for Embeddings {
    fn layer_type(&self) -> &str {
        "Embeddings"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Cache the input; input shape is [1, sequence_length].
        self.cached_input = Some(input.clone());
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();

        // Full-sequence path: position becomes the next decode step's
        // index (the number of tokens embedded so far).
        if !self.step_mode || input.ncols() > 1 {
            self.position = token_ids.len();
            return self.embed_tokens(&token_ids);
        }

        // Decode step: exactly one new token at the recorded position,
        // byte-identical to the full-sequence embedding of that row.
        let token_embed = self.token_embeddings.row(token_ids[0]).to_owned();
        let position_embed = self.positional_embeddings.row(self.position).to_owned();
        self.position += 1;
        (token_embed + position_embed).insert_axis(ndarray::Axis(0))
    }

    fn set_cache_mode(&mut self, active: bool) {
        self.step_mode = active;
        if active {
            self.position = 0;
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Restore the cached input; grads shape is (sequence_length, embedding_dim).
        let input = self.cached_input.as_ref().unwrap();
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        let grads = grads.view();

        // Scatter each position's gradient into its token and position rows.
        let mut token_grads = Array2::zeros(self.token_embeddings.dim());
        let mut positional_grads = Array2::zeros(self.positional_embeddings.dim());
        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= self.token_embeddings.nrows() {
                panic!(
                    "Token ID {} out of bounds for vocab size {}",
                    token_id,
                    self.token_embeddings.nrows()
                );
            }
            let grad_row = grads.row(i);
            let mut token_row = token_grads.row_mut(token_id);
            token_row += &grad_row;
            let mut pos_row = positional_grads.row_mut(i);
            pos_row += &grad_row;
        }

        // Update both embedding tables through their optimizers.
        self.token_optimizer
            .step(&mut self.token_embeddings, &token_grads, lr);
        self.positional_optimizer
            .step(&mut self.positional_embeddings, &positional_grads, lr);

        // The gradient passes through the lookup unchanged.
        grads.to_owned()
    }

    fn parameters(&self) -> usize {
        self.token_embeddings.len() + self.positional_embeddings.len()
    }

    fn parameter_bytes(&self) -> Result<Vec<u8>, EncodeError> {
        encode_to_vec(
            (&self.token_embeddings, &self.positional_embeddings),
            standard(),
        )
    }

    fn load_parameter_bytes(&mut self, bytes: &[u8]) -> Result<(), DecodeError> {
        let (token, positional) =
            decode_from_slice::<(Array2<f32>, Array2<f32>), _>(bytes, standard())?.0;
        self.token_embeddings = token;
        self.positional_embeddings = positional;
        Ok(())
    }
}
