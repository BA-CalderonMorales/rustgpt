use std::cmp::Ordering;
use std::io::IsTerminal;

use ndarray::{Array1, Array2, Axis};

use super::{AnswerScore, DecodeStep, FluencyScore, LLM, Layer, LogitProfile};

/// Literal answer for prompts whose tokens are all outside the vocabulary.
/// The CLI contract pins this string: out-of-vocabulary input must never
/// produce a silent empty output with status:ok.
pub const UNKNOWN_ANSWER: &str = "Assistant : I do not know that word . </s>";
/// Literal answer for prompts that exceed the model's sequence limit.
const TRUNCATED_ANSWER: &str = "Assistant : The input is too long . </s>";
use crate::{
    EMBEDDING_DIM, Embeddings, HIDDEN_DIM, MAX_SEQ_LEN, Vocab, Xorshift,
    output_projection::OutputProjection, transformer::TransformerBlock,
};

impl Default for LLM {
    fn default() -> Self {
        let transformer_block = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let output_projection = OutputProjection::new(EMBEDDING_DIM, Vocab::default_words().len());
        Self {
            vocab: Vocab::default(),
            network: vec![
                Box::new(Embeddings::default()),
                Box::new(transformer_block),
                Box::new(output_projection),
            ],
            max_seq_len: MAX_SEQ_LEN,
            config: crate::configuration::Config::micro(),
        }
    }
}

impl LLM {
    pub fn new(vocab: Vocab, network: Vec<Box<dyn Layer>>) -> Self {
        Self {
            vocab,
            network,
            max_seq_len: MAX_SEQ_LEN,
            config: crate::configuration::Config::micro(),
        }
    }

    pub fn with_config(
        vocab: Vocab,
        network: Vec<Box<dyn Layer>>,
        config: crate::configuration::Config,
    ) -> Self {
        Self {
            vocab,
            network,
            max_seq_len: config.max_seq_len,
            config,
        }
    }
}

impl LLM {
    pub fn network_description(&self) -> String {
        self.network
            .iter()
            .map(|layer| layer.layer_type())
            .collect::<Vec<&str>>()
            .join(", ")
    }

    pub fn total_parameters(&self) -> usize {
        // Sum the parameters across all layers in the network
        self.network
            .iter()
            .map(|layer| layer.parameters())
            .sum::<usize>()
    }

    pub fn predict(&mut self, text: &str) -> String {
        let output_tokens = self.forward(text);
        self.answer_string(text, &output_tokens)
    }

    /// Same generation as `predict`, plus one `DecodeStep` per generated
    /// token (greedy token id and its softmax probability). Prompt tokens
    /// are not steps; an empty-output fallback answer yields no steps.
    pub fn predict_with_steps(&mut self, text: &str) -> (String, Vec<DecodeStep>) {
        let mut steps = Vec::new();
        let output_tokens = self.decode_tokens(text, Some(&mut steps));
        (self.answer_string(text, &output_tokens), steps)
    }

    /// Turn decoded tokens into the answer string. An empty decode is an
    /// explicit report, never a silent empty string: a prompt with no
    /// in-vocabulary token (or only unknown tokens) gets the unknown-word
    /// answer; an over-long prompt gets the truncation answer. This is the
    /// single authoritative home of that empty-output contract.
    fn answer_string(&self, text: &str, output_tokens: &[usize]) -> String {
        if output_tokens.is_empty() {
            let tokens = self.tokenize(text);
            if tokens.is_empty() || (tokens.len() == 1 && self.is_entirely_unknown(&tokens)) {
                return UNKNOWN_ANSWER.to_string();
            }
            return TRUNCATED_ANSWER.to_string();
        }
        output_tokens
            .iter()
            .map(|t| self.vocab.decode[t].clone())
            .collect::<Vec<String>>()
            .join(" ")
    }

    fn forward(&mut self, text: &str) -> Vec<usize> {
        self.decode_tokens(text, None)
    }

    /// Greedy decode with an optional per-step capture: when `capture` is
    /// given, every generated token appends its `DecodeStep`. The shared
    /// engine behind `predict` and `predict_with_steps`.
    fn decode_tokens(
        &mut self,
        text: &str,
        mut capture: Option<&mut Vec<DecodeStep>>,
    ) -> Vec<usize> {
        // Tokenize the input text
        let mut tokenized = self.tokenize(text);
        let mut output_tokens: Vec<usize> = Vec::new();

        // Safety check: ensure we have at least one token; a single
        // entirely-unknown token also decodes nothing (predict emits the
        // literal unknown answer). Multi-token unknown prompts go to the
        // model, which has learned what they mean (greetings, hedges).
        if tokenized.is_empty() || (tokenized.len() == 1 && self.is_entirely_unknown(&tokenized)) {
            return output_tokens;
        }

        let input_len = tokenized.len();

        // Prevent overflow if input_len >= max_seq_len
        if input_len >= self.max_seq_len {
            return output_tokens;
        }

        for _ in 0..(self.max_seq_len - input_len) {
            // Check if we're approaching the maximum sequence length
            if output_tokens.len() >= self.max_seq_len - 1 {
                break;
            }

            let token_input = Array2::from_shape_vec(
                (1, tokenized.len()),
                tokenized.iter().map(|&x| x as f32).collect(),
            )
            .unwrap();
            let mut input = token_input;

            for layer in &mut self.network {
                input = layer.forward(&input);
            }

            let logits = input;

            // Safety check: ensure we have at least one token
            if logits.shape()[0] == 0 {
                break;
            }

            let last_logit = logits
                .row(logits.shape()[0] - 1)
                .to_owned()
                .insert_axis(Axis(0));

            // Softmax - convert activations of each token to a probability distribution over the
            // vocabulary
            let probs = Self::softmax(&last_logit); // 1 x vocab_size

            // Greedy Decode - Choose the highest probability token for each position
            let tokens = Self::greedy_decode(&probs);

            let next_token = tokens[tokens.len() - 1];

            if let Some(steps) = capture.as_deref_mut() {
                steps.push(DecodeStep {
                    token: next_token,
                    prob: probs[[0, next_token]],
                });
            }

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == self.vocab.encode("</s>").unwrap() {
                break;
            }
        }

        output_tokens
    }

    /// Cached decode with a pluggable token selector: prefill fills every
    /// block's K/V cache in one pass, then each step computes only the
    /// newest token's attention row and asks `select` for the next token.
    /// The greedy selector's output is byte-identical to the recompute
    /// path (pinned by tests/kv_cache_test.rs); the weighted selector is
    /// the probability-sampling leg.
    fn decode_cached(
        &mut self,
        text: &str,
        temperature: f32,
        select: &mut dyn FnMut(&Array2<f32>, &[usize]) -> usize,
    ) -> Vec<usize> {
        // Tokenize and guard the degenerate prompts, as the recompute path.
        let mut tokenized = self.tokenize(text);
        let mut output_tokens: Vec<usize> = Vec::new();
        if tokenized.is_empty()
            || self.is_entirely_unknown(&tokenized)
            || tokenized.len() >= self.max_seq_len
        {
            return output_tokens;
        }

        // Switch every layer into decode-cache mode.
        for layer in &mut self.network {
            layer.set_cache_mode(true);
        }

        // Prefill caches every prompt position except the last one; the
        // last prompt token is step 0's input, appended uniformly, so the
        // cache holds exactly the same rows the recompute path attends to.
        // A single-token prompt has no prefill: step 0 appends it directly.
        if tokenized.len() > 1 {
            let prefill = Array2::from_shape_vec(
                (1, tokenized.len() - 1),
                tokenized[..tokenized.len() - 1]
                    .iter()
                    .map(|&x| x as f32)
                    .collect(),
            )
            .unwrap();
            let mut input = prefill;
            for layer in &mut self.network {
                input = layer.forward(&input);
            }
        }

        // One step per generated token, attending against the cache.
        for _ in 0..(self.max_seq_len - tokenized.len()) {
            if output_tokens.len() >= self.max_seq_len - 1 {
                break;
            }
            let step_input = Array2::from_shape_vec(
                (1, 1),
                tokenized.last().map(|&x| x as f32).into_iter().collect(),
            )
            .unwrap();
            let mut input = step_input;
            for layer in &mut self.network {
                input = layer.forward(&input);
            }
            if input.shape()[0] == 0 {
                break;
            }
            let scaled = &input * (1.0 / temperature);
            let next_token = select(&scaled, &output_tokens);
            output_tokens.push(next_token);
            tokenized.push(next_token);
            if next_token == self.vocab.encode("</s>").unwrap() {
                break;
            }
        }

        // Restore the stateless recompute behavior.
        for layer in &mut self.network {
            layer.set_cache_mode(false);
        }
        output_tokens
    }

    /// The pinned greedy path through the cached decoder: softmax then
    /// argmax keeps the exact greedy stream, byte-identical at every T
    /// (scaling preserves the argmax).
    fn forward_cached(&mut self, text: &str, temperature: f32) -> Vec<usize> {
        let mut greedy = |logits: &Array2<f32>, _generated: &[usize]| {
            let probs = Self::softmax(logits);
            let tokens = Self::greedy_decode(&probs);
            tokens[tokens.len() - 1]
        };
        self.decode_cached(text, temperature, &mut greedy)
    }

    /// Greedy prediction through the cached decode path; identical string
    /// to `predict`, produced by the KV-cache decoder.
    pub fn predict_cached(&mut self, text: &str) -> String {
        let output_tokens = self.forward_cached(text, 1.0);
        self.answer_string(text, &output_tokens)
    }

    /// Temperature-scaled greedy decode through the cached path: logits are
    /// divided by `temperature` before the output softmax, then the argmax
    /// is chosen as usual. Scaling preserves the argmax for any positive T,
    /// so the token stream is byte-identical to `predict_cached` at every
    /// temperature (pinned by tests/kv_cache_test.rs); the knob exists so
    /// the collapse gate can be measured under a peaked output softmax.
    pub fn predict_scaled(&mut self, text: &str, temperature: f32) -> String {
        let output_tokens = self.forward_cached(text, temperature);
        self.answer_string(text, &output_tokens)
    }

    /// Probability-weighted temperature sampling through the cached path:
    /// each step draws from the temperature-scaled softmax distribution, so
    /// every token is emitted with exactly its model-given probability
    /// (rank-2+ tokens included). The caller's rng seeds the whole decode,
    /// so a fixed seed reproduces the exact token stream. The greedy
    /// semantics of `predict_scaled` are untouched.
    pub fn predict_weighted(&mut self, text: &str, temperature: f32, rng: &mut Xorshift) -> String {
        let output_tokens =
            self.decode_cached(text, temperature, &mut |logits: &Array2<f32>, _| {
                let probs = Self::softmax(logits);
                Self::weighted_draw(&probs, rng)
            });
        self.answer_string(text, &output_tokens)
    }

    /// Greedy decode with logit-level anti-repetition penalties: every
    /// token present in the generated-so-far gets a flat `presence`
    /// subtraction, and its logit is divided by `repetition` once per
    /// occurrence. The penalty acts before the argmax, so it can break
    /// the frequency-head attractor deterministically. Presence 0.0 and
    /// repetition 1.0 are both off (the stream is then exactly
    /// `predict_scaled`'s).
    pub fn predict_penalized(
        &mut self,
        text: &str,
        temperature: f32,
        presence: f32,
        repetition: f32,
    ) -> String {
        let output_tokens = self.decode_cached(
            text,
            temperature,
            &mut |logits: &Array2<f32>, generated: &[usize]| {
                let adjusted = Self::apply_penalties(logits, generated, presence, repetition);
                let probs = Self::softmax(&adjusted);
                let tokens = Self::greedy_decode(&probs);
                tokens[tokens.len() - 1]
            },
        );
        self.answer_string(text, &output_tokens)
    }

    /// Probability-weighted sampling with the same logit-level penalties
    /// as `predict_penalized`: draws from the penalty-adjusted
    /// temperature-scaled distribution with the caller's seeded rng.
    pub fn predict_weighted_penalized(
        &mut self,
        text: &str,
        temperature: f32,
        presence: f32,
        repetition: f32,
        rng: &mut Xorshift,
    ) -> String {
        let output_tokens = self.decode_cached(
            text,
            temperature,
            &mut |logits: &Array2<f32>, generated: &[usize]| {
                let adjusted = Self::apply_penalties(logits, generated, presence, repetition);
                let probs = Self::softmax(&adjusted);
                Self::weighted_draw(&probs, rng)
            },
        );
        self.answer_string(text, &output_tokens)
    }

    pub fn train(&mut self, data: Vec<&str>, epochs: usize, lr: f32) {
        let _ = self.train_with_progress(data, epochs, lr, false);
    }

    /// Train for `epochs`, returning the average loss of every epoch.
    pub fn train_with_progress(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        progress_to_stderr: bool,
    ) -> Vec<f32> {
        self.train_impl(data, epochs, lr, progress_to_stderr, None)
            .0
    }

    /// Train for `epochs`, sampling the logit-regime profile over
    /// `profile_texts` after every epoch. The two trajectories ride the
    /// same runs, so a verdict (margin rises / entropy falls before the
    /// collapse saturates) is read from one deterministic session.
    pub fn train_with_profile(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        progress_to_stderr: bool,
        profile_texts: &[&str],
    ) -> (Vec<f32>, Vec<LogitProfile>) {
        self.train_impl(data, epochs, lr, progress_to_stderr, Some(profile_texts))
    }

    fn train_impl(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        progress_to_stderr: bool,
        profile_texts: Option<&[&str]>,
    ) -> (Vec<f32>, Vec<LogitProfile>) {
        // Tokenize every example once, truncated to the sequence budget.
        let tokenized_data = data
            .iter()
            .map(|input| {
                let mut tokens = self.tokenize(input);
                tokens.truncate(self.max_seq_len);
                tokens
            })
            .collect::<Vec<Vec<usize>>>();

        // One pass over the corpus per epoch.
        let mut epoch_losses = Vec::with_capacity(epochs);
        let mut profile = Vec::with_capacity(epochs);
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                // Slice the row into input and shifted targets.
                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];

                // Forward pass through every layer.
                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());
                for layer in &mut self.network {
                    input = layer.forward(&input);
                }
                let probs = Self::softmax(&input);
                total_loss += Self::cross_entropy_loss_step(&probs, target_ids);

                // Backward pass: softmax gradient, clipping BEFORE backprop.
                let mut grads_output = Self::compute_gradients_step(&probs, target_ids);
                Self::clip_gradients(&mut grads_output, 5.0);
                for layer in self.network.iter_mut().rev() {
                    grads_output = layer.backward(&grads_output, lr);
                }
            }

            // Sample the logit-regime profile over the probe stream, when the
            // caller asked for it.
            if let Some(texts) = profile_texts {
                profile.push(self.eval_profile(texts));
            }

            // Report the epoch average: live bar on a terminal, plain lines
            // otherwise, on stderr when requested.
            let epoch_loss = total_loss / tokenized_data.len() as f32;
            epoch_losses.push(epoch_loss);
            let message = format!("Epoch {}: Loss = {:.4}", epoch, epoch_loss);
            if progress_to_stderr {
                if std::io::stderr().is_terminal() {
                    use std::io::Write;
                    eprint!(
                        "\rEpoch {}/{} | Loss = {:.4} | {}",
                        epoch + 1,
                        epochs,
                        epoch_loss,
                        Self::progress_bar(epoch + 1, epochs),
                    );
                    let _ = std::io::stderr().flush();
                } else {
                    eprintln!("{message}");
                }
            } else {
                println!("{message}");
            }
        }
        if progress_to_stderr && std::io::stderr().is_terminal() {
            eprintln!();
        }
        (epoch_losses, profile)
    }

    /// Teacher-forced cross-entropy of one full sequence, without training.
    ///
    /// The whole `text` is scored: every token is predicted from its prefix,
    /// exactly the signal training optimizes. This is the trajectory probe
    /// for held-out data.
    pub fn sequence_loss(&mut self, text: &str) -> f32 {
        // Tokenize, truncated to the sequence budget.
        let mut tokens = self.tokenize(text);
        tokens.truncate(self.max_seq_len);
        if tokens.len() < 2 {
            return 0.0;
        }

        // Teacher-forced forward pass over the whole sequence.
        let mut input: Array2<f32> = Array2::zeros((1, tokens.len() - 1));
        input.row_mut(0).assign(
            &tokens[..tokens.len() - 1]
                .iter()
                .map(|&x| x as f32)
                .collect::<Array1<f32>>(),
        );
        for layer in &mut self.network {
            input = layer.forward(&input);
        }

        // Cross-entropy of every token against its prefix.
        let probs = Self::softmax(&input);
        Self::cross_entropy_loss_step(&probs, &tokens[1..])
    }

    /// Teacher-forced logit-regime means over a whole token stream: the
    /// continuous instrument behind the boolean collapse gate. Every
    /// position contributes its top-1 margin, top-2 logit gap, logit norm,
    /// and output entropy, averaged across the stream.
    pub fn eval_profile(&mut self, texts: &[&str]) -> LogitProfile {
        // Accumulators across every position of every text.
        let mut margin_sum = 0.0f64;
        let mut gap_sum = 0.0f64;
        let mut norm_sum = 0.0f64;
        let mut entropy_sum = 0.0f64;
        let mut positions = 0usize;

        // Teacher-forced forward pass over each text, exactly the signal
        // training optimizes.
        for text in texts {
            let mut tokens = self.tokenize(text);
            tokens.truncate(self.max_seq_len);
            if tokens.len() < 2 {
                continue;
            }
            let mut input: Array2<f32> = Array2::zeros((1, tokens.len() - 1));
            input.row_mut(0).assign(
                &tokens[..tokens.len() - 1]
                    .iter()
                    .map(|&x| x as f32)
                    .collect::<Array1<f32>>(),
            );
            for layer in &mut self.network {
                input = layer.forward(&input);
            }

            // Per-position statistics from the raw logits row.
            for row in input.rows() {
                let (margin, gap, norm, entropy) = Self::row_profile(&row);
                margin_sum += margin as f64;
                gap_sum += gap as f64;
                norm_sum += norm as f64;
                entropy_sum += entropy as f64;
                positions += 1;
            }
        }

        // The means over the stream (a degenerate stream yields zeroes).
        LogitProfile {
            top1_margin: (margin_sum / positions as f64) as f32,
            top2_gap: (gap_sum / positions as f64) as f32,
            logit_norm: (norm_sum / positions as f64) as f32,
            entropy: (entropy_sum / positions as f64) as f32,
        }
    }

    /// One row of logits -> top-1 margin, top-2 logit gap, logit norm, and
    /// softmax output entropy. Two scans: rank the top-2 logits, then the
    /// softmax sums for margin and entropy.
    fn row_profile<S: ndarray::Data<Elem = f32>>(
        row: &ndarray::ArrayBase<S, ndarray::Ix1>,
    ) -> (f32, f32, f32, f32) {
        // Rank the top-2 logits and the row norm in one scan.
        let mut best = (f32::NEG_INFINITY, usize::MAX);
        let mut second = (f32::NEG_INFINITY, usize::MAX);
        let mut norm = 0.0f64;
        for (index, &value) in row.iter().enumerate() {
            if value > best.0 {
                second = best;
                best = (value, index);
            } else if value > second.0 {
                second = (value, index);
            }
            norm += value as f64 * value as f64;
        }
        let gap = best.0 - second.0;

        // Softmax sums for the margin and the output entropy.
        let mut exp_sum = 0.0f64;
        let mut exp_best = 0.0f64;
        let mut exp_second = 0.0f64;
        let mut entropy = 0.0f64;
        for (index, &value) in row.iter().enumerate() {
            let exp_value = (value - best.0).exp() as f64;
            exp_sum += exp_value;
            if index == best.1 {
                exp_best = exp_value;
            } else if index == second.1 {
                exp_second = exp_value;
            }
            if exp_value > 0.0 {
                entropy += exp_value * exp_value.ln();
            }
        }

        // Entropy of the normalized distribution, margin from the top-2
        // softmax masses, norm rooted from the accumulated squares.
        let p_sum = exp_sum.max(f32::EPSILON as f64);
        let entropy = -(entropy / p_sum) + p_sum.ln();
        let margin = ((exp_best - exp_second) / p_sum) as f32;
        (margin, gap, norm.sqrt() as f32, entropy as f32)
    }

    /// Score a generated answer against a reference: greedy generation of
    /// `text`, then exact / prefix / per-position accuracy on tokens.
    pub fn answer_score(&mut self, text: &str, reference: &str) -> AnswerScore {
        let predicted = self.predict(text);
        self.score_generated(&predicted, reference)
    }

    /// Score an already-generated string against a reference; the shared
    /// core of `answer_score` and the decode-time compute probe (one
    /// authoritative scoring implementation).
    pub fn score_generated(&self, generated: &str, reference: &str) -> AnswerScore {
        // Tokenize both sides; the generated </s> does not count.
        let mut generated_tokens = self.tokenize(generated);
        let reference_tokens = self.tokenize(reference);
        if generated_tokens.last() == self.vocab.encode("</s>").as_ref() {
            generated_tokens.pop();
        }

        // Matching positions over the longer of the two sequences.
        let matching = generated_tokens
            .iter()
            .zip(&reference_tokens)
            .filter(|(generated, reference)| generated == reference)
            .count();
        let positions = generated_tokens.len().max(reference_tokens.len()).max(1) as f32;

        // Exact, prefix, and per-position verdicts.
        AnswerScore {
            exact: generated_tokens == reference_tokens,
            prefix: !generated_tokens.is_empty()
                && generated_tokens.len() <= reference_tokens.len()
                && generated_tokens
                    .iter()
                    .zip(&reference_tokens)
                    .all(|(generated, reference)| generated == reference),
            accuracy: matching as f32 / positions,
        }
    }

    /// Greedy determinism is a contract; `is_degenerate` is the decode
    /// quality gate: no token repeated three times consecutively and no
    /// back-to-back n-gram window anywhere in the answer.
    pub fn is_degenerate(&self, text: &str) -> bool {
        // Strip the trailing </s>; it is not part of the answer.
        let eos_id = self.vocab.encode("</s>");
        let mut tokens = self.tokenize(text);
        if tokens.last() == eos_id.as_ref() {
            tokens.pop();
        }

        // A token repeated three times consecutively is degenerate.
        if tokens.windows(3).any(|w| w[0] == w[1] && w[1] == w[2]) {
            return true;
        }

        // Any back-to-back n-gram window repeats the answer.
        for len in 2..=(tokens.len() / 2).min(12) {
            for i in 0..=(tokens.len() - 2 * len) {
                if tokens[i..i + len] == tokens[i + len..i + 2 * len] {
                    return true;
                }
            }
        }
        false
    }

    /// Top-k sampled decode driven by the caller's PRNG: identical token
    /// mechanics to `predict`, with the greedy selection replaced by a
    /// seeded draw over the top-k ranks. The greedy contract is untouched.
    pub fn predict_sampled(&mut self, text: &str, k: usize, rng: &mut Xorshift) -> String {
        // Guard the degenerate prompts with the literal fallbacks.
        let mut tokenized = self.tokenize(text);
        if tokenized.is_empty() || (tokenized.len() == 1 && self.is_entirely_unknown(&tokenized)) {
            return UNKNOWN_ANSWER.to_string();
        }
        if tokenized.len() >= self.max_seq_len {
            return TRUNCATED_ANSWER.to_string();
        }

        // Decode step by step, drawing from the top-k ranks.
        let mut output_tokens = Vec::new();
        for _ in 0..(self.max_seq_len - tokenized.len()) {
            if output_tokens.len() >= self.max_seq_len - 1 {
                break;
            }
            let token_input = Array2::from_shape_vec(
                (1, tokenized.len()),
                tokenized.iter().map(|&x| x as f32).collect(),
            )
            .unwrap();
            let mut input = token_input;
            for layer in &mut self.network {
                input = layer.forward(&input);
            }
            if input.shape()[0] == 0 {
                break;
            }
            let last_logit = input
                .row(input.shape()[0] - 1)
                .to_owned()
                .insert_axis(Axis(0));
            let probs = Self::softmax(&last_logit);
            let next_token = Self::top_k_sample(&probs, k, rng);
            output_tokens.push(next_token);
            tokenized.push(next_token);
            if next_token == self.vocab.encode("</s>").unwrap() {
                break;
            }
        }

        output_tokens
            .iter()
            .map(|t| self.vocab.decode[t].clone())
            .collect::<Vec<String>>()
            .join(" ")
    }

    /// Uniform seeded draw over the top-k probability ranks.
    fn top_k_sample(probs: &Array2<f32>, k: usize, rng: &mut Xorshift) -> usize {
        let mut ranked: Vec<(usize, f32)> = probs.row(0).iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        ranked[rng.below(k.min(ranked.len()) as u64) as usize].0
    }

    /// Logit-level anti-repetition penalties over the generated-so-far:
    /// presence subtracts a flat amount from every seen token's logit;
    /// repetition divides a seen token's logit by `repetition` once per
    /// occurrence (1.0 is the identity). Both act before the softmax, so
    /// they can move the argmax.
    fn apply_penalties(
        logits: &Array2<f32>,
        generated: &[usize],
        presence: f32,
        repetition: f32,
    ) -> Array2<f32> {
        // Count occurrences of every generated token.
        let mut counts = std::collections::HashMap::new();
        for &token in generated {
            *counts.entry(token).or_insert(0usize) += 1;
        }

        // Adjust the logit of each seen token by its presence and count.
        let mut adjusted = logits.clone();
        if !counts.is_empty() {
            let mut row = adjusted.row_mut(0);
            for (&token, &count) in &counts {
                let scale = repetition.powi(count as i32);
                row[token] = row[token] / scale - presence;
            }
        }
        adjusted
    }

    /// Seeded probability-weighted draw over a 1-row softmax distribution:
    /// a uniform draw mapped through the cumulative mass, so each token's
    /// emission chance is exactly its model-given probability. Distinct
    /// from `top_k_sample`, which flattens the top-k ranks.
    fn weighted_draw(probs: &Array2<f32>, rng: &mut Xorshift) -> usize {
        // One uniform draw over the row's total mass.
        let row = probs.row(0);
        let total: f32 = row.sum();
        let draw = (rng.next_u64() as f64 / u64::MAX as f64) as f32 * total;

        // Walk the cumulative mass until the draw lands.
        let mut cumulative = 0.0f32;
        for (index, &probability) in row.iter().enumerate() {
            cumulative += probability;
            if draw <= cumulative {
                return index;
            }
        }

        // Floating-point tail: the last token owns the residual mass.
        row.len() - 1
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        // Split by whitespace, then each word by its punctuation.
        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            // Whole-token markers must not split on their angle brackets,
            // even when punctuation is attached ("<unk>?").
            if let Some(rest) = word
                .strip_prefix("</s>")
                .or_else(|| word.strip_prefix("<unk>"))
            {
                if let Some(token_id) = self.vocab.encode(&word[..word.len() - rest.len()]) {
                    tokens.push(token_id);
                }
                for c in rest.chars().filter(|c| c.is_ascii_punctuation()) {
                    self.push_token(&mut tokens, &c.to_string());
                }
                continue;
            }

            // Split the word's alphabetic core from its punctuation.
            let mut current_word = String::new();
            for c in word.chars() {
                if c.is_ascii_punctuation() {
                    if !current_word.is_empty() {
                        self.push_token(&mut tokens, &current_word);
                        current_word.clear();
                    }
                    self.push_token(&mut tokens, &c.to_string());
                } else {
                    current_word.push(c);
                }
            }

            // The trailing alphabetic core, if any.
            if !current_word.is_empty() {
                self.push_token(&mut tokens, &current_word);
            }
        }

        tokens
    }

    /// Push a word token, mapping out-of-vocabulary words to `<unk>` when
    /// the vocabulary has one (the model then learns to hedge); words are
    /// dropped only when the vocabulary lacks `<unk>` entirely.
    fn push_token(&self, tokens: &mut Vec<usize>, word: &str) {
        if let Some(token_id) = self.vocab.encode(word) {
            tokens.push(token_id);
        } else if let Some(unknown) = self.vocab.encode("<unk>") {
            tokens.push(unknown);
        }
    }

    /// True when every token is the unknown-word token: the prompt is
    /// entirely out of vocabulary, so the literal unknown answer applies.
    fn is_entirely_unknown(&self, tokens: &[usize]) -> bool {
        self.vocab.encode("<unk>").is_some_and(|unknown| {
            !tokens.is_empty() && tokens.iter().all(|&token| token == unknown)
        })
    }

    /// Number of whitespace/punctuation tokens `text` splits into, counting
    /// words outside the vocabulary. `tokenize` silently drops out-of-vocab
    /// words, so eval coverage is `tokenize(text).len() / raw_token_count(text)`.
    pub fn raw_token_count(&self, text: &str) -> usize {
        let mut count = 0usize;
        for word in text.split_whitespace() {
            let mut current = String::new();
            for c in word.chars() {
                if c.is_ascii_punctuation() {
                    if !current.is_empty() {
                        count += 1;
                        current.clear();
                    }
                    count += 1;
                } else {
                    current.push(c);
                }
            }
            if !current.is_empty() {
                count += 1;
            }
        }
        count
    }

    /// ASCII progress bar for the training loop's terminal view.
    fn progress_bar(done: usize, total: usize) -> String {
        const WIDTH: usize = 20;
        let filled = done * WIDTH / total.max(1);
        let mut bar = String::with_capacity(WIDTH + 2);
        bar.push('[');
        for i in 0..WIDTH {
            bar.push(if i < filled { '#' } else { '.' });
        }
        bar.push(']');
        bar
    }

    fn softmax(logits: &Array2<f32>) -> Array2<f32> {
        // Row-wise softmax over the vocabulary (logits is seq_len x vocab_size).
        let mut result = logits.clone();
        for mut row in result.rows_mut() {
            // Numerically stable exponentials around the row maximum.
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_values: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();

            // Normalize the row to a probability distribution.
            for (i, &exp_val) in exp_values.iter().enumerate() {
                row[i] = exp_val / sum_exp;
            }
        }

        result
    }

    fn greedy_decode(probs: &Array2<f32>) -> Vec<usize> {
        probs
            .map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(index, _)| index)
                    .unwrap()
            })
            .to_vec()
    }

    fn cross_entropy_loss_step(probs: &Array2<f32>, target: &[usize]) -> f32 {
        let mut loss = 0.0;
        for row_idx in 0..probs.shape()[0] {
            let prob_target = probs[[row_idx, target[row_idx]]]; // Get probability of correct token
            loss -= prob_target.max(1e-15).ln(); // Add numerical stability
        }

        loss / target.len() as f32
    }

    fn compute_gradients_step(probs: &Array2<f32>, target: &[usize]) -> Array2<f32> {
        let mut grads = probs.clone(); // Start with softmax probabilities

        if probs.shape()[0] != target.len() {
            panic!("Probs and target must have the same number of rows");
        }

        let batch_size = target.len() as f32;

        // Compute correct softmax + cross-entropy gradient: softmax - one_hot(target)
        for row_idx in 0..grads.shape()[0] {
            grads[[row_idx, target[row_idx]]] -= 1.0; // Convert to: p - y (where y is one-hot)
        }

        // Normalize by batch size for stable training
        grads.mapv_inplace(|x| x / batch_size);

        grads
    }

    fn clip_gradients(grads: &mut Array2<f32>, max_norm: f32) {
        // Calculate L2 norm of gradients
        let norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // If norm exceeds max_norm, scale gradients down
        if norm > max_norm {
            let scale = max_norm / norm;
            grads.mapv_inplace(|x| x * scale);
        }
    }

    /// The decode-quality yardstick over a batch of generated samples:
    /// per-sample distinct-1, distinct-2, repetition-freeness, sentence-
    /// final punctuation count, and length, averaged across the batch.
    /// This -- not the boolean gate -- is the verdict instrument for
    /// decode levers.
    pub fn fluency_score(&self, samples: &[String]) -> FluencyScore {
        // Per-sample accumulators over the batch.
        let mut distinct_1_sum = 0.0f64;
        let mut distinct_2_sum = 0.0f64;
        let mut repetition_free_count = 0usize;
        let mut sentences_sum = 0.0f64;
        let mut length_sum = 0.0f64;

        // Tokenize each sample, stripping the trailing </s>.
        for sample in samples {
            let mut tokens = self.tokenize(sample);
            if tokens.last() == self.vocab.encode("</s>").as_ref() {
                tokens.pop();
            }
            if tokens.is_empty() {
                continue;
            }

            // Distinct-1 over the sample's tokens, distinct-2 over its pairs.
            let mut unique_tokens = std::collections::HashSet::new();
            let mut unique_pairs = std::collections::HashSet::new();
            for (index, token) in tokens.iter().enumerate() {
                unique_tokens.insert(token);
                if index > 0 {
                    unique_pairs.insert((tokens[index - 1], *token));
                }
            }
            distinct_1_sum += unique_tokens.len() as f64 / tokens.len() as f64;
            distinct_2_sum +=
                unique_pairs.len() as f64 / tokens.len().saturating_sub(1).max(1) as f64;

            // Repetition-free means no adjacent identical pair; sentence-final
            // punctuation marks the multi-sentence signal.
            let has_repeat = tokens.windows(2).any(|window| window[0] == window[1]);
            repetition_free_count += usize::from(!has_repeat);
            let sentences = tokens
                .iter()
                .filter(|token| matches!(self.vocab.decode[*token].as_str(), "." | "!" | "?"))
                .count();
            sentences_sum += sentences as f64;
            length_sum += tokens.len() as f64;
        }

        // Means over the batch; an empty batch yields zeroes.
        let batch = samples.len().max(1) as f64;
        FluencyScore {
            distinct_1: (distinct_1_sum / batch) as f32,
            distinct_2: (distinct_2_sum / batch) as f32,
            repetition_free_rate: repetition_free_count as f32 / batch as f32,
            completion_sentences: (sentences_sum / batch) as f32,
            mean_completion_len: (length_sum / batch) as f32,
        }
    }
}

/// The per-epoch logit-regime profile as the machine-JSON block: one entry
/// per epoch, four continuous instruments that make collapse onset visible.
pub fn profile_json(profile: &[LogitProfile]) -> serde_json::Value {
    serde_json::json!({
        "top1_margin": profile.iter().map(|p| p.top1_margin).collect::<Vec<f32>>(),
        "top2_gap": profile.iter().map(|p| p.top2_gap).collect::<Vec<f32>>(),
        "logit_norm": profile.iter().map(|p| p.logit_norm).collect::<Vec<f32>>(),
        "entropy": profile.iter().map(|p| p.entropy).collect::<Vec<f32>>(),
    })
}
