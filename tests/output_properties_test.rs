//! Output-property suite (P1-P5): hand-rolled, seeded, no dependencies.
//!
//! Rides the same xorshift as the decode probe (`llm::Xorshift`). Runs
//! against the artifact checkpoint (`LLM_MODEL_PATH` env or
//! `models/watercycle-latest.bin`); when the artifact is absent the suite
//! skips, because CI cannot hold gitignored weights. Improvement is the
//! pass-table delta against BASELINE_* below, same seed, same draws.

use std::path::Path;

use llm::{Dataset, DatasetType, LLM, Vocab, Xorshift};

const SUITE_SEED: u64 = 20260816;
const DRAWS: usize = 50;
const DEFAULT_ARTIFACT: &str = "models/watercycle-latest.bin";
const EOS: &str = "</s>";
const NOUNS: &[&str] = &[
    "water", "rain", "cloud", "clouds", "ocean", "river", "rivers", "vapor", "droplet", "droplets",
    "lakes", "sun", "heat", "ice", "air", "ground", "storm", "snow",
];

fn artifact_path() -> Option<String> {
    if let Ok(path) = std::env::var("LLM_MODEL_PATH")
        && Path::new(&path).exists()
    {
        return Some(path);
    }
    if Path::new(DEFAULT_ARTIFACT).exists() {
        return Some(DEFAULT_ARTIFACT.to_string());
    }
    None
}

fn load_model(path: &str) -> LLM {
    llm::load(path).unwrap_or_else(|error| panic!("failed to load artifact {path}: {error}"))
}

fn parse_qa_pairs() -> (Vec<String>, Vec<String>) {
    let dataset = Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        DatasetType::JSON,
    );
    let mut questions = Vec::new();
    let mut answers = Vec::new();
    for example in &dataset.chat_training_data {
        if let Some((question, answer)) = example.split_once(" Assistant: ") {
            questions.push(
                question
                    .strip_prefix("User: ")
                    .unwrap_or(question)
                    .to_string(),
            );
            answers.push(
                answer
                    .strip_suffix(" </s>")
                    .unwrap_or(answer)
                    .trim_end_matches('.')
                    .to_string(),
            );
        }
    }
    let heldout_text = std::fs::read_to_string("data/heldout.json").expect("heldout.json");
    let heldout: Vec<(String, String)> = serde_json::from_str(&heldout_text).expect("pairs");
    for (question, answer) in heldout {
        questions.push(question);
        answers.push(answer);
    }
    (questions, answers)
}

struct PromptGen {
    rng: Xorshift,
    questions: Vec<String>,
    answers: Vec<String>,
    nouns: Vec<String>,
}

impl PromptGen {
    fn new(seed: u64, vocab: &Vocab) -> Self {
        let (questions, answers) = parse_qa_pairs();
        let nouns = NOUNS
            .iter()
            .filter(|noun| vocab.encode(noun).is_some())
            .map(|noun| (*noun).to_string())
            .collect();
        Self {
            rng: Xorshift::new(seed),
            questions,
            answers,
            nouns,
        }
    }

    /// Next (prompt, reference) draw; roughly half of the draws substitute
    /// a corpus noun to widen the prompt distribution.
    fn next(&mut self, vocab: &Vocab) -> (String, String) {
        let index = self.rng.below(self.questions.len() as u64) as usize;
        let mut question = self.questions[index].clone();
        if self.nouns.len() >= 2 && self.rng.below(2) == 0 {
            let original = self
                .nouns
                .iter()
                .find(|noun| question.contains(noun.as_str()));
            if let Some(original) = original {
                let replacement = &self.nouns[self.rng.below(self.nouns.len() as u64) as usize];
                if replacement != original {
                    question = question.replacen(original, replacement, 1);
                }
            }
        }
        let _ = vocab;
        (question, self.answers[index].clone())
    }
}

fn tokenize_roundtrip(llm: &LLM, text: &str) -> Vec<usize> {
    llm.tokenize(text)
}

fn has_degeneracy(tokens: &[usize], eos_id: usize) -> bool {
    let t = if tokens.last() == Some(&eos_id) {
        &tokens[..tokens.len() - 1]
    } else {
        tokens
    };
    if t.windows(3).any(|w| w[0] == w[1] && w[1] == w[2]) {
        return true;
    }
    for len in 2..=(t.len() / 2).min(12) {
        for i in 0..=(t.len() - 2 * len) {
            if t[i..i + len] == t[i + len..i + 2 * len] {
                return true;
            }
        }
    }
    false
}

struct DrawStats {
    terminated: usize,
    formatted: usize,
    short: usize,
    nondegenerate: usize,
}

fn run_draws(llm: &mut LLM, generator: &mut PromptGen) -> DrawStats {
    let eos_id = llm.vocab.encode(EOS).expect("eos in vocab");
    let assistant_id = llm.vocab.encode("Assistant").expect("Assistant in vocab");
    let colon_id = llm.vocab.encode(":").expect("colon in vocab");
    let cap = llm.max_seq_len - 1;
    let mut stats = DrawStats {
        terminated: 0,
        formatted: 0,
        short: 0,
        nondegenerate: 0,
    };
    for _ in 0..DRAWS {
        let (question, reference) = generator.next(&llm.vocab);
        let generated = llm.predict(&format!("User: {question}"));
        let tokens = tokenize_roundtrip(llm, &generated);
        if tokens.last() == Some(&eos_id) || tokens.len() == cap {
            stats.terminated += 1;
        }
        if tokens.len() >= 2 && tokens[0] == assistant_id && tokens[1] == colon_id {
            stats.formatted += 1;
        }
        let ref_tokens = tokenize_roundtrip(llm, &reference);
        if tokens.len() <= ref_tokens.len() * 2 + 6 {
            stats.short += 1;
        }
        if !has_degeneracy(&tokens, eos_id) {
            stats.nondegenerate += 1;
        }
    }
    stats
}

#[test]
fn p1_determinism_same_seed_same_artifact_same_outputs() {
    let Some(path) = artifact_path() else {
        eprintln!("property suite skipped: no artifact at {DEFAULT_ARTIFACT}");
        return;
    };
    let prompts: Vec<String> = {
        let model = load_model(&path);
        let mut generator = PromptGen::new(SUITE_SEED, &model.vocab);
        (0..DRAWS)
            .map(|_| format!("User: {}", generator.next(&model.vocab).0))
            .collect()
    };
    let first = {
        let mut model = load_model(&path);
        prompts.iter().map(|p| model.predict(p)).collect::<Vec<_>>()
    };
    let second = {
        let mut model = load_model(&path);
        prompts.iter().map(|p| model.predict(p)).collect::<Vec<_>>()
    };
    assert_eq!(
        first, second,
        "same seed, same artifact must give identical outputs"
    );
}

#[test]
fn p2_p5_properties_and_pass_table_hold_the_baseline() {
    let Some(path) = artifact_path() else {
        eprintln!("property suite skipped: no artifact at {DEFAULT_ARTIFACT}");
        return;
    };
    let mut model = load_model(&path);
    let mut generator = PromptGen::new(SUITE_SEED, &model.vocab);
    let stats = run_draws(&mut model, &mut generator);

    let rate = |n: usize| n as f64 / DRAWS as f64;
    println!(
        "property pass table (seed {SUITE_SEED}, artifact {path}, {DRAWS} draws)\n\
         P2 termination      {:>3}/{}  ({:.2})\n\
         P3 format           {:>3}/{}  ({:.2})\n\
         P4 budget           {:>3}/{}  ({:.2})\n\
         P5 non-degeneracy   {:>3}/{}  ({:.2})",
        stats.terminated,
        DRAWS,
        rate(stats.terminated),
        stats.formatted,
        DRAWS,
        rate(stats.formatted),
        stats.short,
        DRAWS,
        rate(stats.short),
        stats.nondegenerate,
        DRAWS,
        rate(stats.nondegenerate),
    );

    assert_eq!(stats.terminated, DRAWS, "P2: every answer must terminate");
    assert!(
        rate(stats.formatted) >= 0.5,
        "P3: format rate must hold the 0.5 contract"
    );
    assert!(
        rate(stats.nondegenerate) >= BASELINE_NONDEGENERATE,
        "P5: non-degeneracy must not regress below the baseline"
    );
    assert!(
        rate(stats.short) >= BASELINE_BUDGET,
        "P4: budget rate must not regress below the baseline"
    );
    assert!(
        rate(stats.formatted) >= BASELINE_FORMAT,
        "P3: format rate must not regress below the baseline"
    );
}

// Baseline pass table, recorded 2026-08-16 on the canonical seed-42
// artifact (models/watercycle-latest.bin regenerated from fresh init via
// the interactive recipe), suite seed 20260816, 50 draws. Improvement is
// any rate rising above these; regressions fail the suite.
const BASELINE_FORMAT: f64 = 0.94;
const BASELINE_BUDGET: f64 = 1.0;
const BASELINE_NONDEGENERATE: f64 = 0.92;
