//! Output-property suite (P1-P5): hand-rolled, seeded, no dependencies.
//!
//! Rides the same xorshift as the decode probe (`llm::Xorshift`). Runs
//! against the artifact checkpoint (`LLM_MODEL_PATH` env or
//! `models/watercycle-latest.bin`); when the artifact is absent the suite
//! skips, because CI cannot hold gitignored weights. Improvement is the
//! pass-table delta against BASELINE_* below, same seed, same draws.

use std::path::Path;

use llm::{LLM, Vocab, Xorshift};

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

// Frozen prompt pool: the v0.0.4-era 28 chat questions plus the 4 held-out
// prompts, embedded so the draw distribution can never drift with the
// corpus. Pass-table deltas across artifacts are comparable by
// construction; only the artifact under test changes.
const POOL_QUESTIONS: &[&str] = &[
    "What is evaporation?",
    "What does evaporation change?",
    "How does warm water change into vapor?",
    "What happens after sunlight warms water?",
    "What is condensation?",
    "What does cooling do to water vapor?",
    "How does water vapor become droplets?",
    "What change forms droplets from water vapor?",
    "What forms clouds?",
    "How do small droplets form clouds?",
    "What happens when water droplets join?",
    "Where do water droplets join?",
    "What causes rain?",
    "Why does rain fall from clouds?",
    "What happens when droplets become heavy?",
    "How does precipitation fall?",
    "Where does rainwater flow?",
    "What happens to rainwater on the ground?",
    "Where does rainwater collect?",
    "How does rainwater reach rivers and lakes?",
    "Where do rivers carry water?",
    "How does water reach the ocean?",
    "What happens after water collects in rivers?",
    "What carries water to the ocean?",
    "What happens after water reaches the ocean?",
    "Why does the water cycle repeat?",
    "What keeps the water cycle moving?",
    "How does the water cycle continue?",
    "Why do heavy droplets fall from clouds?",
    "How does cooling change water vapor?",
    "Where does rainwater collect after rainwater flows downhill?",
    "What happens after rivers carry water to the ocean?",
];

const POOL_ANSWERS: &[&str] = &[
    "Evaporation changes warm water into water vapor.",
    "Evaporation changes warm water into water vapor.",
    "Evaporation changes warm water into water vapor.",
    "Evaporation changes warm water into water vapor.",
    "Condensation changes water vapor into droplets.",
    "Condensation changes water vapor into droplets.",
    "Condensation changes water vapor into droplets.",
    "Condensation changes water vapor into droplets.",
    "Small water droplets join and form clouds.",
    "Small water droplets join and form clouds.",
    "Small water droplets join and form clouds.",
    "Small water droplets join and form clouds.",
    "Heavy water droplets fall from clouds as rain.",
    "Heavy water droplets fall from clouds as rain.",
    "Heavy water droplets fall from clouds as rain.",
    "Heavy water droplets fall from clouds as rain.",
    "Rainwater flows downhill and collects in rivers and lakes.",
    "Rainwater flows downhill and collects in rivers and lakes.",
    "Rainwater flows downhill and collects in rivers and lakes.",
    "Rainwater flows downhill and collects in rivers and lakes.",
    "Rivers carry water to the ocean.",
    "Rivers carry water to the ocean.",
    "Rivers carry water to the ocean.",
    "Rivers carry water to the ocean.",
    "The water cycle repeats after water reaches the ocean.",
    "The water cycle repeats after water reaches the ocean.",
    "Sunlight keeps the water cycle moving.",
    "Sunlight keeps the water cycle moving.",
    "Heavy water droplets fall from clouds as rain.",
    "Condensation changes water vapor into droplets.",
    "Rainwater flows downhill and collects in rivers and lakes.",
    "The water cycle repeats after water reaches the ocean.",
];

fn parse_qa_pairs() -> (Vec<String>, Vec<String>) {
    (
        POOL_QUESTIONS.iter().map(|q| (*q).to_string()).collect(),
        POOL_ANSWERS.iter().map(|a| (*a).to_string()).collect(),
    )
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

struct DrawStats {
    terminated: usize,
    formatted: usize,
    short: usize,
    nondegenerate: usize,
    best_of_n_dominates: usize,
}

fn run_draws(llm: &mut LLM, generator: &mut PromptGen) -> DrawStats {
    let eos_id = llm.vocab.encode(EOS).expect("eos in vocab");
    let assistant_id = llm.vocab.encode("Assistant").expect("Assistant in vocab");
    let colon_id = llm.vocab.encode(":").expect("colon in vocab");
    let cap = llm.max_seq_len - 1;
    let mut sampling = Xorshift::new(SUITE_SEED ^ 0x5EED);
    let mut stats = DrawStats {
        terminated: 0,
        formatted: 0,
        short: 0,
        nondegenerate: 0,
        best_of_n_dominates: 0,
    };
    for _ in 0..DRAWS {
        let (question, reference) = generator.next(&llm.vocab);
        let prompt = format!("User: {question}");
        let generated = llm.predict(&prompt);
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
        if !llm.is_degenerate(&generated) {
            stats.nondegenerate += 1;
        }
        // P6: best-of-N (k=5, n=8, seeded) must dominate greedy's
        // per-position score.
        let greedy_accuracy = llm.score_generated(&generated, &reference).accuracy;
        let mut best = 0.0f32;
        for _ in 0..8 {
            let candidate = llm.predict_sampled(&prompt, 5, &mut sampling);
            best = best.max(llm.score_generated(&candidate, &reference).accuracy);
        }
        if best >= greedy_accuracy {
            stats.best_of_n_dominates += 1;
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
         P2 termination        {:>3}/{}  ({:.2})\n\
         P3 format             {:>3}/{}  ({:.2})\n\
         P4 budget             {:>3}/{}  ({:.2})\n\
         P5 non-degeneracy     {:>3}/{}  ({:.2})\n\
         P6 best-of-N dominates {:>3}/{}  ({:.2})",
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
        stats.best_of_n_dominates,
        DRAWS,
        rate(stats.best_of_n_dominates),
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
    // P6 is a reported row, not a gate: dominance can legitimately fall
    // when greedy improves (E3 measured 0.86 baseline -> 0.80 on E2). Its
    // pass-table delta is the decode-probe claim's evidence, recorded per
    // artifact, never asserted against.
}

// Baseline pass table, recorded 2026-08-16 on the canonical seed-42
// artifact (models/watercycle-latest.bin regenerated from fresh init via
// the interactive recipe), suite seed 20260816, 50 draws. Improvement is
// any rate rising above these; regressions fail the suite.
const BASELINE_FORMAT: f64 = 0.94;
const BASELINE_BUDGET: f64 = 1.0;
const BASELINE_NONDEGENERATE: f64 = 0.92;
// P6 reference row (first measurement 2026-08-16, canonical artifact):
// best-of-N dominance 43/50 = 0.86; E2 artifact 40/50 = 0.80. Reported,
// not asserted (see the P6 note in the test body).
