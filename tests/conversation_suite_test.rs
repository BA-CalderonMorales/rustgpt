//! Conversation-surface suite (E9): the interactive lane's score formula.
//!
//! A predeclared, seeded-by-construction probe list across five classes
//! (greeting, out-of-vocabulary, mixed, casual junk, in-vocabulary single
//! words). Each probe reports its tokenization and OOV count so a verdict
//! is interpretable: the model must never answer an out-of-domain prompt
//! with a confident water-cycle sentence or a fragment. Improvement is the
//! pass-table delta between artifacts, same probes, honest counts. Skips
//! when the artifact is absent (CI cannot hold gitignored weights).

use std::path::Path;

use llm::LLM;

const DEFAULT_ARTIFACT: &str = "models/watercycle-latest.bin";
const HEDGE: &str = "Assistant : I do not know that word";

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Class {
    Greeting,
    Oov,
    MixedOov,
    CasualJunk,
    InVocab,
}

struct Probe {
    prompt: &'static str,
    class: Class,
}

const PROBES: &[Probe] = &[
    Probe {
        prompt: "User: hi!",
        class: Class::Greeting,
    },
    Probe {
        prompt: "User: hello",
        class: Class::Greeting,
    },
    Probe {
        prompt: "User: hey!",
        class: Class::Greeting,
    },
    Probe {
        prompt: "User: good morning",
        class: Class::Greeting,
    },
    Probe {
        prompt: "User: howdy",
        class: Class::Greeting,
    },
    Probe {
        prompt: "User: yo!",
        class: Class::Greeting,
    },
    Probe {
        prompt: "User: What is gravity?",
        class: Class::Oov,
    },
    Probe {
        prompt: "User: What is lightning?",
        class: Class::Oov,
    },
    Probe {
        prompt: "User: Why is the moon bright?",
        class: Class::Oov,
    },
    Probe {
        prompt: "User: What does the ocean contain?",
        class: Class::Oov,
    },
    Probe {
        prompt: "User: what's my name?",
        class: Class::MixedOov,
    },
    Probe {
        prompt: "User: how are you today?",
        class: Class::MixedOov,
    },
    Probe {
        prompt: "User: Are you sure?",
        class: Class::MixedOov,
    },
    Probe {
        prompt: "User: What's goign on here?",
        class: Class::MixedOov,
    },
    Probe {
        prompt: "User: do do da doop",
        class: Class::CasualJunk,
    },
    Probe {
        prompt: "User: life can be neat?",
        class: Class::CasualJunk,
    },
    Probe {
        prompt: "User: what else can we use?",
        class: Class::CasualJunk,
    },
    Probe {
        prompt: "User: Water?",
        class: Class::InVocab,
    },
    Probe {
        prompt: "User: clouds!",
        class: Class::InVocab,
    },
    Probe {
        prompt: "User: the ocean",
        class: Class::InVocab,
    },
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

#[derive(Clone, Copy, PartialEq, Debug)]
enum Verdict {
    Hedge,
    JitterHedge,
    PartialHedge,
    Greeting,
    ConfidentAnswer,
    Fragment,
}

fn verdict(model: &mut LLM, prompt: &str) -> (Verdict, String, usize, Vec<String>) {
    let tokens = model.tokenize(prompt);
    let oov_count = tokens
        .iter()
        .filter(|&&t| model.vocab.words[t] == "<unk>")
        .count();
    let decoded: Vec<String> = tokens
        .iter()
        .map(|t| model.vocab.words[*t].clone())
        .collect();
    let output = model.predict(prompt);
    let verdict = if output.contains(HEDGE) {
        if output.starts_with("Assistant :") {
            Verdict::Hedge
        } else {
            Verdict::JitterHedge
        }
    } else if output.contains("I do not know that") {
        Verdict::PartialHedge
    } else if output.contains("Assistant :") {
        if output.contains("Hello") {
            Verdict::Greeting
        } else {
            Verdict::ConfidentAnswer
        }
    } else {
        Verdict::Fragment
    };
    (verdict, output, oov_count, decoded)
}

/// Out-of-domain classes: a response is appropriate when it is a full or
/// jittered hedge or a greeting -- never a confident in-domain sentence or
/// a fragment.
fn is_appropriate(verdict: Verdict) -> bool {
    matches!(
        verdict,
        Verdict::Hedge | Verdict::JitterHedge | Verdict::Greeting
    )
}

#[test]
fn conversation_suite_pass_table_holds_the_baseline() {
    let Some(path) = artifact_path() else {
        eprintln!("conversation suite skipped: no artifact at {DEFAULT_ARTIFACT}");
        return;
    };
    let mut model = llm::load(&path).unwrap_or_else(|e| panic!("load {path}: {e}"));

    println!(
        "conversation surface table (artifact {path}, {} probes)",
        PROBES.len()
    );
    let mut class_counts = std::collections::HashMap::new();
    for probe in PROBES {
        let (verdict, output, oov_count, decoded) = verdict(&mut model, probe.prompt);
        class_counts
            .entry(probe.class)
            .or_insert_with(|| [0usize; 5]);
        let counters = class_counts.get_mut(&probe.class).unwrap();
        counters[verdict as usize] += 1;
        println!(
            "{:?} oov={} {:?} | {:?} -> {}",
            probe.class,
            oov_count,
            decoded,
            verdict,
            &output[..output.len().min(52)],
        );
    }

    let out_of_domain: usize = [
        Class::Greeting,
        Class::Oov,
        Class::MixedOov,
        Class::CasualJunk,
    ]
    .iter()
    .map(|class| class_counts.get(class).map_or(0, |c| c.iter().sum()))
    .sum();
    let appropriate: usize = [
        Class::Greeting,
        Class::Oov,
        Class::MixedOov,
        Class::CasualJunk,
    ]
    .iter()
    .map(|class| {
        class_counts.get(class).map_or(0, |c| {
            c.iter()
                .enumerate()
                .filter(|(i, _)| {
                    is_appropriate(match i {
                        0 => Verdict::Hedge,
                        1 => Verdict::JitterHedge,
                        2 => Verdict::PartialHedge,
                        3 => Verdict::Greeting,
                        4 => Verdict::ConfidentAnswer,
                        _ => Verdict::Fragment,
                    })
                })
                .map(|(_, n)| n)
                .sum::<usize>()
        })
    })
    .sum();
    println!(
        "out-of-domain appropriate rate: {appropriate}/{out_of_domain} (hedge or greeting, \
         never a confident water-cycle answer or fragment)"
    );
    assert!(
        appropriate >= 13,
        "the out-of-domain classes must stay dominated by hedges and greetings"
    );
}
