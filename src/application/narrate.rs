/// The six-stage pipeline tour: the same copy serves `--tiny --train`
/// narration (stderr, machine stdout untouched) and the `--demo` guided
/// tour (stdout). Every line is written for a curious beginner: no
/// unexplained jargon, every number shown with its meaning.
pub(crate) const STAGE_TITLES: [&str; 6] = [
    "DATA",
    "VOCABULARY",
    "MODEL",
    "TRAINING",
    "EVALUATION",
    "USE",
];

pub(crate) const STAGE_COUNT: usize = 6;

/// A stage banner on stderr (the training lane's narration channel).
pub(crate) fn stage(index: usize) {
    eprintln!(
        "\n=== STAGE {index}/{}: {} ===",
        STAGE_COUNT,
        STAGE_TITLES[index - 1]
    );
}

/// A one-line explanation on stderr.
pub(crate) fn note(line: &str) {
    eprintln!("  {line}");
}

/// A stage banner on stdout (the guided demo's narration channel).
pub(crate) fn stage_stdout(index: usize) {
    println!(
        "\n=== STAGE {index}/{}: {} ===",
        STAGE_COUNT,
        STAGE_TITLES[index - 1]
    );
}

/// A one-line explanation on stdout.
pub(crate) fn note_stdout(line: &str) {
    println!("  {line}");
}
