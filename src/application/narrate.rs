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

/// A numbered step header on stdout (the guided demo's channel): the
/// reader always knows which pipeline stage they are watching.
pub(crate) fn step_stdout(number: usize, label: &str) {
    println!("\n{number}) {label}");
}

/// The completion marker under the step that just finished: a dot bar in
/// the trainer's visual grammar, then DONE.
pub(crate) fn done_stdout() {
    println!("   .......... DONE");
}

/// An indented explanation on stdout, under the step it belongs to.
pub(crate) fn note_stdout(line: &str) {
    println!("   {line}");
}
