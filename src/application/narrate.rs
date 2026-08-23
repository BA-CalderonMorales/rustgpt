//! Narration primitives shared by the application layer: the tiny lane
//! speaks on stderr (machine stdout untouched), the guided demo speaks on
//! stdout. Both lanes share one geometry -- a numbered step header,
//! three-space-indented detail lines, and a dot-bar DONE when the step's
//! work completes. Every line is written for a curious beginner: no
//! unexplained jargon, every number shown with its meaning.

/// A numbered step header on stderr (the training lane's channel).
pub(crate) fn step(number: usize, label: &str) {
    eprintln!("\n{number}) {label}");
}

/// The completion marker under the step that just finished: a dot bar in
/// the trainer's visual grammar, then DONE.
pub(crate) fn done() {
    eprintln!("   .......... DONE");
}

/// An indented explanation on stderr.
pub(crate) fn note(line: &str) {
    eprintln!("   {line}");
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
