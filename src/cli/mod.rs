mod interfaces;
mod logic;
mod usage;

pub(crate) use interfaces::{Invocation, Mode};
pub(crate) use logic::parse_invocation;

/// The mode flags in alphabetical order: the single home both contracts
/// derive from -- the mutual-exclusion error here, the usage-line list in
/// usage.rs (which adds each mode's value placeholder).
const MODE_NAMES: [&str; 7] = [
    "--ask", "--demo", "--e2e", "--eval", "--models", "--probe", "--train",
];

/// "a, b, ..., and z are mutually exclusive" -- the pinned parse-error body.
pub(crate) fn exclusive_error() -> String {
    let listed = match MODE_NAMES.len() {
        0 => String::new(),
        1 => MODE_NAMES[0].to_string(),
        n => format!(
            "{}, and {}",
            MODE_NAMES[..n - 1].join(", "),
            MODE_NAMES[n - 1]
        ),
    };
    format!("{listed} are mutually exclusive")
}
