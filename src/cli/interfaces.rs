pub(crate) enum Mode {
    Interactive,
    E2e { prompt: String },
    Eval,
}

pub(crate) struct Invocation {
    pub mode: Mode,
    pub seed: u64,
}
