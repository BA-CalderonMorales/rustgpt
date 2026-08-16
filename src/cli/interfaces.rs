pub(crate) enum Mode {
    Interactive,
    E2e { prompt: String },
    Eval,
    Train { path: String },
    Probe,
}

pub(crate) struct Invocation {
    pub mode: Mode,
    pub seed: u64,
    pub model: Option<String>,
    pub epochs: usize,
    pub tiny: bool,
}
