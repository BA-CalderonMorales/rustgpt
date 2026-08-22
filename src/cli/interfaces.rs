pub(crate) enum Mode {
    Interactive,
    E2e { prompt: String },
    Ask { prompt: String },
    Eval,
    Train { path: String },
    Probe,
    Models,
    Demo,
}

pub(crate) struct Invocation {
    pub mode: Mode,
    pub seed: u64,
    pub model: Option<String>,
    pub epochs: usize,
    pub tiny: bool,
    pub trace: bool,
    pub temperature: f32,
    /// Number of seeded fluency samples for the tiny-lane eval yardstick.
    pub fluency: Option<usize>,
    /// Flat logit penalty per seen token (0.0 = off), tiny-lane eval only.
    pub presence: f32,
    /// Count-scaled logit penalty divisor (1.0 = off), tiny-lane eval only.
    pub repetition: f32,
    /// Nucleus (top-p) mass cutoff (0.0 = off), tiny-lane eval only.
    pub top_p: f32,
    /// Append " </s>" to every --tiny --train row: the E11 termination
    /// lever (default off = byte-identical rows).
    pub eos: bool,
    /// Linear per-epoch LR decay target for --tiny --train (None = constant
    /// LR, the W8 control; default off = byte-identical).
    pub lr_decay: Option<f32>,
}
