use llm::Config;

use super::note_stdout;

/// The guided demo's configuration table: the exact settings about to
/// steer pretraining, what each one decides in plain language, and where
/// to reach when experimenting. Every value comes from the same source
/// the run itself uses -- the table never paraphrases a second truth.
pub(crate) fn print_pretraining(vocab_words: usize, seed: u64, epochs: usize, lr: f32) {
    // Model shape comes straight from the tiny preset the tour builds.
    let config = Config::tiny();

    // The table: setting, applied value, what it decides, where to tweak.
    println!();
    println!(
        "   {:<15} {:<9} {:<34} TWEAK IT AT",
        "SETTING", "VALUE", "WHAT IT DECIDES"
    );
    row(
        "embedding_dim",
        config.embedding_dim,
        "width of each word's meaning",
        "src/configuration/constants.rs",
    );
    row(
        "hidden_dim",
        config.hidden_dim,
        "thinking room inside each block",
        "src/configuration/constants.rs",
    );
    row(
        "max_seq_len",
        config.max_seq_len,
        "how far back the model can look",
        "src/configuration/constants.rs",
    );
    row(
        "blocks",
        config.block_count,
        "reasoning layers, stacked",
        "src/configuration/constants.rs",
    );
    row(
        "vocabulary",
        vocab_words,
        "every word the model can say",
        "your corpus (one word list)",
    );
    row(
        "epochs",
        epochs,
        "full passes over the corpus",
        "--epochs <n>",
    );
    row(
        "learning_rate",
        lr,
        "size of each correction step",
        "--lr-decay <final_lr>",
    );
    row("seed", seed, "pins every random draw", "--seed <n>");

    // Where the levers live on a self-run: one command to copy.
    note_stdout(
        "Your own run sets these levers: llm --tiny --train <corpus.jsonl> --epochs 6 --eos --lr-decay 1e-4",
    );
}

/// One aligned table row; every column shares the header's widths.
fn row<T: std::fmt::Display>(setting: &str, value: T, decides: &str, tweak: &str) {
    println!("   {setting:<15} {value:<9} {decides:<34} {tweak}");
}
