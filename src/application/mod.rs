mod catalog;
mod catalog_table;
mod chat;
mod demo;
mod demo_use;
mod format;
mod logic;
mod narrate;
mod probe;
mod settings_table;
mod tiny;
mod trace;
mod train_lm;

// The facade: every cross-file caller resolves these names here, never
// through another file's internals.
pub(crate) use catalog::{resolve_model_arg, run_models};
pub(crate) use catalog_table::print_catalog_table;
pub(crate) use chat::{chat_loop, render_answer};
pub(crate) use demo::run_demo;
pub(crate) use demo_use::score_and_use;
pub(crate) use format::thousands;
pub(crate) use logic::{
    build_llm, build_tiny_llm, load_datasets, run_headless, run_interactive, save_checkpoint,
    trace_turn,
};
pub(crate) use narrate::{done, done_stdout, note, note_stdout, step, step_stdout};
pub(crate) use probe::run_probe;
pub(crate) use settings_table::print_pretraining;
pub(crate) use tiny::{run_tiny_eval, tiny_eval, tiny_heldout_stories};
pub(crate) use trace::Trace;
