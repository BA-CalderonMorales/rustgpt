mod catalog;
mod chat;
mod demo;
mod logic;
mod narrate;
mod probe;
mod tiny;
mod trace;
mod train_lm;

// The facade: every cross-file caller resolves these names here, never
// through another file's internals.
pub(crate) use catalog::{resolve_model_arg, run_models};
pub(crate) use chat::{chat_loop, render_answer};
pub(crate) use demo::run_demo;
pub(crate) use logic::{
    build_llm, build_tiny_llm, load_datasets, run_headless, run_interactive, save_checkpoint,
    trace_turn,
};
pub(crate) use narrate::{note, note_stdout, stage, stage_stdout};
pub(crate) use probe::run_probe;
pub(crate) use tiny::{run_tiny_eval, tiny_eval, tiny_heldout_stories};
pub(crate) use trace::Trace;
