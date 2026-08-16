mod logic;
mod tiny;

pub(crate) use logic::{build_llm, load_datasets, run};
pub(crate) use tiny::{run_tiny_eval, tiny_eval};
