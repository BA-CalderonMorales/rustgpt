mod application;
mod cli;

fn main() {
    let invocation = cli::parse_invocation();
    llm::set_seed(invocation.seed);
    let dataset = application::load_datasets();
    let train_path = match &invocation.mode {
        cli::Mode::Train { path } => Some(path.as_str()),
        _ => None,
    };
    let mut model = application::build_llm(
        &dataset,
        invocation.model.as_deref(),
        train_path,
        invocation.tiny,
    );
    application::run(invocation, &dataset, &mut model);
}
