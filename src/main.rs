mod application;
mod cli;

fn main() {
    let mode = cli::parse_mode();
    let dataset = application::load_datasets();
    let mut llm = application::build_model(&dataset);
    application::run(mode, &dataset, &mut llm);
}
