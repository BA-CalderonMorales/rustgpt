mod application;
mod cli;

fn main() {
    let invocation = cli::parse_invocation();
    llm::set_seed(invocation.seed);
    let dataset = application::load_datasets();
    let mut model = application::build_model(&dataset);
    application::run(invocation, &dataset, &mut model);
}
