mod application;
mod cli;

fn main() {
    // The CLI owns every argument: mode, seed, model path, epochs, and the
    // trace flag. Machine modes default to seed 42; a bare interactive run
    // draws a random seed (pass --seed 42 for a reproducible session).
    let invocation = cli::parse_invocation();

    llm::set_seed(invocation.seed);

    // The catalog probe and the guided demo are pure reads: no datasets,
    // no water-cycle model build.
    if matches!(invocation.mode, cli::Mode::Models) {
        application::run_models();
        return;
    }
    if matches!(invocation.mode, cli::Mode::Demo) {
        application::run_demo();
        return;
    }

    // Two disjoint views: the interactive lane trains then chats (or, with
    // a loaded checkpoint, chats only); the headless lane serves exactly
    // one JSON object.
    let dataset = application::load_datasets();
    let mut model = application::build_llm(&dataset, &invocation);

    match invocation.mode {
        cli::Mode::Interactive => {
            application::run_interactive(&invocation, &dataset, &mut model);
        }
        _ => {
            application::run_headless(&invocation, &dataset, &mut model);
        }
    }
}
