/// The trained-model catalog: the durable record of what was made and how
/// (path, family, parameters, seed, recipe, eval). `--models` serves it as
/// exactly one JSON object; the probe never loads data or builds a model.
pub(crate) fn run_models() {
    // Read the catalog; a missing or broken catalog is a hard error.
    let text = std::fs::read_to_string("models/catalog.json")
        .expect("models/catalog.json must exist for --models");
    let catalog: serde_json::Value =
        serde_json::from_str(&text).expect("models/catalog.json must be valid JSON");

    // Exactly one JSON object on stdout.
    println!(
        "{}",
        serde_json::json!({
            "status": "ok",
            "catalog": catalog,
        })
    );
}
