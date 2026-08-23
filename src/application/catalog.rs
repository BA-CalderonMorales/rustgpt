/// The trained-model catalog: the durable record of what was made and how
/// (path, family, parameters, seed, recipe, eval). `--models` renders one
/// record per audience: a terminal gets the human table alone; a pipe gets
/// exactly one JSON object on stdout (the machine contract). The probe
/// never loads data or builds a model.
pub(crate) fn run_models() {
    // One catalog, one rendering per audience. A terminal that wanted JSON
    // can redirect: llm --models | cat.
    let catalog = load_catalog();
    let entries = match catalog.as_array() {
        Some(entries) => entries,
        None => {
            eprintln!("error: catalog is not an array");
            std::process::exit(1);
        }
    };

    // The audience split: interactive humans read the table (stderr rides
    // the same screen); machines parse the single JSON object.
    if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        super::print_catalog_table(entries);
    } else {
        println!(
            "{}",
            serde_json::json!({
                "status": "ok",
                "seed": llm::seed(),
                "catalog": catalog,
            })
        );
    }
}

/// The catalog's raw JSON array; a missing or broken catalog is a hard
/// error, because the catalog is the record the CLI names things by.
fn load_catalog() -> serde_json::Value {
    let text = std::fs::read_to_string("models/catalog.json")
        .expect("models/catalog.json must exist for --models");
    serde_json::from_str(&text).expect("models/catalog.json must be valid JSON")
}

/// Resolve a simple model name to its artifact path through the catalog:
/// ids are the namespace `--model` and `make run` share. A name that is
/// not an id resolves to nothing (the caller keeps the original).
pub(crate) fn resolve_model_path(id: &str) -> Option<String> {
    load_catalog()
        .as_array()
        .and_then(|entries| {
            entries
                .iter()
                .find(|entry| entry["id"].as_str() == Some(id))
        })
        .and_then(|entry| entry["path"].as_str().map(String::from))
}

/// Resolve the `--model` argument exactly once, at the parse boundary: a
/// simple name that is neither an existing file nor a path is a catalog id
/// and becomes its artifact path; anything else passes through unchanged.
/// Every later consumer -- loads, interactive's loaded-model check, save
/// targets -- then sees the same real path (the regression this pins: the
/// old code resolved for loading but let interactive re-check the raw id,
/// which fell into the training branch and wrote a stray artifact).
pub(crate) fn resolve_model_arg(model: &mut Option<String>) {
    if let Some(arg) = model {
        let looks_like_path = arg.contains('/') || arg.contains('\\');
        if !looks_like_path
            && !std::path::Path::new(arg).exists()
            && let Some(path) = resolve_model_path(arg)
        {
            *model = Some(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_model_path;

    #[test]
    fn catalog_ids_resolve_to_artifact_paths() {
        assert_eq!(
            resolve_model_path("watercycle-latest").as_deref(),
            Some("models/watercycle-latest.bin")
        );
        assert_eq!(
            resolve_model_path("stories-full").as_deref(),
            Some("models/tinystories/stories-full.bin")
        );
        assert_eq!(
            resolve_model_path("stories-demo").as_deref(),
            Some("models/tinystories/stories-demo.bin")
        );
    }

    #[test]
    fn unknown_ids_resolve_to_nothing() {
        assert_eq!(resolve_model_path("does-not-exist"), None);
    }
}
