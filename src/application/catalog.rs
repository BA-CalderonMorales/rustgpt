/// The trained-model catalog: the durable record of what was made and how
/// (path, family, parameters, seed, recipe, eval). `--models` serves it as
/// a human‑readable table; the probe never loads data or builds a model.
pub(crate) fn run_models() {
    let catalog = load_catalog();
    if let Some(arr) = catalog.as_array() {
        println!("ID          Family      Parameters   Path                Quality");
        println!("---------------------------------------------------------------");
        for entry in arr {
            let id = entry["id"].as_str().unwrap_or("");
            let family = entry["family"].as_str().unwrap_or("");
            let params = entry["parameters"]
                .as_u64()
                .map_or_else(|| "-".to_string(), |v| v.to_string());
            let path = entry["path"].as_str().unwrap_or("");
            let path_display = path.rsplit('/').next().unwrap_or(path);
            let quality = entry["quality"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_else(|| "-".to_string());
            println!(
                "{:12} {:12} {:12} {:25} {}",
                id, family, params, path_display, quality
            );
        }
    } else {
        eprintln!("error: catalog is not an array");
        std::process::exit(1);
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
            resolve_model_path("ts-13m-s42").as_deref(),
            Some("models/tinystories/ts-13m-s42.bin")
        );
    }

    #[test]
    fn unknown_ids_resolve_to_nothing() {
        assert_eq!(resolve_model_path("does-not-exist"), None);
    }
}
