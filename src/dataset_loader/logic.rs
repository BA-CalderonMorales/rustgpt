use std::fs;

use csv::ReaderBuilder;

use super::{Dataset, DatasetType};

impl Dataset {
    pub fn new(
        pretraining_data_path: String,
        chat_training_data_path: String,
        type_of_data: DatasetType,
    ) -> Self {
        // Load both halves through the format's reader.
        let pretraining_data: Vec<String>;
        let chat_training_data: Vec<String>;
        match type_of_data {
            DatasetType::CSV => {
                pretraining_data = get_data_from_csv(pretraining_data_path);
                chat_training_data = get_data_from_csv(chat_training_data_path);
            }
            DatasetType::JSON => {
                pretraining_data = get_data_from_json(pretraining_data_path);
                chat_training_data = get_data_from_json(chat_training_data_path);
            }
        }

        // Assemble the dataset.
        Dataset {
            pretraining_data,
            chat_training_data,
        }
    }
}

fn get_data_from_json(path: String) -> Vec<String> {
    // Read the file and parse it as a flat list of strings.
    let data_json = fs::read_to_string(path).expect("Failed to read data file");
    serde_json::from_str(&data_json).expect("Failed to parse data file")
}

fn get_data_from_csv(path: String) -> Vec<String> {
    // Read the headerless CSV, one string per record.
    let file = fs::File::open(path).expect("Failed to open CSV file");
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result.expect("Failed to read CSV record");
        data.push(record.iter().collect::<Vec<_>>().join(","));
    }
    data
}

/// Load one JSON Lines corpus: one story per line.
///
/// Each line is either a bare string or a JSON object with a `text` field.
/// Lines that fail to parse are skipped, so a file can mix both forms.
pub fn load_jsonl(path: &str) -> Vec<String> {
    // Read the whole corpus; a missing file is a hard error.
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {path}: {error}"));

    // Each line is either a bare string or a JSON object with a `text`
    // field; unparseable lines are skipped.
    content
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .and_then(|value| {
                    value
                        .get("text")
                        .and_then(|text| text.as_str().map(String::from))
                        .or_else(|| value.as_str().map(String::from))
                })
        })
        .collect()
}
