use super::format::thousands;

/// The human side of `--models`, on stderr (stdout stays the machine
/// channel): one row per cataloged artifact, columns
/// sized to the content so long ids never collapse into their neighbors.
/// Parameters carry thousands separators; quality verdicts print verbatim
/// from the catalog (the record, not a paraphrase).
pub(crate) fn print_catalog_table(entries: &[serde_json::Value]) {
    // Gather rows: id, family, parameters, artifact filename, quality.
    // Arrays (not tuples) so column widths can index by position.
    let rows: Vec<[String; 5]> = entries
        .iter()
        .map(|entry| {
            let path = entry["path"].as_str().unwrap_or("");
            [
                entry["id"].as_str().unwrap_or("").to_string(),
                entry["family"].as_str().unwrap_or("").to_string(),
                entry["parameters"]
                    .as_u64()
                    .map_or_else(|| "-".to_string(), thousands),
                path.rsplit('/').next().unwrap_or(path).to_string(),
                entry["quality"]
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_else(|| "-".to_string()),
            ]
        })
        .collect();

    // Column widths from content; the id column is the widest name.
    let width = |column: usize| rows.iter().map(|row| row[column].len()).max().unwrap_or(0);
    let (id_w, family_w, params_w, artifact_w) = (width(0), width(1), width(2), width(3));

    // Families are counted so the header says what the catalog spans.
    let families: Vec<&String> = {
        let mut seen: Vec<&String> = Vec::new();
        for family in rows.iter().map(|row| &row[1]) {
            if !seen.contains(&family) {
                seen.push(family);
            }
        }
        seen
    };

    // Header, rule, rows: two-space gutter, two-space gutters between
    // columns, parameters right-aligned like every number table.
    eprintln!(
        "\nTrained models -- {} artifacts in {} families",
        rows.len(),
        families.len()
    );
    eprintln!(
        "  {id:<id_w$}  {family:<family_w$}  {params:>params_w$}  {artifact:<artifact_w$}  QUALITY",
        id = "ID",
        family = "FAMILY",
        params = "PARAMS",
        artifact = "ARTIFACT",
    );
    for row in &rows {
        eprintln!(
            "  {:<id_w$}  {:<family_w$}  {:>params_w$}  {:<artifact_w$}  {}",
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            id_w = id_w,
            family_w = family_w,
            params_w = params_w,
            artifact_w = artifact_w,
        );
    }
}
