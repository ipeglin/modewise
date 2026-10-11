//! Cohort tables written for MATLAB, R and spreadsheets.

use std::path::Path;

use modewise_study::FeatureTable;

use crate::ResultsError;

/// Tab-separated: `recording`, `subject`, `condition`, the participant columns, then one column
/// per feature. Missing values and NaN are written as `n/a` (BIDS).
pub fn write_feature_table(path: &Path, table: &FeatureTable) -> Result<(), ResultsError> {
    let io = |e: std::io::Error| ResultsError::Io {
        path: path.display().to_string(),
        message: e.to_string(),
    };
    let metadata: Vec<&str> = table.metadata_columns().collect();
    let mut out = String::new();
    let header: Vec<&str> = ["recording", "subject", "condition"]
        .into_iter()
        .chain(metadata.iter().copied())
        .chain(table.feature_names().iter().map(String::as_str))
        .collect();
    out.push_str(&header.join("\t"));
    out.push('\n');
    let text = |v: Option<&str>| v.unwrap_or("n/a").to_string();
    for (row, o) in table.observations().iter().enumerate() {
        let mut cells = vec![
            o.recording.clone(),
            text(o.subject.as_deref()),
            text(o.condition.as_deref()),
        ];
        for column in &metadata {
            let values = table.metadata(column).map_err(|e| ResultsError::Io {
                path: path.display().to_string(),
                message: e.to_string(),
            })?;
            cells.push(text(values[row].as_deref()));
        }
        cells.extend(table.values().row(row).iter().map(|v| {
            if v.is_nan() {
                "n/a".to_string()
            } else {
                v.to_string()
            }
        }));
        out.push_str(&cells.join("\t"));
        out.push('\n');
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    std::fs::write(path, out).map_err(io)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use modewise_study::{Observation, Participants};

    use super::*;

    #[test]
    fn writes_ids_metadata_and_features_with_missing_as_na() {
        let rows = vec![
            Observation {
                recording: "subject-01".into(),
                subject: Some("01".into()),
                condition: Some("face".into()),
            },
            Observation {
                recording: "subject-02".into(),
                subject: None,
                condition: None,
            },
        ];
        let p = Participants::new(
            vec!["group".into()],
            BTreeMap::from([(
                "01".to_string(),
                BTreeMap::from([("group".to_string(), "Patient".to_string())]),
            )]),
        )
        .unwrap();
        let table = FeatureTable::new(
            rows,
            vec!["m0:a~b".into()],
            ndarray::array![[0.25], [f64::NAN]],
        )
        .unwrap()
        .with_participants(&p)
        .unwrap();
        let path =
            std::env::temp_dir().join(format!("modewise-features-{}/t.tsv", std::process::id()));
        write_feature_table(&path, &table).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text,
            "recording\tsubject\tcondition\tgroup\tm0:a~b\nsubject-01\t01\tface\tPatient\t0.25\nsubject-02\tn/a\tn/a\tn/a\tn/a\n"
        );
    }
}
