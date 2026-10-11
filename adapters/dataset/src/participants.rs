//! Participant information from BIDS `participants.tsv` and phenotype tables.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use modewise_study::Participants;
use serde::Deserialize;

use crate::Pattern;

/// Tables with one row per subject, joined on the subject entity.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticipantsSpec {
    /// The recording entity the tables are keyed by.
    #[serde(default = "default_entity")]
    pub entity: String,
    pub tables: Vec<ParticipantTable>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParticipantTable {
    /// Tab-separated, relative to the dataset root.
    pub path: String,
    #[serde(default = "default_id_column")]
    pub id_column: String,
    /// How ids relate to the entity, such as `sub-{subject}`. Default: the id is the entity value.
    pub id_pattern: Option<String>,
    /// Values meaning "no value", besides empty cells and `n/a`, for example `-9` or `999`.
    #[serde(default)]
    pub missing: Vec<String>,
    /// Keep only these columns (default: all but the id).
    pub columns: Option<Vec<String>>,
    /// Put before every column name, to tell apart tables with the same column names.
    #[serde(default)]
    pub prefix: String,
}

fn default_entity() -> String {
    "subject".into()
}

fn default_id_column() -> String {
    "participant_id".into()
}

pub fn read_participants(root: &Path, spec: &ParticipantsSpec) -> Result<Participants, String> {
    let mut joined = Participants::default();
    for table in &spec.tables {
        let read = read_table(&root.join(&table.path), table, &spec.entity)
            .map_err(|e| format!("{}: {e}", table.path))?;
        joined = joined
            .join(read)
            .map_err(|e| format!("{}: {e}", table.path))?;
    }
    Ok(joined)
}

fn read_table(path: &Path, spec: &ParticipantTable, entity: &str) -> Result<Participants, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let pattern_text = spec
        .id_pattern
        .clone()
        .unwrap_or_else(|| format!("{{{entity}}}"));
    let pattern = Pattern::parse(&pattern_text).map_err(|e| e.to_string())?;
    if pattern.entities() != [entity] {
        return Err(format!(
            "id_pattern {pattern_text:?} must contain exactly {{{entity}}}"
        ));
    }
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or("empty file")?
        .split('\t')
        .map(str::trim)
        .collect();
    let id = header
        .iter()
        .position(|h| *h == spec.id_column)
        .ok_or(format!("no column {:?}", spec.id_column))?;
    let keep: Vec<usize> = match &spec.columns {
        Some(wanted) => wanted
            .iter()
            .map(|w| {
                header
                    .iter()
                    .position(|h| h == w)
                    .ok_or(format!("no column {w:?}"))
            })
            .collect::<Result<_, _>>()?,
        None => (0..header.len()).filter(|&i| i != id).collect(),
    };
    let missing: BTreeSet<&str> = spec
        .missing
        .iter()
        .map(String::as_str)
        .chain(["", "n/a", "NA"])
        .collect();
    let mut rows = BTreeMap::new();
    for line in lines {
        let cells: Vec<&str> = line.split('\t').map(str::trim).collect();
        let Some(subject) = cells
            .get(id)
            .and_then(|v| pattern.capture(v))
            .and_then(|m| m.get(entity).cloned())
        else {
            continue;
        };
        let values: BTreeMap<String, String> = keep
            .iter()
            .filter_map(|&i| {
                cells
                    .get(i)
                    .filter(|v| !missing.contains(*v))
                    .map(|v| (format!("{}{}", spec.prefix, header[i]), v.to_string()))
            })
            .collect();
        if rows.insert(subject.clone(), values).is_some() {
            return Err(format!("subject {subject:?} has more than one row"));
        }
    }
    if rows.is_empty() {
        return Err(format!(
            "no id in column {:?} matches {pattern_text:?}",
            spec.id_column
        ));
    }
    let columns = keep
        .iter()
        .map(|&i| format!("{}{}", spec.prefix, header[i]))
        .collect();
    Participants::new(columns, rows).map_err(|e| e.to_string())
}
