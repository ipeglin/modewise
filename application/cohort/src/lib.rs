//! From per-recording results to one cohort table (docs/features.md).

mod designs;
mod extract;

use std::collections::BTreeMap;

use modewise_pipeline::Artifact;
use modewise_study::{FeatureTable, Observation, StudyError};
use ndarray::Array2;
use thiserror::Error;

pub use designs::{Design, NamedComponent, Report, Settings, test};
pub use extract::{Band, Features};

#[derive(Debug, Error, PartialEq)]
pub enum CohortError {
    #[error("recording {recording:?}: {message}")]
    Recording { recording: String, message: String },
    #[error(
        "recording {recording:?} has features {found:?}, but earlier recordings had {expected:?}"
    )]
    Mismatch {
        recording: String,
        expected: usize,
        found: usize,
    },
    #[error("no recordings")]
    Empty,
    #[error(transparent)]
    Study(#[from] StudyError),
}

/// One recording's output, named by its recording id.
pub struct RecordingOutput<'a> {
    pub recording: &'a str,
    pub artifact: &'a Artifact,
}

/// Entities of a recording id such as `subject-01_task-rest_run-1`. Entity values are BIDS
/// labels (letters and digits), so `_` and the first `-` separate them unambiguously.
pub fn entities(recording: &str) -> BTreeMap<String, String> {
    recording
        .split('_')
        .filter_map(|part| part.split_once('-'))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// One row per recording, or per recording and condition when the output was computed on
/// epochs. Every recording must give the same features.
pub fn feature_table(
    outputs: &[RecordingOutput<'_>],
    features: &Features,
    subject_entity: &str,
) -> Result<FeatureTable, CohortError> {
    let mut observations = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut names: Option<Vec<String>> = None;
    for output in outputs {
        let fail = |message: String| CohortError::Recording {
            recording: output.recording.to_string(),
            message,
        };
        let extracted = features.extract(output.artifact).map_err(fail)?;
        let subject = entities(output.recording).remove(subject_entity);
        for (condition, row) in extracted.rows {
            match &names {
                None => names = Some(extracted.names.clone()),
                Some(expected) if *expected != extracted.names => {
                    return Err(CohortError::Mismatch {
                        recording: output.recording.to_string(),
                        expected: expected.len(),
                        found: extracted.names.len(),
                    });
                }
                Some(_) => {}
            }
            observations.push(Observation {
                recording: output.recording.to_string(),
                subject: subject.clone(),
                condition,
            });
            rows.push(row);
        }
    }
    let names = names.ok_or(CohortError::Empty)?;
    let values = Array2::from_shape_fn((rows.len(), names.len()), |(i, j)| rows[i][j]);
    Ok(FeatureTable::new(observations, names, values)?)
}
