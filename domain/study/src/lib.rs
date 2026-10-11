//! Cohort-level data: who the participants are, and features per recording, ready for statistics
//! or machine learning.

mod features;
mod participants;

use thiserror::Error;

pub use features::{FeatureTable, Observation};
pub use participants::Participants;

#[derive(Debug, Error, PartialEq)]
pub enum StudyError {
    #[error("column {0:?} appears in more than one participants table; give one a prefix")]
    DuplicateColumn(String),
    #[error("feature {0:?} appears more than once")]
    DuplicateFeature(String),
    #[error("no column {0:?}")]
    UnknownColumn(String),
    #[error("{column:?} of subject {subject:?} is {value:?}, not a number")]
    NotANumber {
        subject: String,
        column: String,
        value: String,
    },
    #[error("{0}")]
    Shape(String),
}
