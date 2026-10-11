use thiserror::Error;

use crate::{ArtifactKind, StoreError};

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("cannot read pipeline spec: {0}")]
    Spec(String),
    #[error("step id {0:?} is invalid: ids must be non-empty and must not contain '.'")]
    InvalidStepId(String),
    #[error("name {0:?} is used more than once")]
    DuplicateName(String),
    #[error("step {step:?}: unknown operation {op:?}")]
    UnknownOperation { step: String, op: String },
    #[error("step {step:?}: {message}")]
    InvalidParams { step: String, message: String },
    #[error("step {step:?}: {message}")]
    InvalidInputs { step: String, message: String },
    #[error("step {step:?}: input {reference:?} refers to nothing")]
    UnknownReference { step: String, reference: String },
    #[error(
        "step {step:?}: port {port:?} needs {}, but {reference:?} gives {found}",
        kinds(expected)
    )]
    KindMismatch {
        step: String,
        port: String,
        reference: String,
        expected: Vec<ArtifactKind>,
        found: ArtifactKind,
    },
    #[error("step {0:?} has keep = false but no step uses its output, so it would be lost")]
    UnusedTransient(String),
    #[error("steps depend on each other in a cycle: {0:?}")]
    Cycle(Vec<String>),
    #[error("pipeline input {0:?} was not provided")]
    MissingInput(String),
    #[error("pipeline input {name:?} must be {expected}, got {found}")]
    InputKind {
        name: String,
        expected: ArtifactKind,
        found: ArtifactKind,
    },
    #[error("{0:?} is not a declared pipeline input")]
    UnexpectedInput(String),
    #[error("step {step:?} failed: {message}")]
    StepFailed { step: String, message: String },
    #[error(transparent)]
    Store(#[from] StoreError),
}

fn kinds(kinds: &[ArtifactKind]) -> String {
    kinds
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" or ")
}
