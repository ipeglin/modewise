//! Pipeline engine: specs, operation registry, planning, cached execution.

mod artifact;
mod batch;
mod error;
mod executor;
mod manifest;
mod operation;
mod plan;
mod registry;
mod spec;
mod store;
#[cfg(test)]
mod test_support;

pub use artifact::{Artifact, ArtifactKey, ArtifactKind};
pub use batch::{InputSource, RecordingRun, run_batch};
pub use error::PipelineError;
pub use executor::{RunResult, run};
pub use manifest::{RunManifest, StepRecord};
pub use operation::{
    Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params, schema_of,
};
pub use plan::Plan;
pub use registry::Registry;
pub use spec::{InputDecl, InputSpec, PipelineHeader, PipelineSpec, StepSpec};
pub use store::{ArtifactStore, StoreError};
