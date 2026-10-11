use std::sync::Arc;

use thiserror::Error;

use crate::{Artifact, ArtifactKey};

#[derive(Debug, Error)]
#[error("artifact store failed: {0}")]
pub struct StoreError(pub String);

/// Where step results live. Implemented by adapters (memory, Zarr, ...).
pub trait ArtifactStore: Send + Sync {
    fn get(&self, key: &ArtifactKey) -> Result<Option<Arc<Artifact>>, StoreError>;
    fn put(&self, key: &ArtifactKey, artifact: Arc<Artifact>) -> Result<(), StoreError>;
    fn contains(&self, key: &ArtifactKey) -> Result<bool, StoreError>;
}
