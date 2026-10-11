//! Keeps artifacts in memory for the life of the process.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use modewise_pipeline::{Artifact, ArtifactKey, ArtifactStore, StoreError};

#[derive(Default)]
pub struct MemoryStore {
    artifacts: Mutex<HashMap<ArtifactKey, Arc<Artifact>>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.lock().map(|m| m.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> Result<MutexGuard<'_, HashMap<ArtifactKey, Arc<Artifact>>>, StoreError> {
        self.artifacts
            .lock()
            .map_err(|_| StoreError("memory store lock poisoned".into()))
    }
}

impl ArtifactStore for MemoryStore {
    fn get(&self, key: &ArtifactKey) -> Result<Option<Arc<Artifact>>, StoreError> {
        Ok(self.lock()?.get(key).cloned())
    }

    fn put(&self, key: &ArtifactKey, artifact: Arc<Artifact>) -> Result<(), StoreError> {
        self.lock()?.insert(key.clone(), artifact);
        Ok(())
    }

    fn contains(&self, key: &ArtifactKey) -> Result<bool, StoreError> {
        Ok(self.lock()?.contains_key(key))
    }
}

#[cfg(test)]
mod tests {
    use modewise_pipeline::ArtifactKey;
    use modewise_signals::{MultichannelSeries, SamplingRate};
    use ndarray::array;

    use super::*;

    fn key(text: &str) -> ArtifactKey {
        serde_json::from_value(serde_json::json!(text)).unwrap()
    }

    #[test]
    fn stores_and_returns_artifacts() {
        let store = MemoryStore::new();
        assert!(store.is_empty());
        let series = MultichannelSeries::new(
            array![[1.0, 2.0]],
            SamplingRate::new(1.0).unwrap(),
            vec!["a".into()],
        )
        .unwrap();
        let artifact = Arc::new(Artifact::Series(series));
        store.put(&key("k1"), artifact.clone()).unwrap();

        assert!(store.contains(&key("k1")).unwrap());
        assert!(!store.contains(&key("k2")).unwrap());
        assert_eq!(
            store.get(&key("k1")).unwrap().as_deref(),
            Some(artifact.as_ref())
        );
        assert!(store.get(&key("k2")).unwrap().is_none());
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }
}
