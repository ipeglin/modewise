//! Stores artifacts on disk as Zarr v3, readable by zarr-python 3 and xarray.
//!
//! Each artifact is a group at `<first two key characters>/<key>/` holding its
//! arrays (`data`, `modes` or `power`) and its metadata as group attributes.
//! The group's metadata is written last, so an interrupted write never counts as stored.

mod codec;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use modewise_pipeline::{Artifact, ArtifactKey, ArtifactStore, StoreError};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;

pub struct ZarrStore {
    root: PathBuf,
    storage: Arc<FilesystemStore>,
}

fn store_error(e: impl std::fmt::Display) -> StoreError {
    StoreError(e.to_string())
}

impl ZarrStore {
    /// Opens the store at `root`, creating it if needed.
    pub fn open(root: &Path) -> Result<Self, StoreError> {
        std::fs::create_dir_all(root).map_err(store_error)?;
        let storage = Arc::new(FilesystemStore::new(root).map_err(store_error)?);
        if !root.join("zarr.json").exists() {
            let group = GroupBuilder::new()
                .build(storage.clone(), "/")
                .map_err(store_error)?;
            group.store_metadata().map_err(store_error)?;
        }
        Ok(Self {
            root: root.to_path_buf(),
            storage,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn node(key: &ArtifactKey) -> String {
        let key = key.as_str();
        format!("/{}/{key}", &key[..2.min(key.len())])
    }
}

impl ArtifactStore for ZarrStore {
    fn get(&self, key: &ArtifactKey) -> Result<Option<Arc<Artifact>>, StoreError> {
        if !self.contains(key)? {
            return Ok(None);
        }
        codec::read(&self.storage, &Self::node(key)).map(|a| Some(Arc::new(a)))
    }

    fn put(&self, key: &ArtifactKey, artifact: Arc<Artifact>) -> Result<(), StoreError> {
        codec::write(&self.storage, &Self::node(key), &artifact)
    }

    fn contains(&self, key: &ArtifactKey) -> Result<bool, StoreError> {
        let group = Self::node(key);
        Ok(self.root.join(&group[1..]).join("zarr.json").exists())
    }
}
