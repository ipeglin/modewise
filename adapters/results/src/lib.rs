//! A results folder: cached artifacts plus a record of every run, reopenable later.
//!
//! ```text
//! <root>/store/                                   Zarr artifact store
//! <root>/runs/<pipeline>/<recording>.json         run manifest per recording
//! <root>/runs/<pipeline>/summary[-shard-i-of-n].json
//! ```

mod export;
mod tables;

pub use tables::write_feature_table;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use modewise_pipeline::{
    Artifact, ArtifactStore, InputSource, PipelineError, Plan, RecordingRun, RunManifest,
    StoreError, run_batch,
};
use modewise_store_zarr::ZarrStore;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ResultsError {
    #[error("{path}: {message}")]
    Io { path: String, message: String },
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
    #[error(
        "pipeline inputs {pipeline:?} are not all among the inputs the data provides {source_inputs:?}"
    )]
    InputMismatch {
        pipeline: Vec<String>,
        source_inputs: Vec<String>,
    },
    #[error("no run for recording {recording:?} of pipeline {pipeline:?}")]
    MissingRun { pipeline: String, recording: String },
    #[error("step {step:?} has keep = false, so its output {reference:?} was not stored")]
    NotStored { step: String, reference: String },
    #[error("no output {reference:?}; outputs are {available:?}")]
    MissingOutput {
        reference: String,
        available: Vec<String>,
    },
}

fn io(path: &Path, e: impl std::fmt::Display) -> ResultsError {
    ResultsError::Io {
        path: path.display().to_string(),
        message: e.to_string(),
    }
}

/// Which recordings of a batch succeeded, and why the others failed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSummary {
    pub pipeline: String,
    pub shard: Option<(usize, usize)>,
    pub succeeded: Vec<String>,
    pub failed: BTreeMap<String, String>,
}

pub struct ResultsDir {
    root: PathBuf,
    store: ZarrStore,
}

impl ResultsDir {
    /// Opens the folder, creating it if needed.
    pub fn open(root: &Path) -> Result<Self, ResultsError> {
        let store = ZarrStore::open(&root.join("store"))?;
        Ok(Self {
            root: root.to_path_buf(),
            store,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn store(&self) -> &ZarrStore {
        &self.store
    }

    fn runs_dir(&self, pipeline: &str) -> PathBuf {
        self.root.join("runs").join(file_name(pipeline))
    }

    pub fn save_run(
        &self,
        pipeline: &str,
        recording: &str,
        manifest: &RunManifest,
    ) -> Result<(), ResultsError> {
        let dir = self.runs_dir(pipeline);
        std::fs::create_dir_all(&dir).map_err(|e| io(&dir, e))?;
        let path = dir.join(format!("{}.json", file_name(recording)));
        std::fs::write(&path, manifest.to_json()).map_err(|e| io(&path, e))
    }

    pub fn save_summary(&self, summary: &BatchSummary) -> Result<PathBuf, ResultsError> {
        let dir = self.runs_dir(&summary.pipeline).join("summaries");
        std::fs::create_dir_all(&dir).map_err(|e| io(&dir, e))?;
        let name = match summary.shard {
            Some((i, n)) => format!("shard-{i}-of-{n}.json"),
            None => "all.json".to_string(),
        };
        let path = dir.join(name);
        let text = serde_json::to_string_pretty(summary).map_err(|e| io(&path, e))?;
        std::fs::write(&path, text).map_err(|e| io(&path, e))?;
        Ok(path)
    }

    /// Runs `plan` over the recordings in `ids` (all when `None`), saving each run as it
    /// finishes and a summary at the end. `on_done` sees each recording for progress.
    pub fn run_batch(
        &self,
        plan: &Plan,
        source: &dyn InputSource,
        shard: Option<(usize, usize)>,
        workers: usize,
        on_done: &(dyn Fn(&RecordingRun) + Sync),
    ) -> Result<BatchSummary, ResultsError> {
        let mut wanted = plan.inputs().to_vec();
        let mut given = source.input_names();
        wanted.sort();
        given.sort();
        // Inputs the pipeline does not use (such as a channel table) are left out.
        if !wanted.iter().all(|w| given.contains(w)) {
            return Err(ResultsError::InputMismatch {
                pipeline: wanted,
                source_inputs: given,
            });
        }
        let ids: Vec<String> = match shard {
            Some((i, n)) => source
                .recording_ids()
                .into_iter()
                .enumerate()
                .filter(|(j, _)| j % n == i - 1)
                .map(|(_, id)| id)
                .collect(),
            None => source.recording_ids(),
        };
        let save_failures = std::sync::Mutex::new(BTreeMap::new());
        let save = |run: &RecordingRun| {
            if let Ok(result) = &run.result
                && let Err(e) = self.save_run(plan.name(), &run.id, &result.manifest)
                && let Ok(mut failures) = save_failures.lock()
            {
                failures.insert(run.id.clone(), format!("saving run record failed: {e}"));
            }
            on_done(run);
        };
        let runs = run_batch(plan, source, &ids, &self.store, workers, &save)?;
        let save_failures = save_failures.into_inner().unwrap_or_default();

        let mut summary = BatchSummary {
            pipeline: plan.name().to_string(),
            shard,
            succeeded: Vec::new(),
            failed: BTreeMap::new(),
        };
        for run in runs {
            let error = run
                .result
                .err()
                .or_else(|| save_failures.get(&run.id).cloned());
            match error {
                None => summary.succeeded.push(run.id),
                Some(e) => {
                    summary.failed.insert(run.id, e);
                }
            }
        }
        self.save_summary(&summary)?;
        Ok(summary)
    }

    pub fn pipelines(&self) -> Vec<String> {
        list(&self.root.join("runs"), |p| p.is_dir())
    }

    /// Recordings with a saved run, sorted.
    pub fn recordings(&self, pipeline: &str) -> Vec<String> {
        list(&self.runs_dir(pipeline), |p| {
            p.extension().is_some_and(|e| e == "json")
        })
        .into_iter()
        .filter_map(|n| n.strip_suffix(".json").map(String::from))
        .collect()
    }

    pub fn manifest(&self, pipeline: &str, recording: &str) -> Result<RunManifest, ResultsError> {
        let path = self
            .runs_dir(pipeline)
            .join(format!("{}.json", file_name(recording)));
        let text = std::fs::read_to_string(&path).map_err(|_| ResultsError::MissingRun {
            pipeline: pipeline.to_string(),
            recording: recording.to_string(),
        })?;
        serde_json::from_str(&text).map_err(|e| io(&path, e))
    }

    /// A step output, by `step.output`, or by `step` when it has one output.
    pub fn output(
        &self,
        pipeline: &str,
        recording: &str,
        reference: &str,
    ) -> Result<Arc<Artifact>, ResultsError> {
        let manifest = self.manifest(pipeline, recording)?;
        let outputs = manifest.outputs();
        let prefix = format!("{reference}.");
        let mut candidates = outputs
            .iter()
            .filter(|(name, _)| *name == reference || name.starts_with(&prefix));
        let key = match (candidates.next(), candidates.next()) {
            (Some((_, key)), None) => key,
            _ => {
                return Err(ResultsError::MissingOutput {
                    reference: reference.to_string(),
                    available: outputs.keys().cloned().collect(),
                });
            }
        };
        let step = reference.split('.').next().unwrap_or(reference);
        if manifest.steps.iter().any(|s| s.id == step && !s.stored) {
            return Err(ResultsError::NotStored {
                step: step.to_string(),
                reference: reference.to_string(),
            });
        }
        self.store
            .get(key)?
            .ok_or_else(|| ResultsError::MissingOutput {
                reference: reference.to_string(),
                available: outputs.keys().cloned().collect(),
            })
    }

    /// Writes one HDF5 file per recording to `dir`, for MATLAB and other tools.
    pub fn export_hdf5(&self, pipeline: &str, dir: &Path) -> Result<Vec<PathBuf>, ResultsError> {
        std::fs::create_dir_all(dir).map_err(|e| io(dir, e))?;
        self.recordings(pipeline)
            .iter()
            .map(|recording| {
                let manifest = self.manifest(pipeline, recording)?;
                let path = dir.join(format!("{}.h5", file_name(recording)));
                export::write_recording(&self.store, recording, &manifest, &path)?;
                Ok(path)
            })
            .collect()
    }
}

/// Keeps names safe as file names on every platform.
fn file_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-_.".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn list(dir: &Path, keep: impl Fn(&Path) -> bool) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| keep(&e.path()))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}
