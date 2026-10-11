//! Finds recordings on disk by file patterns and loads them as pipeline inputs.
//!
//! Patterns use `{entity}` placeholders, for example
//! `sub-{subject}/func/sub-{subject}_task-{task}_run-{run}_bold.h5`, so BIDS and
//! BIDS-like layouts of any modality are described the same way.

mod channels;
mod events_tsv;
mod participants;
mod pattern;
mod spec;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use modewise_brain::ChannelTable;
use modewise_hdf5::read_series;
use modewise_pipeline::{Artifact, InputSource};
use modewise_signals::{MultichannelSeries, SamplingRate};
use thiserror::Error;

pub use channels::{TableFormat, read_table, schaefer_channel, tian_channel};
pub use events_tsv::read as read_events_tsv;
pub use participants::{ParticipantTable, ParticipantsSpec, read_participants};
pub use pattern::Pattern;
pub use spec::{ChannelsSpec, DatasetSpec, Format, InputSpec, LayoutName, TableSpec};

use spec::SamplingRateSource;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("cannot read dataset spec: {0}")]
    Spec(String),
    #[error("invalid pattern {pattern:?}: {message}")]
    Pattern { pattern: String, message: String },
    #[error("cannot list {path}: {message}")]
    Io { path: String, message: String },
    #[error("filter on {0:?}, which the first input's pattern does not contain")]
    UnknownFilterEntity(String),
}

/// One matched set of entity values, with the file for each input.
#[derive(Debug, Clone, PartialEq)]
pub struct Recording {
    /// Entity values in the order they appear in the first input's pattern.
    pub entities: Vec<(String, String)>,
    pub paths: BTreeMap<String, PathBuf>,
}

impl Recording {
    /// Stable name such as `subject-01_task-rest_run-1`.
    pub fn id(&self) -> String {
        let parts: Vec<String> = self
            .entities
            .iter()
            .map(|(k, v)| format!("{k}-{v}"))
            .collect();
        parts.join("_")
    }

    fn entity_map(&self) -> BTreeMap<String, String> {
        self.entities.iter().cloned().collect()
    }
}

/// A dataset spec with its recordings discovered.
pub struct Dataset {
    spec: DatasetSpec,
    /// The spec's folder, which atlas and channel-table paths are relative to.
    base: PathBuf,
    root: PathBuf,
    recordings: Vec<Recording>,
    channels: Option<ChannelTable>,
    participants: Option<modewise_study::Participants>,
}

impl Dataset {
    /// Reads a dataset TOML file. A relative `root` is resolved against the file's folder.
    pub fn from_file(path: &Path) -> Result<Self, DatasetError> {
        let text = std::fs::read_to_string(path).map_err(|e| DatasetError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        let base = path.parent().unwrap_or(Path::new("."));
        Self::discover(DatasetSpec::from_toml(&text)?, base)
    }

    pub fn discover(spec: DatasetSpec, base: &Path) -> Result<Self, DatasetError> {
        let root = base.join(&spec.root);
        let Some(first) = spec.inputs.first() else {
            return Err(DatasetError::Spec("declare at least one [[input]]".into()));
        };

        let primary = Pattern::parse(&first.pattern)?;
        if let Some(unknown) = spec.filter.keys().find(|k| !primary.entities().contains(k)) {
            return Err(DatasetError::UnknownFilterEntity(unknown.clone()));
        }
        let others = spec.inputs[1..]
            .iter()
            .map(|i| Ok((i.name.clone(), Pattern::parse(&i.pattern)?)))
            .collect::<Result<Vec<_>, DatasetError>>()?;

        let mut recordings = Vec::new();
        for (path, entities) in primary.find(&root)? {
            let keep = spec.filter.iter().all(|(entity, allowed)| {
                entities
                    .iter()
                    .any(|(k, v)| k == entity && allowed.contains(v))
            });
            if !keep {
                continue;
            }
            let values: BTreeMap<String, String> = entities.iter().cloned().collect();
            let mut paths = BTreeMap::from([(first.name.clone(), path)]);
            for (name, pattern) in &others {
                paths.insert(name.clone(), root.join(pattern.fill(&values)?));
            }
            recordings.push(Recording { entities, paths });
        }
        let channels = match &spec.channels {
            Some(c) => {
                let tables = c
                    .tables
                    .iter()
                    .map(|t| read_table(&base.join(&t.path), t.format))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(DatasetError::Spec)?;
                Some(ChannelTable::concat(tables).map_err(|e| DatasetError::Spec(e.to_string()))?)
            }
            None => None,
        };
        let participants = match &spec.participants {
            Some(p) => Some(read_participants(&root, p).map_err(DatasetError::Spec)?),
            None => None,
        };
        Ok(Self {
            spec,
            base: base.to_path_buf(),
            root,
            recordings,
            channels,
            participants,
        })
    }

    /// Participant information from `[participants]`, if the spec has one.
    pub fn participants(&self) -> Option<&modewise_study::Participants> {
        self.participants.as_ref()
    }

    /// The channel table from `[channels]`, if the spec has one.
    pub fn channels(&self) -> Option<&ChannelTable> {
        self.channels.as_ref()
    }

    pub fn recordings(&self) -> &[Recording] {
        &self.recordings
    }

    pub fn recording(&self, id: &str) -> Option<&Recording> {
        self.recordings.iter().find(|r| r.id() == id)
    }

    fn load_input(&self, input: &InputSpec, recording: &Recording) -> Result<Artifact, String> {
        let path = &recording.paths[&input.name];
        match input.format {
            Format::Hdf5 => self.load_series(input, recording).map(Artifact::Series),
            Format::Nifti => self.load_nifti(input, recording).map(Artifact::Series),
            Format::EventsTsv => {
                let column = input.label_column.as_deref().unwrap_or("trial_type");
                events_tsv::read(path, column).map(Artifact::Events)
            }
        }
    }

    /// Channel names: a labels file, or the channel table when `names_for` lists this input.
    fn channel_names(&self, input: &InputSpec) -> Result<Option<Vec<String>>, String> {
        let named = self
            .spec
            .channels
            .as_ref()
            .is_some_and(|c| c.names_for.contains(&input.name));
        Ok(match (&input.labels_file, &self.channels) {
            (Some(file), _) => Some(read_labels(&self.root.join(file))?),
            (None, Some(table)) if named => Some(table.names()),
            _ => None,
        })
    }

    fn sampling_rate(
        &self,
        input: &InputSpec,
        recording: &Recording,
    ) -> Result<Option<f64>, String> {
        Ok(match input.sampling_rate_source() {
            SamplingRateSource::Hz(hz) if hz.is_nan() => None,
            SamplingRateSource::Hz(hz) => Some(hz),
            SamplingRateSource::Sidecar(sidecar) => {
                let relative = Pattern::parse(sidecar)
                    .and_then(|p| p.fill(&recording.entity_map()))
                    .map_err(|e| e.to_string())?;
                Some(sidecar_rate(&self.root.join(relative))?)
            }
        })
    }

    fn load_nifti(
        &self,
        input: &InputSpec,
        recording: &Recording,
    ) -> Result<MultichannelSeries, String> {
        let bold =
            modewise_nifti::read_bold(&recording.paths[&input.name]).map_err(|e| e.to_string())?;
        let hz =
            match self.sampling_rate(input, recording)? {
                Some(hz) => hz,
                None => 1.0 / bold.repetition_time_s.ok_or(
                    "the NIfTI header has no repetition time; give `sampling_rate_hz` or `sidecar`",
                )?,
            };
        let standardize = match input.standardize {
            spec::StandardizeName::None => modewise_parcellate::Standardize::None,
            spec::StandardizeName::ZscoreSample => modewise_parcellate::Standardize::ZscoreSample,
        };
        let mut rows = Vec::new();
        for atlas in &input.atlases {
            let path = self.base.join(atlas);
            let labels = modewise_nifti::read_labels(&path).map_err(|e| e.to_string())?;
            let (signals, _) = modewise_parcellate::parcellate(
                bold.data.view(),
                &bold.affine,
                labels.data.view(),
                &labels.affine,
                standardize,
            )
            .map_err(|e| format!("{}: {e}", path.display()))?;
            rows.push(signals);
        }
        let views: Vec<_> = rows.iter().map(|r| r.view()).collect();
        let data = ndarray::concatenate(ndarray::Axis(0), &views).map_err(|e| e.to_string())?;
        let labels = self
            .channel_names(input)?
            .unwrap_or_else(|| (0..data.nrows()).map(|c| format!("ch{c}")).collect());
        let fs = SamplingRate::new(hz).map_err(|e| e.to_string())?;
        MultichannelSeries::new(data, fs, labels).map_err(|e| e.to_string())
    }

    fn load_series(
        &self,
        input: &InputSpec,
        recording: &Recording,
    ) -> Result<MultichannelSeries, String> {
        let hz = self.sampling_rate(input, recording)?.unwrap_or(f64::NAN);
        let fs = SamplingRate::new(hz).map_err(|e| e.to_string())?;
        let labels = self.channel_names(input)?;
        let dataset = input.dataset.as_deref().unwrap_or_default();
        read_series(
            &recording.paths[&input.name],
            dataset,
            input.layout.into(),
            fs,
            labels,
        )
        .map_err(|e| e.to_string())
    }
}

impl InputSource for Dataset {
    fn input_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.spec.inputs.iter().map(|i| i.name.clone()).collect();
        names.extend(self.spec.channels.as_ref().map(|c| c.name.clone()));
        names
    }

    fn recording_ids(&self) -> Vec<String> {
        self.recordings.iter().map(Recording::id).collect()
    }

    fn load(&self, id: &str) -> Result<BTreeMap<String, Artifact>, String> {
        let recording = self
            .recording(id)
            .ok_or_else(|| format!("no recording {id:?}"))?;
        let mut loaded = self
            .spec
            .inputs
            .iter()
            .map(|input| {
                let series = self
                    .load_input(input, recording)
                    .map_err(|e| format!("input {:?}: {e}", input.name))?;
                Ok((input.name.clone(), series))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        if let (Some(spec), Some(table)) = (&self.spec.channels, &self.channels) {
            loaded.insert(spec.name.clone(), Artifact::Channels(table.clone()));
        }
        Ok(loaded)
    }
}

/// Reads `SamplingFrequency` (Hz) or `RepetitionTime` (s) from a BIDS JSON sidecar.
fn sidecar_rate(path: &Path) -> Result<f64, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("sidecar {}: {e}", path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("sidecar {}: {e}", path.display()))?;
    if let Some(hz) = json.get("SamplingFrequency").and_then(|v| v.as_f64()) {
        return Ok(hz);
    }
    if let Some(tr) = json.get("RepetitionTime").and_then(|v| v.as_f64()) {
        return Ok(1.0 / tr);
    }
    Err(format!(
        "sidecar {} has neither SamplingFrequency nor RepetitionTime",
        path.display()
    ))
}

/// One channel label per non-empty line.
fn read_labels(path: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("labels file {}: {e}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect())
}
