use std::collections::BTreeMap;

use modewise_hdf5::Layout;

use crate::channels::TableFormat;
use serde::Deserialize;

use crate::DatasetError;

/// Where a dataset lives and how to find and read its recordings.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetSpec {
    pub root: String,
    /// Keep only recordings whose entity takes one of the listed values.
    #[serde(default)]
    pub filter: BTreeMap<String, Vec<String>>,
    /// The first input defines which recordings exist; later inputs must exist for each.
    #[serde(rename = "input")]
    pub inputs: Vec<InputSpec>,
    /// What each channel is (region, hemisphere, tags), the same for every recording.
    pub channels: Option<ChannelsSpec>,
    /// Information about each subject, for group statistics and machine learning.
    pub participants: Option<crate::participants::ParticipantsSpec>,
}

/// Channel tables, read once and offered to pipelines as one more input.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelsSpec {
    /// Pipeline input name for the table.
    #[serde(default = "default_channels_name")]
    pub name: String,
    /// Label files in channel order, relative to the dataset spec's folder; concatenated.
    pub tables: Vec<TableSpec>,
    /// Signal inputs whose channels are named by the table, in order, instead of `ch0`, `ch1`, ….
    #[serde(default)]
    pub names_for: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableSpec {
    pub path: String,
    pub format: TableFormat,
}

fn default_channels_name() -> String {
    "channels".into()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSpec {
    /// Pipeline input this file feeds.
    pub name: String,
    pub pattern: String,
    pub format: Format,
    /// HDF5: path of the 2D dataset inside the file.
    pub dataset: Option<String>,
    #[serde(default)]
    pub layout: LayoutName,
    /// Fixed sampling rate. Give this or `sidecar`.
    pub sampling_rate_hz: Option<f64>,
    /// Pattern of a BIDS JSON sidecar holding `SamplingFrequency` or `RepetitionTime`.
    pub sidecar: Option<String>,
    /// Text file with one channel label per line, relative to the dataset root.
    pub labels_file: Option<String>,
    /// events_tsv: column holding each event's label (BIDS default: `trial_type`).
    pub label_column: Option<String>,
    /// nifti: label atlases (3D NIfTI), relative to the dataset spec's folder. Each gives one
    /// channel per label, in ascending label order; atlases are concatenated in this order.
    #[serde(default)]
    pub atlases: Vec<String>,
    /// nifti: scaling of each region's signal after averaging.
    #[serde(default)]
    pub standardize: StandardizeName,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StandardizeName {
    #[default]
    None,
    /// Minus the mean, divided by the sample SD (ddof = 1), as nilearn's `zscore_sample`.
    ZscoreSample,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Hdf5,
    /// A 4D NIfTI run, averaged per atlas label at load time (docs/volumes.md).
    Nifti,
    /// BIDS `events.tsv`: tab-separated `onset`, `duration` and a label column.
    EventsTsv,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutName {
    #[default]
    ChannelsByTime,
    TimeByChannels,
}

impl From<LayoutName> for Layout {
    fn from(name: LayoutName) -> Self {
        match name {
            LayoutName::ChannelsByTime => Layout::ChannelsByTime,
            LayoutName::TimeByChannels => Layout::TimeByChannels,
        }
    }
}

pub(crate) enum SamplingRateSource<'a> {
    Hz(f64),
    Sidecar(&'a str),
}

impl InputSpec {
    pub(crate) fn sampling_rate_source(&self) -> SamplingRateSource<'_> {
        match (&self.sampling_rate_hz, &self.sidecar) {
            (Some(hz), _) => SamplingRateSource::Hz(*hz),
            (None, Some(sidecar)) => SamplingRateSource::Sidecar(sidecar),
            (None, None) => SamplingRateSource::Hz(f64::NAN),
        }
    }

    fn validate(&self) -> Result<(), DatasetError> {
        let invalid = |m: &str| DatasetError::Spec(format!("input {:?}: {m}", self.name));
        match self.format {
            Format::Hdf5 => {
                if self.sampling_rate_hz.is_some() == self.sidecar.is_some() {
                    return Err(invalid(
                        "give exactly one of `sampling_rate_hz` or `sidecar`",
                    ));
                }
                if self.dataset.is_none() {
                    return Err(invalid(
                        "hdf5 inputs need `dataset`, the path inside the file",
                    ));
                }
                if self.label_column.is_some() {
                    return Err(invalid("`label_column` only applies to events_tsv inputs"));
                }
                if !self.atlases.is_empty() || self.standardize != StandardizeName::None {
                    return Err(invalid(
                        "`atlases` and `standardize` only apply to nifti inputs",
                    ));
                }
            }
            Format::Nifti => {
                if self.sampling_rate_hz.is_some() && self.sidecar.is_some() {
                    return Err(invalid(
                        "give at most one of `sampling_rate_hz` or `sidecar`; without either, the NIfTI header's TR is used",
                    ));
                }
                if self.atlases.is_empty() {
                    return Err(invalid(
                        "nifti inputs need `atlases`, the label volumes to average over",
                    ));
                }
                if self.dataset.is_some() || self.label_column.is_some() {
                    return Err(invalid(
                        "`dataset` and `label_column` do not apply to nifti inputs",
                    ));
                }
            }
            Format::EventsTsv => {
                let signal_only = [
                    ("dataset", self.dataset.is_some()),
                    ("sampling_rate_hz", self.sampling_rate_hz.is_some()),
                    ("sidecar", self.sidecar.is_some()),
                    ("labels_file", self.labels_file.is_some()),
                    ("atlases", !self.atlases.is_empty()),
                    ("standardize", self.standardize != StandardizeName::None),
                ];
                if let Some((field, _)) = signal_only.iter().find(|(_, set)| *set) {
                    return Err(invalid(&format!(
                        "`{field}` does not apply to events_tsv inputs"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl DatasetSpec {
    pub fn from_toml(text: &str) -> Result<Self, DatasetError> {
        let spec: Self = toml::from_str(text).map_err(|e| DatasetError::Spec(e.to_string()))?;
        if spec.inputs.is_empty() {
            return Err(DatasetError::Spec("declare at least one [[input]]".into()));
        }
        for input in &spec.inputs {
            input.validate()?;
        }
        if let Some(channels) = &spec.channels {
            channels.validate(&spec.inputs)?;
        }
        Ok(spec)
    }
}

impl ChannelsSpec {
    fn validate(&self, inputs: &[InputSpec]) -> Result<(), DatasetError> {
        let invalid = |m: String| DatasetError::Spec(format!("[channels]: {m}"));
        if self.tables.is_empty() {
            return Err(invalid("list at least one table".into()));
        }
        if inputs.iter().any(|i| i.name == self.name) {
            return Err(invalid(format!("name {:?} is already an input", self.name)));
        }
        for target in &self.names_for {
            match inputs.iter().find(|i| &i.name == target) {
                None => return Err(invalid(format!("names_for: no input {target:?}"))),
                Some(i) if i.format == Format::EventsTsv => {
                    return Err(invalid(format!(
                        "names_for: {target:?} is not a signal input"
                    )));
                }
                Some(i) if i.labels_file.is_some() => {
                    return Err(invalid(format!(
                        "names_for: {target:?} already has a labels_file"
                    )));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }
}
