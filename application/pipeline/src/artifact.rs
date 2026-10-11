use std::fmt;

use modewise_brain::ChannelTable;
use modewise_connectivity::FunctionalConnectivity;
use modewise_decomposition::{Decomposition, InstantaneousModes, Loadings};
use modewise_epochs::{Epochs, Events};
use modewise_signals::MultichannelSeries;
use modewise_spectral::TimeFrequencyMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Data that flows between pipeline steps.
#[derive(Debug, Clone, PartialEq)]
pub enum Artifact {
    Series(MultichannelSeries),
    Decomposition(Decomposition),
    TimeFrequency(TimeFrequencyMap),
    Events(Events),
    Epochs(Epochs),
    Instantaneous(InstantaneousModes),
    Loadings(Loadings),
    FunctionalConnectivity(FunctionalConnectivity),
    Channels(ChannelTable),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Series,
    Decomposition,
    TimeFrequency,
    Events,
    Epochs,
    Instantaneous,
    Loadings,
    FunctionalConnectivity,
    Channels,
}

impl Artifact {
    pub fn kind(&self) -> ArtifactKind {
        match self {
            Self::Series(_) => ArtifactKind::Series,
            Self::Decomposition(_) => ArtifactKind::Decomposition,
            Self::TimeFrequency(_) => ArtifactKind::TimeFrequency,
            Self::Events(_) => ArtifactKind::Events,
            Self::Epochs(_) => ArtifactKind::Epochs,
            Self::Instantaneous(_) => ArtifactKind::Instantaneous,
            Self::Loadings(_) => ArtifactKind::Loadings,
            Self::FunctionalConnectivity(_) => ArtifactKind::FunctionalConnectivity,
            Self::Channels(_) => ArtifactKind::Channels,
        }
    }
}

impl fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Series => "series",
            Self::Decomposition => "decomposition",
            Self::TimeFrequency => "time_frequency",
            Self::Events => "events",
            Self::Epochs => "epochs",
            Self::Instantaneous => "instantaneous",
            Self::Loadings => "loadings",
            Self::FunctionalConnectivity => "functional_connectivity",
            Self::Channels => "channels",
        };
        f.write_str(name)
    }
}

/// SHA-256 content address of an artifact, as lowercase hex.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ArtifactKey(String);

impl ArtifactKey {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        Self(digest.iter().map(|b| format!("{b:02x}")).collect())
    }

    /// Keys a pipeline input by its content, so changed data never hits a stale cache.
    pub(crate) fn of_series(series: &MultichannelSeries) -> Self {
        let mut bytes = b"series".to_vec();
        bytes.extend(series.sampling_rate().hz().to_le_bytes());
        bytes.extend(series.n_channels().to_le_bytes());
        bytes.extend(series.n_samples().to_le_bytes());
        for label in series.labels() {
            bytes.extend(label.len().to_le_bytes());
            bytes.extend(label.as_bytes());
        }
        for value in series.data().iter() {
            bytes.extend(value.to_le_bytes());
        }
        Self::from_bytes(&bytes)
    }

    pub(crate) fn of_events(events: &Events) -> Self {
        let mut bytes = b"events".to_vec();
        for e in events.as_slice() {
            bytes.extend(e.onset_s.to_le_bytes());
            bytes.extend(e.duration_s.to_le_bytes());
            bytes.extend(e.label.len().to_le_bytes());
            bytes.extend(e.label.as_bytes());
        }
        Self::from_bytes(&bytes)
    }

    pub(crate) fn of_channels(table: &ChannelTable) -> Self {
        let mut bytes = b"channels".to_vec();
        let mut text = |s: &str| {
            bytes.extend(s.len().to_le_bytes());
            bytes.extend(s.as_bytes());
        };
        for c in table.channels() {
            text(c.name());
            text(c.region().map_or("\0", |r| r.as_str()));
            text(c.hemisphere().map_or("\0", |h| h.as_str()));
            for (k, v) in c.tags() {
                text(k);
                text(v);
            }
            text("\0end");
        }
        Self::from_bytes(&bytes)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ArtifactKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_lowercase_sha256_hex() {
        let key = ArtifactKey::from_bytes(b"abc");
        assert_eq!(
            key.as_str(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(key.to_string(), key.as_str());
    }
}
