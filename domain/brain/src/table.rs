use std::collections::{BTreeMap, HashSet};

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ChannelError {
    #[error("channel {0:?} appears more than once")]
    Duplicate(String),
    #[error("channel {0:?} is not in the channel table")]
    Unknown(String),
    #[error("{0}")]
    Invalid(String),
}

/// Which side of the brain a channel sits on. Closed on purpose: every atlas agrees there are
/// two. Midline channels (brainstem, vermis, EEG Cz) have none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Hemisphere {
    Left,
    Right,
}

impl Hemisphere {
    /// Accepts the spellings atlases use: left/right, L/R, LH/RH, lh/rh.
    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "left" | "l" | "lh" => Some(Self::Left),
            "right" | "r" | "rh" => Some(Self::Right),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

/// A region name as the atlas states it. Open, because AAL, Desikan-Killiany, Schaefer, Tian and
/// Glasser share almost no names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Region(String);

impl Region {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One channel: a name, optionally a region and hemisphere, and open tags such as
/// `yeo17 = DefaultB` or `lobe = frontal`, so several region definitions can sit side by side.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelInfo {
    name: String,
    region: Option<Region>,
    hemisphere: Option<Hemisphere>,
    tags: BTreeMap<String, String>,
}

impl ChannelInfo {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            region: None,
            hemisphere: None,
            tags: BTreeMap::new(),
        }
    }

    pub fn with_region(mut self, region: Region) -> Self {
        self.region = Some(region);
        self
    }

    pub fn with_hemisphere(mut self, hemisphere: Hemisphere) -> Self {
        self.hemisphere = Some(hemisphere);
        self
    }

    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn region(&self) -> Option<&Region> {
        self.region.as_ref()
    }

    pub fn hemisphere(&self) -> Option<Hemisphere> {
        self.hemisphere
    }

    pub fn tag(&self, key: &str) -> Option<&str> {
        self.tags.get(key).map(String::as_str)
    }

    pub fn tags(&self) -> &BTreeMap<String, String> {
        &self.tags
    }
}

/// The channels of a recording in order, with unique names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelTable(Vec<ChannelInfo>);

impl ChannelTable {
    pub fn new(channels: Vec<ChannelInfo>) -> Result<Self, ChannelError> {
        let mut seen = HashSet::new();
        if let Some(dup) = channels.iter().find(|c| !seen.insert(c.name.as_str())) {
            return Err(ChannelError::Duplicate(dup.name.clone()));
        }
        Ok(Self(channels))
    }

    /// Tables in order, as when cortex, subcortex and cerebellum come from different atlases.
    pub fn concat(tables: impl IntoIterator<Item = Self>) -> Result<Self, ChannelError> {
        Self::new(tables.into_iter().flat_map(|t| t.0).collect())
    }

    pub fn channels(&self) -> &[ChannelInfo] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn names(&self) -> Vec<String> {
        self.0.iter().map(|c| c.name.clone()).collect()
    }

    /// Channel info for each label, in the labels' order; every label must be in the table.
    pub fn lookup<'a>(&'a self, labels: &[String]) -> Result<Vec<&'a ChannelInfo>, ChannelError> {
        labels
            .iter()
            .map(|l| {
                self.0
                    .iter()
                    .find(|c| &c.name == l)
                    .ok_or_else(|| ChannelError::Unknown(l.clone()))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hemispheres_parse_common_spellings() {
        for (text, h) in [
            ("left", Hemisphere::Left),
            ("LH", Hemisphere::Left),
            ("l", Hemisphere::Left),
            ("Right", Hemisphere::Right),
            ("rh", Hemisphere::Right),
            ("R", Hemisphere::Right),
        ] {
            assert_eq!(Hemisphere::parse(text), Some(h), "{text}");
        }
        assert_eq!(Hemisphere::parse("midline"), None);
        assert_eq!(
            (Hemisphere::Left.as_str(), Hemisphere::Right.as_str()),
            ("left", "right")
        );
    }

    #[test]
    fn channel_info_carries_every_description() {
        let c = ChannelInfo::new("17networks_LH_DefaultB_PFCm_1")
            .with_region(Region::new("PFCm"))
            .with_hemisphere(Hemisphere::Left)
            .with_tag("yeo17", "DefaultB");
        assert_eq!(c.name(), "17networks_LH_DefaultB_PFCm_1");
        assert_eq!(c.region().map(Region::as_str), Some("PFCm"));
        assert_eq!(c.hemisphere(), Some(Hemisphere::Left));
        assert_eq!((c.tag("yeo17"), c.tag("lobe")), (Some("DefaultB"), None));
        assert_eq!(
            c.tags(),
            &BTreeMap::from([("yeo17".to_string(), "DefaultB".to_string())])
        );
    }

    #[test]
    fn tables_keep_order_reject_duplicates_and_look_up_labels() {
        let a = ChannelTable::new(vec![ChannelInfo::new("a"), ChannelInfo::new("b")]).unwrap();
        let c = ChannelTable::new(vec![ChannelInfo::new("c")]).unwrap();
        let both = ChannelTable::concat([a.clone(), c]).unwrap();
        assert_eq!(both.names(), ["a", "b", "c"]);
        assert_eq!((both.len(), both.is_empty()), (3, false));
        assert_eq!(
            ChannelTable::concat([a.clone(), a]).unwrap_err(),
            ChannelError::Duplicate("a".into())
        );
        let found = both.lookup(&["c".into(), "a".into()]).unwrap();
        assert_eq!(
            found.iter().map(|c| c.name()).collect::<Vec<_>>(),
            ["c", "a"]
        );
        assert_eq!(
            both.lookup(&["z".into()]).unwrap_err(),
            ChannelError::Unknown("z".into())
        );
        assert_eq!(both.channels().len(), 3);
        assert!(ChannelTable::new(vec![]).unwrap().is_empty());
    }
}
