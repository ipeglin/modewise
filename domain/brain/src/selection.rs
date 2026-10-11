use std::collections::BTreeMap;

use crate::table::{ChannelError, ChannelInfo, ChannelTable, Hemisphere};

/// Matches text exactly, or with `*` standing for any run of characters (`*AMY*`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern(String);

impl Pattern {
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    pub fn matches(&self, text: &str) -> bool {
        let parts: Vec<&str> = self.0.split('*').collect();
        if parts.len() == 1 {
            return self.0 == text;
        }
        let (first, last) = (parts[0], parts[parts.len() - 1]);
        if !text.starts_with(first)
            || text.len() < first.len() + last.len()
            || !text.ends_with(last)
        {
            return false;
        }
        let mut rest = &text[first.len()..text.len() - last.len()];
        for part in &parts[1..parts.len() - 1] {
            match rest.find(part) {
                Some(at) => rest = &rest[at + part.len()..],
                None => return false,
            }
        }
        true
    }
}

/// Channels matching every field that is set. A hemisphere rule leaves out midline channels,
/// and a region or tag rule leaves out channels that lack that description.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Rule {
    pub name: Option<Pattern>,
    pub region: Option<Pattern>,
    pub hemisphere: Option<Hemisphere>,
    pub tags: BTreeMap<String, Pattern>,
}

impl Rule {
    pub fn matches(&self, channel: &ChannelInfo) -> bool {
        self.name.as_ref().is_none_or(|p| p.matches(channel.name()))
            && self
                .region
                .as_ref()
                .is_none_or(|p| channel.region().is_some_and(|r| p.matches(r.as_str())))
            && self
                .hemisphere
                .is_none_or(|h| channel.hemisphere() == Some(h))
            && self
                .tags
                .iter()
                .all(|(key, p)| channel.tag(key).is_some_and(|v| p.matches(v)))
    }
}

/// Channels matching any of its rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection(Vec<Rule>);

impl Selection {
    pub fn new(rules: Vec<Rule>) -> Result<Self, ChannelError> {
        if rules.is_empty() {
            return Err(ChannelError::Invalid(
                "a selection needs at least one rule; an empty rule selects every channel".into(),
            ));
        }
        Ok(Self(rules))
    }

    pub fn matches(&self, channel: &ChannelInfo) -> bool {
        self.0.iter().any(|r| r.matches(channel))
    }

    /// Positions in `labels` of the selected channels, in the labels' order.
    pub fn positions(
        &self,
        table: &ChannelTable,
        labels: &[String],
    ) -> Result<Vec<usize>, ChannelError> {
        let infos = table.lookup(labels)?;
        let picked: Vec<usize> = infos
            .iter()
            .enumerate()
            .filter(|(_, c)| self.matches(c))
            .map(|(i, _)| i)
            .collect();
        if picked.is_empty() {
            return Err(ChannelError::Invalid(
                "the selection matches no channel".into(),
            ));
        }
        Ok(picked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::Region;

    fn table() -> ChannelTable {
        let ctx = |name: &str, h, net, region: Option<&str>| {
            let c = ChannelInfo::new(name)
                .with_hemisphere(h)
                .with_tag("yeo17", net);
            match region {
                Some(r) => c.with_region(Region::new(r)),
                None => c,
            }
        };
        ChannelTable::new(vec![
            ctx(
                "LH_DefaultB_PFCm_1",
                Hemisphere::Left,
                "DefaultB",
                Some("PFCm"),
            ),
            ctx(
                "RH_DefaultB_PFCm_1",
                Hemisphere::Right,
                "DefaultB",
                Some("PFCm"),
            ),
            ctx(
                "LH_LimbicA_TempPole_1",
                Hemisphere::Left,
                "LimbicA",
                Some("TempPole"),
            ),
            ctx("LH_SomMotB_1", Hemisphere::Left, "SomMotB", None),
            ChannelInfo::new("lAMY-rh")
                .with_region(Region::new("lAMY"))
                .with_hemisphere(Hemisphere::Right),
            ChannelInfo::new("vermis"),
        ])
        .unwrap()
    }

    fn picked(rules: Vec<Rule>) -> Vec<String> {
        let t = table();
        let positions = Selection::new(rules)
            .unwrap()
            .positions(&t, &t.names())
            .unwrap();
        positions.iter().map(|&i| t.names()[i].clone()).collect()
    }

    #[test]
    fn patterns_match_exactly_or_with_wildcards() {
        let p = Pattern::new;
        assert!(p("PFCm").matches("PFCm") && !p("PFCm").matches("PFCm2"));
        assert!(
            p("*AMY*").matches("lAMY") && p("*AMY*").matches("AMY") && !p("*AMY*").matches("HIP")
        );
        assert!(p("PFC*").matches("PFCv") && !p("PFC*").matches("aPFC"));
        assert!(p("*-rh").matches("lAMY-rh") && !p("*-rh").matches("lAMY-lh"));
        assert!(p("a*b*c").matches("axxbyyc") && !p("a*b*c").matches("acb"));
        assert!(
            !p("ab*ba").matches("aba"),
            "prefix and suffix may not overlap"
        );
        assert!(p("*").matches(""));
        assert!(
            !p("a*a").matches("a"),
            "one character cannot be both prefix and suffix"
        );
        assert!(
            !p("*aab*b*").matches("xaab"),
            "the second part must come after the first"
        );
    }

    #[test]
    fn rules_intersect_their_fields_and_selections_unite_rules() {
        let region = |r: &str| Rule {
            region: Some(Pattern::new(r)),
            ..Rule::default()
        };
        assert_eq!(
            picked(vec![region("PFCm")]),
            ["LH_DefaultB_PFCm_1", "RH_DefaultB_PFCm_1"]
        );
        let left_pfcm = Rule {
            hemisphere: Some(Hemisphere::Left),
            ..region("PFCm")
        };
        assert_eq!(picked(vec![left_pfcm]), ["LH_DefaultB_PFCm_1"]);
        assert_eq!(
            picked(vec![region("PFCm"), region("*AMY*")]),
            ["LH_DefaultB_PFCm_1", "RH_DefaultB_PFCm_1", "lAMY-rh"]
        );
        let limbic = Rule {
            tags: BTreeMap::from([("yeo17".into(), Pattern::new("Limbic*"))]),
            ..Rule::default()
        };
        assert_eq!(picked(vec![limbic]), ["LH_LimbicA_TempPole_1"]);
        let by_name = Rule {
            name: Some(Pattern::new("LH_*")),
            ..Rule::default()
        };
        assert_eq!(picked(vec![by_name]).len(), 3);
    }

    #[test]
    fn missing_descriptions_never_match_and_empty_rules_match_everything() {
        let left = Rule {
            hemisphere: Some(Hemisphere::Left),
            ..Rule::default()
        };
        assert!(
            !picked(vec![left]).contains(&"vermis".to_string()),
            "midline has no hemisphere"
        );
        let any_region = Rule {
            region: Some(Pattern::new("*")),
            ..Rule::default()
        };
        assert!(!picked(vec![any_region]).contains(&"LH_SomMotB_1".to_string()));
        assert_eq!(picked(vec![Rule::default()]).len(), 6);
        assert!(Selection::new(vec![]).is_err());
        let t = table();
        let none = Selection::new(vec![Rule {
            region: Some(Pattern::new("Nowhere")),
            ..Rule::default()
        }])
        .unwrap();
        assert!(none.positions(&t, &t.names()).is_err());
        assert!(none.positions(&t, &["unknown".into()]).is_err());
    }

    #[test]
    fn positions_follow_the_labels_not_the_table() {
        let t = table();
        let pfcm = Selection::new(vec![Rule {
            region: Some(Pattern::new("PFCm")),
            ..Rule::default()
        }])
        .unwrap();
        let labels = vec![
            "vermis".to_string(),
            "RH_DefaultB_PFCm_1".into(),
            "LH_DefaultB_PFCm_1".into(),
        ];
        assert_eq!(pfcm.positions(&t, &labels).unwrap(), [1, 2]);
    }
}
