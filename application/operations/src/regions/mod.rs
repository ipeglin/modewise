//! Regional selection and summaries (docs/regions.md).

mod subset;

use std::collections::BTreeMap;

use modewise_brain::{ChannelTable, Hemisphere, Pattern, Rule, Selection};
use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;
use subset::{
    Average, Keep, average_connectivity, combine_channels, keep_connectivity, keep_table, labels,
};

const WITH_CHANNELS: &[ArtifactKind] = &[
    ArtifactKind::Series,
    ArtifactKind::Decomposition,
    ArtifactKind::Instantaneous,
    ArtifactKind::TimeFrequency,
    ArtifactKind::Epochs,
    ArtifactKind::FunctionalConnectivity,
    ArtifactKind::Loadings,
    ArtifactKind::Channels,
];

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Side {
    Left,
    Right,
}

#[derive(Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
struct RuleParams {
    /// Channel name, exactly or with `*` wildcards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    /// Region label, exactly or with `*` wildcards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    region: Option<String>,
    /// `left` or `right`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hemisphere: Option<Side>,
    /// Tag values that must all match, for example `{ yeo17 = "Default*" }`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    tags: BTreeMap<String, String>,
}

impl RuleParams {
    fn rule(&self) -> Rule {
        Rule {
            name: self.name.as_deref().map(Pattern::new),
            region: self.region.as_deref().map(Pattern::new),
            hemisphere: self.hemisphere.map(|s| match s {
                Side::Left => Hemisphere::Left,
                Side::Right => Hemisphere::Right,
            }),
            tags: self
                .tags
                .iter()
                .map(|(k, v)| (k.clone(), Pattern::new(v)))
                .collect(),
        }
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SelectParams {
    /// Channels matching any rule are kept. A rule has any of `name`, `region`, `hemisphere` (`left`/`right`) and `tags` (a table); all of them must match. Text matches exactly, or with `*` wildcards.
    rules: Vec<RuleParams>,
}

impl SelectParams {
    fn selection(&self) -> Result<Selection, String> {
        Selection::new(self.rules.iter().map(RuleParams::rule).collect()).map_err(|e| e.to_string())
    }
}

fn channel_table(inputs: &Inputs) -> Result<&ChannelTable, String> {
    match input(inputs, "channels")? {
        Artifact::Channels(t) => Ok(t),
        other => Err(format!(
            "input \"channels\" must be a channel table, not a {}",
            other.kind()
        )),
    }
}

fn data_labels(data: &Artifact) -> Result<Vec<String>, String> {
    match data {
        Artifact::Channels(t) => Ok(t.names()),
        other => labels(other)
            .map(<[String]>::to_vec)
            .ok_or_else(|| format!("a {} has no channels", other.kind())),
    }
}

/// Keeps the channels that match any rule; works before or after decomposition.
pub struct SelectOperation;

impl Operation for SelectOperation {
    fn summary(&self) -> &'static str {
        "Keeps the channels whose name, region, hemisphere or tags match the rules (docs/regions.md)."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<SelectParams>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "select",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[
                Port {
                    name: "data",
                    kinds: WITH_CHANNELS,
                },
                Port {
                    name: "channels",
                    kinds: &[ArtifactKind::Channels],
                },
            ],
            outputs: &[Port {
                name: "selected",
                kinds: WITH_CHANNELS,
            }],
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        let (typed, resolved) = resolve_params::<SelectParams>(params)?;
        typed.selection()?;
        Ok(resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let selection = serde_json::from_value::<SelectParams>(params.clone())
            .map_err(|e| e.to_string())?
            .selection()?;
        let data = input(inputs, "data")?;
        let names = data_labels(data)?;
        let positions = selection
            .positions(channel_table(inputs)?, &names)
            .map_err(|e| e.to_string())?;
        let selected = match data {
            Artifact::FunctionalConnectivity(c) => {
                Artifact::FunctionalConnectivity(keep_connectivity(c, &positions)?)
            }
            Artifact::Channels(t) => Artifact::Channels(keep_table(t, &positions)?),
            other => combine_channels(
                other,
                &Keep {
                    positions: &positions,
                    labels: &names,
                },
            )?,
        };
        Ok(Outputs::from([("selected".into(), selected)]))
    }

    fn output_kind(&self, port: &Port, inputs: &BTreeMap<String, ArtifactKind>) -> ArtifactKind {
        inputs
            .get("data")
            .copied()
            .unwrap_or_else(|| port.output_kind())
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum GroupBy {
    Region,
    Hemisphere,
    Tag,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct AverageParams {
    /// Group channels by `region`, `hemisphere` or `tag`.
    by: GroupBy,
    /// Tag key to group by when `by = "tag"`, for example `yeo17`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tag: Option<String>,
    /// Keep left and right apart: `PFCm_left`, `PFCm_right`.
    #[serde(default)]
    split_hemispheres: bool,
}

impl AverageParams {
    fn check(&self) -> Result<(), String> {
        match (self.by, &self.tag) {
            (GroupBy::Tag, None) => Err("by = \"tag\" needs `tag`, the tag key to group by".into()),
            (GroupBy::Region | GroupBy::Hemisphere, Some(_)) => {
                Err("`tag` applies only with by = \"tag\"".into())
            }
            (GroupBy::Hemisphere, None) if self.split_hemispheres => {
                Err("split_hemispheres has no effect when grouping by hemisphere".into())
            }
            _ => Ok(()),
        }
    }

    /// Groups of positions in `names`, in order of first appearance. Channels without the
    /// grouping description are left out.
    fn groups(
        &self,
        table: &ChannelTable,
        names: &[String],
    ) -> Result<Vec<(String, Vec<usize>)>, String> {
        let infos = table.lookup(names).map_err(|e| e.to_string())?;
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, c) in infos.iter().enumerate() {
            let key = match self.by {
                GroupBy::Region => c.region().map(|r| r.as_str().to_string()),
                GroupBy::Hemisphere => c.hemisphere().map(|h| h.as_str().to_string()),
                GroupBy::Tag => self
                    .tag
                    .as_deref()
                    .and_then(|t| c.tag(t))
                    .map(str::to_string),
            };
            let Some(mut key) = key else { continue };
            if self.split_hemispheres
                && let Some(h) = c.hemisphere()
            {
                key = format!("{key}_{}", h.as_str());
            }
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, members)) => members.push(i),
                None => groups.push((key, vec![i])),
            }
        }
        if groups.is_empty() {
            return Err("no channel has the grouping description".into());
        }
        Ok(groups)
    }
}

/// Averages channels within each region, hemisphere or tag group.
pub struct RegionAverageOperation;

impl Operation for RegionAverageOperation {
    fn summary(&self) -> &'static str {
        "Averages channels by region, hemisphere or tag (functional connectivity in Fisher z) (docs/regions.md)."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<AverageParams>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        const AVERAGEABLE: &[ArtifactKind] = &[
            ArtifactKind::Series,
            ArtifactKind::Decomposition,
            ArtifactKind::TimeFrequency,
            ArtifactKind::Epochs,
            ArtifactKind::FunctionalConnectivity,
        ];
        OperationDescriptor {
            name: "region_average",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[
                Port {
                    name: "data",
                    kinds: AVERAGEABLE,
                },
                Port {
                    name: "channels",
                    kinds: &[ArtifactKind::Channels],
                },
            ],
            outputs: &[Port {
                name: "averaged",
                kinds: AVERAGEABLE,
            }],
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        let (typed, resolved) = resolve_params::<AverageParams>(params)?;
        typed.check()?;
        Ok(resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let p: AverageParams = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
        let data = input(inputs, "data")?;
        if let Artifact::Epochs(e) = data
            && matches!(e.source(), modewise_epochs::EpochSource::Instantaneous(_))
        {
            return Err(
                "averaging instantaneous frequency across channels is not meaningful".into(),
            );
        }
        let groups = p.groups(channel_table(inputs)?, &data_labels(data)?)?;
        let averaged = match data {
            Artifact::FunctionalConnectivity(c) => {
                Artifact::FunctionalConnectivity(average_connectivity(c, &groups)?)
            }
            other => combine_channels(other, &Average { groups })?,
        };
        Ok(Outputs::from([("averaged".into(), averaged)]))
    }

    fn output_kind(&self, port: &Port, inputs: &BTreeMap<String, ArtifactKind>) -> ArtifactKind {
        inputs
            .get("data")
            .copied()
            .unwrap_or_else(|| port.output_kind())
    }
}
