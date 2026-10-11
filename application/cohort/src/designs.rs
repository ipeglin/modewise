//! Statistical designs built from a feature table (docs/statistics.md).

use std::collections::{BTreeMap, BTreeSet};

use modewise_stats::{
    Component, Edge, Exchange, Fdr, Glm, PermutationTest, Statistic, Tail, adjust,
};
use modewise_study::FeatureTable;
use ndarray::{Array1, Array2};

/// Which comparison to test.
#[derive(Debug, Clone, PartialEq)]
pub enum Design {
    /// Group `a` minus group `b` of a participant column. `welch` (no covariates) uses Welch's t.
    Groups {
        column: String,
        a: String,
        b: String,
        welch: bool,
    },
    /// Condition `a` minus condition `b` within each recording, by sign flipping.
    Paired { a: String, b: String },
    /// The slope of a numeric participant column.
    Correlation { score: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub covariates: Vec<String>,
    /// Use only rows of this condition (for tables built from epochs).
    pub condition: Option<String>,
    pub tail: Tail,
    pub n_permutations: usize,
    pub seed: u64,
    pub fdr: Fdr,
    /// Network-based statistic over edge features at this primary threshold.
    pub nbs_threshold: Option<f64>,
}

/// A component of the network-based statistic, with its edges by feature name.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedComponent {
    pub network: usize,
    pub edges: Vec<String>,
    pub p: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub feature_names: Vec<String>,
    pub t: Array1<f64>,
    pub df: Array1<f64>,
    pub p_parametric: Array1<f64>,
    pub p_permutation: Array1<f64>,
    pub p_fwer: Array1<f64>,
    pub q_fdr: Array1<f64>,
    pub components: Vec<NamedComponent>,
    pub n_observations: usize,
    /// Rows left out because a value the design needs is missing.
    pub dropped: usize,
    pub n_rearrangements: usize,
    pub exhaustive: bool,
    pub regressors: Vec<String>,
}

/// Rows chosen for the design, the response and the regressors.
struct Prepared {
    y: Array2<f64>,
    columns: Vec<(String, Vec<f64>)>,
    tested: usize,
    exchange: Exchange,
    in_a: Option<Vec<bool>>,
    dropped: usize,
}

pub fn test(table: &FeatureTable, design: &Design, settings: &Settings) -> Result<Report, String> {
    let prepared = prepare(table, design, settings)?;
    let n = prepared.y.nrows();
    let regressors: Vec<String> = prepared
        .columns
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    let statistic = match prepared.in_a {
        Some(in_a) => {
            if in_a.iter().filter(|&&g| g).count() < 2 || in_a.iter().filter(|&&g| !g).count() < 2 {
                return Err("each group needs at least two observations".into());
            }
            Statistic::Welch { in_a }
        }
        None => {
            let design = Array2::from_shape_fn((n, prepared.columns.len()), |(i, c)| {
                prepared.columns[c].1[i]
            });
            Statistic::Glm(Glm::new(design, prepared.tested).map_err(|e| e.to_string())?)
        }
    };
    let edges = settings
        .nbs_threshold
        .map(|threshold| (edge_map(table.feature_names()), threshold));
    let result = PermutationTest {
        statistic,
        exchange: prepared.exchange,
        tail: settings.tail,
        n_permutations: settings.n_permutations,
        seed: settings.seed,
        nbs: edges,
    }
    .run(prepared.y.view())
    .map_err(|e| e.to_string())?;
    let q = adjust(
        result.p_permutation.as_slice().unwrap_or_default(),
        settings.fdr,
    );
    let names = table.feature_names();
    let components = result
        .components
        .into_iter()
        .map(|c: Component| NamedComponent {
            network: c.network,
            edges: c.edges.iter().map(|&f| names[f].clone()).collect(),
            p: c.p,
        });
    Ok(Report {
        feature_names: names.to_vec(),
        t: result.t,
        df: result.df,
        p_parametric: result.p_parametric,
        p_permutation: result.p_permutation,
        p_fwer: result.p_fwer,
        q_fdr: Array1::from(q),
        components: components.collect(),
        n_observations: n,
        dropped: prepared.dropped,
        n_rearrangements: result.n_rearrangements,
        exhaustive: result.exhaustive,
        regressors,
    })
}

fn prepare(table: &FeatureTable, design: &Design, settings: &Settings) -> Result<Prepared, String> {
    let rows = table.observations();
    let conditions: BTreeSet<&str> = rows.iter().filter_map(|o| o.condition.as_deref()).collect();
    let in_condition = |i: usize| match &settings.condition {
        Some(c) => rows[i].condition.as_deref() == Some(c.as_str()),
        None => true,
    };
    if let Some(c) = &settings.condition
        && !conditions.contains(c.as_str())
    {
        return Err(format!("no condition {c:?}; the table has {conditions:?}"));
    }
    let metadata = |column: &str| table.metadata(column).map_err(|e| e.to_string());
    match design {
        Design::Paired { a, b } => {
            let mut pairs = Vec::new();
            let recordings: BTreeSet<&str> = rows.iter().map(|o| o.recording.as_str()).collect();
            for recording in recordings {
                let row = |cond: &str| {
                    rows.iter().position(|o| {
                        o.recording == recording && o.condition.as_deref() == Some(cond)
                    })
                };
                if let (Some(i), Some(j)) = (row(a), row(b)) {
                    pairs.push((i, j));
                }
            }
            if pairs.is_empty() {
                return Err(format!(
                    "no recording has both conditions {a:?} and {b:?}; the table has {conditions:?}"
                ));
            }
            check_one_per_subject(table, pairs.iter().map(|(i, _)| *i))?;
            let mut columns = vec![("intercept".to_string(), Vec::new())];
            let (kept, covariates, dropped) = covariate_columns(
                table,
                &settings.covariates,
                pairs.iter().map(|(i, _)| *i).collect(),
            )?;
            columns[0].1 = vec![1.0; kept.len()];
            columns.extend(covariates);
            let partner: BTreeMap<usize, usize> = pairs.into_iter().collect();
            let values = table.values();
            let y = Array2::from_shape_fn((kept.len(), values.ncols()), |(r, f)| {
                let i = kept[r];
                values[[i, f]] - values[[partner[&i], f]]
            });
            Ok(Prepared {
                y,
                columns,
                tested: 0,
                exchange: Exchange::SignFlip,
                in_a: None,
                dropped,
            })
        }
        Design::Groups {
            column,
            a,
            b,
            welch,
        } => {
            if *welch && !settings.covariates.is_empty() {
                return Err("Welch's t takes no covariates; use the GLM (welch = false)".into());
            }
            require_condition(&conditions, settings)?;
            let labels = metadata(column)?;
            let candidates: Vec<usize> = (0..rows.len()).filter(|&i| in_condition(i)).collect();
            let missing = candidates.iter().filter(|&&i| labels[i].is_none()).count();
            let chosen: Vec<usize> = candidates
                .into_iter()
                .filter(|&i| matches!(labels[i].as_deref(), Some(v) if v == a || v == b))
                .collect();
            check_one_per_subject(table, chosen.iter().copied())?;
            let (kept, covariates, dropped) =
                covariate_columns(table, &settings.covariates, chosen)?;
            let in_a: Vec<bool> = kept
                .iter()
                .map(|&i| labels[i].as_deref() == Some(a.as_str()))
                .collect();
            let mut columns = vec![
                ("intercept".to_string(), vec![1.0; kept.len()]),
                (
                    format!("{column}={a}"),
                    in_a.iter().map(|&g| if g { 1.0 } else { 0.0 }).collect(),
                ),
            ];
            columns.extend(covariates);
            let y = table.values().select(ndarray::Axis(0), &kept);
            Ok(Prepared {
                y,
                columns,
                tested: 1,
                exchange: Exchange::Permute,
                in_a: welch.then_some(in_a),
                dropped: dropped + missing,
            })
        }
        Design::Correlation { score } => {
            require_condition(&conditions, settings)?;
            let candidates: Vec<usize> = (0..rows.len()).filter(|&i| in_condition(i)).collect();
            check_one_per_subject(table, candidates.iter().copied())?;
            let mut all = vec![score.clone()];
            all.extend(settings.covariates.iter().cloned());
            let (kept, mut columns, dropped) = covariate_columns(table, &all, candidates)?;
            if columns.first().map(|(name, _)| name.as_str()) != Some(score.as_str()) {
                return Err(format!("{score:?} must be numeric to be a score"));
            }
            columns.insert(0, ("intercept".to_string(), vec![1.0; kept.len()]));
            let y = table.values().select(ndarray::Axis(0), &kept);
            Ok(Prepared {
                y,
                columns,
                tested: 1,
                exchange: Exchange::Permute,
                in_a: None,
                dropped,
            })
        }
    }
}

fn require_condition(conditions: &BTreeSet<&str>, settings: &Settings) -> Result<(), String> {
    if !conditions.is_empty() && settings.condition.is_none() {
        return Err(format!(
            "the table has one row per condition {conditions:?}; choose one with `condition`"
        ));
    }
    Ok(())
}

fn check_one_per_subject(
    table: &FeatureTable,
    rows: impl Iterator<Item = usize>,
) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for i in rows {
        if let Some(s) = &table.observations()[i].subject
            && !seen.insert(s.clone())
        {
            return Err(format!(
                "subject {s:?} appears more than once; rows must be exchangeable, so keep one row per subject (average runs, or filter first)"
            ));
        }
    }
    Ok(())
}

/// Centred covariate columns for the rows that have every value. Numeric columns are used as
/// they are; text columns are dummy-coded against their first level. Returns the kept rows,
/// the columns and how many rows were dropped.
#[allow(clippy::type_complexity)]
fn covariate_columns(
    table: &FeatureTable,
    names: &[String],
    rows: Vec<usize>,
) -> Result<(Vec<usize>, Vec<(String, Vec<f64>)>, usize), String> {
    let values: Vec<&[Option<String>]> = names
        .iter()
        .map(|c| table.metadata(c).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let kept: Vec<usize> = rows
        .iter()
        .copied()
        .filter(|&i| values.iter().all(|v| v[i].is_some()))
        .collect();
    let dropped = rows.len() - kept.len();
    if kept.is_empty() {
        return Err("no rows have every value the design needs".into());
    }
    let mut columns = Vec::new();
    for (name, v) in names.iter().zip(&values) {
        let text: Vec<&str> = kept
            .iter()
            .map(|&i| v[i].as_deref().unwrap_or_default())
            .collect();
        let numbers: Option<Vec<f64>> = text.iter().map(|t| t.trim().parse().ok()).collect();
        match numbers {
            Some(x) => columns.push((name.clone(), centre(x))),
            None => {
                let levels: BTreeSet<&str> = text.iter().copied().collect();
                for level in levels.iter().skip(1) {
                    let dummy = text
                        .iter()
                        .map(|t| if t == level { 1.0 } else { 0.0 })
                        .collect();
                    columns.push((format!("{name}={level}"), centre(dummy)));
                }
            }
        }
    }
    Ok((kept, columns, dropped))
}

fn centre(x: Vec<f64>) -> Vec<f64> {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    x.into_iter().map(|v| v - mean).collect()
}

/// Edge features named `m{k}:{a}~{b}` (or `{a}~{b}`), mapped to networks and node indices.
fn edge_map(names: &[String]) -> Vec<Option<Edge>> {
    let mut nodes: BTreeMap<String, usize> = BTreeMap::new();
    let mut node = |label: &str| {
        let next = nodes.len();
        *nodes.entry(label.to_string()).or_insert(next)
    };
    names
        .iter()
        .map(|name| {
            let (network, pair) = match name.split_once(':') {
                Some((m, rest)) => (m.strip_prefix('m')?.parse().ok()?, rest),
                None => (0, name.as_str()),
            };
            let (a, b) = pair.split_once('~')?;
            Some(Edge {
                network,
                a: node(a),
                b: node(b),
            })
        })
        .collect()
}
