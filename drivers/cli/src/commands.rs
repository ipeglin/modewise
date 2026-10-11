use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use modewise_dataset::Dataset;
use modewise_operations::builtin_registry;
use modewise_pipeline::{Artifact, InputSource, PipelineSpec, Plan, RecordingRun};
use modewise_results::ResultsDir;

use crate::{OK, PARTIAL};

pub(crate) fn run(
    pipeline: &Path,
    dataset: &Path,
    out: &Path,
    shard: Option<(usize, usize)>,
    workers: usize,
) -> Result<u8> {
    let text = std::fs::read_to_string(pipeline)
        .with_context(|| format!("reading {}", pipeline.display()))?;
    let plan = Plan::new(&PipelineSpec::from_toml(&text)?, &builtin_registry())?;
    let dataset = Dataset::from_file(dataset)?;
    let total = dataset.recording_ids().len();
    let shard_note = shard.map_or(String::new(), |(i, n)| format!(", shard {i}/{n}"));
    println!("{}: {total} recordings found{shard_note}", plan.name());

    let results = ResultsDir::open(out)?;
    let started = Instant::now();
    let report = |run: &RecordingRun| match &run.result {
        Ok(result) => {
            let steps = &result.manifest.steps;
            let cached = steps.iter().filter(|s| s.cached).count();
            println!(
                "  ok      {} ({cached}/{} steps cached)",
                run.id,
                steps.len()
            );
        }
        Err(e) => println!("  FAILED  {}: {e}", run.id),
    };
    let summary = results.run_batch(&plan, &dataset, shard, workers, &report)?;
    println!(
        "{} succeeded, {} failed in {:.1?}. Results: {}",
        summary.succeeded.len(),
        summary.failed.len(),
        started.elapsed(),
        out.display()
    );
    Ok(if summary.failed.is_empty() {
        OK
    } else {
        PARTIAL
    })
}

pub(crate) fn inspect(dataset: &Path, load: bool) -> Result<()> {
    let dataset = Dataset::from_file(dataset)?;
    let recordings = dataset.recordings();
    println!(
        "{} recordings, inputs {:?}",
        recordings.len(),
        dataset.input_names()
    );
    let mut values: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for recording in recordings {
        for (entity, value) in &recording.entities {
            let seen = values.entry(entity).or_default();
            if !seen.contains(&value.as_str()) {
                seen.push(value);
            }
        }
    }
    for (entity, seen) in &values {
        let shown: Vec<&str> = seen.iter().take(8).copied().collect();
        let more = seen.len().saturating_sub(shown.len());
        let more = if more > 0 {
            format!(" and {more} more")
        } else {
            String::new()
        };
        println!(
            "  {entity}: {} values: {}{more}",
            seen.len(),
            shown.join(", ")
        );
    }
    if load && let Some(first) = recordings.first() {
        let id = first.id();
        let inputs = dataset.load(&id).map_err(anyhow::Error::msg)?;
        println!("first recording {id}:");
        for (name, input) in inputs {
            match input {
                Artifact::Series(series) => println!(
                    "  {name}: {} channels x {} samples at {} Hz ({:.1} s)",
                    series.n_channels(),
                    series.n_samples(),
                    series.sampling_rate().hz(),
                    series.duration_s()
                ),
                Artifact::Events(events) => {
                    let mut labels: Vec<&str> =
                        events.as_slice().iter().map(|e| e.label.as_str()).collect();
                    labels.sort_unstable();
                    labels.dedup();
                    println!(
                        "  {name}: {} events, labels: {}",
                        events.len(),
                        labels.join(", ")
                    );
                }
                other => println!("  {name}: {}", other.kind()),
            }
        }
    }
    Ok(())
}

pub(crate) fn ops(json: bool, markdown: bool) {
    if json {
        println!("{:#}", modewise_operations::catalogue());
        return;
    }
    if markdown {
        print!("{}", modewise_operations::reference_markdown());
        return;
    }
    let registry = builtin_registry();
    for name in registry.names() {
        let Some(op) = registry.get(name) else {
            continue;
        };
        let d = op.descriptor();
        let ports = |ports: &[modewise_pipeline::Port]| {
            ports
                .iter()
                .map(|p| {
                    let kinds: Vec<String> = p.kinds.iter().map(ToString::to_string).collect();
                    format!("{} ({})", p.name, kinds.join(" or "))
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!(
            "{name} {}\n  {}\n  inputs:  {}\n  outputs: {}",
            d.version,
            op.summary(),
            ports(d.inputs),
            ports(d.outputs)
        );
    }
}

pub(crate) fn export(results: &Path, pipeline: Option<&str>, to: &Path) -> Result<()> {
    if !results.join("runs").is_dir() {
        bail!("{} is not a results folder (no runs/)", results.display());
    }
    let dir = ResultsDir::open(results)?;
    let pipelines = match pipeline {
        Some(p) => vec![p.to_string()],
        None => dir.pipelines(),
    };
    for pipeline in pipelines {
        let files = dir.export_hdf5(&pipeline, &to.join(&pipeline))?;
        println!(
            "{pipeline}: {} files in {}",
            files.len(),
            to.join(&pipeline).display()
        );
    }
    Ok(())
}
