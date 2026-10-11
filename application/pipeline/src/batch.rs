use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::{Artifact, ArtifactStore, PipelineError, Plan, RunResult, run};

/// Supplies pipeline inputs for many recordings, for example files found on disk.
pub trait InputSource: Sync {
    /// Names of the inputs each recording provides.
    fn input_names(&self) -> Vec<String>;
    fn recording_ids(&self) -> Vec<String>;
    fn load(&self, id: &str) -> Result<BTreeMap<String, Artifact>, String>;
}

/// Outcome for one recording. A failure here does not stop the others.
pub struct RecordingRun {
    pub id: String,
    pub result: Result<RunResult, String>,
}

/// Runs a plan once per recording, `workers` recordings at a time (0 = all cores).
/// `on_done` sees each recording as soon as it finishes, for progress or saving.
pub fn run_batch(
    plan: &Plan,
    source: &dyn InputSource,
    ids: &[String],
    store: &dyn ArtifactStore,
    workers: usize,
    on_done: &(dyn Fn(&RecordingRun) + Sync),
) -> Result<Vec<RecordingRun>, PipelineError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| PipelineError::Spec(format!("cannot start {workers} workers: {e}")))?;
    let runs = pool.install(|| {
        ids.par_iter()
            .map(|id| {
                let result = source
                    .load(id)
                    .map_err(|e| format!("loading failed: {e}"))
                    .and_then(|mut inputs| {
                        // A source may offer more than this pipeline uses (such as a channel table).
                        inputs.retain(|name, _| plan.inputs().contains(name));
                        run(plan, &inputs, store).map_err(|e| e.to_string())
                    });
                let run = RecordingRun {
                    id: id.clone(),
                    result,
                };
                on_done(&run);
                run
            })
            .collect()
    });
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PipelineSpec;
    use crate::test_support::{TestStore, ops, series};

    struct Source;

    impl InputSource for Source {
        fn input_names(&self) -> Vec<String> {
            vec!["x".into()]
        }

        fn recording_ids(&self) -> Vec<String> {
            vec!["a".into(), "bad".into(), "b".into()]
        }

        fn load(&self, id: &str) -> Result<BTreeMap<String, Artifact>, String> {
            match id {
                "bad" => Err("file missing".into()),
                _ => Ok(BTreeMap::from([(
                    "x".into(),
                    Artifact::Series(series(&[1.0, id.len() as f64])),
                )])),
            }
        }
    }

    #[test]
    fn runs_each_recording_and_keeps_going_after_failures() {
        let spec = PipelineSpec::from_toml(
            "[pipeline]\nname = \"b\"\ninputs = [\"x\"]\n[[step]]\nid = \"s\"\nop = \"scale\"\ninput = \"x\"\n",
        )
        .unwrap();
        let ops = ops();
        let plan = Plan::new(&spec, &ops.registry).unwrap();
        let ids = Source.recording_ids();
        let done = std::sync::atomic::AtomicUsize::new(0);
        let count = |_: &RecordingRun| {
            done.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        };
        let runs = run_batch(&plan, &Source, &ids, &TestStore::default(), 2, &count).unwrap();
        assert_eq!(done.into_inner(), 3);

        let outcome: Vec<(&str, bool)> = runs
            .iter()
            .map(|r| (r.id.as_str(), r.result.is_ok()))
            .collect();
        assert_eq!(outcome, [("a", true), ("bad", false), ("b", true)]);
        assert!(
            runs[1]
                .result
                .as_ref()
                .err()
                .unwrap()
                .contains("file missing")
        );
    }
}
