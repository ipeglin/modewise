use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use serde_json::json;

use crate::plan::{PlannedStep, Source};
use crate::{
    Artifact, ArtifactKey, ArtifactKind, ArtifactStore, Inputs, PipelineError, Plan, RunManifest,
    StepRecord,
};

pub struct RunResult {
    pub manifest: RunManifest,
    outputs: BTreeMap<String, ArtifactKey>,
}

impl RunResult {
    /// All step outputs, by `step.output`.
    pub fn outputs(&self) -> &BTreeMap<String, ArtifactKey> {
        &self.outputs
    }

    /// Key of a step output, by `step.output`, or by `step` when it has one output.
    pub fn output_key(&self, reference: &str) -> Option<&ArtifactKey> {
        if let Some(key) = self.outputs.get(reference) {
            return Some(key);
        }
        let prefix = format!("{reference}.");
        let mut matches = self
            .outputs
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix));
        match (matches.next(), matches.next()) {
            (Some((_, key)), None) => Some(key),
            _ => None,
        }
    }
}

/// Runs a plan. Steps whose outputs are already in the store are skipped.
pub fn run(
    plan: &Plan,
    inputs: &BTreeMap<String, Artifact>,
    store: &dyn ArtifactStore,
) -> Result<RunResult, PipelineError> {
    if let Some(extra) = inputs.keys().find(|k| !plan.inputs.contains(k)) {
        return Err(PipelineError::UnexpectedInput(extra.clone()));
    }
    let mut keys = BTreeMap::new();
    for name in &plan.inputs {
        let artifact = inputs
            .get(name)
            .ok_or_else(|| PipelineError::MissingInput(name.clone()))?;
        let key = match (plan.input_kind(name), artifact) {
            (Some(ArtifactKind::Series), Artifact::Series(s)) => ArtifactKey::of_series(s),
            (Some(ArtifactKind::Events), Artifact::Events(e)) => ArtifactKey::of_events(e),
            (Some(ArtifactKind::Channels), Artifact::Channels(t)) => ArtifactKey::of_channels(t),
            (expected, found) => {
                return Err(PipelineError::InputKind {
                    name: name.clone(),
                    expected: expected.unwrap_or(ArtifactKind::Series),
                    found: found.kind(),
                });
            }
        };
        if !store.contains(&key)? {
            store.put(&key, Arc::new(artifact.clone()))?;
        }
        keys.insert(name.clone(), key);
    }

    let mut manifest = RunManifest {
        pipeline: plan.name.clone(),
        modewise_version: env!("CARGO_PKG_VERSION").to_string(),
        inputs: keys.clone(),
        steps: Vec::with_capacity(plan.steps.len()),
    };
    // Every key follows from the recipe alone, so they are all known before anything runs.
    let mut outputs = BTreeMap::new();
    let mut step_keys = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        let k = keys_of(step, &keys);
        for (name, key) in &k.outputs {
            let full = format!("{}.{name}", step.id);
            keys.insert(full.clone(), key.clone());
            outputs.insert(full, key.clone());
        }
        step_keys.push(k);
    }

    // A stored step runs when its outputs are missing; an unstored one only when a step that
    // runs needs it. Later steps decide first, so walk backwards.
    let consumers = |i: usize| {
        let id = &plan.steps[i].id;
        (i + 1..plan.steps.len()).filter(move |&j| {
            plan.steps[j]
                .inputs
                .values()
                .any(|s| matches!(s, Source::Step { step, .. } if step == id))
        })
    };
    let mut needed = vec![false; plan.steps.len()];
    for i in (0..plan.steps.len()).rev() {
        needed[i] = if plan.steps[i].keep {
            let mut missing = false;
            for key in step_keys[i].outputs.values() {
                missing |= !store.contains(key)?;
            }
            missing
        } else {
            consumers(i).any(|j| needed[j])
        };
    }

    // Unstored outputs live here until their last consumer has run.
    let mut scratch: HashMap<ArtifactKey, Arc<Artifact>> = HashMap::new();
    let mut uses: HashMap<ArtifactKey, usize> = HashMap::new();
    for (_, k) in step_keys.iter().enumerate().filter(|(i, _)| needed[*i]) {
        for key in k.inputs.values() {
            *uses.entry(key.clone()).or_default() += 1;
        }
    }
    for (i, step) in plan.steps.iter().enumerate() {
        let k = &step_keys[i];
        let started = Instant::now();
        if needed[i] {
            execute(step, k, store, &mut scratch)?;
            for key in k.inputs.values() {
                if let Some(n) = uses.get_mut(key) {
                    *n -= 1;
                    if *n == 0 {
                        scratch.remove(key);
                    }
                }
            }
        }
        let descriptor = step.op.descriptor();
        manifest.steps.push(StepRecord {
            id: step.id.clone(),
            op: descriptor.name.to_string(),
            op_version: descriptor.version.to_string(),
            params: step.params.clone(),
            inputs: k.inputs.clone(),
            outputs: k.outputs.clone(),
            cached: !needed[i],
            stored: step.keep,
            seconds: started.elapsed().as_secs_f64(),
        });
    }
    Ok(RunResult { manifest, outputs })
}

struct StepKeys {
    inputs: BTreeMap<String, ArtifactKey>,
    outputs: BTreeMap<String, ArtifactKey>,
}

/// A step's input and output keys: a hash of the operation, its version, its parameters and its
/// input keys, so equal recipes share outputs.
fn keys_of(step: &PlannedStep, keys: &BTreeMap<String, ArtifactKey>) -> StepKeys {
    let descriptor = step.op.descriptor();
    let inputs: BTreeMap<String, ArtifactKey> = step
        .inputs
        .iter()
        .map(|(port, source)| (port.clone(), keys[&source.name()].clone()))
        .collect();
    let identity = json!({
        "op": descriptor.name,
        "version": descriptor.version,
        "params": step.params,
        "inputs": inputs,
    });
    let step_key = ArtifactKey::from_bytes(identity.to_string().as_bytes());
    let outputs = descriptor
        .outputs
        .iter()
        .map(|p| {
            let key = ArtifactKey::from_bytes(format!("{step_key}/{}", p.name).as_bytes());
            (p.name.to_string(), key)
        })
        .collect();
    StepKeys { inputs, outputs }
}

fn execute(
    step: &PlannedStep,
    keys: &StepKeys,
    store: &dyn ArtifactStore,
    scratch: &mut HashMap<ArtifactKey, Arc<Artifact>>,
) -> Result<(), PipelineError> {
    let descriptor = step.op.descriptor();
    let failed = |message: String| PipelineError::StepFailed {
        step: step.id.clone(),
        message,
    };
    let mut inputs = Inputs::new();
    for (port, key) in &keys.inputs {
        let artifact = match scratch.get(key) {
            Some(a) => Arc::clone(a),
            None => store
                .get(key)?
                .ok_or_else(|| failed(format!("input {port:?} is missing from the store")))?,
        };
        inputs.insert(port.clone(), artifact);
    }
    let mut produced = step.op.run(&step.params, &inputs).map_err(failed)?;
    for port in descriptor.outputs {
        let artifact = produced
            .remove(port.name)
            .ok_or_else(|| failed(format!("operation did not produce {:?}", port.name)))?;
        let expected = step.output_kinds[port.name];
        if artifact.kind() != expected {
            return Err(failed(format!(
                "output {:?} should be {}, got {}",
                port.name,
                expected,
                artifact.kind()
            )));
        }
        let key = &keys.outputs[port.name];
        if step.keep {
            store.put(key, Arc::new(artifact))?;
        } else {
            scratch.insert(key.clone(), Arc::new(artifact));
        }
    }
    if let Some(extra) = produced.keys().next() {
        return Err(failed(format!(
            "operation produced undeclared output {extra:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::PipelineSpec;
    use crate::test_support::{TestStore, ops, series};

    const CHAIN: &str = r#"
        [pipeline]
        name = "chain"
        inputs = ["x"]

        [[step]]
        id = "first"
        op = "scale"
        input = "x"
        params = { factor = 2.0 }

        [[step]]
        id = "second"
        op = "scale"
        input = "first"
        params = { factor = FACTOR }
    "#;

    fn chain(factor: f64) -> PipelineSpec {
        PipelineSpec::from_toml(&CHAIN.replace("FACTOR", &format!("{factor:?}"))).unwrap()
    }

    fn inputs(values: &[f64]) -> BTreeMap<String, Artifact> {
        BTreeMap::from([("x".to_string(), Artifact::Series(series(values)))])
    }

    fn cached(result: &RunResult) -> Vec<bool> {
        result.manifest.steps.iter().map(|s| s.cached).collect()
    }

    fn unkept(second_factor: f64) -> PipelineSpec {
        let text = CHAIN
            .replace("FACTOR", &format!("{second_factor:?}"))
            .replacen(
                "params = { factor = 2.0 }",
                "params = { factor = 2.0 }\n        keep = false",
                1,
            );
        PipelineSpec::from_toml(&text).unwrap()
    }

    #[test]
    fn unkept_outputs_stay_in_memory_and_are_recomputed_only_when_needed() {
        let ops = ops();
        let store = TestStore::default();
        let plan = Plan::new(&unkept(3.0), &ops.registry).unwrap();
        let first = run(&plan, &inputs(&[1.0, 2.0]), &store).unwrap();
        assert_eq!(cached(&first), [false, false]);
        assert!(
            !store.contains(first.output_key("first").unwrap()).unwrap(),
            "never written"
        );
        let Some(Artifact::Series(out)) = store
            .get(first.output_key("second").unwrap())
            .unwrap()
            .as_deref()
            .cloned()
        else {
            panic!("missing output");
        };
        assert_eq!(
            out.data().row(0).to_vec(),
            [6.0, 12.0],
            "computed from the in-memory output"
        );
        assert_eq!(
            first
                .manifest
                .steps
                .iter()
                .map(|s| s.stored)
                .collect::<Vec<_>>(),
            [false, true]
        );

        let again = run(&plan, &inputs(&[1.0, 2.0]), &store).unwrap();
        assert_eq!(
            cached(&again),
            [true, true],
            "nothing downstream is missing, so nothing runs"
        );
        assert_eq!(ops.scale.runs.load(Ordering::SeqCst), 2);

        let changed = run(
            &Plan::new(&unkept(4.0), &ops.registry).unwrap(),
            &inputs(&[1.0, 2.0]),
            &store,
        )
        .unwrap();
        assert_eq!(
            cached(&changed),
            [false, false],
            "the unkept step reruns because the next one must"
        );
        assert_eq!(ops.scale.runs.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn an_unkept_step_nothing_uses_is_refused() {
        let ops = ops();
        let text = CHAIN.replace("FACTOR", "3.0").replace(
            "params = { factor = 3.0 }",
            "params = { factor = 3.0 }\n        keep = false",
        );
        let err = Plan::new(&PipelineSpec::from_toml(&text).unwrap(), &ops.registry)
            .err()
            .unwrap();
        assert!(
            matches!(err, PipelineError::UnusedTransient(ref id) if id == "second"),
            "{err}"
        );
    }

    #[test]
    fn an_unkept_output_serves_every_step_that_uses_it() {
        let ops = ops();
        let store = TestStore::default();
        let text = r#"
            [pipeline]
            name = "fork"
            inputs = ["x"]

            [[step]]
            id = "a"
            op = "scale"
            input = "x"
            params = { factor = 2.0 }
            keep = false

            [[step]]
            id = "b"
            op = "scale"
            input = "a"
            params = { factor = 3.0 }

            [[step]]
            id = "c"
            op = "scale"
            input = "a"
            params = { factor = 5.0 }
        "#;
        let plan = Plan::new(&PipelineSpec::from_toml(text).unwrap(), &ops.registry).unwrap();
        let result = run(&plan, &inputs(&[1.0, 2.0]), &store).unwrap();
        for (id, want) in [("b", [6.0, 12.0]), ("c", [10.0, 20.0])] {
            let Some(Artifact::Series(out)) = store
                .get(result.output_key(id).unwrap())
                .unwrap()
                .as_deref()
                .cloned()
            else {
                panic!("missing {id}");
            };
            assert_eq!(out.data().row(0).to_vec(), want);
        }
    }

    #[test]
    fn chains_of_unkept_steps_are_computed_in_memory() {
        let ops = ops();
        let store = TestStore::default();
        let text = r#"
            [pipeline]
            name = "three"
            inputs = ["x"]

            [[step]]
            id = "a"
            op = "scale"
            input = "x"
            params = { factor = 2.0 }
            keep = false

            [[step]]
            id = "b"
            op = "scale"
            input = "a"
            params = { factor = 3.0 }
            keep = false

            [[step]]
            id = "c"
            op = "scale"
            input = "b"
            params = { factor = 5.0 }
        "#;
        let plan = Plan::new(&PipelineSpec::from_toml(text).unwrap(), &ops.registry).unwrap();
        let result = run(&plan, &inputs(&[1.0, 2.0]), &store).unwrap();
        let Some(Artifact::Series(out)) = store
            .get(result.output_key("c").unwrap())
            .unwrap()
            .as_deref()
            .cloned()
        else {
            panic!("missing output");
        };
        assert_eq!(out.data().row(0).to_vec(), [30.0, 60.0]);
        for id in ["a", "b"] {
            assert!(!store.contains(result.output_key(id).unwrap()).unwrap());
        }
    }

    #[test]
    fn computes_then_reuses_everything() {
        let ops = ops();
        let store = TestStore::default();
        let plan = Plan::new(&chain(3.0), &ops.registry).unwrap();

        let first = run(&plan, &inputs(&[1.0, 2.0]), &store).unwrap();
        assert_eq!(cached(&first), [false, false]);
        let key = first.output_key("second").unwrap();
        let Some(Artifact::Series(out)) = store.get(key).unwrap().as_deref().cloned() else {
            panic!("missing output");
        };
        assert_eq!(out.data().row(0).to_vec(), [6.0, 12.0]);

        let second = run(&plan, &inputs(&[1.0, 2.0]), &store).unwrap();
        assert_eq!(cached(&second), [true, true]);
        assert_eq!(ops.scale.runs.load(Ordering::SeqCst), 2);
        assert_eq!(second.output_key("second"), first.output_key("second"));
    }

    #[test]
    fn param_change_reruns_only_that_step_and_later() {
        let ops = ops();
        let store = TestStore::default();
        run(
            &Plan::new(&chain(3.0), &ops.registry).unwrap(),
            &inputs(&[1.0, 2.0]),
            &store,
        )
        .unwrap();

        let changed = run(
            &Plan::new(&chain(4.0), &ops.registry).unwrap(),
            &inputs(&[1.0, 2.0]),
            &store,
        )
        .unwrap();
        assert_eq!(cached(&changed), [true, false]);
    }

    #[test]
    fn new_input_data_reruns_everything() {
        let ops = ops();
        let store = TestStore::default();
        let plan = Plan::new(&chain(3.0), &ops.registry).unwrap();
        run(&plan, &inputs(&[1.0, 2.0]), &store).unwrap();
        let changed = run(&plan, &inputs(&[1.0, 2.5]), &store).unwrap();
        assert_eq!(cached(&changed), [false, false]);
    }

    #[test]
    fn rejects_missing_and_unexpected_inputs() {
        let ops = ops();
        let store = TestStore::default();
        let plan = Plan::new(&chain(3.0), &ops.registry).unwrap();
        let missing = run(&plan, &BTreeMap::new(), &store);
        assert!(matches!(missing, Err(PipelineError::MissingInput(n)) if n == "x"));

        let mut extra = inputs(&[1.0, 2.0]);
        extra.insert("y".into(), Artifact::Series(series(&[1.0, 2.0])));
        assert!(
            matches!(run(&plan, &extra, &store), Err(PipelineError::UnexpectedInput(n)) if n == "y")
        );
    }

    fn single_step(op: &str, params: &str) -> PipelineSpec {
        PipelineSpec::from_toml(&format!(
            "[pipeline]\nname = \"t\"\ninputs = [\"x\"]\n[[step]]\nid = \"s\"\nop = \"{op}\"\ninput = \"x\"\nparams = {params}\n"
        ))
        .unwrap()
    }

    #[test]
    fn rejects_operations_that_break_their_contract() {
        let ops = ops();
        for (lie, expected) in [
            ("missing", "did not produce"),
            ("wrong_kind", "should be series"),
            ("extra", "undeclared output"),
            ("fail", "asked to fail"),
        ] {
            let plan = Plan::new(
                &single_step("liar", &format!("{{ lie = \"{lie}\" }}")),
                &ops.registry,
            )
            .unwrap();
            let err = run(&plan, &inputs(&[1.0, 2.0]), &TestStore::default())
                .err()
                .unwrap();
            assert!(
                matches!(&err, PipelineError::StepFailed { step, message } if step == "s" && message.contains(expected)),
                "{lie}: {err}"
            );
        }
    }

    #[test]
    fn missing_input_in_store_fails_the_step() {
        let ops = ops();
        let plan = Plan::new(&chain(3.0), &ops.registry).unwrap();
        let err = run(
            &plan,
            &inputs(&[1.0, 2.0]),
            &crate::test_support::ForgetfulStore,
        )
        .err()
        .unwrap();
        assert!(
            matches!(&err, PipelineError::StepFailed { message, .. } if message.contains("missing from the store")),
            "{err}"
        );
    }

    #[test]
    fn output_keys_resolve_by_full_or_short_name() {
        let ops = ops();
        let spec = PipelineSpec::from_toml(
            "[pipeline]\nname = \"t\"\ninputs = [\"x\"]\n[[step]]\nid = \"split\"\nop = \"split\"\ninput = \"x\"\n[[step]]\nid = \"one\"\nop = \"scale\"\ninput = \"x\"\n",
        )
        .unwrap();
        let result = run(
            &Plan::new(&spec, &ops.registry).unwrap(),
            &inputs(&[1.0, 2.0]),
            &TestStore::default(),
        )
        .unwrap();
        assert_eq!(result.outputs().len(), 3);
        assert!(result.output_key("split.low").is_some());
        assert!(
            result.output_key("split").is_none(),
            "two outputs, so the short name is ambiguous"
        );
        assert_eq!(result.output_key("one"), result.output_key("one.signal"));
        assert!(result.output_key("nope").is_none());
    }

    #[test]
    fn manifest_records_resolved_params() {
        let ops = ops();
        let store = TestStore::default();
        let plan = Plan::new(&chain(3.0), &ops.registry).unwrap();
        let result = run(&plan, &inputs(&[1.0, 2.0]), &store).unwrap();
        let outputs = result.manifest.outputs();
        assert_eq!(
            outputs.keys().collect::<Vec<_>>(),
            ["first.signal", "second.signal"]
        );
        assert_eq!(&outputs, result.outputs());
        let json = result.manifest.to_json();
        assert!(json.contains("\"factor\": 3.0"), "{json}");
        assert!(json.contains("\"pipeline\": \"chain\""));
    }

    mod properties {
        use proptest::prelude::*;

        use super::*;

        proptest! {
            #[test]
            fn keys_are_stable_and_sensitive_to_every_sample(
                values in prop::collection::vec(-1e6f64..1e6, 2..32),
                at in any::<prop::sample::Index>(),
            ) {
                let ops = ops();
                let plan = Plan::new(&chain(3.0), &ops.registry).unwrap();
                let first = run(&plan, &inputs(&values), &TestStore::default()).unwrap();
                let again = run(&plan, &inputs(&values), &TestStore::default()).unwrap();
                prop_assert_eq!(first.outputs(), again.outputs());

                let mut changed = values.clone();
                let i = at.index(changed.len());
                changed[i] += 1.0;
                let other = run(&plan, &inputs(&changed), &TestStore::default()).unwrap();
                prop_assert_ne!(first.output_key("second"), other.output_key("second"));
            }
        }
    }
}
