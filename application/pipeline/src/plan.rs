use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::Value;

use crate::spec::{InputSpec, StepSpec};
use crate::{ArtifactKind, Operation, OperationDescriptor, PipelineError, PipelineSpec, Registry};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Source {
    Input(String),
    Step { step: String, output: String },
}

impl Source {
    /// Name under which the executor tracks this artifact.
    pub(crate) fn name(&self) -> String {
        match self {
            Self::Input(name) => name.clone(),
            Self::Step { step, output } => format!("{step}.{output}"),
        }
    }
}

pub(crate) struct PlannedStep {
    pub id: String,
    pub op: Arc<dyn Operation>,
    pub params: Value,
    pub inputs: BTreeMap<String, Source>,
    /// The kind each output will have, decided when the plan is checked.
    pub output_kinds: BTreeMap<String, ArtifactKind>,
    /// Whether the outputs go to the store (`keep` in the spec).
    pub keep: bool,
    /// Ports with the references bound to them, kept for checking kinds in dependency order.
    bindings: Vec<(crate::Port, String)>,
}

/// A checked pipeline: operations exist, parameters are valid, references resolve,
/// kinds match, and steps are in dependency order.
pub struct Plan {
    pub(crate) name: String,
    pub(crate) inputs: Vec<String>,
    pub(crate) input_kinds: BTreeMap<String, ArtifactKind>,
    pub(crate) steps: Vec<PlannedStep>,
}

impl Plan {
    pub fn new(spec: &PipelineSpec, registry: &Registry) -> Result<Self, PipelineError> {
        check_names(spec)?;
        let ops = spec
            .steps
            .iter()
            .map(|s| {
                let op = registry
                    .get(&s.op)
                    .ok_or_else(|| PipelineError::UnknownOperation {
                        step: s.id.clone(),
                        op: s.op.clone(),
                    })?;
                Ok((s.id.as_str(), op))
            })
            .collect::<Result<BTreeMap<_, _>, PipelineError>>()?;

        let mut planned = Vec::with_capacity(spec.steps.len());
        for step in &spec.steps {
            let op = Arc::clone(&ops[step.id.as_str()]);
            let descriptor = op.descriptor();
            let params = op
                .resolve(Value::Object(step.params.clone()))
                .map_err(|message| PipelineError::InvalidParams {
                    step: step.id.clone(),
                    message,
                })?;
            let bindings = bind_ports(step, &descriptor)?;
            let mut inputs = BTreeMap::new();
            for (port, reference) in &bindings {
                inputs.insert(
                    port.name.to_string(),
                    resolve_reference(spec, &ops, &step.id, reference)?,
                );
            }
            planned.push(PlannedStep {
                id: step.id.clone(),
                op,
                params,
                inputs,
                output_kinds: BTreeMap::new(),
                keep: step.keep,
                bindings,
            });
        }
        let mut steps = in_dependency_order(planned)?;
        check_kinds(spec, &mut steps)?;
        // An output that is neither stored nor used would be computed for nothing.
        if let Some(lost) = steps.iter().find(|s| {
            !s.keep
                && !steps.iter().any(|c| {
                    c.inputs
                        .values()
                        .any(|src| matches!(src, Source::Step { step, .. } if *step == s.id))
                })
        }) {
            return Err(PipelineError::UnusedTransient(lost.id.clone()));
        }

        let declared = spec.pipeline.inputs.kinds();
        Ok(Self {
            name: spec.pipeline.name.clone(),
            inputs: declared.iter().map(|(n, _)| n.clone()).collect(),
            input_kinds: declared.into_iter().collect(),
            steps,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    /// The kind of data a pipeline input expects.
    pub fn input_kind(&self, name: &str) -> Option<ArtifactKind> {
        self.input_kinds.get(name).copied()
    }

    /// Step ids in the order they run.
    pub fn step_ids(&self) -> impl Iterator<Item = &str> {
        self.steps.iter().map(|s| s.id.as_str())
    }
}

fn check_names(spec: &PipelineSpec) -> Result<(), PipelineError> {
    let mut seen = BTreeSet::new();
    let inputs = spec.pipeline.inputs.kinds();
    for (name, kind) in &inputs {
        if !matches!(
            kind,
            ArtifactKind::Series | ArtifactKind::Events | ArtifactKind::Channels
        ) {
            return Err(PipelineError::Spec(format!(
                "input {name:?}: only series, events and channels can be pipeline inputs, not {kind}"
            )));
        }
    }
    let names = inputs
        .iter()
        .map(|(n, _)| n)
        .chain(spec.steps.iter().map(|s| &s.id));
    for name in names {
        if name.is_empty() || name.contains('.') {
            return Err(PipelineError::InvalidStepId(name.clone()));
        }
        if !seen.insert(name) {
            return Err(PipelineError::DuplicateName(name.clone()));
        }
    }
    Ok(())
}

fn bind_ports(
    step: &StepSpec,
    descriptor: &OperationDescriptor,
) -> Result<Vec<(crate::Port, String)>, PipelineError> {
    let invalid = |message: String| PipelineError::InvalidInputs {
        step: step.id.clone(),
        message,
    };
    let port_names = || {
        descriptor
            .inputs
            .iter()
            .map(|p| p.name)
            .collect::<Vec<_>>()
            .join(", ")
    };
    match &step.input {
        InputSpec::Single(reference) => match descriptor.inputs {
            [port] => Ok(vec![(*port, reference.clone())]),
            _ => Err(invalid(format!(
                "operation {:?} takes inputs [{}]; name each one in a table",
                descriptor.name,
                port_names()
            ))),
        },
        InputSpec::Named(map) => {
            if let Some(unknown) = map
                .keys()
                .find(|k| !descriptor.inputs.iter().any(|p| p.name == *k))
            {
                return Err(invalid(format!(
                    "operation {:?} has no input {unknown:?}; it takes [{}]",
                    descriptor.name,
                    port_names()
                )));
            }
            descriptor
                .inputs
                .iter()
                .map(|port| {
                    map.get(port.name)
                        .map(|r| (*port, r.clone()))
                        .ok_or_else(|| invalid(format!("input {:?} is not connected", port.name)))
                })
                .collect()
        }
    }
}

fn resolve_reference(
    spec: &PipelineSpec,
    ops: &BTreeMap<&str, Arc<dyn Operation>>,
    step: &str,
    reference: &str,
) -> Result<Source, PipelineError> {
    if spec
        .pipeline
        .inputs
        .kinds()
        .iter()
        .any(|(n, _)| n == reference)
    {
        return Ok(Source::Input(reference.to_string()));
    }
    let unknown = || PipelineError::UnknownReference {
        step: step.to_string(),
        reference: reference.to_string(),
    };
    let (target, output) = match reference.split_once('.') {
        Some((target, output)) => (target, Some(output)),
        None => (reference, None),
    };
    let outputs = ops.get(target).ok_or_else(unknown)?.descriptor().outputs;
    let port = match (output, outputs) {
        (Some(name), _) => outputs
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(unknown)?,
        (None, [only]) => only,
        (None, _) => {
            let names: Vec<_> = outputs
                .iter()
                .map(|p| format!("{target}.{}", p.name))
                .collect();
            return Err(PipelineError::InvalidInputs {
                step: step.to_string(),
                message: format!("{reference:?} has several outputs; pick one of {names:?}"),
            });
        }
    };
    Ok(Source::Step {
        step: target.to_string(),
        output: port.name.to_string(),
    })
}

/// Checks that every input gets a kind its port accepts, and decides each step's output kinds.
/// Steps are in dependency order, so the kinds of their inputs are already known.
fn check_kinds(spec: &PipelineSpec, steps: &mut [PlannedStep]) -> Result<(), PipelineError> {
    let declared: BTreeMap<String, ArtifactKind> =
        spec.pipeline.inputs.kinds().into_iter().collect();
    let mut produced: BTreeMap<String, ArtifactKind> = BTreeMap::new();
    for step in steps.iter_mut() {
        let mut bound = BTreeMap::new();
        for (port, reference) in &step.bindings {
            let source = &step.inputs[port.name];
            let found = match source {
                Source::Input(name) => declared[name],
                Source::Step { .. } => produced[&source.name()],
            };
            if !port.kinds.contains(&found) {
                return Err(PipelineError::KindMismatch {
                    step: step.id.clone(),
                    port: port.name.to_string(),
                    reference: reference.clone(),
                    expected: port.kinds.to_vec(),
                    found,
                });
            }
            bound.insert(port.name.to_string(), found);
        }
        for port in step.op.descriptor().outputs {
            let kind = step.op.output_kind(port, &bound);
            step.output_kinds.insert(port.name.to_string(), kind);
            produced.insert(format!("{}.{}", step.id, port.name), kind);
        }
    }
    Ok(())
}

fn in_dependency_order(mut pending: Vec<PlannedStep>) -> Result<Vec<PlannedStep>, PipelineError> {
    let mut done = BTreeSet::new();
    let mut ordered = Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        let ready = pending.iter().position(|s| {
            s.inputs.values().all(|src| match src {
                Source::Input(_) => true,
                Source::Step { step, .. } => done.contains(step),
            })
        });
        match ready {
            Some(i) => {
                let step = pending.remove(i);
                done.insert(step.id.clone());
                ordered.push(step);
            }
            None => {
                return Err(PipelineError::Cycle(
                    pending.into_iter().map(|s| s.id).collect(),
                ));
            }
        }
    }
    Ok(ordered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ops;

    fn plan(toml: &str) -> Result<Plan, PipelineError> {
        let text = format!("[pipeline]\nname = \"t\"\ninputs = [\"x\"]\n{toml}");
        Plan::new(&PipelineSpec::from_toml(&text)?, &ops().registry)
    }

    #[test]
    fn orders_steps_by_dependency_not_by_listing() {
        let p = plan(
            r#"
            [[step]]
            id = "b"
            op = "scale"
            input = "a"
            [[step]]
            id = "a"
            op = "scale"
            input = "x"
            "#,
        )
        .unwrap();
        assert_eq!(p.step_ids().collect::<Vec<_>>(), ["a", "b"]);
    }

    #[test]
    fn fills_parameter_defaults() {
        let p = plan("[[step]]\nid = \"a\"\nop = \"scale\"\ninput = \"x\"\n").unwrap();
        assert_eq!(p.steps[0].params["factor"], 1.0);
    }

    #[test]
    fn rejects_cycles() {
        let err = plan(
            r#"
            [[step]]
            id = "a"
            op = "scale"
            input = "b"
            [[step]]
            id = "b"
            op = "scale"
            input = "a"
            "#,
        )
        .err()
        .unwrap();
        assert!(matches!(err, PipelineError::Cycle(ids) if ids == ["a", "b"]));
    }

    #[test]
    fn rejects_invalid_specs() {
        let cases = [
            (
                "duplicate id",
                "[[step]]\nid = \"x\"\nop = \"scale\"\ninput = \"x\"\n",
            ),
            (
                "dot in id",
                "[[step]]\nid = \"a.b\"\nop = \"scale\"\ninput = \"x\"\n",
            ),
            (
                "unknown op",
                "[[step]]\nid = \"a\"\nop = \"nope\"\ninput = \"x\"\n",
            ),
            (
                "unknown param",
                "[[step]]\nid = \"a\"\nop = \"scale\"\ninput = \"x\"\nparams = { factr = 2 }\n",
            ),
            (
                "unknown reference",
                "[[step]]\nid = \"a\"\nop = \"scale\"\ninput = \"y\"\n",
            ),
            (
                "kind mismatch",
                "[[step]]\nid = \"a\"\nop = \"needs_modes\"\ninput = \"x\"\n",
            ),
            (
                "single input to pair",
                "[[step]]\nid = \"a\"\nop = \"pair\"\ninput = \"x\"\n",
            ),
            (
                "unconnected port",
                "[[step]]\nid = \"a\"\nop = \"pair\"\ninput = { left = \"x\" }\n",
            ),
            (
                "unknown port",
                "[[step]]\nid = \"a\"\nop = \"scale\"\ninput = { sig = \"x\" }\n",
            ),
            ("no input at all", "[[step]]\nid = \"a\"\nop = \"scale\"\n"),
            (
                "extra port beside a connected one",
                "[[step]]\nid = \"a\"\nop = \"scale\"\ninput = { signal = \"x\", extra = \"x\" }\n",
            ),
            (
                "ambiguous output",
                "[[step]]\nid = \"s\"\nop = \"split\"\ninput = \"x\"\n[[step]]\nid = \"a\"\nop = \"scale\"\ninput = \"s\"\n",
            ),
        ];
        for (case, toml) in cases {
            assert!(plan(toml).is_err(), "accepted: {case}");
        }
    }

    #[test]
    fn wires_the_named_output_of_a_multi_output_step() {
        for output in ["low", "high"] {
            let p = plan(&format!(
                "[[step]]\nid = \"s\"\nop = \"split\"\ninput = \"x\"\n[[step]]\nid = \"a\"\nop = \"scale\"\ninput = \"s.{output}\"\n"
            ))
            .unwrap();
            let consumer = p.steps.iter().find(|s| s.id == "a").unwrap();
            assert_eq!(
                consumer.inputs["signal"],
                Source::Step {
                    step: "s".into(),
                    output: output.into()
                }
            );
        }
    }

    #[test]
    fn accepts_every_port_named() {
        let p =
            plan("[[step]]\nid = \"a\"\nop = \"pair\"\ninput = { left = \"x\", right = \"x\" }\n")
                .unwrap();
        assert_eq!(p.steps[0].inputs.len(), 2);
        let single = plan("[[step]]\nid = \"a\"\nop = \"scale\"\ninput = { signal = \"x\" }\n");
        assert!(single.is_ok());
    }

    #[test]
    fn exposes_name_and_inputs() {
        let p = plan("").unwrap();
        assert_eq!((p.name(), p.inputs()), ("t", ["x".to_string()].as_slice()));
    }

    mod properties {
        use proptest::prelude::*;

        use super::*;

        proptest! {
            /// Each step reads either the pipeline input or an earlier step; listing order is shuffled.
            #[test]
            fn steps_always_run_after_their_inputs(
                sources in prop::collection::vec(any::<prop::sample::Index>(), 1..12),
                shuffle_seed in any::<u64>(),
            ) {
                let n = sources.len();
                // Step i reads the pipeline input or one of the steps 0..i.
                let source = |i: usize| match sources[i].index(i + 1) {
                    0 => "x".to_string(),
                    j => format!("s{}", j - 1),
                };
                let mut order: Vec<usize> = (0..n).collect();
                let mut state = shuffle_seed;
                for i in (1..n).rev() {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    order.swap(i, (state >> 33) as usize % (i + 1));
                }
                let steps: String = order
                    .iter()
                    .map(|&i| format!("[[step]]\nid = \"s{i}\"\nop = \"scale\"\ninput = \"{}\"\n", source(i)))
                    .collect();
                let p = plan(&steps).unwrap();
                let position: BTreeMap<&str, usize> = p.step_ids().enumerate().map(|(k, id)| (id, k)).collect();
                for i in 0..n {
                    let src = source(i);
                    if src != "x" {
                        let step = format!("s{i}");
                        prop_assert!(position[src.as_str()] < position[step.as_str()], "{} must run before {}", src, step);
                    }
                }
            }
        }
    }
}
