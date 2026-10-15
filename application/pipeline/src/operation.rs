use std::collections::BTreeMap;
use std::sync::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{Artifact, ArtifactKind};

/// A named input or output. Inputs may accept several kinds. An output lists one kind, or
/// several when `Operation::output_kind` picks one from the kinds of the inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Port {
    pub name: &'static str,
    pub kinds: &'static [ArtifactKind],
}

impl Port {
    /// The kind an output port produces.
    pub fn output_kind(&self) -> ArtifactKind {
        self.kinds.first().copied().unwrap_or(ArtifactKind::Series)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OperationDescriptor {
    pub name: &'static str,
    /// Part of every cache key. Bump it when results of the operation change.
    pub version: &'static str,
    pub inputs: &'static [Port],
    pub outputs: &'static [Port],
}

pub type Inputs = BTreeMap<String, Arc<Artifact>>;
pub type Outputs = BTreeMap<String, Artifact>;

/// One kind of pipeline step, for example a decomposition method.
pub trait Operation: Send + Sync {
    fn descriptor(&self) -> OperationDescriptor;

    /// Checks parameters and fills in defaults. The result is what gets cached and recorded.
    fn resolve(&self, params: Value) -> Result<Value, String>;

    /// Runs on parameters that `resolve` returned.
    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String>;

    /// The kind an output will have, given the kinds bound to each input. By default the first
    /// kind the port lists; operations whose output follows their input override this.
    fn output_kind(&self, port: &Port, _inputs: &BTreeMap<String, ArtifactKind>) -> ArtifactKind {
        port.output_kind()
    }

    /// Inputs a step may leave unconnected; `run` then gets no entry for them.
    fn optional_inputs(&self) -> &'static [&'static str] {
        &[]
    }

    /// One sentence on what the operation does, for `modewise ops` and the generated reference.
    fn summary(&self) -> &'static str {
        ""
    }

    /// JSON Schema of the parameters: types, descriptions from doc comments, and defaults. The
    /// generated reference (docs/reference/operations.md) and tools read it.
    fn params_schema(&self) -> Value {
        Value::Object(Default::default())
    }
}

/// The JSON Schema of a parameter struct.
pub fn schema_of<T: schemars::JsonSchema>() -> Value {
    schemars::schema_for!(T).to_value()
}

/// Round-trips raw parameters through a typed struct, which applies defaults and rejects unknown fields.
pub fn resolve_params<T: DeserializeOwned + Serialize>(
    params: Value,
) -> Result<(T, Value), String> {
    let typed: T = serde_json::from_value(params).map_err(|e| e.to_string())?;
    let resolved = serde_json::to_value(&typed).map_err(|e| e.to_string())?;
    Ok((typed, resolved))
}
