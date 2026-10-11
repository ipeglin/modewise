use modewise_decomposition::ModeDecomposer;
use modewise_emd::{Ceemdan, Eemd, Emd, Memd, Stop};
use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
enum StopName {
    Rilling,
    SNumber,
    Fixed,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// When sifting stops: `rilling` (thresholds sd, sd2, tol), `s_number` (count) or `fixed` (siftings); `fixed` for eemd, `rilling` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stop: Option<StopName>,
    /// rilling: the envelope mean relative to the mode amplitude must stay below this on all but a fraction tol of samples.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sd: Option<f64>,
    /// rilling: the same ratio must stay below this everywhere (0.5; memd 0.75).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    sd2: Option<f64>,
    /// rilling: the fraction of samples allowed above sd (0.05; memd 0.075).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tol: Option<f64>,
    /// s_number: consecutive siftings with extrema and zero crossings within one (2 by default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    count: Option<usize>,
    /// fixed: siftings per IMF (10 by default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    siftings: Option<usize>,
    /// Stop after this many IMFs (the rest is the residual); all by default, at most ⌊log₂ n⌋ − 1 for eemd.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_imfs: Option<usize>,
    /// memd only: projection directions; use at least twice the channels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    num_directions: Option<usize>,
    /// eemd and ceemdan: noise realisations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    trials: Option<usize>,
    /// eemd only: noise SD relative to the signal's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    noise_sd: Option<f64>,
    /// ceemdan only: noise scale ε.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    epsilon: Option<f64>,
    /// eemd and ceemdan: noise seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Emd,
    Memd,
    Eemd,
    Ceemdan,
}

fn only(name: &str, owner: &str) -> String {
    format!("`{name}` only applies to {owner}")
}

impl Params {
    /// Fills the defaults of the method and its stop rule, and rejects settings that do not apply.
    fn complete(mut self, variant: Variant) -> Result<Self, String> {
        let default_stop = if variant == Variant::Eemd {
            StopName::Fixed
        } else {
            StopName::Rilling
        };
        let stop = *self.stop.get_or_insert(default_stop);
        let rilling = [self.sd, self.sd2, self.tol];
        let foreign = match stop {
            StopName::Rilling => self.count.is_some() || self.siftings.is_some(),
            StopName::SNumber => rilling.iter().any(Option::is_some) || self.siftings.is_some(),
            StopName::Fixed => rilling.iter().any(Option::is_some) || self.count.is_some(),
        };
        if foreign {
            return Err("only the chosen stop's settings apply: sd, sd2 and tol for rilling, count for s_number, siftings for fixed".into());
        }
        match stop {
            StopName::Rilling => {
                let default = if variant == Variant::Memd {
                    Stop::RILLING_MEMD
                } else {
                    Stop::RILLING_EMD
                };
                if let Stop::Rilling { sd, sd2, tol } = default {
                    self.sd.get_or_insert(sd);
                    self.sd2.get_or_insert(sd2);
                    self.tol.get_or_insert(tol);
                }
            }
            StopName::SNumber => {
                self.count.get_or_insert(2);
            }
            StopName::Fixed => {
                self.siftings.get_or_insert(10);
            }
        }
        let noisy = matches!(variant, Variant::Eemd | Variant::Ceemdan);
        if variant == Variant::Memd {
            self.num_directions.get_or_insert(64);
        } else if self.num_directions.is_some() {
            return Err(only("num_directions", "memd"));
        }
        if noisy {
            self.trials.get_or_insert(100);
            self.seed.get_or_insert(0);
        } else if self.trials.is_some() || self.seed.is_some() {
            return Err("`trials` and `seed` only apply to eemd and ceemdan".into());
        }
        if variant == Variant::Eemd {
            self.noise_sd.get_or_insert(0.2);
        } else if self.noise_sd.is_some() {
            return Err(only("noise_sd", "eemd"));
        }
        if variant == Variant::Ceemdan {
            self.epsilon.get_or_insert(0.2);
        } else if self.epsilon.is_some() {
            return Err(only("epsilon", "ceemdan"));
        }
        Ok(self)
    }

    fn stop(&self) -> Stop {
        match self.stop.unwrap_or(StopName::Rilling) {
            StopName::Rilling => Stop::Rilling {
                sd: self.sd.unwrap_or_default(),
                sd2: self.sd2.unwrap_or_default(),
                tol: self.tol.unwrap_or_default(),
            },
            StopName::SNumber => Stop::SNumber {
                count: self.count.unwrap_or_default(),
            },
            StopName::Fixed => Stop::Fixed {
                siftings: self.siftings.unwrap_or_default(),
            },
        }
    }

    fn method(&self, variant: Variant) -> Result<Box<dyn ModeDecomposer + Send + Sync>, String> {
        let (stop, max_imfs) = (self.stop(), self.max_imfs);
        let (trials, seed) = (
            self.trials.unwrap_or_default(),
            self.seed.unwrap_or_default(),
        );
        let method: Result<Box<dyn ModeDecomposer + Send + Sync>, _> = match variant {
            Variant::Emd => Emd::new(stop, max_imfs).map(|m| Box::new(m) as _),
            Variant::Memd => Memd::new(stop, max_imfs, self.num_directions.unwrap_or_default())
                .map(|m| Box::new(m) as _),
            Variant::Eemd => {
                let noise_sd = self.noise_sd.unwrap_or_default();
                Eemd {
                    stop,
                    max_imfs,
                    trials,
                    noise_sd,
                    seed,
                }
                .validated()
                .map(|m| Box::new(m) as _)
            }
            Variant::Ceemdan => {
                let epsilon = self.epsilon.unwrap_or_default();
                Ceemdan {
                    stop,
                    max_imfs,
                    trials,
                    epsilon,
                    seed,
                }
                .validated()
                .map(|m| Box::new(m) as _)
            }
        };
        method.map_err(|e| e.to_string())
    }
}

/// The shared schema narrowed to what `variant` accepts, with its defaults filled in by the
/// same completion that `resolve` uses.
fn variant_schema(variant: Variant) -> Value {
    let mut schema = modewise_pipeline::schema_of::<Params>();
    let applies = |field: &str| match field {
        "num_directions" => variant == Variant::Memd,
        "trials" | "seed" => matches!(variant, Variant::Eemd | Variant::Ceemdan),
        "noise_sd" => variant == Variant::Eemd,
        "epsilon" => variant == Variant::Ceemdan,
        _ => true,
    };
    let defaults = serde_json::from_value::<Params>(Value::Object(Default::default()))
        .ok()
        .and_then(|p| p.complete(variant).ok())
        .and_then(|p| serde_json::to_value(p).ok())
        .unwrap_or_default();
    if let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) {
        properties.retain(|field, _| applies(field));
        for (field, spec) in properties.iter_mut() {
            if let (Some(default), Some(spec)) = (defaults.get(field), spec.as_object_mut()) {
                spec.insert("default".into(), default.clone());
            }
        }
    }
    schema
}

/// `emd`, `eemd` and `ceemdan` (each channel on its own) and `memd` (multivariate).
pub struct EmdOperation;
pub struct MemdOperation;
pub struct EemdOperation;
pub struct CeemdanOperation;

fn descriptor(name: &'static str) -> OperationDescriptor {
    OperationDescriptor {
        name,
        version: env!("CARGO_PKG_VERSION"),
        inputs: &[Port {
            name: "signal",
            kinds: &[ArtifactKind::Series],
        }],
        outputs: &[Port {
            name: "modes",
            kinds: &[ArtifactKind::Decomposition],
        }],
    }
}

fn resolve(variant: Variant, params: Value) -> Result<Value, String> {
    let (typed, _) = resolve_params::<Params>(params)?;
    let typed = typed.complete(variant)?;
    typed.method(variant)?;
    serde_json::to_value(&typed).map_err(|e| e.to_string())
}

fn run(variant: Variant, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
    let method = serde_json::from_value::<Params>(params.clone())
        .map_err(|e| e.to_string())?
        .complete(variant)?
        .method(variant)?;
    let Artifact::Series(signal) = input(inputs, "signal")? else {
        return Err("input \"signal\" must be a series".into());
    };
    let modes = method.decompose(signal).map_err(|e| e.to_string())?;
    Ok(Outputs::from([(
        "modes".into(),
        Artifact::Decomposition(modes),
    )]))
}

macro_rules! emd_family {
    ($operation:ident, $name:literal, $variant:expr, $summary:literal) => {
        impl Operation for $operation {
            fn summary(&self) -> &'static str {
                $summary
            }

            fn params_schema(&self) -> serde_json::Value {
                variant_schema($variant)
            }

            fn descriptor(&self) -> OperationDescriptor {
                descriptor($name)
            }

            fn resolve(&self, params: Value) -> Result<Value, String> {
                resolve($variant, params)
            }

            fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
                run($variant, params, inputs)
            }
        }
    };
}

emd_family!(
    EmdOperation,
    "emd",
    Variant::Emd,
    "Empirical mode decomposition of each channel on its own (docs/emd.md)."
);
emd_family!(
    MemdOperation,
    "memd",
    Variant::Memd,
    "Multivariate EMD: the same IMFs in every channel, from projections on many directions (docs/emd.md)."
);
emd_family!(
    EemdOperation,
    "eemd",
    Variant::Eemd,
    "Ensemble EMD: the mean of EMDs of the signal plus seeded white noise (docs/emd.md)."
);
emd_family!(
    CeemdanOperation,
    "ceemdan",
    Variant::Ceemdan,
    "CEEMDAN, improved form: noise-assisted EMD whose modes sum to the signal (docs/emd.md)."
);
