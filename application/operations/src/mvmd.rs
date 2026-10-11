use modewise_decomposition::ModeDecomposer;
use modewise_mvmd::{FrequencyInit, Mvmd, MvmdParams, NaMvmd, NoiseParams, Vmd};
use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum Init {
    Zero,
    Linear,
    Exponential,
    Uniform,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// Number of modes K.
    num_modes: usize,
    /// Bandwidth penalty; larger means narrower modes.
    alpha: f64,
    /// Starting centre frequencies: `zero`, `linear` (evenly over [0, 0.5], as Dmocrito/mvmd), `exponential`, or `uniform` ((0.5/K)·k, as the original MATLAB VMD and MVMD codes and vmdpy).
    #[serde(default = "defaults::init")]
    init: Init,
    /// Stop when the mean squared update falls below this.
    #[serde(default = "defaults::tolerance")]
    tolerance: f64,
    /// Dual ascent step; 0 lets noise stay out of the modes.
    #[serde(default = "defaults::tau")]
    tau: f64,
    /// Iteration cap.
    #[serde(default = "defaults::max_iterations")]
    max_iterations: usize,
    /// na_mvmd only: white-noise channels appended before decomposing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    noise_channels: Option<usize>,
    /// na_mvmd only: noise SD relative to the unit-SD signal channels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    noise_sd: Option<f64>,
    /// na_mvmd only: seed for the noise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

mod defaults {
    use modewise_mvmd::MvmdParams;

    fn reference() -> MvmdParams {
        MvmdParams::new(1, 1.0)
    }
    pub fn init() -> super::Init {
        super::Init::Zero
    }
    pub fn tolerance() -> f64 {
        reference().tolerance
    }
    pub fn tau() -> f64 {
        reference().tau
    }
    pub fn max_iterations() -> usize {
        reference().max_iterations
    }
}

/// The methods that share the MVMD solver and its parameters.
#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Mvmd,
    Vmd,
    NaMvmd,
}

impl Params {
    fn mvmd_params(&self) -> MvmdParams {
        let init = match self.init {
            Init::Zero => FrequencyInit::Zero,
            Init::Linear => FrequencyInit::Linear,
            Init::Exponential => FrequencyInit::Exponential,
            Init::Uniform => FrequencyInit::Uniform,
        };
        MvmdParams {
            num_modes: self.num_modes,
            alpha: self.alpha,
            init,
            tolerance: self.tolerance,
            tau: self.tau,
            max_iterations: self.max_iterations,
        }
    }

    /// Fills the noise defaults for na_mvmd, and rejects noise settings elsewhere.
    fn complete(mut self, variant: Variant) -> Result<Self, String> {
        let has_noise =
            self.noise_channels.is_some() || self.noise_sd.is_some() || self.seed.is_some();
        match variant {
            Variant::NaMvmd => {
                let d = NoiseParams::default();
                self.noise_channels.get_or_insert(d.channels);
                self.noise_sd.get_or_insert(d.sd);
                self.seed.get_or_insert(d.seed);
            }
            _ if has_noise => {
                return Err("`noise_channels`, `noise_sd` and `seed` only apply to na_mvmd".into());
            }
            _ => {}
        }
        Ok(self)
    }

    fn method(&self, variant: Variant) -> Result<Box<dyn ModeDecomposer + Send + Sync>, String> {
        let params = self.mvmd_params();
        let method: Box<dyn ModeDecomposer + Send + Sync> = match variant {
            Variant::Mvmd => Box::new(Mvmd::new(params).map_err(|e| e.to_string())?),
            Variant::Vmd => Box::new(Vmd::new(params).map_err(|e| e.to_string())?),
            Variant::NaMvmd => {
                let d = NoiseParams::default();
                let noise = NoiseParams {
                    channels: self.noise_channels.unwrap_or(d.channels),
                    sd: self.noise_sd.unwrap_or(d.sd),
                    seed: self.seed.unwrap_or(d.seed),
                };
                Box::new(NaMvmd::new(params, noise).map_err(|e| e.to_string())?)
            }
        };
        Ok(method)
    }
}

/// The shared schema narrowed to what `variant` accepts, with the noise defaults filled in by
/// the same completion that `resolve` uses.
fn variant_schema(variant: Variant) -> Value {
    let mut schema = modewise_pipeline::schema_of::<Params>();
    let noisy = matches!(variant, Variant::NaMvmd);
    let defaults =
        serde_json::from_value::<Params>(serde_json::json!({"num_modes": 1, "alpha": 1.0}))
            .ok()
            .and_then(|p| p.complete(variant).ok())
            .and_then(|p| serde_json::to_value(p).ok())
            .unwrap_or_default();
    if let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) {
        properties.retain(|field, _| {
            noisy || !matches!(field.as_str(), "noise_channels" | "noise_sd" | "seed")
        });
        for field in ["noise_channels", "noise_sd", "seed"] {
            if let (Some(default), Some(spec)) = (
                defaults.get(field),
                properties.get_mut(field).and_then(Value::as_object_mut),
            ) {
                spec.insert("default".into(), default.clone());
            }
        }
    }
    schema
}

/// `mvmd`, `vmd` (each channel on its own) and `na_mvmd` (noise-assisted).
pub struct MvmdOperation;
pub struct VmdOperation;
pub struct NaMvmdOperation;

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

macro_rules! mvmd_family {
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

mvmd_family!(
    MvmdOperation,
    "mvmd",
    Variant::Mvmd,
    "Multivariate variational mode decomposition: K modes shared by all channels (Rehman and Aftab 2019)."
);
mvmd_family!(
    VmdOperation,
    "vmd",
    Variant::Vmd,
    "Variational mode decomposition of each channel on its own (Dragomiretskiy and Zosso 2014)."
);
mvmd_family!(
    NaMvmdOperation,
    "na_mvmd",
    Variant::NaMvmd,
    "Noise-assisted MVMD: MVMD with white-noise channels appended, less sensitive to K (Zisou et al. 2021)."
);
