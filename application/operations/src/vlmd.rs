use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use modewise_vlmd::{Vlmd, VlmdParams};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// Number of modes K.
    num_modes: usize,
    /// Number of latent components L, at most the number of channels; one per channel by default.
    #[serde(default)]
    num_latents: Option<usize>,
    /// Bandwidth penalty; larger means narrower modes.
    #[serde(default = "defaults::alpha")]
    alpha: f64,
    /// Sparsity of the loadings, on the reference code's scale.
    #[serde(default = "defaults::reg_lambda")]
    reg_lambda: f64,
    /// ADMM penalty.
    #[serde(default = "defaults::reg_rho")]
    reg_rho: f64,
    /// Dual step, in [0, 1].
    #[serde(default = "defaults::tau")]
    tau: f64,
    /// Stop when the summed squared relative change of the centre frequencies falls to this.
    #[serde(default = "defaults::tolerance")]
    tolerance: f64,
    /// Iteration cap.
    #[serde(default = "defaults::max_iterations")]
    max_iterations: usize,
}

mod defaults {
    use modewise_vlmd::VlmdParams;

    fn reference() -> VlmdParams {
        VlmdParams::new(1)
    }
    pub fn alpha() -> f64 {
        reference().alpha
    }
    pub fn reg_lambda() -> f64 {
        reference().reg_lambda
    }
    pub fn reg_rho() -> f64 {
        reference().reg_rho
    }
    pub fn tau() -> f64 {
        reference().tau
    }
    pub fn tolerance() -> f64 {
        reference().tolerance
    }
    pub fn max_iterations() -> usize {
        reference().max_iterations
    }
}

impl Params {
    fn method(self) -> Result<Vlmd, String> {
        Vlmd::new(VlmdParams {
            num_modes: self.num_modes,
            num_latents: self.num_latents,
            alpha: self.alpha,
            reg_lambda: self.reg_lambda,
            reg_rho: self.reg_rho,
            tau: self.tau,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
        })
        .map_err(|e| e.to_string())
    }
}

/// Variational Latent Mode Decomposition (docs/vlmd.md).
pub struct VlmdOperation;

impl Operation for VlmdOperation {
    fn summary(&self) -> &'static str {
        "Variational latent mode decomposition: modes of a few latent signals and sparse channel loadings (docs/vlmd.md)."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<Params>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "vlmd",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[Port {
                name: "signal",
                kinds: &[ArtifactKind::Series],
            }],
            outputs: &[
                Port {
                    name: "modes",
                    kinds: &[ArtifactKind::Decomposition],
                },
                Port {
                    name: "latent_modes",
                    kinds: &[ArtifactKind::Decomposition],
                },
                Port {
                    name: "loadings",
                    kinds: &[ArtifactKind::Loadings],
                },
            ],
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        let (typed, resolved) = resolve_params::<Params>(params)?;
        typed.method()?;
        Ok(resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let method = serde_json::from_value::<Params>(params.clone())
            .map_err(|e| e.to_string())?
            .method()?;
        let Artifact::Series(signal) = input(inputs, "signal")? else {
            return Err("input \"signal\" must be a series".into());
        };
        let out = method.decompose_latent(signal).map_err(|e| e.to_string())?;
        Ok(Outputs::from([
            ("modes".into(), Artifact::Decomposition(out.modes)),
            (
                "latent_modes".into(),
                Artifact::Decomposition(out.latent_modes),
            ),
            ("loadings".into(), Artifact::Loadings(out.loadings)),
        ]))
    }
}
