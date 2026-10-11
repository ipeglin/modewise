use modewise_method::MethodError;
use ndarray::{Array2, ArrayView2};

/// How strongly each latent component contributes to each channel, laid out as latent × channel
/// (the coefficient matrix `A` of a latent model `X = ZA`).
#[derive(Debug, Clone, PartialEq)]
pub struct Loadings {
    values: Array2<f64>,
    latent_labels: Vec<String>,
    channel_labels: Vec<String>,
}

impl Loadings {
    pub fn new(
        values: Array2<f64>,
        latent_labels: Vec<String>,
        channel_labels: Vec<String>,
    ) -> Result<Self, MethodError> {
        let (n_latents, n_channels) = values.dim();
        if latent_labels.len() != n_latents || channel_labels.len() != n_channels {
            return Err(MethodError::ShapeMismatch(format!(
                "loadings are {n_latents} × {n_channels} but there are {} latent and {} channel labels",
                latent_labels.len(),
                channel_labels.len()
            )));
        }
        Ok(Self {
            values,
            latent_labels,
            channel_labels,
        })
    }

    pub fn values(&self) -> ArrayView2<'_, f64> {
        self.values.view()
    }

    pub fn latent_labels(&self) -> &[String] {
        &self.latent_labels
    }

    pub fn channel_labels(&self) -> &[String] {
        &self.channel_labels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_counts_must_match_the_matrix() {
        let labels = |n: usize| (0..n).map(|i| i.to_string()).collect::<Vec<_>>();
        let ok = Loadings::new(Array2::ones((2, 3)), labels(2), labels(3)).unwrap();
        assert_eq!(
            (ok.latent_labels(), ok.channel_labels()),
            (&labels(2)[..], &labels(3)[..])
        );
        assert_eq!(ok.values(), Array2::<f64>::ones((2, 3)));
        assert!(Loadings::new(Array2::zeros((2, 3)), labels(3), labels(3)).is_err());
        assert!(Loadings::new(Array2::zeros((2, 3)), labels(2), labels(2)).is_err());
    }
}
