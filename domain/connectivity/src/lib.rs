//! Connectivity between channels (docs/connectivity.md).
//!
//! Functional connectivity is statistical dependence: undirected, symmetric, and silent about
//! cause. Effective (directed) connectivity is a different claim and will get its own type.

use modewise_method::MethodError;
use ndarray::{Array2, Array4, ArrayView2, ArrayView4};

/// How the dependence between two channels is measured. Every measure is undirected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Measure {
    /// Pearson correlation, optionally Fisher z-transformed.
    Pearson { fisher_z: bool },
    /// Phase locking value `|mean e^{iΔφ}|` over the window, from full-run analytic signals.
    PhaseLockingValue,
}

impl Measure {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pearson { fisher_z: false } => "pearson",
            Self::Pearson { fisher_z: true } => "pearson_fisher_z",
            Self::PhaseLockingValue => "plv",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "pearson" => Some(Self::Pearson { fisher_z: false }),
            "pearson_fisher_z" => Some(Self::Pearson { fisher_z: true }),
            "plv" => Some(Self::PhaseLockingValue),
            _ => None,
        }
    }
}

/// Where each matrix's window came from, when it was computed per epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowInfo {
    pub label: String,
    pub event_onset_s: f64,
}

/// Functional connectivity laid out as window × component × channel × channel. Each matrix is symmetric.
///
/// A whole run is one window; a series is one component, a decomposition one per mode.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionalConnectivity {
    values: Array4<f64>,
    measure: Measure,
    labels: Vec<String>,
    centre_frequencies_hz: Option<Vec<f64>>,
    windows: Option<Vec<WindowInfo>>,
    resolvable: Option<Array2<bool>>,
}

impl FunctionalConnectivity {
    pub fn new(
        values: Array4<f64>,
        measure: Measure,
        labels: Vec<String>,
    ) -> Result<Self, MethodError> {
        let (_, _, rows, columns) = values.dim();
        if rows != columns || labels.len() != rows {
            return Err(MethodError::ShapeMismatch(format!(
                "matrices are {rows} × {columns} but there are {} labels",
                labels.len()
            )));
        }
        Ok(Self {
            values,
            measure,
            labels,
            centre_frequencies_hz: None,
            windows: None,
            resolvable: None,
        })
    }

    /// One centre frequency per component (modes of a decomposition).
    pub fn with_centre_frequencies(mut self, hz: Vec<f64>) -> Result<Self, MethodError> {
        if hz.len() != self.n_components() {
            return Err(MethodError::ShapeMismatch(format!(
                "{} components but {} centre frequencies",
                self.n_components(),
                hz.len()
            )));
        }
        self.centre_frequencies_hz = Some(hz);
        Ok(self)
    }

    /// One entry per window, and optionally window × component flags for whether the window
    /// holds enough cycles of the component (see docs/epochs.md).
    pub fn with_windows(
        mut self,
        windows: Vec<WindowInfo>,
        resolvable: Option<Array2<bool>>,
    ) -> Result<Self, MethodError> {
        if windows.len() != self.n_windows() {
            return Err(MethodError::ShapeMismatch(format!(
                "{} windows but {} window entries",
                self.n_windows(),
                windows.len()
            )));
        }
        if let Some(r) = &resolvable
            && r.dim() != (self.n_windows(), self.n_components())
        {
            return Err(MethodError::ShapeMismatch(format!(
                "resolvable flags are {:?}, expected {:?}",
                r.dim(),
                (self.n_windows(), self.n_components())
            )));
        }
        self.windows = Some(windows);
        self.resolvable = resolvable;
        Ok(self)
    }

    pub fn values(&self) -> ArrayView4<'_, f64> {
        self.values.view()
    }

    pub fn matrix(&self, window: usize, component: usize) -> ArrayView2<'_, f64> {
        self.values.slice(ndarray::s![window, component, .., ..])
    }

    pub fn measure(&self) -> Measure {
        self.measure
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn centre_frequencies_hz(&self) -> Option<&[f64]> {
        self.centre_frequencies_hz.as_deref()
    }

    pub fn windows(&self) -> Option<&[WindowInfo]> {
        self.windows.as_deref()
    }

    pub fn resolvable(&self) -> Option<ArrayView2<'_, bool>> {
        self.resolvable.as_ref().map(|r| r.view())
    }

    pub fn n_windows(&self) -> usize {
        self.values.dim().0
    }

    pub fn n_components(&self) -> usize {
        self.values.dim().1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("c{i}")).collect()
    }

    #[test]
    fn shapes_are_checked() {
        let m = Measure::Pearson { fisher_z: false };
        assert!(FunctionalConnectivity::new(Array4::zeros((1, 2, 3, 3)), m, labels(3)).is_ok());
        assert!(FunctionalConnectivity::new(Array4::zeros((1, 2, 3, 2)), m, labels(3)).is_err());
        assert!(FunctionalConnectivity::new(Array4::zeros((1, 2, 3, 3)), m, labels(2)).is_err());

        let c = FunctionalConnectivity::new(Array4::ones((2, 3, 4, 4)), m, labels(4)).unwrap();
        assert!(c.clone().with_centre_frequencies(vec![0.1; 2]).is_err());
        let c = c.with_centre_frequencies(vec![0.01, 0.05, 0.1]).unwrap();
        let w = |n: usize| {
            (0..n)
                .map(|i| WindowInfo {
                    label: "x".into(),
                    event_onset_s: i as f64,
                })
                .collect::<Vec<_>>()
        };
        assert!(c.clone().with_windows(w(3), None).is_err());
        assert!(
            c.clone()
                .with_windows(w(2), Some(Array2::from_elem((3, 2), true)))
                .is_err()
        );
        let c = c
            .with_windows(w(2), Some(Array2::from_elem((2, 3), true)))
            .unwrap();
        assert_eq!(
            (c.n_windows(), c.n_components(), c.labels().len()),
            (2, 3, 4)
        );
        assert_eq!(c.centre_frequencies_hz(), Some(&[0.01, 0.05, 0.1][..]));
        assert_eq!(c.windows().unwrap()[1].event_onset_s, 1.0);
        assert_eq!(c.resolvable().unwrap().dim(), (2, 3));
        assert_eq!(c.matrix(1, 2).dim(), (4, 4));
        assert_eq!(c.values().dim(), (2, 3, 4, 4));
        assert_eq!(c.measure(), m);
    }

    #[test]
    fn measure_names_round_trip() {
        for m in [
            Measure::Pearson { fisher_z: false },
            Measure::Pearson { fisher_z: true },
            Measure::PhaseLockingValue,
        ] {
            assert_eq!(Measure::from_name(m.name()), Some(m));
        }
        assert_eq!(Measure::from_name("coherence"), None);
    }
}
