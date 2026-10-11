//! Noise-assisted EMD: EEMD (Wu and Huang 2009) and CEEMDAN in its improved form (Colominas,
//! Schlotthauer and Torres 2014), each following its authors' code (eemd.m, iceemdan.m).

use modewise_decomposition::{Decomposition, ModeDecomposer};
use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::MultichannelSeries;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::decomposers::validate;
use crate::layout::{Channel, emd1, per_channel};
use crate::sift::Stop;

fn sd(x: ArrayView1<f64>) -> f64 {
    x.std(1.0)
}

fn mean_of(parts: Vec<Array1<f64>>, n: usize) -> Array1<f64> {
    let count = parts.len() as f64;
    parts.into_iter().fold(Array1::zeros(n), |a, b| a + b) / count
}

/// Unit-variance white noise, one realisation per trial; channel `c` has its own stream.
fn white_noise(seed: u64, channel: usize, trials: usize, n: usize) -> Vec<Array1<f64>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(channel as u64);
    (0..trials)
        .map(|_| Array1::from_shape_simple_fn(n, || StandardNormal.sample(&mut rng)))
        .collect()
}

fn check_noise(trials: usize, scale: f64, name: &'static str) -> Result<(), MethodError> {
    if trials == 0 {
        return Err(MethodError::invalid("trials", "must be at least 1"));
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(MethodError::invalid(name, "must be finite and positive"));
    }
    Ok(())
}

/// EEMD of one channel with the given noise realisations (already scaled): the mean over trials
/// of each IMF index and of the residual.
pub(crate) fn eemd_channel(
    x: ArrayView1<f64>,
    noises: &[Array1<f64>],
    stop: Stop,
    max_imfs: usize,
) -> Channel {
    let n = x.len();
    let trials: Vec<Channel> = noises
        .par_iter()
        .map(|w| emd1((&x + w).view(), stop, Some(max_imfs)))
        .collect();
    let count = trials.iter().map(|(imfs, _)| imfs.len()).max().unwrap_or(0);
    let imfs = (0..count)
        .map(|j| {
            mean_of(
                trials
                    .iter()
                    .map(|(imfs, _)| imfs.get(j).cloned().unwrap_or_else(|| Array1::zeros(n)))
                    .collect(),
                n,
            )
        })
        .collect();
    let residual = mean_of(trials.into_iter().map(|(_, r)| r).collect(), n);
    (imfs, residual)
}

/// Ensemble EMD (Wu and Huang 2009), each channel on its own: the mean of EMDs of the signal
/// plus white noise with SD `noise_sd`·SD(signal). Every trial takes at most ⌊log₂ n⌋ − 1 IMFs
/// unless set, as Wu and Huang's eemd.m; unlike it, a trial stops early once its residual no
/// longer oscillates. Modes sum to the signal plus the mean noise, which shrinks with more trials.
#[derive(Debug, Clone)]
pub struct Eemd {
    pub stop: Stop,
    pub max_imfs: Option<usize>,
    pub trials: usize,
    pub noise_sd: f64,
    pub seed: u64,
}

impl Eemd {
    /// Wu and Huang's settings: 10 siftings per IMF, 100 trials, noise SD 0.2.
    pub fn new(trials: usize, noise_sd: f64, seed: u64) -> Result<Self, MethodError> {
        Self {
            stop: Stop::Fixed { siftings: 10 },
            max_imfs: None,
            trials,
            noise_sd,
            seed,
        }
        .validated()
    }

    pub fn validated(self) -> Result<Self, MethodError> {
        validate(self.stop, self.max_imfs)?;
        check_noise(self.trials, self.noise_sd, "noise_sd")?;
        Ok(self)
    }
}

impl ModeDecomposer for Eemd {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "eemd",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: false,
        }
    }

    fn decompose(&self, signal: &MultichannelSeries) -> Result<Decomposition, MethodError> {
        let n = signal.n_samples();
        let k = self
            .max_imfs
            .unwrap_or_else(|| (n.ilog2() as usize).saturating_sub(1).max(1));
        per_channel(signal, |c, x| {
            let scale = self.noise_sd * sd(x);
            let noises: Vec<_> = white_noise(self.seed, c, self.trials, n)
                .into_iter()
                .map(|w| w * scale)
                .collect();
            eemd_channel(x, &noises, self.stop, k)
        })
    }
}

/// The local mean M(y): what is left after the first IMF (y itself when it has none).
fn local_mean(y: ArrayView1<f64>, stop: Stop) -> Array1<f64> {
    emd1(y, stop, Some(1)).1
}

fn has_imf(y: ArrayView1<f64>, stop: Stop) -> bool {
    !emd1(y, stop, Some(1)).0.is_empty()
}

/// CEEMDAN of one channel with unit-variance noise realisations, as iceemdan.m (SNRFlag 1).
pub(crate) fn ceemdan_channel(
    x: ArrayView1<f64>,
    noises: &[Array1<f64>],
    stop: Stop,
    epsilon: f64,
    max_imfs: Option<usize>,
) -> Channel {
    let n = x.len();
    let scale = sd(x);
    if scale == 0.0 || !scale.is_finite() {
        return (Vec::new(), x.to_owned());
    }
    let x = &x / scale;
    // Each realisation's EMD, with its residual as the last row, as MATLAB emd returns it.
    let noise_modes: Vec<Vec<Array1<f64>>> = noises
        .par_iter()
        .map(|w| {
            let (mut imfs, residual) = emd1(w.view(), stop, None);
            imfs.push(residual);
            imfs
        })
        .collect();
    let first: Vec<Array1<f64>> = noise_modes
        .par_iter()
        .map(|e| {
            let xi = &x + &(&e[0] * (epsilon / sd(e[0].view())));
            local_mean(xi.view(), stop)
        })
        .collect();
    let mut r = mean_of(first, n);
    let mut modes = vec![&x - &r];
    let mut k = 1;
    while max_imfs.is_none_or(|m| modes.len() < m) && has_imf(r.view(), stop) {
        let beta = epsilon * sd(r.view());
        let means: Vec<Array1<f64>> = noise_modes
            .par_iter()
            .map(|e| match e.get(k) {
                Some(noise) => local_mean((&r + &(noise * beta)).view(), stop),
                None => local_mean(r.view(), stop),
            })
            .collect();
        let next = mean_of(means, n);
        modes.push(&r - &next);
        r = next;
        k += 1;
    }
    (modes.into_iter().map(|m| m * scale).collect(), r * scale)
}

/// Complete ensemble EMD with adaptive noise, improved form (Colominas et al. 2014), each channel
/// on its own. Mode k is r_{k−1} − ⟨M(r_{k−1} + ε·SD(r_{k−1})·E_k(w))⟩, so modes sum to the
/// signal exactly.
#[derive(Debug, Clone)]
pub struct Ceemdan {
    pub stop: Stop,
    pub max_imfs: Option<usize>,
    pub trials: usize,
    pub epsilon: f64,
    pub seed: u64,
}

impl Ceemdan {
    /// The authors' settings: emd.m's Rilling stop, ε = 0.2.
    pub fn new(trials: usize, epsilon: f64, seed: u64) -> Result<Self, MethodError> {
        Self {
            stop: Stop::RILLING_EMD,
            max_imfs: None,
            trials,
            epsilon,
            seed,
        }
        .validated()
    }

    pub fn validated(self) -> Result<Self, MethodError> {
        validate(self.stop, self.max_imfs)?;
        check_noise(self.trials, self.epsilon, "epsilon")?;
        Ok(self)
    }
}

impl ModeDecomposer for Ceemdan {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "ceemdan",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: false,
        }
    }

    fn decompose(&self, signal: &MultichannelSeries) -> Result<Decomposition, MethodError> {
        let n = signal.n_samples();
        per_channel(signal, |c, x| {
            let noises = white_noise(self.seed, c, self.trials, n);
            ceemdan_channel(x, &noises, self.stop, self.epsilon, self.max_imfs)
        })
    }
}
