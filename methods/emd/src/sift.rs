//! The sifting engine shared by EMD and MEMD, following memd.m: envelopes of projections along
//! direction vectors, averaged; one channel with direction [1] is plain EMD.

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::boundary::mirror;
use crate::extrema::{local_peaks, zero_crossings};
use crate::spline::interpolate;

/// When sifting one mode stops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stop {
    /// Rilling, Flandrin and Gonçalvès (2003): with σ = |envelope mean| / mode amplitude, stop
    /// once σ < `sd` on all but a fraction `tol` of samples and σ < `sd2` everywhere.
    Rilling { sd: f64, sd2: f64, tol: f64 },
    /// Huang et al. (2003) S-number: stop after `count` consecutive siftings in which the
    /// extrema and zero crossings differ by at most one (memd.m `fix_h`).
    SNumber { count: usize },
    /// A fixed number of siftings per mode (Wu and Huang 2009).
    Fixed { siftings: usize },
}

impl Stop {
    /// emd.m's default.
    pub const RILLING_EMD: Self = Self::Rilling {
        sd: 0.05,
        sd2: 0.5,
        tol: 0.05,
    };
    /// memd.m's default.
    pub const RILLING_MEMD: Self = Self::Rilling {
        sd: 0.075,
        sd2: 0.75,
        tol: 0.075,
    };
}

/// Limits from the reference codes.
pub(crate) const MAX_SIFTINGS: usize = 1000;

pub(crate) struct Envelope {
    mean: Array2<f64>,
    amplitude: Array1<f64>,
    extrema: Vec<usize>,
    zeros: Vec<usize>,
}

fn project(m: ArrayView2<f64>, direction: &[f64]) -> Vec<f64> {
    m.rows()
        .into_iter()
        .map(|row| row.iter().zip(direction).map(|(a, b)| a * b).sum())
        .collect()
}

/// Mean envelope over all directions with enough extrema, or `None` when an envelope cannot be
/// built (the reference codes then stop sifting with a zero mean).
fn envelope(m: ArrayView2<f64>, directions: &[Vec<f64>], flat_rule: bool) -> Option<Envelope> {
    let (n, dims) = m.dim();
    let mut mean = Array2::zeros((n, dims));
    let mut amplitude = Array1::zeros(n);
    let (mut extrema, mut zeros) = (
        Vec::with_capacity(directions.len()),
        Vec::with_capacity(directions.len()),
    );
    let mut used = 0usize;
    for direction in directions {
        let y = project(m, direction);
        let (indmin, indmax) = local_peaks(&y, flat_rule);
        extrema.push(indmin.len() + indmax.len());
        zeros.push(zero_crossings(&y));
        if indmin.len() + indmax.len() < 3 {
            continue;
        }
        let (lower, upper) = mirror(&indmin, &indmax, &y).ok()?;
        let lo = interpolate(
            &lower.positions,
            m.select(Axis(0), &lower.samples).view(),
            n,
        );
        let hi = interpolate(
            &upper.positions,
            m.select(Axis(0), &upper.samples).view(),
            n,
        );
        let half = (&hi - &lo) / 2.0;
        amplitude += &half.map_axis(Axis(1), |r| r.dot(&r).sqrt());
        mean += &((&hi + &lo) / 2.0);
        used += 1;
    }
    if used == 0 {
        return Some(Envelope {
            mean: Array2::zeros((n, dims)),
            amplitude: Array1::zeros(n),
            extrema: vec![0; directions.len()],
            zeros,
        });
    }
    mean /= used as f64;
    amplitude /= used as f64;
    Some(Envelope {
        mean,
        amplitude,
        extrema,
        zeros,
    })
}

/// Whether to stop sifting `m`, and the envelope mean to subtract if not.
struct Check {
    stop: bool,
    mean: Array2<f64>,
}

struct StopState {
    rule: Stop,
    counter: usize,
    siftings: usize,
}

impl StopState {
    fn new(rule: Stop) -> Self {
        Self {
            rule,
            counter: 0,
            siftings: 0,
        }
    }

    fn check(&mut self, m: ArrayView2<f64>, directions: &[Vec<f64>], flat_rule: bool) -> Check {
        let Some(env) = envelope(m, directions, flat_rule) else {
            return Check {
                stop: true,
                mean: Array2::zeros(m.dim()),
            };
        };
        let stop = match self.rule {
            Stop::Rilling { sd, sd2, tol } => {
                let mut sx = env.mean.map_axis(Axis(1), |r| r.dot(&r).sqrt());
                // As the reference: scale only when every sample has an amplitude.
                if env.amplitude.iter().all(|&a| a != 0.0) {
                    sx /= &env.amplitude;
                }
                let n = sx.len() as f64;
                let above = sx.iter().filter(|&&v| v > sd).count() as f64 / n;
                let wild = sx.iter().any(|&v| v > sd2);
                !((above > tol || wild) && env.extrema.iter().any(|&e| e > 2))
            }
            Stop::SNumber { count } => {
                let unsettled = env
                    .extrema
                    .iter()
                    .zip(&env.zeros)
                    .all(|(&e, &z)| e.abs_diff(z) > 1);
                if unsettled {
                    self.counter = 0;
                    false
                } else {
                    self.counter += 1;
                    self.counter >= count
                }
            }
            Stop::Fixed { siftings } => self.siftings >= siftings,
        };
        Check {
            stop,
            mean: env.mean,
        }
    }
}

/// True when every projection has fewer than three extrema: what is left is the residual.
fn is_residual(r: ArrayView2<f64>, directions: &[Vec<f64>], flat_rule: bool) -> bool {
    directions.iter().all(|d| {
        let (indmin, indmax) = local_peaks(&project(r, d), flat_rule);
        indmin.len() + indmax.len() < 3
    })
}

/// Intrinsic mode functions (samples × dims each), highest frequency first, and the residual.
pub(crate) fn decompose(
    x: ArrayView2<f64>,
    directions: &[Vec<f64>],
    rule: Stop,
    max_imfs: Option<usize>,
    flat_rule: bool,
) -> (Vec<Array2<f64>>, Array2<f64>) {
    let scale = x.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let mut r = x.to_owned();
    let mut imfs = Vec::new();
    while max_imfs.is_none_or(|k| imfs.len() < k) && !is_residual(r.view(), directions, flat_rule) {
        let mut m = r.clone();
        let mut state = StopState::new(rule);
        let mut check = state.check(m.view(), directions, flat_rule);
        // A mode this small would only add spurious extrema from rounding.
        if m.iter().fold(0.0f64, |a, v| a.max(v.abs())) < 1e-10 * scale {
            break;
        }
        while !check.stop && state.siftings < MAX_SIFTINGS {
            m -= &check.mean;
            state.siftings += 1;
            check = state.check(m.view(), directions, flat_rule);
        }
        r -= &m;
        imfs.push(m);
    }
    (imfs, r)
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    fn column(f: impl Fn(f64) -> f64) -> Array2<f64> {
        Array2::from_shape_fn((200, 1), |(t, _)| f(t as f64))
    }

    #[test]
    fn s_number_counts_only_siftings_with_extrema_and_zero_crossings_within_one() {
        let up = vec![vec![1.0]];
        // Riding on an offset: many extrema, no zero crossings, so not settled.
        let riding = column(|t| (0.3 * t).sin() + 3.0);
        let mut state = StopState::new(Stop::SNumber { count: 1 });
        assert!(!state.check(riding.view(), &up, false).stop);
        assert_eq!(state.counter, 0);
        // Centred: extrema and zero crossings agree.
        let centred = column(|t| (0.3 * t).sin());
        assert!(state.check(centred.view(), &up, false).stop);
        let mut two = StopState::new(Stop::SNumber { count: 2 });
        assert!(!two.check(centred.view(), &up, false).stop);
        assert!(two.check(centred.view(), &up, false).stop);
    }
}
