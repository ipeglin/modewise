//! Extrema and zero crossings, as memd.m (`local_peaks`, `zero_crossings`). A plateau counts
//! once, at its middle sample.

/// Indices of local minima and maxima. With `memd_flat_rule`, a signal whose values are all
/// below 1e-5 has none, as memd.m's `local_peaks` (which so treats wholly negative projections
/// as flat; off by default, see docs/emd.md).
pub(crate) fn local_peaks(x: &[f64], memd_flat_rule: bool) -> (Vec<usize>, Vec<usize>) {
    let n = x.len();
    if memd_flat_rule && x.iter().all(|&v| v < 1e-5) {
        return (Vec::new(), Vec::new());
    }
    // Samples where the signal changes; a run of equal samples is represented by its middle.
    let mut a: Vec<usize> = (0..n - 1).filter(|&i| x[i + 1] != x[i]).collect();
    let shifts: Vec<(usize, usize)> = (1..a.len())
        .filter(|&j| a[j] - a[j - 1] != 1)
        .map(|j| (j, (a[j] - a[j - 1]) / 2))
        .collect();
    for (j, shift) in shifts {
        a[j] -= shift;
    }
    a.push(n - 1);
    if a.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    let ya: Vec<f64> = a.iter().map(|&i| x[i]).collect();
    let peaks = |sign: f64| -> Vec<usize> {
        let d: Vec<f64> = ya.windows(2).map(|w| sign * (w[1] - w[0])).collect();
        (0..d.len().saturating_sub(1))
            .filter(|&j| d[j] > 0.0 && d[j + 1] < 0.0)
            .map(|j| a[j + 1])
            .collect()
    };
    (peaks(-1.0), peaks(1.0))
}

/// Number of zero crossings; a run of zeros counts once.
pub(crate) fn zero_crossings(x: &[f64]) -> usize {
    let strict = x.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
    let mut runs = 0;
    let mut previous_zero = false;
    for &v in x {
        let zero = v == 0.0;
        if zero && !previous_zero {
            runs += 1;
        }
        previous_zero = zero;
    }
    strict + runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plateaus_count_once_at_their_middle_and_zero_runs_once() {
        assert_eq!(
            local_peaks(&[0.0, 2.0, 2.0, 2.0, 0.0, -1.0, 0.0], false),
            (vec![5], vec![2])
        );
        assert_eq!(local_peaks(&[1.0, 1.0, 1.0], false), (vec![], vec![]));
        // Plateaus of four and five samples away from the start: the middle, rounded down.
        let x = [
            0.0, 1.0, 3.0, 3.0, 3.0, 3.0, 1.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0,
        ];
        assert_eq!(local_peaks(&x, false), (vec![9], vec![3]));
        assert_eq!(
            local_peaks(&[-3.0, -1.0, -3.0, -1.0], false),
            (vec![2], vec![1])
        );
        assert_eq!(
            local_peaks(&[-3.0, -1.0, -3.0, -1.0], true),
            (vec![], vec![]),
            "memd.m flat rule"
        );
        assert_eq!(zero_crossings(&[1.0, -1.0, 0.0, 0.0, 2.0, 0.0, -1.0]), 3);
    }
}
