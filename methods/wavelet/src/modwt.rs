//! Maximal overlap DWT and its multiresolution analysis (Percival and Walden 2000, ch. 5;
//! Percival and Mofjeld 1997), with circular filtering.

/// MODWT filters: the DWT filters divided by √2.
pub(crate) struct Filters {
    pub scaling: Vec<f64>,
    pub wavelet: Vec<f64>,
}

impl Filters {
    /// From a DWT scaling filter g; the wavelet filter is h_l = (−1)^l g_{L−1−l}.
    pub fn from_scaling(g: &[f64]) -> Self {
        let l = g.len();
        let scaling = g
            .iter()
            .map(|v| v / std::f64::consts::SQRT_2)
            .collect::<Vec<_>>();
        let wavelet = (0..l)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 } * scaling[l - 1 - i])
            .collect();
        Self { scaling, wavelet }
    }
}

/// One pyramid step at level j (stride 2^{j−1}): wavelet and scaling coefficients of `v`.
fn forward(v: &[f64], f: &Filters, stride: usize) -> (Vec<f64>, Vec<f64>) {
    let n = v.len();
    let (mut w, mut s) = (vec![0.0; n], vec![0.0; n]);
    for t in 0..n {
        for (l, (h, g)) in f.wavelet.iter().zip(&f.scaling).enumerate() {
            let x = v[(t + n - (stride * l) % n) % n];
            w[t] += h * x;
            s[t] += g * x;
        }
    }
    (w, s)
}

/// The inverse step: level j − 1 scaling coefficients from level j wavelet and scaling ones.
fn inverse(w: &[f64], s: &[f64], f: &Filters, stride: usize) -> Vec<f64> {
    let n = w.len();
    let mut v = vec![0.0; n];
    for (t, out) in v.iter_mut().enumerate() {
        for (l, (h, g)) in f.wavelet.iter().zip(&f.scaling).enumerate() {
            let i = (t + stride * l) % n;
            *out += h * w[i] + g * s[i];
        }
    }
    v
}

/// The MRA of `x` to level J: details D_1..D_J and the smooth S_J, which sum to `x`.
pub(crate) fn mra(x: &[f64], f: &Filters, levels: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = x.len();
    let mut coefficients = Vec::with_capacity(levels);
    let mut v = x.to_vec();
    for j in 1..=levels {
        let (w, s) = forward(&v, f, 1 << (j - 1));
        coefficients.push(w);
        v = s;
    }
    let zeros = vec![0.0; n];
    // Synthesise from one level's coefficients only, back to level 0.
    let synthesise = |level: usize, w: &[f64], s: &[f64]| {
        let mut out = inverse(w, s, f, 1 << (level - 1));
        for j in (1..level).rev() {
            out = inverse(&zeros, &out, f, 1 << (j - 1));
        }
        out
    };
    let details = coefficients
        .iter()
        .enumerate()
        .map(|(i, w)| synthesise(i + 1, w, &zeros))
        .collect();
    let smooth = synthesise(levels, &zeros, &v);
    (details, smooth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scaling_filter_is_orthonormal() {
        // Σ g = √2, Σ g² = 1 and Σ g_l g_{l+2k} = 0 for k ≠ 0 (Percival and Walden 2000, ch. 4).
        for (name, g) in crate::filters::FILTERS {
            assert!(
                (g.iter().sum::<f64>() - std::f64::consts::SQRT_2).abs() < 1e-10,
                "{name}"
            );
            for k in 0..g.len() / 2 {
                let dot: f64 = (0..g.len() - 2 * k).map(|l| g[l] * g[l + 2 * k]).sum();
                let want = if k == 0 { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-10, "{name} shift {}: {dot}", 2 * k);
            }
        }
    }

    #[test]
    fn coefficients_partition_the_energy() {
        // Percival and Walden (2000), eq. 171c: ‖X‖² = Σ_j ‖W_j‖² + ‖V_J‖².
        let x: Vec<f64> = (0..100)
            .map(|t| ((t as f64) * 0.37).sin() + 0.01 * (t * t) as f64)
            .collect();
        let f = Filters::from_scaling(
            crate::filters::FILTERS
                .iter()
                .find(|(n, _)| *n == "db3")
                .unwrap()
                .1,
        );
        let mut v = x.clone();
        let mut energy = 0.0;
        for j in 1..=4 {
            let (w, s) = forward(&v, &f, 1 << (j - 1));
            energy += w.iter().map(|a| a * a).sum::<f64>();
            v = s;
        }
        energy += v.iter().map(|a| a * a).sum::<f64>();
        let total: f64 = x.iter().map(|a| a * a).sum();
        assert!((energy - total).abs() < 1e-9 * total);
    }
}
