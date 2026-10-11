//! False discovery rate adjustment.

/// Which false discovery rate procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fdr {
    /// Benjamini & Hochberg (1995): valid for independent or positively dependent tests.
    BenjaminiHochberg,
    /// Benjamini & Yekutieli (2001): valid under any dependence, more conservative.
    BenjaminiYekutieli,
}

/// Adjusted p-values (q-values) in the original order. NaN p-values stay NaN and are not counted.
pub fn adjust(p: &[f64], method: Fdr) -> Vec<f64> {
    let mut order: Vec<usize> = (0..p.len()).filter(|&i| !p[i].is_nan()).collect();
    let m = order.len() as f64;
    let scale = match method {
        Fdr::BenjaminiHochberg => 1.0,
        Fdr::BenjaminiYekutieli => (1..=order.len()).map(|i| 1.0 / i as f64).sum(),
    };
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut q = vec![f64::NAN; p.len()];
    let mut running = 1.0f64;
    for (rank, &i) in order.iter().enumerate().rev() {
        running = running.min(p[i] * m * scale / (rank + 1) as f64);
        q[i] = running.min(1.0);
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bh_and_by_by_hand() {
        let p = [0.01, 0.04, 0.03, f64::NAN, 0.5];
        let bh = adjust(&p, Fdr::BenjaminiHochberg);
        // Sorted 0.01, 0.03, 0.04, 0.5 with m = 4: 0.04, 0.0533…, 0.0533…, 0.5.
        assert!((bh[0] - 0.04).abs() < 1e-15);
        assert!((bh[1] - 0.04 * 4.0 / 3.0).abs() < 1e-15 && (bh[2] - bh[1]).abs() < 1e-15);
        assert!(bh[3].is_nan() && (bh[4] - 0.5).abs() < 1e-15);
        let by = adjust(&p, Fdr::BenjaminiYekutieli);
        let c: f64 = 1.0 + 0.5 + 1.0 / 3.0 + 0.25;
        assert!((by[0] - (0.04 * c).min(1.0)).abs() < 1e-15);
        assert_eq!(adjust(&[0.9, 0.95], Fdr::BenjaminiYekutieli), [1.0, 1.0]);
        assert!(adjust(&[], Fdr::BenjaminiHochberg).is_empty());
    }
}
