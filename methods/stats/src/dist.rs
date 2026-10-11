//! Student's t distribution through the regularised incomplete beta function.

/// Upper tail probability P(T > t) for Student's t with `df` degrees of freedom.
pub fn t_upper(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if t.is_infinite() {
        return if t > 0.0 { 0.0 } else { 1.0 };
    }
    let x = df / (df + t * t);
    let half = 0.5 * incomplete_beta(0.5 * df, 0.5, x);
    if t > 0.0 { half } else { 1.0 - half }
}

/// Regularised incomplete beta I_x(a, b) by its continued fraction (Lentz's method), using the
/// symmetry I_x(a, b) = 1 − I_{1−x}(b, a) where the fraction converges faster.
pub fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * continued_fraction(a, b, x) / a
    } else {
        1.0 - front * continued_fraction(b, a, 1.0 - x) / b
    }
}

fn continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=1000 {
        let m = m as f64;
        for numerator in [
            m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0)),
        ] {
            d = 1.0 + numerator * d;
            if d.abs() < TINY {
                d = TINY;
            }
            c = 1.0 + numerator / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            h *= d * c;
        }
        if (d * c - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// ln Γ(x) for x > 0, by the Lanczos approximation (g = 7, nine coefficients).
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let series = G[1..]
        .iter()
        .enumerate()
        .fold(G[0], |s, (i, g)| s + g / (x + i as f64 + 1.0));
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + series.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-12);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-12);
        assert!(
            (ln_gamma(0.25) - 3.625_609_908_221_908f64.ln()).abs() < 1e-12,
            "reflection below 0.5"
        );
        assert_eq!(
            (
                incomplete_beta(2.0, 3.0, 0.0),
                incomplete_beta(2.0, 3.0, 1.0)
            ),
            (0.0, 1.0)
        );
        // I_x(1, 1) = x and I_x(a, 1) = x^a.
        assert!((incomplete_beta(1.0, 1.0, 0.3) - 0.3).abs() < 1e-14);
        assert!((incomplete_beta(3.0, 1.0, 0.6) - 0.216).abs() < 1e-14);
        assert!((t_upper(0.0, 7.0) - 0.5).abs() < 1e-15);
        assert_eq!(
            (t_upper(f64::INFINITY, 3.0), t_upper(f64::NEG_INFINITY, 3.0)),
            (0.0, 1.0)
        );
        assert!(t_upper(f64::NAN, 3.0).is_nan() && t_upper(1.0, 0.0).is_nan());
        // Cauchy (df = 1): P(T > 1) = 1/4.
        assert!((t_upper(1.0, 1.0) - 0.25).abs() < 1e-14);
        assert!((t_upper(-1.0, 1.0) - 0.75).abs() < 1e-14);
    }
}
