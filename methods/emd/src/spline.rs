//! Cubic spline with not-a-knot ends, as MATLAB `spline` and scipy `CubicSpline` (default).
//! Two knots give a line and three a parabola, as both do.

use ndarray::{Array2, ArrayView2};

/// Interpolates each column of `values` (knots × columns) at the sample positions `0..n`.
/// Knots must be strictly increasing and span every position.
pub(crate) fn interpolate(knots: &[f64], values: ArrayView2<f64>, n: usize) -> Array2<f64> {
    let k = knots.len();
    let cols = values.ncols();
    let mut out = Array2::zeros((n, cols));
    let h: Vec<f64> = knots.windows(2).map(|w| w[1] - w[0]).collect();
    for c in 0..cols {
        let y: Vec<f64> = values.column(c).to_vec();
        let m = second_derivatives(&h, &y);
        let mut seg = 0;
        for t in 0..n {
            let x = t as f64;
            while seg + 2 < k && x > knots[seg + 1] {
                seg += 1;
            }
            let (a, b) = (knots[seg + 1] - x, x - knots[seg]);
            let hs = h[seg];
            out[[t, c]] = (m[seg] * a.powi(3) + m[seg + 1] * b.powi(3)) / (6.0 * hs)
                + (y[seg] / hs - m[seg] * hs / 6.0) * a
                + (y[seg + 1] / hs - m[seg + 1] * hs / 6.0) * b;
        }
    }
    out
}

/// Second derivatives at the knots. Not-a-knot: the third derivative is continuous at the
/// second and second-last knots. The two end unknowns are eliminated, leaving a tridiagonal
/// system for the interior ones.
fn second_derivatives(h: &[f64], y: &[f64]) -> Vec<f64> {
    let k = y.len();
    let slope: Vec<f64> = (0..k - 1).map(|i| (y[i + 1] - y[i]) / h[i]).collect();
    match k {
        2 => return vec![0.0; 2],
        3 => {
            // One parabola through three points: constant second derivative.
            let m = 2.0 * (slope[1] - slope[0]) / (h[0] + h[1]);
            return vec![m; 3];
        }
        _ => {}
    }
    let n = k - 2;
    let (mut lower, mut diag, mut upper, mut rhs) =
        (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for r in 0..n {
        let i = r + 1;
        lower[r] = h[i - 1];
        diag[r] = 2.0 * (h[i - 1] + h[i]);
        upper[r] = h[i];
        rhs[r] = 6.0 * (slope[i] - slope[i - 1]);
    }
    // M0 = ((h0 + h1) M1 − h0 M2) / h1, folded into the first row.
    let (h0, h1) = (h[0], h[1]);
    diag[0] += h0 * (h0 + h1) / h1;
    upper[0] -= h0 * h0 / h1;
    // M_{k−1} = ((h_{k−2} + h_{k−3}) M_{k−2} − h_{k−2} M_{k−3}) / h_{k−3}, folded into the last.
    let (ha, hb) = (h[k - 2], h[k - 3]);
    diag[n - 1] += ha * (ha + hb) / hb;
    lower[n - 1] -= ha * ha / hb;
    // Four or more knots leave at least two interior unknowns.
    let interior = thomas(&lower, &diag, &upper, &rhs);
    let mut m = Vec::with_capacity(k);
    m.push(((h0 + h1) * interior[0] - h0 * interior[1]) / h1);
    m.extend(&interior);
    m.push(((ha + hb) * interior[n - 1] - ha * interior[n - 2]) / hb);
    m
}

fn thomas(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let (mut c, mut d) = (vec![0.0; n], vec![0.0; n]);
    c[0] = upper[0] / diag[0];
    d[0] = rhs[0] / diag[0];
    for i in 1..n {
        let denom = diag[i] - lower[i] * c[i - 1];
        c[i] = upper[i] / denom;
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d[i] - c[i] * x[i + 1];
    }
    x
}
