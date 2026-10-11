//! Mirror extrema across the ends so envelopes reach the first and last samples (Rilling,
//! Flandrin and Gonçalvès 2003; the `boundary_conditions` of emd.m and memd.m, PyEMD "simple").

const NBSYM: usize = 2;

/// Envelope knots: positions and the samples whose values they take.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Knots {
    pub positions: Vec<f64>,
    pub samples: Vec<usize>,
}

/// Extrema too few or too close to an end to mirror; the reference codes stop sifting then.
#[derive(Debug, PartialEq)]
pub(crate) struct Unmirrorable;

fn rev(v: &[usize]) -> Vec<usize> {
    v.iter().rev().copied().collect()
}

fn reflect(centre: usize, idx: &[usize]) -> Vec<f64> {
    idx.iter()
        .map(|&i| 2.0 * centre as f64 - i as f64)
        .collect()
}

/// Knots for the lower and upper envelopes of `x`, given its minima and maxima.
pub(crate) fn mirror(
    indmin: &[usize],
    indmax: &[usize],
    x: &[f64],
) -> Result<(Knots, Knots), Unmirrorable> {
    let (Some(&min0), Some(&max0), Some(&min1), Some(&max1)) =
        (indmin.first(), indmax.first(), indmin.last(), indmax.last())
    else {
        return Err(Unmirrorable);
    };
    let lx = x.len() - 1;
    let (nmin, nmax) = (indmin.len(), indmax.len());
    let head = |v: &[usize], from: usize, to: usize| {
        rev(&v[from.min(v.len())..to.min(v.len()).max(from.min(v.len()))])
    };
    let tail = |v: &[usize], from: usize, cut_last: bool| {
        let end = if cut_last {
            v.len().saturating_sub(1)
        } else {
            v.len()
        };
        rev(&v[from.min(end)..end])
    };

    let (mut lmax, mut lmin, mut lsym);
    if max0 < min0 {
        if x[0] > x[min0] {
            (lmax, lmin, lsym) = (head(indmax, 1, NBSYM + 1), head(indmin, 0, NBSYM), max0);
        } else {
            let mut m = head(indmin, 0, NBSYM - 1);
            m.push(0);
            (lmax, lmin, lsym) = (head(indmax, 0, NBSYM), m, 0);
        }
    } else if x[0] < x[max0] {
        (lmax, lmin, lsym) = (head(indmax, 0, NBSYM), head(indmin, 1, NBSYM + 1), min0);
    } else {
        let mut m = head(indmax, 0, NBSYM - 1);
        m.push(0);
        (lmax, lmin, lsym) = (m, head(indmin, 0, NBSYM), 0);
    }

    let (mut rmax, mut rmin, mut rsym);
    if max1 < min1 {
        if x[lx] < x[max1] {
            (rmax, rmin, rsym) = (
                tail(indmax, (nmax + 1).saturating_sub(NBSYM + 1), false),
                tail(indmin, nmin.saturating_sub(NBSYM + 1), true),
                min1,
            );
        } else {
            let mut m = vec![lx];
            m.extend(tail(indmax, (nmax + 1).saturating_sub(NBSYM), false));
            (rmax, rmin, rsym) = (m, tail(indmin, nmin.saturating_sub(NBSYM), false), lx);
        }
    } else if x[lx] > x[min1] {
        (rmax, rmin, rsym) = (
            tail(indmax, nmax.saturating_sub(NBSYM + 1), true),
            tail(indmin, nmin.saturating_sub(NBSYM), false),
            max1,
        );
    } else {
        let mut m = vec![lx];
        m.extend(tail(indmin, (nmin + 1).saturating_sub(NBSYM), false));
        (rmax, rmin, rsym) = (tail(indmax, nmax.saturating_sub(NBSYM), false), m, lx);
    }

    let first = |v: &[f64]| v.first().copied().ok_or(Unmirrorable);
    let last = |v: &[f64]| v.last().copied().ok_or(Unmirrorable);

    // When the mirrored extrema do not reach past an end, mirror about the end sample instead.
    if first(&reflect(lsym, &lmin))? > 0.0 || first(&reflect(lsym, &lmax))? > 0.0 {
        if lsym == max0 {
            lmax = head(indmax, 0, NBSYM);
        } else {
            lmin = head(indmin, 0, NBSYM);
        }
        if lsym == 0 {
            return Err(Unmirrorable);
        }
        lsym = 0;
    }
    if last(&reflect(rsym, &rmin))? < lx as f64 || last(&reflect(rsym, &rmax))? < lx as f64 {
        if rsym == max1 {
            rmax = tail(indmax, nmax.saturating_sub(NBSYM), false);
        } else {
            rmin = tail(indmin, nmin.saturating_sub(NBSYM), false);
        }
        if rsym == lx {
            return Err(Unmirrorable);
        }
        rsym = lx;
    }

    let knots = |left: &[usize], inner: &[usize], right: &[usize]| -> Result<Knots, Unmirrorable> {
        let mut positions = reflect(lsym, left);
        positions.extend(inner.iter().map(|&i| i as f64));
        positions.extend(reflect(rsym, right));
        let samples: Vec<usize> = left.iter().chain(inner).chain(right).copied().collect();
        if positions.windows(2).any(|w| w[1] <= w[0]) {
            return Err(Unmirrorable);
        }
        Ok(Knots { positions, samples })
    };
    Ok((knots(&lmin, indmin, &rmin)?, knots(&lmax, indmax, &rmax)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knots_reach_past_both_ends_and_degenerate_extrema_are_refused() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.7).sin()).collect();
        let (min, max) = crate::extrema::local_peaks(&x, false);
        let (lower, upper) = mirror(&min, &max, &x).unwrap();
        for k in [&lower, &upper] {
            assert!(k.positions[0] <= 0.0 && *k.positions.last().unwrap() >= 39.0);
            assert_eq!(k.positions.len(), k.samples.len());
        }
        assert_eq!(mirror(&[], &[3], &x), Err(Unmirrorable));
    }
}
