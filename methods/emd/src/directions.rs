//! Direction vectors on the unit (n − 1)-sphere from a Hammersley sequence, as memd.m.

use std::f64::consts::PI;

/// The Hammersley coordinate for points 1..=n: the radical inverse in a prime `base`, or
/// (i mod (n + 1) + ½) / n for the first coordinate.
fn hammersley(n: usize, base: Option<u64>) -> Vec<f64> {
    (1..=n)
        .map(|i| match base {
            Some(b) => {
                let (mut seed, mut inv, mut value) = (i as u64, 1.0 / b as f64, 0.0);
                while seed != 0 {
                    value += (seed % b) as f64 * inv;
                    inv /= b as f64;
                    seed /= b;
                }
                value
            }
            None => ((i % (n + 1)) as f64 + 0.5) / n as f64,
        })
        .collect()
}

fn primes(count: usize) -> Vec<u64> {
    let mut found = Vec::with_capacity(count);
    let mut candidate = 2u64;
    while found.len() < count {
        if found
            .iter()
            .take_while(|&&p| p * p <= candidate)
            .all(|&p| !candidate.is_multiple_of(p))
        {
            found.push(candidate);
        }
        candidate += 1;
    }
    found
}

/// `count` unit directions in `dims` dimensions.
pub(crate) fn directions(count: usize, dims: usize) -> Vec<Vec<f64>> {
    if dims == 1 {
        return vec![vec![1.0]];
    }
    if dims == 3 {
        let (z, phi) = (hammersley(count, None), hammersley(count, Some(2)));
        return z
            .iter()
            .zip(&phi)
            .map(|(&s, &p)| {
                let t = (2.0 * s - 1.0).clamp(-1.0, 1.0);
                let (st, angle) = ((1.0 - t * t).sqrt(), p * 2.0 * PI);
                vec![st * angle.cos(), st * angle.sin(), t]
            })
            .collect();
    }
    let mut columns = vec![hammersley(count, None)];
    columns.extend(
        primes(dims - 1)
            .into_iter()
            .map(|p| hammersley(count, Some(p))),
    );
    (0..count)
        .map(|it| {
            let b: Vec<f64> = columns.iter().map(|c| 2.0 * c[it] - 1.0).collect();
            // θ_j = atan2(‖b_{j+1..}‖, b_j); hyperspherical coordinates.
            let theta: Vec<f64> = (0..dims - 1)
                .map(|j| {
                    b[j + 1..]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt()
                        .atan2(b[j])
                })
                .collect();
            let mut v = Vec::with_capacity(dims);
            let mut product = 1.0;
            for t in &theta {
                v.push(product * t.cos());
                product *= t.sin();
            }
            v.push(product);
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directions_are_unit_vectors_and_the_sequence_matches_its_definition() {
        assert_eq!(hammersley(4, Some(2)), [0.5, 0.25, 0.75, 0.125]);
        assert_eq!(hammersley(2, None), [0.75, 1.25]);
        assert_eq!(primes(5), [2, 3, 5, 7, 11]);
        for dims in [1, 2, 3, 4, 7] {
            for d in directions(16, dims) {
                assert!((d.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
            }
        }
    }
}
