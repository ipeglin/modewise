//! Rearrangements of observations: permutations or sign flips, free or restricted by
//! exchangeability blocks (Winkler et al. 2015).

use modewise_method::MethodError;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::permutation::Exchange;

/// Exchangeability blocks: each observation's block label. Observations are exchanged only
/// within their block, or, with `whole`, blocks are exchanged (or sign-flipped) as units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blocks {
    pub block: Vec<usize>,
    pub whole: bool,
}

/// One rearrangement: row i of the result is row `source[i]` of the data, negated if `flip[i]`.
pub(crate) struct Rearrangement {
    pub source: Vec<usize>,
    pub flip: Vec<bool>,
}

pub(crate) struct Rearrangements {
    exchange: Exchange,
    n: usize,
    /// Sets of units permuted among themselves; a unit is a list of rows, of equal length
    /// within a set. Sign flips act on every unit.
    sets: Vec<Vec<Vec<usize>>>,
}

impl Rearrangements {
    pub fn new(n: usize, exchange: Exchange, blocks: Option<&Blocks>) -> Result<Self, MethodError> {
        let rows = |r: Vec<usize>| r.into_iter().map(|i| vec![i]).collect::<Vec<_>>();
        let sets = match blocks {
            None => vec![rows((0..n).collect())],
            Some(b) => {
                if b.block.len() != n {
                    return Err(MethodError::ShapeMismatch(format!(
                        "{} block labels for {n} observations",
                        b.block.len()
                    )));
                }
                let mut members: Vec<(usize, Vec<usize>)> = Vec::new();
                for (i, &label) in b.block.iter().enumerate() {
                    match members.iter_mut().find(|(l, _)| *l == label) {
                        Some((_, m)) => m.push(i),
                        None => members.push((label, vec![i])),
                    }
                }
                let members: Vec<Vec<usize>> = members.into_iter().map(|(_, m)| m).collect();
                if !b.whole {
                    members.into_iter().map(rows).collect()
                } else {
                    if exchange == Exchange::Permute
                        && members.iter().any(|m| m.len() != members[0].len())
                    {
                        return Err(MethodError::invalid(
                            "blocks",
                            "whole blocks can only be permuted when they are the same size",
                        ));
                    }
                    vec![members]
                }
            }
        };
        Ok(Self { exchange, n, sets })
    }

    /// Distinct rearrangements, the identity included; `None` if too many to count.
    pub fn count(&self) -> Option<u128> {
        match self.exchange {
            Exchange::SignFlip => 1u128.checked_shl(self.units().count() as u32),
            Exchange::Permute => self
                .sets
                .iter()
                .try_fold(1u128, |c, set| c.checked_mul(factorial(set.len())?)),
        }
    }

    /// Rearrangement number `index` of `count()`; 0 is the identity.
    pub fn enumerated(&self, mut index: u128) -> Rearrangement {
        match self.exchange {
            Exchange::SignFlip => {
                let flips = self.units().map(|_| {
                    let bit = index & 1 == 1;
                    index >>= 1;
                    bit
                });
                self.flipped(flips.collect())
            }
            Exchange::Permute => {
                let orders = self.sets.iter().map(|set| {
                    let k = set.len();
                    let radix = factorial(k).unwrap_or(u128::MAX);
                    let order = lehmer(index % radix, k);
                    index /= radix;
                    order
                });
                self.permuted(orders.collect())
            }
        }
    }

    /// A rearrangement drawn uniformly at random.
    pub fn random(&self, rng: &mut impl Rng) -> Rearrangement {
        match self.exchange {
            Exchange::SignFlip => {
                let flips = self.units().map(|_| rng.random::<bool>()).collect();
                self.flipped(flips)
            }
            Exchange::Permute => {
                let orders = self
                    .sets
                    .iter()
                    .map(|set| {
                        let mut order: Vec<usize> = (0..set.len()).collect();
                        order.shuffle(rng);
                        order
                    })
                    .collect();
                self.permuted(orders)
            }
        }
    }

    fn units(&self) -> impl Iterator<Item = &Vec<usize>> {
        self.sets.iter().flatten()
    }

    fn flipped(&self, unit_flips: Vec<bool>) -> Rearrangement {
        let mut flip = vec![false; self.n];
        for (unit, f) in self.units().zip(unit_flips) {
            for &i in unit {
                flip[i] = f;
            }
        }
        Rearrangement {
            source: (0..self.n).collect(),
            flip,
        }
    }

    fn permuted(&self, orders: Vec<Vec<usize>>) -> Rearrangement {
        let mut source: Vec<usize> = (0..self.n).collect();
        for (set, order) in self.sets.iter().zip(orders) {
            for (unit, &from) in set.iter().zip(&order) {
                for (&i, &j) in unit.iter().zip(&set[from]) {
                    source[i] = j;
                }
            }
        }
        Rearrangement {
            source,
            flip: vec![false; self.n],
        }
    }
}

fn factorial(k: usize) -> Option<u128> {
    (1..=k as u128).try_fold(1u128, |f, i| f.checked_mul(i))
}

/// Permutation number `index` of `k` items in lexicographic order (index < k!).
fn lehmer(mut index: u128, k: usize) -> Vec<usize> {
    let mut left: Vec<usize> = (0..k).collect();
    (0..k)
        .map(|i| {
            let f = factorial(k - 1 - i).unwrap_or(u128::MAX);
            let digit = (index / f) as usize;
            index %= f;
            left.remove(digit)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn all(r: &Rearrangements) -> BTreeSet<(Vec<usize>, Vec<bool>)> {
        (0..r.count().unwrap())
            .map(|i| {
                let a = r.enumerated(i);
                (a.source, a.flip)
            })
            .collect()
    }

    #[test]
    fn lehmer_codes_enumerate_every_permutation_once() {
        let seen: BTreeSet<Vec<usize>> = (0..24).map(|i| lehmer(i, 4)).collect();
        assert_eq!(seen.len(), 24);
        assert_eq!(lehmer(0, 4), [0, 1, 2, 3]);
        assert_eq!(lehmer(23, 4), [3, 2, 1, 0]);
        assert_eq!((factorial(0), factorial(34).is_some()), (Some(1), true));
        assert_eq!(factorial(35), None);
    }

    #[test]
    fn blocks_restrict_what_moves() {
        let within = Blocks {
            block: vec![7, 7, 3, 3, 3],
            whole: false,
        };
        let r = Rearrangements::new(5, Exchange::Permute, Some(&within)).unwrap();
        assert_eq!(r.count(), Some(2 * 6));
        let found = all(&r);
        assert_eq!(found.len(), 12);
        for (source, _) in &found {
            assert!(source[..2].iter().all(|&j| j < 2) && source[2..].iter().all(|&j| j >= 2));
        }
        assert_eq!(r.enumerated(0).source, [0, 1, 2, 3, 4]);

        let whole = Blocks {
            block: vec![0, 0, 1, 1, 2, 2],
            whole: true,
        };
        let r = Rearrangements::new(6, Exchange::Permute, Some(&whole)).unwrap();
        assert_eq!(r.count(), Some(6));
        for (source, _) in all(&r) {
            for b in 0..3 {
                assert_eq!(source[2 * b] + 1, source[2 * b + 1], "{source:?}");
            }
        }
        let flips = Rearrangements::new(6, Exchange::SignFlip, Some(&whole)).unwrap();
        assert_eq!(flips.count(), Some(8));
        for (_, flip) in all(&flips) {
            assert!((0..3).all(|b| flip[2 * b] == flip[2 * b + 1]));
        }
        let mut rng = rand::rng();
        let drawn = flips.random(&mut rng);
        assert!((0..3).all(|b| drawn.flip[2 * b] == drawn.flip[2 * b + 1]));
        let drawn = r.random(&mut rng);
        assert!((0..3).all(|b| drawn.source[2 * b] + 1 == drawn.source[2 * b + 1]));
    }

    #[test]
    fn invalid_blocks_are_refused() {
        let uneven = Blocks {
            block: vec![0, 0, 1],
            whole: true,
        };
        assert!(Rearrangements::new(3, Exchange::Permute, Some(&uneven)).is_err());
        assert!(Rearrangements::new(3, Exchange::SignFlip, Some(&uneven)).is_ok());
        assert!(Rearrangements::new(4, Exchange::SignFlip, Some(&uneven)).is_err());
        let free = Rearrangements::new(40, Exchange::Permute, None).unwrap();
        assert_eq!(free.count(), None);
        let free = Rearrangements::new(200, Exchange::SignFlip, None).unwrap();
        assert_eq!(free.count(), None);
    }
}
