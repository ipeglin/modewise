# Statistics

Group-level tests on a feature table (`docs/features.md`), with permutation inference and correction for many features. No single source is trusted on its own (see `TESTING.md`).

## Designs

| Function | Tests | Model |
|---|---|---|
| `test_groups(table, column, (a, b), covariates=[...])` | group `a` minus group `b` | GLM: intercept, a 0/1 dummy for `a`, centred covariates |
| `test_groups(..., welch=True)` | the same, without covariates | Welch's t |
| `test_paired(table, (a, b), covariates=[...])` | condition `a` minus `b` within each recording | one-sample GLM on the differences, by sign flipping |
| `test_correlation(table, score, covariates=[...])` | the slope of a numeric column | GLM: intercept, the score, centred covariates |

- **Covariates:** numeric columns are centred. Text columns (such as sex) are dummy-coded against their first level, then centred.
- **Missing values:** rows missing any value the design needs are dropped, and `report.dropped` counts them.
- **One row per subject:** every design needs exactly one row per subject, because rows must be exchangeable under the null. Average runs, or filter, first. A table built from epochs has one row per condition, so `test_groups` and `test_correlation` need `condition=`.

## Inference

| Choice | Sources | Strength |
|---|---|---|
| GLM t for one coefficient, with nuisance covariates handled by Freedman–Lane: permute (or sign-flip) the residuals of the covariates-only model | Freedman & Lane 1983; Anderson & Robinson 2001; Winkler et al. 2014 (NeuroImage 92), whose PALM implements it | consensus in neuroimaging |
| Welch's t for two groups without covariates | Delacre, Lakens & Leys 2017; Ruxton 2006 | consensus |
| Sign flipping for paired differences, assuming symmetric errors | Winkler et al. 2014; Nichols & Holmes 2002 | consensus |
| Permutation p = (b + 1) / (m + 1), never zero | Phipson & Smyth 2010 | consensus |
| Exact p by enumerating every sign flip when 2^n ≤ `n_permutations` | follows from the above | exact |
| Family-wise error by the null distribution of the maximum statistic over all features | Nichols & Holmes 2002; Westfall & Young 1993 | consensus |
| FDR: Benjamini–Hochberg (independent or positively dependent tests), or Benjamini–Yekutieli (`fdr="by"`, any dependence) | Benjamini & Hochberg 1995; Benjamini & Yekutieli 2001 | consensus |
| Network-based statistic: connected components of suprathreshold edges, size counted in edges, p from the null distribution of the largest component | Zalesky, Fornito & Bullmore 2010 | consensus |

**Tails.** `tail="both"` uses |t|; `"greater"` and `"less"` test one direction. For NBS, one direction per test is the usual practice (Zalesky 2010; the NBS toolbox), so pass `tail="greater"` or `"less"` together with `nbs_threshold`.

**NBS across modes.** Each mode's edges form a separate network (`m{k}`). Components never cross networks. The maximum is taken over all networks, so FWER covers every mode tested.

**Ties.** A rearranged statistic within a relative 10⁻¹² of the observed one counts as at least as large. Rearrangements that are equal in exact arithmetic (flipping every sign in a two-sided test gives exactly −t) must not be split by rounding, which differs between builds. The numpy reference enumeration uses the same rule.

**Reproducibility.** The same `seed` gives the same p-values on any machine and any number of threads, because each rearrangement draws from its own random stream (ChaCha8).

**Equal variances.** The GLM assumes equal residual variance across groups. With unequal group sizes and variances, permutation p-values for group differences can be inexact. Use `welch=True` when there are no covariates; Winkler's variance-group v statistic, the general fix, is planned.

## Evidence

| Check | Result |
|---|---|
| GLM t and two-sided p vs statsmodels OLS (with a covariate) | within 1e-10 and 1e-9 |
| Welch t, df and p vs `scipy.stats.ttest_ind(equal_var=False)` | within 1e-12, 1e-10 and 1e-9 |
| Incomplete beta and t tails vs scipy | within 1e-12 |
| BH and BY vs statsmodels `multipletests` | within 1e-14 |
| Exact sign-flip p and FWER vs a numpy enumeration of all 256 flips | equal |
| NBS components vs bctpy `get_components` | equal |
| Permutation p vs parametric p on Gaussian data (4000 permutations) | within Monte Carlo error |
| A planted 3-node subnetwork | found by NBS, with p < 0.01 |
| Design matrices | t equals a GLM on a hand-built matrix, in Rust and against NumPy least squares in Python |

## Planned

- General contrasts and F tests.
- Winkler's v statistic for unequal variances with covariates.
- Restricted exchangeability (blocks) for several rows per subject.
