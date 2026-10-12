# Statistics

Group-level tests on a feature table ([features](features.md)), with permutation inference and correction for testing many features at once.

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

- **Test statistic.** Each design is a general linear model, and the statistic is the t of the
  coefficient of interest. Nuisance covariates are handled by the Freedman–Lane procedure: the
  residuals of the covariates-only model are permuted (or sign-flipped) and added back to its fit
  (Freedman & Lane, 1983; Anderson & Robinson, 2001; Winkler et al., 2014). With `welch=True` and
  no covariates, two groups are compared with Welch's t, which does not assume equal variances
  (Delacre, Lakens & Leys, 2017).
- **Paired designs** flip the signs of the within-subject differences, assuming they are
  symmetric around zero under the null (Nichols & Holmes, 2002).
- **p-values** are (b + 1) / (m + 1), where b of m rearrangements reach the observed statistic, so
  they are never zero (Phipson & Smyth, 2010). When every sign flip fits in `n_permutations`
  (2ⁿ of them for n subjects), all are enumerated and the p-value is exact.
- **Family-wise error** uses the null distribution of the largest statistic over all features
  (Nichols & Holmes, 2002; Westfall & Young, 1993).
- **False discovery rate:** Benjamini–Hochberg by default, for independent or positively
  dependent tests, or Benjamini–Yekutieli (`fdr="by"`) for any dependence (Benjamini & Hochberg,
  1995; Benjamini & Yekutieli, 2001).
- **Network-based statistic:** edges above `nbs_threshold` form connected components, each sized
  by its number of edges, and each component's p comes from the null distribution of the largest
  component (Zalesky, Fornito & Bullmore, 2010).

**Tails.** `tail="both"` uses |t|; `"greater"` and `"less"` test one direction. For NBS, test one direction at a time (Zalesky, Fornito & Bullmore, 2010): pass `tail="greater"` or `"less"` together with `nbs_threshold`.

**NBS across modes.** Each mode's edges form a separate network (`m{k}`). Components never cross networks. The maximum is taken over all networks, so FWER covers every mode tested.

**Ties.** A rearranged statistic within a relative 10⁻¹² of the observed one counts as at least as large. Rearrangements that are equal in exact arithmetic (flipping every sign in a two-sided test gives exactly −t) must not be split by rounding, which differs between builds.

**Reproducibility.** The same `seed` gives the same p-values on any machine and any number of threads, because each rearrangement draws from its own random stream (ChaCha8).

**Equal variances.** The GLM assumes equal residual variance across groups. With unequal group sizes and variances, permutation p-values for group differences can be inexact. Use `welch=True` when there are no covariates; Winkler's variance-group v statistic, the general fix, is planned.

## Planned

- General contrasts and F tests.
- Winkler's v statistic for unequal variances with covariates.
- Restricted exchangeability (blocks) for several rows per subject.

## References

- Anderson, M. J., & Robinson, J. (2001). Permutation tests for linear models. *Australian & New Zealand Journal of Statistics*, 43(1), 75–88. https://doi.org/10.1111/1467-842X.00156
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165–1188. https://doi.org/10.1214/aos/1013699998
- Delacre, M., Lakens, D., & Leys, C. (2017). Why psychologists should by default use Welch's t-test instead of Student's t-test. *International Review of Social Psychology*, 30(1), 92–101. https://doi.org/10.5334/irsp.82
- Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported significance levels. *Journal of Business & Economic Statistics*, 1(4), 292–298.
- Nichols, T. E., & Holmes, A. P. (2002). Nonparametric permutation tests for functional neuroimaging: A primer with examples. *Human Brain Mapping*, 15(1), 1–25. https://doi.org/10.1002/hbm.1058
- Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39. https://doi.org/10.2202/1544-6115.1585
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing*. Wiley.
- Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014). Permutation inference for the general linear model. *NeuroImage*, 92, 381–397. https://doi.org/10.1016/j.neuroimage.2014.01.060
- Zalesky, A., Fornito, A., & Bullmore, E. T. (2010). Network-based statistic: Identifying differences in brain networks. *NeuroImage*, 53(4), 1197–1207. https://doi.org/10.1016/j.neuroimage.2010.06.041
