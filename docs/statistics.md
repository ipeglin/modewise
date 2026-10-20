# Statistics

Group-level tests on a feature table ([features](features.md)), with permutation inference and correction for testing many features at once.

## Designs

| Function | Tests | Model |
|---|---|---|
| `test_groups(table, column, (a, b), covariates=[...])` | group `a` minus group `b` (t) | GLM: intercept, a 0/1 dummy for `a`, centred covariates |
| `test_groups(table, column, (a, b, c, ...))` | all groups equal (F) | GLM: intercept, a dummy for every group but the last, centred covariates |
| `test_groups(..., welch=True)` | the same, each group with its own variance | v (two groups) or G (more) |
| `test_paired(table, (a, b), covariates=[...])` | condition `a` minus `b` within each recording | one-sample GLM on the differences, by sign flipping |
| `test_correlation(table, score, covariates=[...])` | the slope of a numeric column | GLM: intercept, the score, centred covariates |
| `test_glm(table, design, contrast)` | any contrast of your own design matrix | the matrix as given (no intercept or centring added) |

- **Covariates:** numeric columns are centred. Text columns (such as sex) are dummy-coded against their first level, then centred.
- **Missing values:** rows missing any value the design needs are dropped, and `report.dropped` counts them.
- **One row per subject:** the designs above need exactly one row per subject, because rows must be exchangeable under the null. Average runs, or filter, first, or use `test_glm` with `blocks`. A table built from epochs has one row per condition, so `test_groups` and `test_correlation` need `condition=`.

## Contrasts and design matrices

`test_glm` takes a design matrix with one row per table row, and a contrast with one weight per
column. A 1-D contrast `c` tests cβ = 0 with t. A 2-D contrast `C` tests every row at once,
Cβ = 0, with F (Winkler et al., 2014). For example, three group columns (one-hot, no intercept)
with `[[1, -1, 0], [0, 1, -1]]` test that all three means are equal.

```python
import numpy as np
import modewise

rng = np.random.default_rng(0)
site = ["a", "b", "c"] * 6
table = modewise.FeatureTable(
    rng.standard_normal((18, 4)), ["f0", "f1", "f2", "f3"],
    [f"sub-{i:02d}" for i in range(18)], subjects=[f"{i:02d}" for i in range(18)],
    metadata={"site": site},
)
cells = np.array([[s == level for level in "abc"] for s in site], float)
report = modewise.test_glm(table, cells, [[1, -1, 0], [0, 1, -1]], regressors=["a", "b", "c"])
print(report.statistic, report.rank, report.df[0])  # F 2 15.0
```

`report.statistic` names the statistic in `report.t`: t, F, or, with variance groups, v and G.
F has no direction, so it needs `tail="both"`. `exchange="signflip"` flips signs instead of
permuting, for one-sample designs such as within-subject differences.

The nuisance space under the null is every combination Mb with Cb = 0, so the Freedman–Lane
residuals do not depend on how the design is parametrised.

## Exchangeability blocks

`blocks` names a metadata column (or `"subject"`) whose rows depend on each other, such as
several runs of one subject. Rows are then exchanged only within their block, or, with
`whole_blocks=True`, blocks are exchanged as units (Winkler et al., 2015):

- **Within blocks:** for an effect that varies within each block, such as condition within a
  subject. Each block's rows are permuted among themselves. With a column per subject in the
  design, permuting two conditions within each subject gives exactly the paired sign-flip test.
- **Whole blocks:** for an effect that is constant within each block, such as a patient group
  with two runs per subject. Blocks move together and keep their order inside, so they must be the
  same size. With `exchange="signflip"`, each block's signs flip together, and sizes may differ.

Without `blocks`, `test_glm` needs one row per subject, like the other designs. Parametric
p-values still treat rows as independent, so with blocks only the permutation p-values account
for the dependence.

## Unequal variances

With `welch=True` (`test_groups`) or `variance_groups=` (`test_glm`), each group has its own
error variance. The statistic is then G (Winkler et al., 2014):

G = (Cβ)ᵀ (C (MᵀWM)⁻¹ Cᵀ)⁻¹ (Cβ) / (Λ s)

β is the least-squares fit and s the number of contrast rows. W is diagonal: each
observation's weight is its group's residual degrees of freedom (the sum of the residual-forming
matrix's diagonal over the group) divided by the group's residual sum of squares, that is, one
over the group's variance (Horn et al., 1975). Λ = 1 + 2(s − 1)/(s(s + 2)) · Σ_g (1 − w_g/Σw)² /
r_g, where w_g is group g's total weight and r_g its residual degrees of freedom (Welch, 1951).
For one row, v = sign(cβ)·√G is the Aspin–Welch v. With no covariates, v is Welch's t and G is
Welch's ANOVA F. Parametric p-values use F(s, ν) with ν = s(s + 2) / (3 Σ_g (1 − w_g/Σw)² / r_g),
which for one row is the Welch–Satterthwaite degrees of freedom (Welch, 1947).

Permutation under unequal variances is not exact, but v and G keep the permutation p-values
close to nominal where t and F do not (Winkler et al., 2014).

## Inference

- **Test statistic.** Each design is a general linear model, and the statistic is the t (or F)
  of the contrast of interest. Nuisance covariates are handled by the Freedman–Lane procedure: the
  residuals of the covariates-only model are permuted (or sign-flipped) and added back to its fit
  (Freedman & Lane, 1983; Anderson & Robinson, 2001; Winkler et al., 2014). With `welch=True`,
  groups are compared with v or G, which do not assume equal variances; without covariates, two
  groups give Welch's t (Delacre, Lakens & Leys, 2017).
- **Paired designs** flip the signs of the within-subject differences, assuming they are
  symmetric around zero under the null (Nichols & Holmes, 2002).
- **p-values** are (b + 1) / (m + 1), where b of m rearrangements reach the observed statistic, so
  they are never zero (Phipson & Smyth, 2010). When every rearrangement fits in `n_permutations`
  (2ⁿ sign flips or n! permutations for n rows, fewer within blocks), all are enumerated and the
  p-value is exact.
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

**Equal variances.** The GLM assumes equal residual variance across groups. With unequal group sizes and variances, permutation p-values for group differences can be inexact. Use `welch=True` or `variance_groups=` ([unequal variances](#unequal-variances)).

**Other packages.** FSL PALM (Winkler et al., 2014) gives the same t, F, v and G and the same degrees of freedom for F and G. It differs in three details:

- **v degrees of freedom:** PALM gives v one third of the Welch–Satterthwaite value that scipy, statsmodels' Welch ANOVA and PALM's own G with one row use. modewise follows Welch, so its parametric p for v differs from PALM's. Permutation p-values are not affected.
- **Ties:** PALM counts the mirrored rearrangement, which reaches the observed |t| exactly in exact arithmetic, as smaller when rounding makes it so. modewise counts it as a tie.
- **Which permutations:** PALM enumerates only permutations that change the tested regressors. modewise permutes all rows. The two give the same p-values when the nuisance is only an intercept (or block means), and different but equally valid null distributions with other nuisance regressors.

**Not supported yet.** Multi-level blocks (blocks within blocks, such as twins within families; Winkler et al., 2015) and variance groups that follow from blocks automatically.

## References

- Anderson, M. J., & Robinson, J. (2001). Permutation tests for linear models. *Australian & New Zealand Journal of Statistics*, 43(1), 75–88. https://doi.org/10.1111/1467-842X.00156
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society B*, 57(1), 289–300.
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165–1188. https://doi.org/10.1214/aos/1013699998
- Delacre, M., Lakens, D., & Leys, C. (2017). Why psychologists should by default use Welch's t-test instead of Student's t-test. *International Review of Social Psychology*, 30(1), 92–101. https://doi.org/10.5334/irsp.82
- Freedman, D., & Lane, D. (1983). A nonstochastic interpretation of reported significance levels. *Journal of Business & Economic Statistics*, 1(4), 292–298.
- Horn, S. D., Horn, R. A., & Duncan, D. B. (1975). Estimating heteroscedastic variances in linear models. *Journal of the American Statistical Association*, 70(350), 380–385. https://doi.org/10.1080/01621459.1975.10479877
- Nichols, T. E., & Holmes, A. P. (2002). Nonparametric permutation tests for functional neuroimaging: A primer with examples. *Human Brain Mapping*, 15(1), 1–25. https://doi.org/10.1002/hbm.1058
- Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39. https://doi.org/10.2202/1544-6115.1585
- Welch, B. L. (1947). The generalization of 'Student's' problem when several different population variances are involved. *Biometrika*, 34(1–2), 28–35. https://doi.org/10.1093/biomet/34.1-2.28
- Welch, B. L. (1951). On the comparison of several mean values: An alternative approach. *Biometrika*, 38(3–4), 330–336. https://doi.org/10.1093/biomet/38.3-4.330
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing*. Wiley.
- Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014). Permutation inference for the general linear model. *NeuroImage*, 92, 381–397. https://doi.org/10.1016/j.neuroimage.2014.01.060
- Winkler, A. M., Webster, M. A., Vidaurre, D., Nichols, T. E., & Smith, S. M. (2015). Multi-level block permutation. *NeuroImage*, 123, 253–268. https://doi.org/10.1016/j.neuroimage.2015.05.092
- Zalesky, A., Fornito, A., & Bullmore, E. T. (2010). Network-based statistic: Identifying differences in brain networks. *NeuroImage*, 53(4), 1197–1207. https://doi.org/10.1016/j.neuroimage.2010.06.041
