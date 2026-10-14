# Nuisance regression

How modewise removes confound signals (head motion, physiological noise, scanner drift) from every channel before decomposition or connectivity.

## How it works

The `regress` step fits each channel by least squares on one design matrix and keeps the residual:

- **Constant:** always included, so the result has mean 0.
- **Linear trend:** with `detrend = true` (the default).
- **Cosine drifts:** with `high_pass_hz`, the discrete cosine regressors cos(πk(t + ½)/n) for k = 1 … ⌊2n·high_pass_hz/fs⌋. Every frequency up to the cutoff is removed. This is SPM's high-pass filter (Friston et al., 2007) and nilearn's `filter="cosine"` (Abraham et al., 2014).
- **Confounds:** the columns of the `regressors` input, in their order.

All of these go into **one fit**. A filter applied before or after a regression can bring back noise that the other step removed. Fitting the trend, the drifts and the confounds together keeps every removed part out (Lindquist et al., 2019). The result equals nilearn's `signal.clean` with `detrend=True`, `filter="cosine"` and the same confounds.

The fit uses an orthonormal basis of the design's columns, built by Gram–Schmidt with column pivoting. Columns that add nothing to the others are dropped, so a duplicate or all-zero confound does not break the fit. When the design leaves no samples to spare, the step fails.

## Confounds as an input

fMRIPrep writes each run's confounds to `*_desc-confounds_timeseries.tsv`, one column per regressor (Esteban et al., 2019). Read the columns you want with a `timeseries_tsv` input ([datasets](datasets.md)):

```toml
[[input]]
name = "bold"
pattern = "derivatives/sub-{subject}/func/sub-{subject}_task-{task}_bold.h5"
format = "hdf5"
dataset = "timeseries"
sidecar = "sub-{subject}/func/sub-{subject}_task-{task}_bold.json"

[[input]]
name = "confounds"
pattern = "derivatives/sub-{subject}/func/sub-{subject}_task-{task}_desc-confounds_timeseries.tsv"
format = "timeseries_tsv"
columns = ["trans_*", "rot_*", "white_matter", "csf"]
sidecar = "sub-{subject}/func/sub-{subject}_task-{task}_bold.json"
```

- **Columns** are names or globs (`*` for any run of characters, `?` for one). In fMRIPrep output, `trans_*` and `rot_*` match the six motion parameters with their derivatives and squares: the 24-parameter model (Friston et al., 1996).
- **Other common columns:** `white_matter` and `csf` (mean signals), `a_comp_cor_*` (anatomical CompCor; Behzadi et al., 2007) and `global_signal`. Ciric et al. (2017) compare these strategies on motion benchmarks.
- **Missing first values:** derivatives and framewise displacement have no value at the first volume, written `n/a`. modewise uses the second volume's value there, as nilearn's `load_confounds` does. `n/a` in any other row is an error.

The pipeline then cleans the signal before anything else:

```toml
[pipeline]
name = "clean_fc"
inputs = { bold = "series", confounds = "series" }

[[step]]
id = "clean"
op = "regress"
input = { data = "bold", regressors = "confounds" }
params = { detrend = true, high_pass_hz = 0.008 }

[[step]]
id = "fc"
op = "functional_connectivity"
input = "clean"
params = { measure = "pearson" }
```

SPM's default cutoff is a 128 s period, about 0.0078 Hz. Choose the cutoff below the slowest band you study ([epochs](epochs.md) lists the slow bands).

## Head motion as a covariate

Head motion changes functional connectivity, and regression does not remove all of its effect (Power et al., 2012; Satterthwaite et al., 2012). Groups often move differently, so group comparisons usually add each recording's mean framewise displacement (FD) as a covariate. FD sums the absolute changes of the six motion parameters between volumes, with rotations turned into millimetres on a 50 mm sphere (Power et al., 2012). fMRIPrep writes it as `framewise_displacement`.

```{code-block} python
import pandas as pd
import modewise

dataset = modewise.Dataset.from_file("dataset.toml")
table = modewise.Results("results").feature_table(
    "clean_fc", "fc", "edges", participants=dataset.participants
)

def mean_fd(recording):
    e = dataset.entities(recording)
    path = (f"derivatives/sub-{e['subject']}/func/"
            f"sub-{e['subject']}_task-{e['task']}_desc-confounds_timeseries.tsv")
    return pd.read_csv(path, sep="\t")["framewise_displacement"].mean()  # skips the first n/a

table = modewise.FeatureTable(
    table.values, table.feature_names, table.recordings,
    subjects=table.subjects, conditions=table.conditions,
    metadata={**table.metadata, "mean_fd": [str(mean_fd(r)) for r in table.recordings]},
)
report = modewise.test_groups(table, "group", ("patient", "control"), covariates=["age", "mean_fd"])
```

## References

- Abraham, A., Pedregosa, F., Eickenberg, M., et al. (2014). Machine learning for neuroimaging with scikit-learn. *Frontiers in Neuroinformatics*, 8, 14. https://doi.org/10.3389/fninf.2014.00014
- Behzadi, Y., Restom, K., Liau, J., & Liu, T. T. (2007). A component based noise correction method (CompCor) for BOLD and perfusion based fMRI. *NeuroImage*, 37(1), 90–101. https://doi.org/10.1016/j.neuroimage.2007.04.042
- Ciric, R., Wolf, D. H., Power, J. D., et al. (2017). Benchmarking of participant-level confound regression strategies for the control of motion artifact in studies of functional connectivity. *NeuroImage*, 154, 174–187. https://doi.org/10.1016/j.neuroimage.2017.03.020
- Esteban, O., Markiewicz, C. J., Blair, R. W., et al. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. *Nature Methods*, 16(1), 111–116. https://doi.org/10.1038/s41592-018-0235-4
- Friston, K. J., Ashburner, J. T., Kiebel, S. J., Nichols, T. E., & Penny, W. D. (Eds.). (2007). *Statistical Parametric Mapping: The Analysis of Functional Brain Images*. Academic Press.
- Friston, K. J., Williams, S., Howard, R., Frackowiak, R. S. J., & Turner, R. (1996). Movement-related effects in fMRI time-series. *Magnetic Resonance in Medicine*, 35(3), 346–355. https://doi.org/10.1002/mrm.1910350312
- Lindquist, M. A., Geuter, S., Wager, T. D., & Caffo, B. S. (2019). Modular preprocessing pipelines can reintroduce artifacts into fMRI data. *Human Brain Mapping*, 40(8), 2358–2376. https://doi.org/10.1002/hbm.24528
- Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L., & Petersen, S. E. (2012). Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. *NeuroImage*, 59(3), 2142–2154. https://doi.org/10.1016/j.neuroimage.2011.10.018
- Satterthwaite, T. D., Wolf, D. H., Loughead, J., et al. (2012). Impact of in-scanner head motion on multiple measures of functional connectivity: relevance for studies of neurodevelopment in youth. *NeuroImage*, 60(1), 623–632. https://doi.org/10.1016/j.neuroimage.2011.12.063
