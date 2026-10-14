# Task GLM

How modewise estimates each channel's response to the conditions of a task with a general linear model (GLM), the standard first-level analysis in fMRI.

## How it works

The `glm` step models each channel's signal as a sum of regressors plus noise, and estimates one weight (beta) per regressor by least squares (Friston et al., 1995).

- **Task regressors:** one per condition (event label, in sorted order). Each event is a box from its onset to onset + duration. The boxes are convolved with the canonical haemodynamic response and read at the sample times i / fs. An event with duration 0 is a unit-area impulse.
- **Canonical response:** SPM's double gamma. A gamma density with shape 6 (peak at 5 s) minus one sixth of a gamma density with shape 16 (peak at 15 s), both with a 1 s scale, cut at 32 s and scaled to unit area (Friston et al., 1998; Friston et al., 2007). modewise convolves in closed form, as a difference of gamma distribution functions, so there is no time grid. A long block therefore rises to exactly 1.
- **Nuisance regressors:** the same ones as [regression](regression.md): the mean, a linear trend (`detrend = true`, the default), cosine drifts below `high_pass_hz`, and the columns of the optional `confounds` input. They are fitted together with the task regressors, so motion that correlates with the task does not bias the betas.
- **Noise:** fMRI noise is autocorrelated. Treating samples as independent understates the variance of the betas and inflates t (Purdon & Weisskoff, 1998). With `noise = "ar1"` (the default), modewise fits by ordinary least squares, estimates each channel's lag-1 autocorrelation ρ from the residuals (Yule–Walker), then whitens the signal and the design and fits again. The whitening scales the first sample by √(1 − ρ²) and replaces every later sample x_t with x_t − ρ·x_{t−1}. This is the exact generalised least squares fit for AR(1) noise. `noise = "ols"` skips the whitening.

The step gives two outputs, each condition × channel: `betas` and `t` (beta over its standard error, with n minus the design's rank as degrees of freedom).

Packages differ in the details:

- **First sample:** nilearn leaves the first sample unwhitened (Abraham et al., 2014), statsmodels' `GLSAR` drops it, and modewise scales it.
- **Grouping:** nilearn rounds ρ to steps of 0.01 to share fits between voxels; modewise fits each channel with its own ρ.
- **Event impulses:** SPM gives an event of duration 0 unit area when every event in the design has duration 0. nilearn gives it the area of one step of its fine time grid.
- **Sample time:** SPM samples regressors in the middle of each volume by default. nilearn and modewise sample them at the start, matching onsets in seconds from the first stored sample, as in BIDS ([epochs](epochs.md)).
- **High-pass cutoff:** SPM's default is a 128 s period (about 0.0078 Hz), FSL's 100 s and nilearn's 0.01 Hz. modewise sets none by default, as in `regress`.
- **Remaining bias:** the lag-1 estimate from least-squares residuals is biased towards 0 (Worsley et al., 2002), so AR(1) t values stay slightly inflated. In a synthetic test with ρ = 0.5 and 200 samples, 6.7% of null channels pass p < 0.05, against 25% without whitening. SPM uses a global ReML estimate and FSL a smoothed, tapered autocorrelation estimate instead (Woolrich et al., 2001).

## A pipeline

```toml
[pipeline]
name = "faces"
inputs = { bold = "series", trials = "events", confounds = "series" }

[[step]]
id = "glm"
op = "glm"
input = { data = "bold", events = "trials", confounds = "confounds" }
params = { high_pass_hz = 0.0078 }
```

Leave out `confounds` (in `inputs` and in the step's `input`) when the data are already cleaned. The events come from an `events_tsv` input and the confounds from a `timeseries_tsv` input ([datasets](datasets.md)).

## Group analysis

Each subject's betas go into a feature table with `features="coefficients"`: one row per condition and subject, the channels as features ([features](features.md)). A paired test across subjects of two conditions' betas is the summary-statistics approach to a random-effects group analysis (Holmes & Friston, 1998):

```{code-block} python
import modewise

dataset = modewise.Dataset.from_file("dataset.toml")
betas = modewise.Results("results").feature_table(
    "faces", "glm.betas", "coefficients", participants=dataset.participants
)
report = modewise.test_paired(betas, ("face", "shape"), n_permutations=10000, seed=1)
```

The faces case study ([case studies](case-studies.md)) runs this contrast both from block averages and from the GLM.

## References

- Abraham, A., Pedregosa, F., Eickenberg, M., et al. (2014). Machine learning for neuroimaging with scikit-learn. *Frontiers in Neuroinformatics*, 8, 14. https://doi.org/10.3389/fninf.2014.00014
- Friston, K. J., Holmes, A. P., Worsley, K. J., Poline, J.-P., Frith, C. D., & Frackowiak, R. S. J. (1995). Statistical parametric maps in functional imaging: a general linear approach. *Human Brain Mapping*, 2(4), 189–210. https://doi.org/10.1002/hbm.460020402
- Friston, K. J., Fletcher, P., Josephs, O., Holmes, A., Rugg, M. D., & Turner, R. (1998). Event-related fMRI: characterizing differential responses. *NeuroImage*, 7(1), 30–40. https://doi.org/10.1006/nimg.1997.0306
- Friston, K. J., Ashburner, J. T., Kiebel, S. J., Nichols, T. E., & Penny, W. D. (Eds.). (2007). *Statistical Parametric Mapping: The Analysis of Functional Brain Images*. Academic Press.
- Holmes, A. P., & Friston, K. J. (1998). Generalisability, random effects and population inference. *NeuroImage*, 7(4), S754.
- Purdon, P. L., & Weisskoff, R. M. (1998). Effect of temporal autocorrelation due to physiological noise and stimulus paradigm on voxel-level false-positive rates in fMRI. *Human Brain Mapping*, 6(4), 239–249.
- Woolrich, M. W., Ripley, B. D., Brady, M., & Smith, S. M. (2001). Temporal autocorrelation in univariate linear modeling of FMRI data. *NeuroImage*, 14(6), 1370–1386. https://doi.org/10.1006/nimg.2001.0931
- Worsley, K. J., Liao, C. H., Aston, J., et al. (2002). A general statistical analysis for fMRI data. *NeuroImage*, 15(1), 1–15. https://doi.org/10.1006/nimg.2001.0933
