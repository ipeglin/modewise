# Frequency-specific amplitude: report

<!-- stamp: modewise 0.2.0, commit 80177ec, 2026-10-11, 241 subjects -->

241 subjects of TCP (ds005237), first restAP run. Method: `pipeline.toml` and `analysis.py`.

## Slow-4 vs slow-5 ALFF per region (paired sign-flip test, FWER over 434 regions)

- slow-5 > slow-4: 428 regions
- slow-4 > slow-5: 0 regions
- no FWER-significant difference: 6 regions

| region | t (slow4 − slow5) | FWER p |
|---|---|---|
| pGP-lh | -1.0 | 1.000 |
| NAc-core-lh | -1.3 | 1.000 |
| NAc-core-rh | -1.4 | 1.000 |
| aGP-lh | -2.6 | 0.447 |
| pGP-rh | -3.1 | 0.166 |
| aGP-rh | -3.1 | 0.163 |
| THA-DP-lh | -3.7 | 0.028 |
| lAMY-rh | -3.8 | 0.020 |

## log(slow-4 / slow-5 ALFF) by structure (mean ± SE over subjects)

| structure | regions | ALFF ratio |
|---|---|---|
| cortex | 400 | -0.329 ± 0.008 |
| striatum | 16 | -0.071 ± 0.008 |
| thalamus | 8 | -0.101 ± 0.011 |
| medial_temporal_lobe | 8 | -0.144 ± 0.011 |
| cerebellum | 2 | -0.387 ± 0.020 |

Striatum − cortex: +0.258 ± 0.008, positive in 99% of subjects; paired t = 33.1, permutation p = 0.0001.

## Second method: MODWT MRA (sym4), log(SD of D4 / SD of D6)

Striatum − cortex: +0.351 ± 0.009, positive in 99% of subjects; paired t = 37.2, permutation p = 0.0001.
Across regions, the wavelet ratio correlates with the ALFF ratio at r = 0.99.

## Cortical networks, most slow-5 first

DefaultA -0.38 · DefaultB -0.37 · DorsAttnA -0.37 · VisCent -0.37 · ContA -0.35 · VisPeri -0.35 · DorsAttnB -0.35 · ContB -0.35 · SalVentAttnB -0.35 · TempPar -0.34 · SalVentAttnA -0.32 · ContC -0.31 · SomMotA -0.29 · DefaultC -0.29 · LimbicB -0.27 · SomMotB -0.26 · LimbicA -0.22
