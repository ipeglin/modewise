# Faces against shapes: report

<!-- stamp: modewise 0.2.0, commit 31facd1, 2026-10-14, 219 subjects -->

219 subjects of TCP (ds005237), hammerAP run. Method: `pipeline.toml` and `analysis.py`.

Not analysed: 7 runs that have parcellated data but no BIDS events or sidecar on OpenNeuro, so their blocks cannot be defined.

## Faces − shapes per region (paired sign-flip test, FWER over 434 regions)

- faces > shapes: 84 regions
- shapes > faces: 250 regions

| region | t (face − shape) | FWER p |
|---|---|---|
| **lAMY-lh** | +12.7 | 0.0001 |
| **mAMY-lh** | +22.6 | 0.0001 |
| **lAMY-rh** | +17.3 | 0.0001 |
| **mAMY-rh** | +25.9 | 0.0001 |

Strongest faces > shapes:

| region | t | FWER p |
|---|---|---|
| 17networks_RH_VisCent_ExStr_9 | +41.6 | 0.0001 |
| 17networks_RH_VisCent_ExStr_10 | +38.9 | 0.0001 |
| 17networks_LH_VisCent_ExStr_10 | +37.3 | 0.0001 |
| 17networks_LH_VisCent_ExStr_9 | +36.1 | 0.0001 |
| 17networks_RH_VisCent_ExStr_5 | +35.4 | 0.0001 |
| 17networks_LH_VisCent_ExStr_5 | +33.2 | 0.0001 |
| 17networks_RH_VisCent_ExStr_7 | +31.6 | 0.0001 |
| 17networks_RH_VisCent_ExStr_4 | +30.5 | 0.0001 |
| 17networks_RH_VisCent_ExStr_6 | +28.9 | 0.0001 |
| 17networks_RH_DorsAttnA_TempOcc_5 | +28.8 | 0.0001 |

## Faces − shapes from a task GLM (AR(1) noise, paired sign-flip test on betas, FWER over 434 regions)

- faces > shapes: 84 regions
- shapes > faces: 259 regions
- Across regions, the t values of the GLM and of the block averages correlate at r = 0.998.

| region | t (face − shape) | FWER p |
|---|---|---|
| **lAMY-lh** | +12.6 | 0.0001 |
| **mAMY-lh** | +23.1 | 0.0001 |
| **lAMY-rh** | +17.8 | 0.0001 |
| **mAMY-rh** | +26.5 | 0.0001 |

## Exploratory: the amygdala response (mean of four amygdala parcels, face − shape)

- Mean response: +23.0 (SD 13.6, in the TCP files' signal units), positive in 98% of subjects.
- Against SHAPS anhedonia (age and sex as covariates): t = -1.24, permutation p = 0.226, 203 subjects (16 without a score).
- Patients against the general population (age and sex as covariates): t = +0.62, permutation p = 0.542, 219 subjects.

## Notes

- The TCP series had the global signal regressed out. That shifts every region's signal by the brain-wide mean, so regions with no face response tend to show shapes > faces; the count above reflects this, not a shape response.
- Blocks are averages of the BOLD signal 5 s after each block's events (a fixed lag). The GLM instead models every trial with the canonical haemodynamic response, so the two methods share only the data.
