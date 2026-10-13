# Patients versus the general population: report

<!-- stamp: modewise 0.2.0, commit 1706882, 2026-10-12, 241 subjects -->

241 subjects of TCP (ds005237), first restAP run: 149 patients and 92 from the general population. Network-based statistic on Fisher-z edges, covariates age, sex, Site, 5000 permutations. Method: `pipeline.toml` and `analysis.py`.

| band | direction | t threshold | components | largest component (p) |
|---|---|---|---|---|
| full band | Patient < GenPop | 3.0 | 6 | 691 edges (0.002) |
| full band | Patient < GenPop | 3.5 | 8 | 149 edges (0.003) |
| full band | Patient > GenPop | 3.0 | 6 | 431 edges (0.017) |
| full band | Patient > GenPop | 3.5 | 10 | 70 edges (0.026) |
| D6, about 0.010–0.020 Hz | Patient < GenPop | 3.0 | 9 | 491 edges (0.004) |
| D6, about 0.010–0.020 Hz | Patient < GenPop | 3.5 | 18 | 46 edges (0.044) |
| D6, about 0.010–0.020 Hz | Patient > GenPop | 3.0 | 15 | 216 edges (0.088) |
| D6, about 0.010–0.020 Hz | Patient > GenPop | 3.5 | 15 | 10 edges (0.350) |
| D5, about 0.020–0.039 Hz | Patient < GenPop | 3.0 | 17 | 360 edges (0.023) |
| D5, about 0.020–0.039 Hz | Patient < GenPop | 3.5 | 21 | 38 edges (0.067) |
| D5, about 0.020–0.039 Hz | Patient > GenPop | 3.0 | 14 | 254 edges (0.060) |
| D5, about 0.020–0.039 Hz | Patient > GenPop | 3.5 | 10 | 37 edges (0.067) |
| D4, about 0.039–0.078 Hz | Patient < GenPop | 3.0 | 10 | 400 edges (0.014) |
| D4, about 0.039–0.078 Hz | Patient < GenPop | 3.5 | 22 | 65 edges (0.022) |
| D4, about 0.039–0.078 Hz | Patient > GenPop | 3.0 | 11 | 219 edges (0.085) |
| D4, about 0.039–0.078 Hz | Patient > GenPop | 3.5 | 17 | 9 edges (0.341) |

## Components with p < 0.05

- full band, Patient < GenPop, t > 3.0: p = 0.002; 691 edges, 302 regions; hubs 17networks_LH_ContA_PFCl_5 (22), 17networks_LH_DorsAttnB_PrCv (21), aCAU-rh (21); network pairs DefaultB – DorsAttnB (41), DefaultA – VisCent (41), DefaultB – SomMotB (34)
- full band, Patient < GenPop, t > 3.5: p = 0.003; 149 edges, 122 regions; hubs 17networks_LH_DorsAttnB_PrCv (11), 17networks_LH_SalVentAttnB_Ins_2 (10), 17networks_LH_VisCent_ExStr_1 (8); network pairs DefaultA – VisCent (11), DefaultB – DorsAttnB (10), ContA – striatum (10)
- full band, Patient > GenPop, t > 3.0: p = 0.017; 431 edges, 233 regions; hubs 17networks_RH_VisCent_Striate (22), THA-DA-lh (21), 17networks_RH_LimbicA_TempOcc (18); network pairs SomMotB – VisCent (32), DorsAttnA – VisPeri (26), ContA – LimbicA (24)
- full band, Patient > GenPop, t > 3.5: p = 0.026; 70 edges, 65 regions; hubs 17networks_RH_LimbicA_TempOcc (7), 17networks_RH_ContC_PCC_1 (6), 17networks_RH_VisCent_Striate (6); network pairs ContA – LimbicA (9), DorsAttnA – thalamus (7), DorsAttnA – VisPeri (5)
- D6, about 0.010–0.020 Hz, Patient < GenPop, t > 3.0: p = 0.004; 491 edges, 271 regions; hubs NAc-core-lh (63), 17networks_LH_TempPar_Temp_2 (17), 17networks_RH_SomMotA_PostC (16); network pairs SalVentAttnA – striatum (43), DefaultA – SalVentAttnA (28), ContA – striatum (23)
- D6, about 0.010–0.020 Hz, Patient < GenPop, t > 3.5: p = 0.044; 46 edges, 44 regions; hubs NAc-core-lh (31), NAc-shell-rh (5), 17networks_RH_SalVentAttnA_IPL_2 (4); network pairs SalVentAttnA – striatum (20), DorsAttnB – striatum (5), SomMotB – striatum (4)
- D5, about 0.020–0.039 Hz, Patient < GenPop, t > 3.0: p = 0.023; 360 edges, 222 regions; hubs 17networks_LH_DorsAttnA_TempOcc_5 (16), 17networks_RH_VisPeri_PHC (14), 17networks_LH_DefaultB_PFCv_2 (14); network pairs DefaultB – DorsAttnA (32), DorsAttnA – TempPar (20), ContA – SalVentAttnB (17)
- D4, about 0.039–0.078 Hz, Patient < GenPop, t > 3.0: p = 0.014; 400 edges, 233 regions; hubs 17networks_LH_ContA_PFCl_4 (31), mAMY-lh (27), pHIP-lh (19); network pairs SomMotA – VisPeri (36), ContA – SomMotB (26), SalVentAttnA – medial_temporal_lobe (26)
- D4, about 0.039–0.078 Hz, Patient < GenPop, t > 3.5: p = 0.022; 65 edges, 52 regions; hubs mAMY-lh (17), 17networks_LH_ContA_PFCl_4 (11), 17networks_RH_SomMotB_ParOper_2 (8); network pairs ContA – SomMotA (15), SomMotB – medial_temporal_lobe (15), SalVentAttnA – medial_temporal_lobe (9)

## Notes

- Each component's p is family-wise over all edges of that band, for one direction and one threshold. Sixteen tests are reported (four bands, two directions, two thresholds), without correction across them, so treat a single component near p = 0.05 with care.
- The t threshold is a free choice of the network-based statistic; results at both thresholds are shown so the dependence is visible.
- Dropped edges, not finite in some recording: cerebellum_left~cerebellum_right. In the TCP parcellated files the two cerebellum channels hold identical signals, so their correlation is 1 and its Fisher z infinite.
- Head motion often differs between patients and controls and can change connectivity. These files are denoised with global signal regression, but motion is not a covariate here.
- Patients in the TCP have a range of diagnoses; a difference between the groups need not hold for any single diagnosis.
