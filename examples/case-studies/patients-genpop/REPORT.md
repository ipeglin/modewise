# Patients versus the general population: report

<!-- stamp: modewise 0.2.0, commit 370fa74, 2026-10-14, 241 subjects -->

241 subjects of TCP (ds005237), first restAP run: 149 patients and 92 from the general population. Network-based statistic on Fisher-z edges, covariates age, sex, Site, mean_fd, 5000 permutations. Median mean framewise displacement: 0.052 mm in patients, 0.043 mm in the general population. Method: `pipeline.toml` and `analysis.py`.

| band | direction | t threshold | components | largest component (p) |
|---|---|---|---|---|
| full band | Patient < GenPop | 3.0 | 6 | 887 edges (0.001) |
| full band | Patient < GenPop | 3.5 | 9 | 207 edges (0.002) |
| full band | Patient > GenPop | 3.0 | 5 | 427 edges (0.017) |
| full band | Patient > GenPop | 3.5 | 12 | 77 edges (0.021) |
| D6, about 0.010–0.020 Hz | Patient < GenPop | 3.0 | 12 | 529 edges (0.003) |
| D6, about 0.010–0.020 Hz | Patient < GenPop | 3.5 | 16 | 72 edges (0.015) |
| D6, about 0.010–0.020 Hz | Patient > GenPop | 3.0 | 6 | 229 edges (0.073) |
| D6, about 0.010–0.020 Hz | Patient > GenPop | 3.5 | 12 | 13 edges (0.269) |
| D5, about 0.020–0.039 Hz | Patient < GenPop | 3.0 | 11 | 452 edges (0.012) |
| D5, about 0.020–0.039 Hz | Patient < GenPop | 3.5 | 20 | 86 edges (0.015) |
| D5, about 0.020–0.039 Hz | Patient > GenPop | 3.0 | 13 | 271 edges (0.053) |
| D5, about 0.020–0.039 Hz | Patient > GenPop | 3.5 | 8 | 44 edges (0.054) |
| D4, about 0.039–0.078 Hz | Patient < GenPop | 3.0 | 6 | 472 edges (0.007) |
| D4, about 0.039–0.078 Hz | Patient < GenPop | 3.5 | 21 | 73 edges (0.018) |
| D4, about 0.039–0.078 Hz | Patient > GenPop | 3.0 | 20 | 186 edges (0.129) |
| D4, about 0.039–0.078 Hz | Patient > GenPop | 3.5 | 20 | 10 edges (0.305) |

## Components with p < 0.05

- full band, Patient < GenPop, t > 3.0: p = 0.001; 887 edges, 323 regions; hubs aCAU-rh (28), 17networks_LH_DorsAttnB_PrCv (27), 17networks_RH_VisCent_ExStr_1 (27); network pairs DefaultA – VisCent (55), DefaultB – DorsAttnB (52), DefaultA – DorsAttnB (49)
- full band, Patient < GenPop, t > 3.5: p = 0.002; 207 edges, 146 regions; hubs 17networks_LH_DorsAttnB_PrCv (17), aCAU-rh (14), 17networks_LH_DefaultA_PFCd_3 (13); network pairs DefaultB – DorsAttnB (19), DefaultA – DorsAttnA (19), DefaultA – VisCent (17)
- full band, Patient > GenPop, t > 3.0: p = 0.017; 427 edges, 227 regions; hubs 17networks_RH_ContC_pCun_2 (19), 17networks_RH_VisCent_Striate (19), 17networks_RH_DefaultC_PHC_1 (17); network pairs SomMotB – VisCent (36), DorsAttnA – VisPeri (29), ContA – VisPeri (26)
- full band, Patient > GenPop, t > 3.5: p = 0.021; 77 edges, 62 regions; hubs 17networks_RH_LimbicA_TempOcc (8), 17networks_RH_DefaultC_PHC_1 (7), 17networks_LH_DorsAttnA_TempOcc_5 (7); network pairs DorsAttnA – VisPeri (11), ContA – LimbicA (8), ContA – DefaultC (6)
- D6, about 0.010–0.020 Hz, Patient < GenPop, t > 3.0: p = 0.003; 529 edges, 280 regions; hubs NAc-core-lh (72), aCAU-rh (25), 17networks_LH_TempPar_Temp_2 (17); network pairs SalVentAttnA – striatum (49), DefaultA – SalVentAttnA (31), ContA – striatum (27)
- D6, about 0.010–0.020 Hz, Patient < GenPop, t > 3.5: p = 0.015; 72 edges, 63 regions; hubs NAc-core-lh (43), NAc-shell-rh (6), aCAU-rh (6); network pairs SalVentAttnA – striatum (27), ContA – striatum (12), DorsAttnB – striatum (7)
- D5, about 0.020–0.039 Hz, Patient < GenPop, t > 3.0: p = 0.012; 452 edges, 244 regions; hubs 17networks_LH_DorsAttnA_TempOcc_5 (20), 17networks_LH_DefaultA_PCC_4 (19), 17networks_RH_DefaultC_PHC_1 (17); network pairs DefaultB – DorsAttnA (49), DorsAttnA – TempPar (27), DefaultA – DorsAttnA (27)
- D5, about 0.020–0.039 Hz, Patient < GenPop, t > 3.5: p = 0.015; 86 edges, 72 regions; hubs 17networks_RH_DefaultC_PHC_1 (8), 17networks_LH_DefaultA_PCC_4 (8), 17networks_RH_VisPeri_PHC (7); network pairs DefaultB – DorsAttnA (11), DefaultA – DorsAttnA (10), DorsAttnA – TempPar (9)
- D4, about 0.039–0.078 Hz, Patient < GenPop, t > 3.0: p = 0.007; 472 edges, 270 regions; hubs 17networks_LH_ContA_PFCl_4 (32), pHIP-lh (22), 17networks_LH_DefaultC_RSC (20); network pairs SomMotA – VisPeri (30), SalVentAttnA – medial_temporal_lobe (29), ContA – SomMotA (23)
- D4, about 0.039–0.078 Hz, Patient < GenPop, t > 3.5: p = 0.018; 73 edges, 60 regions; hubs 17networks_LH_ContA_PFCl_4 (11), 17networks_LH_DefaultC_RSC (8), 17networks_RH_SomMotB_ParOper_2 (7); network pairs ContA – SomMotA (15), SalVentAttnA – medial_temporal_lobe (8), SomMotB – medial_temporal_lobe (8)

## Notes

- Each component's p is family-wise over all edges of that band, for one direction and one threshold. Sixteen tests are reported (four bands, two directions, two thresholds), without correction across them, so treat a single component near p = 0.05 with care.
- The t threshold is a free choice of the network-based statistic; results at both thresholds are shown so the dependence is visible.
- Dropped edges, not finite in some recording: cerebellum_left~cerebellum_right. In the TCP parcellated files the two cerebellum channels hold identical signals, so their correlation is 1 and its Fisher z infinite.
- Head motion changes connectivity and often differs between groups. The files are denoised with global signal regression, and each subject's mean framewise displacement, from the dataset's motion_FD table, is a covariate.
- Patients in the TCP have a range of diagnoses; a difference between the groups need not hold for any single diagnosis.
