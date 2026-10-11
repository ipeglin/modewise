# Connectome fingerprinting: report

<!-- stamp: modewise 0.2.0, commit 80177ec, 2026-10-11, 241 subjects -->

241 subjects of TCP (ds005237): first restAP run against first restPA run. Chance is 1/241 = 0.4%. Method: `pipeline.toml` and `analysis.py`.

| band | restAP → restPA | restPA → restAP | best chance-level accuracy in 1000 shuffles |
|---|---|---|---|
| full band | 80.9% | 78.8% | 2.1% |
| D6, about 0.010–0.020 Hz | 31.5% | 35.3% | 2.1% |
| D4, about 0.039–0.078 Hz | 66.8% | 60.6% | 2.1% |

## Notes

- Identification correlates Fisher-z edge vectors (434 regions, 93,961 edges); a target is identified when its best match is the same subject, as in Finn et al. (2015).
- Finn et al. reported 92.9% and 94.4% between two 14-minute resting runs recorded on different days, with 268 regions and no global signal regression. Here the runs are 6.5 minutes from one session with opposite phase encoding (AP and PA), with 434 regions and global signal regression, so the numbers are not directly comparable.
- A 6.5-minute run holds only about 4 to 8 cycles at 0.010–0.020 Hz, so connectivity in that band is estimated from little data; its lower accuracy says as much about estimation as about how individual the band is.
