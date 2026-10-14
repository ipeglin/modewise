# Case studies

Analyses of real data with modewise: three reproduce published findings, and one is exploratory. They need data that is
too large to download on every build, so they run by hand; each report is stamped with the
modewise version and commit that made it, and a test flags reports made by an older version.

To rerun one:

```bash
python examples/case-studies/fetch_tcp.py restAP restPA hammerAP   # OpenNeuro ds005237, checked against the bucket's MD5s
python examples/case-studies/frequency-amplitude/analysis.py
python examples/case-studies/faces-amygdala/analysis.py
python examples/case-studies/fingerprinting/analysis.py        # about 2.2 GB of results
python examples/case-studies/patients-genpop/analysis.py       # about 1.4 GB of results
```

## Frequency-specific amplitude (Zuo et al. 2010)

Resting BOLD fluctuates mostly below 0.1 Hz. Zuo et al. split the classic 0.01–0.08 Hz range into
slow-5 (0.01–0.027 Hz) and slow-4 (0.027–0.073 Hz) and found the balance differs by region:
slow-5 dominates cortex, most of all the default-mode network, while the basal ganglia carry
relatively more slow-4. This case study repeats the comparison on the first resting-state run of
241 subjects of the Transdiagnostic Connectome Project, with `spectrum`, `band_amplitude` and
`test_paired`, and checks it with an independent method (the MODWT MRA). The files are in
`examples/case-studies/frequency-amplitude/`.

What it shows: the regional pattern replicates, and both methods agree across regions
(r = 0.99). One difference: Zuo et al. found slow-4 larger than slow-5 in the basal ganglia; here
the two are about level there, with slow-5 larger everywhere else.

```{include} ../examples/case-studies/frequency-amplitude/REPORT.md
:start-line: 4
:heading-offset: 1
```

## Faces against shapes in the amygdala (Hariri et al. 2002)

In the emotional face-matching task, people match faces with fearful or angry expressions, or
match shapes. Faces drive the amygdala more than shapes, one of the most replicated results in
affective neuroimaging, along with the ventral visual stream that processes faces. This case
study cuts each TCP subject's face-matching run into face and shape blocks with a 5 s
haemodynamic lag (`epoch`), averages each block (the `mean` feature), and tests faces against
shapes in every region with `test_paired`. It repeats the contrast with a [task GLM](glm.md),
which models every trial with the canonical haemodynamic response. Exploratory follow-ups ask whether the amygdala
response tracks anhedonia (SHAPS) or differs between patients and the general population, using a
`FeatureTable` built from the per-subject contrasts. The files are in
`examples/case-studies/faces-amygdala/`.

What it shows: all four amygdala parcels respond more to faces (t = 12.7 to 25.9, FWER p < 0.001),
and the strongest responses are in extrastriate visual cortex. The GLM gives the same picture: its
t values correlate with those of the block averages at r = 0.998 across regions. The exploratory
tests find no relation with anhedonia and no group difference.

```{include} ../examples/case-studies/faces-amygdala/REPORT.md
:start-line: 4
:heading-offset: 1
```

## Connectome fingerprinting (Finn et al. 2015)

Finn et al. showed that functional connectivity identifies a person: correlate the connectome
of one scan with every connectome from another session, and the best match is usually the same
person. This case study matches each TCP subject's first restAP run against everyone's first
restPA run (and the reverse), for the full band and for two MODWT octaves. The modes and band
signals are computed with `keep = false`, so only the connectivity matrices are stored. The files
are in `examples/case-studies/fingerprinting/`.

What it shows: about 80% of 241 subjects are identified in the full band, against a chance level
of 0.4% (the best of 1000 shuffles reaches 2.1%). The 0.039–0.078 Hz octave identifies more
subjects than the 0.010–0.020 Hz octave, which a 6.5-minute run samples with only a few cycles.

```{include} ../examples/case-studies/fingerprinting/REPORT.md
:start-line: 4
:heading-offset: 1
```

## Patients against the general population (exploratory)

The Transdiagnostic Connectome Project recruited patients with a range of psychiatric diagnoses
and a comparison group from the general population. This case study compares their resting-state
connectivity edge by edge, for the full band and three MODWT octaves, with the network-based
statistic (`test_groups` with `nbs_threshold`), adjusting for age, sex, site and head motion (mean
framewise displacement; [regression](regression.md)). It tests no hypothesis; it shows how a
whole-connectome group comparison runs and reports what comes out. The files are in
`examples/case-studies/patients-genpop/`.

What it shows: every band, at both thresholds, has a network of edges that is weaker in patients
(full band at t > 3.0: 887 edges, p = 0.001). In the slowest octave (0.010–0.020 Hz) it centres on
the left nucleus accumbens and its links to the salience network. A smaller network is stronger in
patients in the full band only. These are exploratory results: the patients span several
diagnoses, and sixteen tests are reported without correction across them.

```{include} ../examples/case-studies/patients-genpop/REPORT.md
:start-line: 4
:heading-offset: 1
```
