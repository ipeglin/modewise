# Epochs and decompositions

How modewise cuts signals into epochs (task blocks, trials) when frequency components are involved, and why. This is the design basis for roadmap step S3c. It records where the original thesis pipeline fell short, so those choices are not repeated.

## The problem in one picture

Imagine judging how fast a wave moves by looking through a window that is shorter than one wave. You see a rising or falling slope, never a full rise and fall. Two windows on the same wave can show opposite slopes. If you call that slope "the wave's strength in this window", you are mostly measuring where the window happens to fall, not the wave.

A frequency component with period `P` behaves the same way inside an epoch shorter than `P`. This applies to every modality. In EEG the periods are milliseconds and the epochs seconds, so it bites mainly for slow rhythms (delta, below 4 Hz) in short trials. In fMRI the periods are tens of seconds and blocks are short, so it bites almost everywhere below about 0.05 Hz.

## The problem

An epoch of length `T` seconds cannot show a component whose period is longer than `T`. Inside the epoch, such a component looks like a trend or an offset, not an oscillation. Its "power in this epoch" then depends mostly on where in its cycle the epoch happens to fall. That is noise, not a condition effect.

**Example from the thesis data** (TCP face task, TR = 0.8 s): face blocks last about 18 s (23–24 volumes).

| Band ([Zuo et al. 2010](https://researchwith.njit.edu/en/publications/the-oscillating-brain-complex-and-reliable/)) | Range | Period | Fits in an 18 s block? |
|---|---|---|---|
| slow-5 | 0.010–0.027 Hz | 37–100 s | no |
| slow-4 | 0.027–0.073 Hz | 14–37 s | only above ~0.055 Hz |
| slow-3 | 0.073–0.198 Hz | 5–14 s | yes |

## What the thesis pipeline did

The thesis is one source among those weighed here. These points are where its choices differ from what the other sources agree on.

1. **Per-block decomposition (in the thesis text).** The thesis decomposes each block on its own and lists this as a limitation (`DiscussionMethodologicalLimitations.tex`, "Block-Wise Rather Than Full-Run Analysis"). The legacy code was later changed to decompose the full run and then slice it. So the thesis results and the latest legacy outputs may come from different procedures.
2. **No haemodynamic lag.** Blocks were cut at the recorded stimulus times (`floor(onset·fs)` to `ceil(end·fs)`) with no shift. The BOLD response peaks several seconds after the stimulus, and that latency varies across subjects and regions ([Handwerker et al. 2004](https://fim.nimh.nih.gov/publications/handwerker_et_al_2004_0)). Each window therefore started too early and cut off the end of the response.
3. **No check that a component fits in the block.** Modes with periods longer than the block were summarised per block like any other mode.

## Evidence

- **Window length.** Sliding-window estimates show spurious fluctuations when the window is shorter than the largest wavelength present, even for signals with a fixed relationship. Rule of thumb: window `w ≥ 1/f_min`. ([Leonardi & Van De Ville 2015, NeuroImage 104:430–436](https://miplab.epfl.ch/pub/leonardi1402p.pdf), quoted from the paper.)
- **Transform first, then cut.** In EEG and MEG time-frequency work, common practice is to transform the longer continuous data and epoch afterwards. This avoids edge loss of about half a window at each epoch boundary. The lowest frequency an epoch can carry is roughly cycles divided by epoch length. This is practitioner guidance from tool mailing lists, not a formal standard: [EEGLAB](https://sccn.ucsd.edu/pipermail/eeglablist/2017/012589.html), [EEGLAB](https://sccn.ucsd.edu/pipermail/eeglablist/2012/004539.html), [FieldTrip](https://mailman.science.ru.nl/pipermail/fieldtrip/2017-August/011705.html), [MNE](https://mne.discourse.group/t/edge-effects-and-trial-based-epoching/8790).
- **Mode decomposition on continuous fMRI.** Mode decompositions (EMD, VMD, MEMD) give narrow-band components suitable for instantaneous phase and amplitude without choosing a band-pass first. This was shown on resting-state fMRI ([Honari & Lindquist 2022, arXiv:2203.13955](https://arxiv.org/abs/2203.13955)).
- **The haemodynamic lag varies.** Response latency and shape differ widely across subjects and regions ([Handwerker et al. 2004, NeuroImage 21](https://fim.nimh.nih.gov/publications/handwerker_et_al_2004_0)). Lag differences of several seconds are common (["Resting-state fMRI signals contain spectral signatures of local hemodynamic response timing", 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10506795/)). A fixed shift is common practice, for example 6 s in [this decoding study](https://arxiv.org/pdf/1603.03162), but it is an assumption, and sensitivity to it should be checked.
- **Cross-validation.** For decoding, prefer repeated random splits to leave-one-out, and use nested cross-validation when tuning ([Varoquaux et al. 2017, NeuroImage 145](https://arxiv.org/abs/1606.05201)). One further point is our own reasoning, not a cited finding: epochs cut from one full-run decomposition share information through the decomposition and through slow autocorrelation, so splits should keep a whole run (or subject) on one side.

## How strong is the evidence?

| Claim | Independent sources | Strength |
|---|---|---|
| Windows shorter than a component's period give unreliable estimates | Leonardi & Van De Ville 2015 (analytic); Hindriks et al. 2016 (simulation and data); wavelet edge effects in Torrence & Compo 1998; EEG/MEG practitioner guidance | consensus |
| Transform the continuous data, then epoch | EEG/MEG practitioner guidance (EEGLAB, FieldTrip, MNE lists); Torrence & Compo edge effects; the thesis's own limitations section | consensus, mostly practice rather than formal studies |
| Haemodynamic latency varies; a lag must be chosen | Handwerker et al. 2004; the 2023 PMC study; a decoding study using 6 s | consensus that it varies; no consensus on one value, hence no default |
| `min_cycles = 1` as the threshold | Leonardi & Van De Ville 2015 only; EEG habit suggests more (Cohen 2019 discusses cycles) | single source, so it is a provisional default and open to review |
| Sample timing: sample `i` at `i / fs`, onsets from the first stored sample | BIDS specification; MNE's epoching follows the same time base | consensus (standard) |
| Epochs from one decomposition leak information across CV splits | our reasoning; Varoquaux et al. 2017 on CV pitfalls in general | weak; needs a direct source or our own simulation |

## Rules for modewise

| # | Rule | How it is enforced |
|---|---|---|
| E1 | Decompose or transform the full continuous run, then epoch. | `epoch` outputs a new kind, `Epochs`. Decomposers accept only `Series`, so a pipeline that decomposes epochs fails at planning. |
| E2 | Mark every component whose period does not fit in the epoch. A component is resolvable when `centre_frequency × epoch_length ≥ min_cycles`. | Each epoch records its length. Each mode or frequency bin gets a `resolvable` flag. Epoch summaries skip unresolvable components unless the user opts in. The threshold is a parameter and is written to the manifest. |
| E3 | The haemodynamic lag is explicit. | `epoch` takes `lag_s`, the shift applied to event times. It has no hidden default for fMRI, and it is recorded in the manifest. Cached pipelines make a lag sweep (for example 0, 2, 4, 6 s) cheap, so sensitivity is easy to report. |
| E4 | No padding at the edges of the run. | An epoch that runs past the start or end of the recording is dropped and reported, never padded or mirrored. |
| E5 | Epochs are slices of full-run results. | Values inside an epoch equal the matching samples of the full-run output exactly, with no recomputation. Tested. |
| E6 | Short-window spectra must follow `w ≥ 1/f`. | Spectral estimates inside an epoch (STFT) flag bins with frequencies below `min_cycles / w` as unresolvable. |
| E7 | The ML handover carries the split rule. | Features from epochs record the run and subject they came from, and that they share one decomposition. `brainkyb-ml` must split by run or subject, not by epoch. |

## Conventions (implemented in S3c)

| Convention | Source |
|---|---|
| Sample `i` is at `i / fs` s. Event onsets are seconds from the first stored sample. | [BIDS events](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html) |
| An epoch holds the samples whose times fall in `[start, end)`, so back-to-back blocks never share a sample. | our choice. MNE includes "the closest or matching samples" with `tmax` inclusive; we use half-open windows so blocks tile without overlap. |
| `window = "duration"` uses the event's own duration (blocks). `window = { tmin_s, tmax_s }` uses a fixed window around each onset (trials). | MNE `tmin`/`tmax` ([MNE Epochs](https://mne.tools/stable/generated/mne.Epochs.html)) |
| Epochs that reach outside the recording, or hold fewer than 2 samples, are dropped and listed with a reason. | MNE drops epochs outside the data and logs why (same page) |
| `lag_s` is required for every modality (EEG uses 0). | rule E3 |
| Trials can be merged into blocks with `merge_gap_s`: consecutive same-label events whose gap is at most that many seconds. | a data transformation, not a method claim |
| `events.tsv` rows with `n/a` onset or duration are rejected, not guessed. | BIDS allows `n/a`, but no safe value can be invented |

## Tests S3c must include

- **Resolvability.** A synthetic run with 0.02 Hz and 0.2 Hz tones, decomposed in full and cut into 18 s epochs: the 0.02 Hz mode is flagged unresolvable and the 0.2 Hz mode is not.
- **Slices match.** Epoch values equal the full-run output at `[floor((onset+lag)·fs), ceil((end+lag)·fs))`.
- **Lag.** With `lag_s = 4` at 1.25 Hz (exactly 5 samples), every window starts 5 samples later. A lag that is not a whole number of samples moves each window to the first sample at or after `onset + lag`. Changing the lag reruns only `epoch` and the steps after it.
- **Edges.** An epoch that runs past the end of the recording is dropped and listed in the result. Nothing is padded.
- **Types.** A pipeline that feeds `Epochs` into `mvmd` is rejected at planning time.
- **Thesis comparison.** The acceptance test reruns the thesis analysis with the full run, then the lag, then the resolvability flags, and reports how the results change. Each difference is explained against the sources above. Neither the thesis nor the new pipeline is assumed right by default.

## Decisions (2026-10-10)

1. **`min_cycles` defaults to 1**, the Leonardi & Van De Ville minimum. It can be raised per pipeline. The check is not tied to modality, so it applies to EEG, MEG and fNIRS as well as fMRI. Revisit the default after reading the sources below.
2. **The haemodynamic lag is required for `epoch` on fMRI data, with no default.** The docs suggest 4–6 s and a lag sweep.
3. **A GLM-style step** (block regressors convolved with a haemodynamic response, fitted to mode amplitudes) comes as its own slice after S3c.

## EEG and other time-frequency methods

- **STFT:** a bin is resolvable when its frequency × window length ≥ `min_cycles` (rule E6).
- **CWT (planned):** a wavelet's support grows with its period, so near the edges of a recording or epoch, low frequencies are contaminated. The standard map of where this happens is the *cone of influence*. The CWT step must report it, and epoch summaries must treat values inside it as unresolvable.
- **Wavelet width.** Choosing a wavelet "number of cycles" trades time precision for frequency precision. Cohen (2019) recommends describing it by its full width at half maximum instead.

## Reading list

In suggested order, from the core idea to the details:

1. **Leonardi & Van De Ville (2015).** "On spurious and real fluctuations of dynamic functional connectivity during rest." *NeuroImage* 104:430–436. [PDF](https://miplab.epfl.ch/pub/leonardi1402p.pdf). Short and readable; shows, with simple sine waves, exactly how windows shorter than a period create fake fluctuations. Source of `w ≥ 1/f_min`.
2. **Hindriks et al. (2016).** "Can sliding-window correlations reveal dynamic functional connectivity in resting-state fMRI?" *NeuroImage* 127:242–256. [Open copy](https://repositori.upf.edu/items/8e94bb51-dcdb-4c4f-82ff-d402b27ce6ff/full). Shows how little a short window can establish statistically, and why testing against a null model matters.
3. **Torrence & Compo (1998).** "A practical guide to wavelet analysis." *Bulletin of the American Meteorological Society* 79(1):61–78. [Record](https://experts.colorado.edu/display/pubid_107382). The classic introduction to wavelet power, significance and the cone of influence, so edge effects become concrete.
4. **Cohen (2019).** "A better way to define and describe Morlet wavelets for time-frequency analysis." *NeuroImage* 199:81–86. [Preprint](https://www.biorxiv.org/content/10.1101/397182.full.pdf). The EEG/MEG view: what "number of cycles" really controls, and how to choose it.
5. **Keil et al. (2014).** "Committee report: publication guidelines and recommendations for studies using electroencephalography and magnetoencephalography." *Psychophysiology* 51(1):1–21. [Record](https://experts.illinois.edu/en/publications/committee-report-publication-guidelines-and-recommendations-for-s/). What EEG/MEG papers are expected to report about their processing, including the order of steps such as filtering, epoching and transforms.
6. **Handwerker et al. (2004).** "Variation of BOLD hemodynamic responses across subjects and brain regions and their effects on statistical analyses." *NeuroImage* 21. [Record](https://fim.nimh.nih.gov/publications/handwerker_et_al_2004_0). Why one fixed haemodynamic lag is an assumption, and why modewise asks for it explicitly.
