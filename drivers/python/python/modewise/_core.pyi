from os import PathLike
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: str

class Series:
    """Signals laid out as channels × samples, with sampling rate and channel labels."""

    data: NDArray[np.float64]
    sampling_rate: float
    labels: list[str]

    def __init__(self, data: ArrayLike, sampling_rate: float, labels: Sequence[str] | None = None) -> None:
        """Raises ValueError on NaN values, a non-positive sampling rate or bad labels."""

class Decomposition:
    """Result of a mode decomposition."""

    modes: NDArray[np.float64]
    """Modes as mode × channel × sample, sorted by centre frequency."""
    centre_frequencies_hz: NDArray[np.float64]
    channel_centre_frequencies_hz: NDArray[np.float64] | None
    """Mode × channel, for methods that decompose each channel on its own (NaN: no such mode)."""
    aligned_across_channels: bool
    """True when mode k means the same oscillation in every channel."""
    labels: list[str]
    sampling_rate: float
    method: str
    n_modes: int
    def __init__(
        self,
        modes: ArrayLike,
        centre_frequencies_hz: Sequence[float] | None,
        sampling_rate: float,
        labels: Sequence[str] | None = None,
        method: str = "external",
        *,
        channel_centre_frequencies_hz: ArrayLike | None = None,
    ) -> None:
        """Modes computed elsewhere (mode × channel × sample), so modewise can work on them. Give
        channel_centre_frequencies_hz (mode × channel) instead of centre_frequencies_hz for modes
        found per channel; each mode's frequency is then the channel median."""


class TimeFrequencyMap:
    """Time-frequency power of a signal."""

    power: NDArray[np.float64]
    """Power as channel × frequency × time."""
    frequencies_hz: NDArray[np.float64]
    times_s: NDArray[np.float64]
    labels: list[str]
    method: str
    sampling_rate: float | None
    """Samples per second when there is one time column per sample (CWT, Hilbert spectrum); None for STFT frames."""
    reliable: NDArray[np.bool_] | None
    """Frequency × time: False where edge effects make values unreliable (cone of influence)."""

def emd(
    signal: ArrayLike,
    sampling_rate: float,
    *,
    stop: Literal["rilling", "s_number", "fixed"] = "rilling",
    thresholds: tuple[float, float, float] | None = None,
    count: int = 2,
    siftings: int = 10,
    max_imfs: int | None = None,
    labels: Sequence[str] | None = None,
) -> Decomposition:
    """Empirical mode decomposition (Huang et al., 1998) of each channel on its own (docs/emd.md).

    stop: "rilling" (thresholds = (sd, sd2, tol), default (0.05, 0.5, 0.05)), "s_number"
    (count consecutive siftings) or "fixed" (siftings per IMF). Modes are aligned by IMF index
    from the top; mode 0 is the residual. Channels with fewer IMFs get zero modes with NaN in
    channel_centre_frequencies_hz.
    """

def eemd(
    signal: ArrayLike,
    sampling_rate: float,
    *,
    trials: int = 100,
    noise_sd: float = 0.2,
    seed: int = 0,
    stop: Literal["rilling", "s_number", "fixed"] = "fixed",
    thresholds: tuple[float, float, float] | None = None,
    count: int = 2,
    siftings: int = 10,
    max_imfs: int | None = None,
    labels: Sequence[str] | None = None,
) -> Decomposition:
    """Ensemble EMD (Wu and Huang, 2009) of each channel on its own (docs/emd.md).

    The mean of EMDs of the signal plus white noise with SD noise_sd·SD(signal), at most
    ⌊log₂ n⌋ − 1 IMFs unless max_imfs is set. Modes sum to the signal plus the mean noise.
    """

def ceemdan(
    signal: ArrayLike,
    sampling_rate: float,
    *,
    trials: int = 100,
    epsilon: float = 0.2,
    seed: int = 0,
    stop: Literal["rilling", "s_number", "fixed"] = "rilling",
    thresholds: tuple[float, float, float] | None = None,
    count: int = 2,
    siftings: int = 10,
    max_imfs: int | None = None,
    labels: Sequence[str] | None = None,
) -> Decomposition:
    """CEEMDAN in its improved form (Colominas et al., 2014) of each channel on its own
    (docs/emd.md). Modes sum to the signal exactly."""

def memd(
    signal: ArrayLike,
    sampling_rate: float,
    *,
    num_directions: int = 64,
    stop: Literal["rilling", "s_number", "fixed"] = "rilling",
    thresholds: tuple[float, float, float] | None = None,
    count: int = 2,
    siftings: int = 10,
    max_imfs: int | None = None,
    labels: Sequence[str] | None = None,
) -> Decomposition:
    """Multivariate EMD (Rehman and Mandic, 2010) on at least 3 channels (docs/emd.md).

    Envelopes come from projections along num_directions Hammersley directions; use at least
    twice the number of channels. Rilling thresholds default to (0.075, 0.75, 0.075), as memd.m.
    """

def modwt_mra(
    signal: ArrayLike,
    sampling_rate: float,
    *,
    wavelet: str = "sym4",
    levels: int | None = None,
    boundary: Literal["periodic", "reflection"] = "periodic",
    labels: Sequence[str] | None = None,
) -> Decomposition:
    """MODWT multiresolution analysis (Percival and Walden, 2000) of each channel (docs/wavelets.md).

    Mode 0 is the smooth S_J and the last mode the finest detail D_1; detail D_j covers about
    [fs/2^(j+1), fs/2^j]. Modes sum to the signal. levels defaults to the deepest level whose
    filter fits the series. Wavelets: haar, db1-db10, sym2-sym10, coif1-coif5.
    """

def operations() -> list[dict[str, Any]]:
    """Every built-in pipeline operation: name, version, summary, inputs and outputs (port names
    and data kinds) and params (a JSON Schema with types, descriptions and defaults), as
    `modewise ops --json`. docs/reference/operations.md is generated from the same data."""

def mvmd(
    signal: ArrayLike,
    sampling_rate: float,
    num_modes: int,
    alpha: float,
    *,
    init: Literal["zero", "linear", "exponential", "uniform"] = "zero",
    tolerance: float = 1e-3,
    tau: float = 1e-2,
    max_iterations: int = 500,
    labels: Sequence[str] | None = None,
) -> Decomposition:
    """Multivariate Variational Mode Decomposition (Rehman and Aftab, 2019).

    signal: channels × samples. alpha: bandwidth penalty, larger gives narrower modes.
    tau: dual ascent step, 0 lets noise stay out of the modes.
    Raises ValueError on invalid input or parameters.
    """

def vmd(
    signal: ArrayLike,
    sampling_rate: float,
    num_modes: int,
    alpha: float,
    *,
    init: Literal["zero", "linear", "exponential", "uniform"] = "zero",
    tolerance: float = 1e-3,
    tau: float = 1e-2,
    max_iterations: int = 500,
    labels: Sequence[str] | None = None,
) -> Decomposition:
    """Variational Mode Decomposition (Dragomiretskiy and Zosso, 2014) of each channel on its own.

    Modes are not aligned across channels: channel_centre_frequencies_hz gives each channel's
    frequencies, and centre_frequencies_hz their median. init="uniform" matches vmdpy's init=1.
    """

def na_mvmd(
    signal: ArrayLike,
    sampling_rate: float,
    num_modes: int,
    alpha: float,
    *,
    noise_channels: int = 1,
    noise_sd: float = 0.8,
    seed: int = 0,
    init: Literal["zero", "linear", "exponential", "uniform"] = "zero",
    tolerance: float = 1e-3,
    tau: float = 1e-2,
    max_iterations: int = 500,
    labels: Sequence[str] | None = None,
) -> Decomposition:
    """Noise-assisted MVMD (Zisou, Apostolidis and Hadjileontiadis, 2021).

    Each channel is scaled to unit SD, noise_channels white-noise channels with SD noise_sd are
    appended, MVMD runs on all of them, and the noise channels are dropped.
    """

def vlmd(
    signal: ArrayLike,
    sampling_rate: float,
    num_modes: int,
    *,
    num_latents: int | None = None,
    alpha: float = 100.0,
    reg_lambda: float = 0.01,
    reg_rho: float = 1.0,
    tau: float = 0.9,
    tolerance: float = 1e-3,
    max_iterations: int = 1000,
    labels: Sequence[str] | None = None,
) -> tuple[Decomposition, Decomposition, Loadings]:
    """Variational Latent Mode Decomposition (Morante and Rehman, 2025); see docs/vlmd.md.

    signal: channels × samples. Returns (modes, latent_modes, loadings): channel modes, the modes of
    num_latents latent components (default: one per channel), and their sparse latent × channel loadings.
    The channel modes sum to the latent fit, not exactly to the signal.
    alpha: bandwidth penalty. reg_lambda: sparsity of the loadings. reg_rho: ADMM penalty.
    Raises ValueError on invalid input or parameters.
    """

def spectrum(
    signal: ArrayLike,
    sampling_rate: float,
    *,
    segment_s: float | None = None,
    overlap_s: float | None = None,
    window: Literal["boxcar", "hann"] = "boxcar",
    detrend: Literal["none", "constant", "linear"] = "constant",
    labels: Sequence[str] | None = None,
) -> TimeFrequencyMap:
    """Power spectral density per channel as one frame, as scipy.signal: a periodogram by
    default, or Welch's method with segment_s (and overlap_s, half a segment by default)."""

def stft(
    signal: ArrayLike,
    sampling_rate: float,
    window_len: int,
    hop: int,
    *,
    labels: Sequence[str] | None = None,
) -> TimeFrequencyMap:
    """Short-time Fourier transform with a periodic Hann window. Power is ``|X|²``.

    signal: channels × samples. window_len and hop are in samples.
    Raises ValueError on invalid input or parameters.
    """

class MemoryStore:
    """Keeps step results in memory so reruns skip unchanged steps."""

    def __init__(self) -> None: ...
    def __len__(self) -> int: ...

class PipelineRun:
    """Outcome of one pipeline run."""

    outputs: list[str]
    """Output names, as `step.output`."""
    cached: dict[str, bool]
    """Which steps were reused from the store instead of computed."""
    manifest: dict[str, Any]
    """Provenance record: operations, parameters, data keys and timings."""

    def output(self, reference: str) -> Series | Decomposition | TimeFrequencyMap | Events | Epochs | Instantaneous:
        """A step output, by `step.output`, or by `step` when it has one output."""

class Pipeline:
    """A checked pipeline of named steps. Invalid specs raise ValueError."""

    name: str
    inputs: list[str]
    steps: list[str]
    """Step ids in the order they run."""

    @staticmethod
    def from_toml(text: str) -> Pipeline: ...
    @staticmethod
    def from_file(path: str | PathLike[str]) -> Pipeline: ...
    @staticmethod
    def from_dict(spec: Mapping[str, Any]) -> Pipeline:
        """Same layout as the TOML file: {"pipeline": {...}, "step": [{...}, ...]}."""

    def run_dataset(
        self,
        dataset: Dataset,
        out: str | PathLike[str],
        *,
        shard: tuple[int, int] | None = None,
        workers: int = 0,
        progress: bool = True,
    ) -> BatchRun:
        """Runs over every recording of `dataset`, saving results to the folder `out`.

        Unchanged steps from earlier runs into the same folder are reused. One
        failing recording does not stop the others; see `BatchRun.failed`.
        `shard=(i, n)` takes every n-th recording starting at i (1-based), for
        splitting a cohort across Slurm array jobs. `workers=0` uses all cores.
        """

    def run(
        self,
        inputs: Mapping[str, Series | Events | ChannelTable | ArrayLike],
        sampling_rate: float | None = None,
        *,
        labels: Mapping[str, Sequence[str]] | None = None,
        store: MemoryStore | None = None,
    ) -> PipelineRun:
        """Runs the pipeline on Series or channels × samples arrays.

        Plain arrays need `sampling_rate`; a Series carries its own.

        Pass the same `store` to later runs to reuse unchanged steps.
        Step failures raise RuntimeError.
        """

def read_hdf5(
    path: str | PathLike[str],
    dataset: str,
    sampling_rate: float,
    *,
    layout: Literal["channels_by_time", "time_by_channels"] = "channels_by_time",
    labels: Sequence[str] | None = None,
) -> Series:
    """Reads one 2D numeric dataset from an HDF5 file. No HDF5 C library needed.

    Raises OSError if the file cannot be opened, ValueError for a bad dataset.
    """

class Dataset:
    """Recordings found on disk by file patterns, ready to load or run pipelines over."""

    recordings: list[str]
    inputs: list[str]

    @staticmethod
    def from_file(path: str | PathLike[str]) -> Dataset:
        """Reads a dataset spec. A relative `root` is resolved against the file's folder."""
    @staticmethod
    def from_toml(text: str, base: str | PathLike[str] | None = None) -> Dataset: ...
    def entities(self, recording: str) -> dict[str, str]: ...
    def load(self, recording: str) -> dict[str, Series | Events]:
        """Reads every input of one recording: Series for signals, Events for events files."""
    participants: Participants | None
    """From the spec's [participants] section."""
    def __len__(self) -> int: ...

class Results:
    """A results folder written by `Pipeline.run_dataset` or `modewise run`."""

    path: Path
    pipelines: list[str]

    def __init__(self, path: str | PathLike[str]) -> None: ...
    def recordings(self, pipeline: str) -> list[str]: ...
    def manifest(self, pipeline: str, recording: str) -> dict[str, Any]: ...
    def output(self, pipeline: str, recording: str, reference: str) -> Series | Decomposition | TimeFrequencyMap: ...
    def feature_table(
        self,
        pipeline: str,
        output: str,
        features: Literal["edges", "mode_power", "band_amplitude", "mean"] = "edges",
        *,
        resolvable_only: bool = True,
        participants: Participants | None = None,
        subject_entity: str = "subject",
        bands: Mapping[str, tuple[float, float]] | None = None,
        relative: bool = False,
    ) -> FeatureTable:
        """One table over every recording (docs/features.md): one row per recording, or per
        recording and condition for epoch outputs. edges: functional connectivity edges (Pearson
        as Fisher z). mode_power: each mode's variance per channel and its centre frequency. band_amplitude: from a spectrum, the mean amplitude
        (√power) of each channel in each named band [low, high) Hz, ALFF, or with relative=True
        its share of all frequencies above 0, fALFF; one row per recording and band. mean: from
        epochs of a series, each condition's mean signal per channel (block averages).
        resolvable_only skips epochs too short for a component."""
    def stack(self, pipeline: str, output: str) -> tuple[list[str], NDArray[np.float64]]:
        """One output across recordings as a single array, recording first (for deep learning)."""
    def export_hdf5(self, pipeline: str, to: str | PathLike[str]) -> list[Path]:
        """Writes one HDF5 file per recording (for MATLAB and other tools)."""

class BatchRun:
    """Outcome of running a pipeline over a dataset."""

    succeeded: list[str]
    failed: dict[str, str]
    """Recording id -> error message."""
    results: Results

    def output(self, recording: str, reference: str) -> Series | Decomposition | TimeFrequencyMap: ...

def cli(args: Sequence[str]) -> int:
    """Runs the `modewise` command with the given arguments (the first is the program name)."""

class Events:
    """Events with onsets and durations in seconds from the first sample (BIDS events.tsv)."""

    onsets_s: NDArray[np.float64]
    durations_s: NDArray[np.float64]
    labels: list[str]

    def __init__(self, onsets_s: Sequence[float], durations_s: Sequence[float], labels: Sequence[str]) -> None:
        """Raises ValueError on non-finite onsets, negative durations, empty labels or length mismatch."""
    def __len__(self) -> int: ...

class Epochs:
    """Epochs cut from a full-run series or decomposition (rules: docs/epochs.md).

    Each epoch covers samples start_sample[i] .. end_sample[i] (end exclusive) of `source`.
    """

    start_sample: NDArray[np.uint64]
    end_sample: NDArray[np.uint64]
    event_onsets_s: NDArray[np.float64]
    labels: list[str]
    durations_s: NDArray[np.float64]
    resolvable: NDArray[np.bool_] | None
    """Epochs × modes: True where the mode's period fits min_cycles times. None for series sources."""
    dropped: list[dict[str, Any]]
    """Events that produced no epoch: label, event_onset_s, reason."""
    lag_s: float
    min_cycles: float
    source: Series | Decomposition | Instantaneous | TimeFrequencyMap

    def data(self, i: int) -> NDArray[np.float64] | dict[str, NDArray[np.float64]]:
        """Epoch i: channels × samples (series), modes × channels × samples (decomposition),
        {"amplitude", "frequency_hz"} arrays (instantaneous), or channel × frequency × samples (time-frequency)."""
    def __len__(self) -> int: ...

def read_events_tsv(path: str | PathLike[str], *, label_column: str = "trial_type") -> Events:
    """Reads a BIDS events.tsv. Raises OSError if missing, ValueError on bad rows (including n/a onsets or durations)."""

class FunctionalConnectivity:
    """Functional connectivity (undirected, symmetric), window × component × channel × channel."""

    values: NDArray[np.float64]
    measure: str
    """ "pearson", "pearson_fisher_z" or "plv"."""
    labels: list[str]
    centre_frequencies_hz: NDArray[np.float64] | None
    """One per component when computed on a decomposition."""
    window_labels: list[str] | None
    window_onsets_s: NDArray[np.float64] | None
    resolvable: NDArray[np.bool_] | None
    """Window × component: True where the window holds enough cycles of the component."""

class ChannelTable:
    """What each channel is: name, region, hemisphere ("left", "right" or None) and open tags,
    so several region definitions (atlas regions, Yeo networks, EEG lobes) sit side by side."""

    names: list[str]
    regions: list[str | None]
    hemispheres: list[str | None]
    tags: list[dict[str, str]]
    @staticmethod
    def from_records(records: Sequence[Mapping[str, object]]) -> ChannelTable:
        """Dicts with "name" and optionally "region", "hemisphere" and "tags"."""
    def select(self, rules: Sequence[Mapping[str, object]]) -> list[str]:
        """Names of the channels matching any rule. A rule has any of "name", "region",
        "hemisphere" and "tags" (all must match); text matches exactly or with * wildcards.
        A hemisphere rule leaves out midline channels."""
    def __add__(self, other: ChannelTable) -> ChannelTable: ...
    def __len__(self) -> int: ...

class Participants:
    """Information about each subject, joined from participants.tsv and phenotype tables."""

    columns: list[str]
    subjects: list[str]
    def value(self, subject: str, column: str) -> str | None: ...
    def to_pandas(self) -> Any:
        """A pandas DataFrame indexed by subject (needs pandas)."""
    def __len__(self) -> int: ...

class FeatureTable:
    """Observations × named features with participant columns (docs/features.md)."""

    values: NDArray[np.float64]
    feature_names: list[str]
    recordings: list[str]
    subjects: list[str | None]
    conditions: list[str | None]
    metadata: dict[str, list[str | None]]
    def __init__(
        self,
        values: ArrayLike,
        feature_names: Sequence[str],
        recordings: Sequence[str],
        *,
        subjects: Sequence[str | None] | None = None,
        conditions: Sequence[str | None] | None = None,
        metadata: Mapping[str, Sequence[str | None]] | None = None,
        participants: Participants | None = None,
    ) -> None:
        """A table from your own values (observations × features), for example ratios or
        contrasts computed in numpy, so they can go to test_groups, test_paired or
        test_correlation. conditions labels rows for paired tests; metadata and participants add
        the columns designs use for groups and covariates."""
    def to_pandas(self) -> Any:
        """recording, subject, condition, participant columns, then features (needs pandas)."""
    def to_tsv(self, path: str | PathLike[str]) -> None:
        """Tab-separated, missing values as n/a, for MATLAB, R or spreadsheets."""

class StatsReport:
    """Per-feature results of a permutation test (docs/statistics.md)."""

    feature_names: list[str]
    t: NDArray[np.float64]
    df: NDArray[np.float64]
    p_parametric: NDArray[np.float64]
    """From Student's t (Welch–Satterthwaite df for Welch); for reference."""
    p_permutation: NDArray[np.float64]
    """(b + 1) / (m + 1) over m rearrangements; exact when every sign flip was enumerated."""
    p_fwer: NDArray[np.float64]
    """Family-wise error over all features, by the maximum statistic."""
    q_fdr: NDArray[np.float64]
    components: list[dict[str, Any]]
    """Network-based statistic: dicts with "network", "edges" (feature names) and "p"."""
    n_observations: int
    dropped: int
    n_rearrangements: int
    exhaustive: bool
    regressors: list[str]
    def to_pandas(self) -> Any: ...

class Loadings:
    """How strongly each latent component contributes to each channel (latent × channel)."""

    values: NDArray[np.float64]
    latent_labels: list[str]
    channel_labels: list[str]

class Instantaneous:
    """Instantaneous amplitude and frequency of modes, each mode × channel × sample."""

    amplitude: NDArray[np.float64]
    frequency_hz: NDArray[np.float64]
    centre_frequencies_hz: NDArray[np.float64]
    labels: list[str]
    sampling_rate: float

def hilbert(modes: Decomposition, *, padding: Literal["mirror", "none"] = "mirror") -> Instantaneous:
    """Instantaneous amplitude and frequency of every mode (docs/time-frequency.md).

    Frequency is the centred difference of the unwrapped phase; "none" padding matches scipy.signal.hilbert.
    """

def hilbert_spectrum(
    attributes: Instantaneous,
    f_min: float,
    f_max: float,
    n_bins: int,
    *,
    scale: Literal["linear", "log"] = "linear",
    weight: Literal["power", "amplitude"] = "power",
) -> TimeFrequencyMap:
    """Hilbert spectrum: each sample's power (or amplitude) in the bin of its instantaneous frequency,
    summed over modes. Bins are [low, high); samples outside the grid are dropped."""

def cwt(
    signal: ArrayLike,
    sampling_rate: float,
    f_min: float,
    f_max: float,
    n_bins: int,
    *,
    omega0: float = 6.0,
    rectify: bool = False,
    labels: Sequence[str] | None = None,
) -> TimeFrequencyMap:
    """Morlet continuous wavelet transform (Torrence & Compo 1998), checked against pycwt.

    Frequencies are the geometric centres of n_bins log-spaced bins over [f_min, f_max].
    `reliable` marks values outside the cone of influence. `rectify` divides power by scale
    (Liu et al. 2007).
    """

def functional_connectivity(
    data: Series | Decomposition | Epochs,
    measure: Literal["pearson", "plv"] = "pearson",
    *,
    fisher_z: bool = False,
) -> FunctionalConnectivity:
    """Functional connectivity (docs/connectivity.md): statistical dependence, undirected and
    symmetric, not direction or cause. One matrix for a Series, one per mode for a
    Decomposition, one per epoch (and mode) for Epochs.

    plv is the phase locking value ``|mean(exp(iΔφ))|``; phases come from the full run, so epoch edges add
    no Hilbert edge effects. PLV is inflated on narrow-band data, so compare it against a null.
    fisher_z (pearson only) returns atanh(r) with an undefined (NaN) diagonal.
    """

def read_channels(
    path: str | PathLike[str], format: Literal["tsv", "schaefer", "tian", "names"] = "tsv"
) -> ChannelTable:
    """Reads a channel table from an atlas label file (docs/regions.md).

    tsv: name, region, hemisphere and tag columns (covers BIDS dseg.tsv). schaefer: Schaefer or
    Yan labels such as 17Networks_LH_DefaultB_PFCm_1. tian: Tian labels such as lAMY-rh.
    names: one name per line.
    """

def test_groups(
    table: FeatureTable,
    column: str,
    groups: tuple[str, str],
    *,
    covariates: Sequence[str] = (),
    condition: str | None = None,
    welch: bool = False,
    tail: Literal["both", "greater", "less"] = "both",
    n_permutations: int = 5000,
    seed: int = 0,
    fdr: Literal["bh", "by"] = "bh",
    nbs_threshold: float | None = None,
) -> StatsReport:
    """groups[0] minus groups[1] of a participant column, per feature (docs/statistics.md).

    GLM with centred covariates (text columns dummy-coded), Freedman–Lane permutation.
    welch=True (no covariates) uses Welch's t. One row per subject is required.
    """

def test_paired(
    table: FeatureTable,
    conditions: tuple[str, str],
    *,
    covariates: Sequence[str] = (),
    tail: Literal["both", "greater", "less"] = "both",
    n_permutations: int = 5000,
    seed: int = 0,
    fdr: Literal["bh", "by"] = "bh",
    nbs_threshold: float | None = None,
) -> StatsReport:
    """conditions[0] minus conditions[1] within each recording, by sign flipping; every flip is
    enumerated (exact p) when 2^n is within n_permutations."""

def test_correlation(
    table: FeatureTable,
    score: str,
    *,
    covariates: Sequence[str] = (),
    condition: str | None = None,
    tail: Literal["both", "greater", "less"] = "both",
    n_permutations: int = 5000,
    seed: int = 0,
    fdr: Literal["bh", "by"] = "bh",
    nbs_threshold: float | None = None,
) -> StatsReport:
    """The slope of a numeric participant column per feature, adjusting for covariates."""

def read_nifti(
    path: str | PathLike[str],
    atlases: Sequence[str | PathLike[str]],
    *,
    sampling_rate: float | None = None,
    standardize: Literal["none", "zscore_sample"] = "none",
    labels: Sequence[str] | None = None,
) -> Series:
    """Region signals from a 4D NIfTI run (docs/volumes.md): the mean over each atlas label's
    voxels, as nilearn's NiftiLabelsMasker (atlas resampled to the data by nearest neighbour,
    0 = background, labels ascending). Atlases are concatenated in order. The TR comes from the
    header unless sampling_rate is given."""
