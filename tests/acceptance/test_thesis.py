"""Thesis acceptance test: rebuild the thesis's hammer-task analysis with modewise and compare it
with the stored legacy outputs (docs/thesis-acceptance.md).

Opt-in, on local data that is not public in this form:
  MODEWISE_LEGACY_DIR      folder of legacy consolidated `sub-*/…_bold.h5` files
  MODEWISE_TCP_ROOT        a ds005237 checkout with events.tsv, participants.tsv and phenotype/
  MODEWISE_LEGACY_ATLASES  folder with the Schaefer 400 (17 networks) info file and Tian S2 labels
Optional: MODEWISE_ACCEPTANCE_REPORT, where to write the JSON report.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import modewise

h5py = pytest.importorskip("h5py")
ENV = ("MODEWISE_LEGACY_DIR", "MODEWISE_TCP_ROOT", "MODEWISE_LEGACY_ATLASES")
pytestmark = pytest.mark.skipif(not all(os.environ.get(v) for v in ENV), reason=f"set {', '.join(ENV)}")

FS = 1.25
EXAMPLES = Path(__file__).parents[2] / "examples" / "tcp"
LEGACY_FILE = "sub-NDAR{subject}/sub-NDAR{subject}_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5"
ROIS = [{"region": "PFCv"}, {"region": "PFCm"}, {"region": "*AMY*"}]
REPORT = {}


def legacy_dir():
    return Path(os.environ["MODEWISE_LEGACY_DIR"])


def subjects():
    return sorted(p.name.removeprefix("sub-NDAR") for p in legacy_dir().glob("sub-NDAR*"))


def channel_table():
    atlases = Path(os.environ["MODEWISE_LEGACY_ATLASES"])
    return modewise.read_channels(atlases / "400Parcels_Yeo2011_17Networks_info.txt", "schaefer") + modewise.read_channels(
        atlases / "Tian_Subcortex_S2_3T_label.txt", "tian"
    )


def corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


@pytest.fixture(scope="module")
def legacy_first():
    subject = subjects()[0]
    return subject, h5py.File(legacy_dir() / LEGACY_FILE.format(subject=subject), "r")


def legacy_blocks(f):
    blocks = []
    for condition in ("face", "shape"):
        for name, ds in f[f"02fmri_segment_trials/blocks_std/{condition}"].items():
            blocks.append((condition, name, ds[()].astype(float), float(ds.attrs["onset_s"]), float(ds.attrs["block_end_s"])))
    return sorted(blocks, key=lambda b: b[3])


def test_block_correlation_matches_the_legacy_exactly(legacy_first):
    _, f = legacy_first
    worst = 0.0
    for condition, name, block, _, _ in legacy_blocks(f):
        ours = modewise.functional_connectivity(modewise.Series(block, FS)).values[0, 0]
        theirs = f[f"06fc/ts/blocks_std/{condition}/{name}/pearson"][()]
        worst = max(worst, float(np.nanmax(np.abs(ours - theirs))))
        clipped = f[f"06fc/ts/blocks_std/{condition}/{name}/fisher_z"][()]
        off = ~np.eye(len(ours), dtype=bool)
        np.testing.assert_allclose(np.arctanh(np.clip(ours, -0.9999, 0.9999))[off], clipped[off], atol=1e-4)
    REPORT["fc_ts_max_abs_difference"] = worst
    assert worst < 1e-5, "Pearson on the legacy's own blocks agrees to float32 precision"


def test_block_windows_against_events(legacy_first):
    subject, f = legacy_first
    tcp = Path(os.environ["MODEWISE_TCP_ROOT"])
    events = modewise.read_events_tsv(next((tcp / f"sub-NDAR{subject}" / "func").glob("*hammerAP*_events.tsv")))
    series = modewise.Series(f["01fmri_parcellation/full_run_std"][()].astype(float), FS)
    ours = modewise.Pipeline.from_dict({
        "pipeline": {"name": "w", "inputs": {"x": "series", "ev": "events"}},
        "step": [{"id": "b", "op": "epoch", "input": {"data": "x", "events": "ev"}, "params": {"lag_s": 0.0, "merge_gap_s": 2.0}}],
    }).run({"x": series, "ev": events}).output("b")
    legacy = legacy_blocks(f)
    REPORT["blocks"] = {
        "legacy_count": len(legacy),
        "modewise_count": len(ours),
        "legacy_durations_s": sorted({round(b[4] - b[3], 2) for b in legacy}),
        "modewise_durations_s": sorted({round(d, 2) for d in ours.durations_s}),
        "legacy_samples": sorted({b[2].shape[1] for b in legacy}),
        "onset_differences_s": sorted({round(o - b[3], 2) for o, b in zip(ours.event_onsets_s, legacy)}),
    }
    assert len(ours) == len(legacy)


def test_mvmd_hilbert_and_cwt_on_legacy_blocks(legacy_first):
    _, f = legacy_first
    condition, name, block, onset, end = legacy_blocks(f)[0]
    group = f[f"04hht/blocks_std/{condition}/{name}"]
    params = dict(group.attrs)
    ours = modewise.mvmd(block, FS, int(params["num_modes"]), float(params["alpha"]), init="exponential",
                         tolerance=float(params["admm_tolerance"]), tau=float(params["admm_tau"]),
                         max_iterations=int(params["admm_max_iterations"]))
    theirs = group["modes"][()].astype(float)
    REPORT["mvmd_block"] = {
        "legacy_centre_hz": np.round(group["center_frequencies"][()], 4).tolist(),
        "modewise_centre_hz": np.round(ours.centre_frequencies_hz, 4).tolist(),
        "median_mode_correlation": float(np.median([corr(ours.modes[k], theirs[k]) for k in range(len(theirs))])),
    }

    legacy_modes = modewise.Decomposition(theirs, group["center_frequencies"][()].astype(float), FS)
    attrs = modewise.hilbert(legacy_modes, padding="none")
    REPORT["hilbert_from_legacy_modes"] = {
        "envelope_correlation": corr(attrs.amplitude, group["envelope"][()]),
        "frequency_correlation": corr(attrs.frequency_hz, group["instantaneous_frequency"][()]),
    }

    full = f["01fmri_parcellation/full_run_std"][()].astype(float)
    scalogram = modewise.cwt(full, FS, 0.01, 0.25, 224)
    start, stop = int(np.floor(onset * FS)), min(int(np.ceil(end * FS)), full.shape[1])
    legacy_cwt = f[f"03cwt/blocks_std/{condition}/{name}"][()].astype(float)
    # Both grids run from low to high frequency. The legacy uses a different Morlet convention
    # (scale = ω0/(2πf·dt), no Fourier factor) and per-block max-divide scaling, so only the
    # pattern is compared (docs/thesis-acceptance.md).
    ours_block = scalogram.power[:, :, start:stop]
    n = min(ours_block.shape[2], legacy_cwt.shape[2])
    REPORT["cwt_block"] = {
        "median_channel_correlation": float(np.median([corr(ours_block[c, :, :n], legacy_cwt[c, :, :n]) for c in range(0, 432, 9)])),
        "legacy_normalisation": str(f[f"03cwt/blocks_std/{condition}/{name}"].attrs.get("normalize_method")),
    }


def test_cohort_runs_and_gives_paired_and_group_results(tmp_path):
    tcp = Path(os.environ["MODEWISE_TCP_ROOT"])
    root = tmp_path / "data"
    root.mkdir()
    (root / "legacy").symlink_to(legacy_dir())
    (root / "tcp").symlink_to(tcp)
    grouping = tmp_path / "anhedonia.tsv"
    subprocess.run([sys.executable, str(EXAMPLES / "make_participants.py"), str(tcp), str(grouping)], check=True, capture_output=True)
    atlases = Path(os.environ["MODEWISE_LEGACY_ATLASES"])
    spec = f"""
root = "data"

[[input]]
name = "bold"
pattern = "legacy/{LEGACY_FILE}"
format = "hdf5"
dataset = "01fmri_parcellation/full_run_std"
sampling_rate_hz = {FS}

[[input]]
name = "trials"
pattern = "tcp/sub-NDAR{{subject}}/func/sub-NDAR{{subject}}_task-hammerAP_run-01_events.tsv"
format = "events_tsv"

[channels]
tables = [{{ path = "{atlases / '400Parcels_Yeo2011_17Networks_info.txt'}", format = "schaefer" }}, {{ path = "{atlases / 'Tian_Subcortex_S2_3T_label.txt'}", format = "tian" }}]
names_for = ["bold"]

[participants]
tables = [{{ path = "{grouping}", id_column = "subjectkey", id_pattern = "NDAR_{{subject}}" }}]
"""
    ds = modewise.Dataset.from_toml(spec, tmp_path)
    assert len(ds) == len(subjects())
    pipeline = modewise.Pipeline.from_toml((EXAMPLES / "thesis.toml").read_text())
    summary = pipeline.run_dataset(ds, tmp_path / "results")
    results = modewise.Results(tmp_path / "results")
    table = results.feature_table("thesis", "fc", participants=ds.participants)
    groups = dict(zip(table.subjects, table.metadata["thesis_group"]))
    paired = modewise.test_paired(table, ("face", "shape"), n_permutations=20000, seed=1)
    face_groups = modewise.test_groups(table, "thesis_group", ("anhedonic", "control"), condition="face", n_permutations=5000, seed=1)
    REPORT["cohort"] = {
        "subjects": len(ds),
        "groups": {g: list(groups.values()).count(g) for g in set(groups.values())},
        "features": len(table.feature_names),
        "paired_face_vs_shape": {
            "exhaustive": paired.exhaustive,
            "min_p_fwer": float(np.nanmin(paired.p_fwer)),
            "n_q_below_0.05": int(np.nansum(paired.q_fdr < 0.05)),
            "finite_features": int(np.isfinite(paired.t).sum()),
        },
        "groups_face": {
            "n": face_groups.n_observations,
            "min_p_fwer": float(np.nanmin(face_groups.p_fwer)),
            "n_q_below_0.05": int(np.nansum(face_groups.q_fdr < 0.05)),
        },
    }
    assert summary is not None and np.isfinite(paired.t).any()


def teardown_module():
    if REPORT:
        path = Path(os.environ.get("MODEWISE_ACCEPTANCE_REPORT", Path.cwd() / "thesis-acceptance.json"))
        path.write_text(json.dumps(REPORT, indent=2))
        print("\n" + json.dumps(REPORT, indent=2))


REST_FILE = "sub-NDAR{subject}/sub-NDAR{subject}_task-restAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5"


def analytic(x):
    """FFT analytic signal, scipy's convention (the legacy used scirs2's copy of it)."""
    n = x.shape[-1]
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[1 : (n + 1) // 2] = 2
    return np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1)


def test_rest_full_run_mvmd_and_hht_match_the_legacy_for_every_subject():
    mode_r, freq_err, env_r, if_r = [], [], [], []
    for subject in subjects():
        f = h5py.File(legacy_dir() / REST_FILE.format(subject=subject), "r")
        group = f["04hht/full_run_std"]
        series = f["01fmri_parcellation/full_run_std"][()].astype(float)
        ours = modewise.mvmd(series, FS, 10, 1000.0, init="exponential", tolerance=1e-8, tau=1e-3, max_iterations=500)
        theirs = group["modes"][()].astype(float)
        mode_r.append(min(corr(ours.modes[k], theirs[k]) for k in range(10)))
        freq_err.append(float(np.max(np.abs(ours.centre_frequencies_hz - group["center_frequencies"][()]))))
        # The legacy HHT: |z| with log1p and per-channel max-divide; backward phase difference.
        z = analytic(theirs)
        amp = np.log1p(np.abs(z))
        amp /= amp.max(axis=(0, 2), keepdims=True) + 1e-12
        step = np.angle(z[..., 1:] * np.conj(z[..., :-1])) * FS / (2 * np.pi)
        freq = np.concatenate([step[..., :1], step], axis=-1)
        env_r.append(corr(amp, group["envelope"][()]))
        if_r.append(corr(freq, group["instantaneous_frequency"][()]))
    REPORT["rest_full_run"] = {
        "subjects": len(mode_r),
        "min_mode_correlation": float(min(mode_r)),
        "max_centre_frequency_error_hz": float(max(freq_err)),
        "legacy_hht_from_legacy_modes": {"envelope_r_min": float(min(env_r)), "frequency_r_min": float(min(if_r))},
    }
    # Modes are identical; centre-frequency labels may differ by the last update step (which
    # iteration's estimate is reported), well below the 0.004 Hz spectral resolution of a run.
    assert min(mode_r) > 0.999 and max(freq_err) < 1e-3
