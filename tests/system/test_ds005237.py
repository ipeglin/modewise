"""System test on public data: ds005237 (Transdiagnostic Connectome Project, CC0) from OpenNeuro.

Downloads two runs of one subject (about 3.4 MB), checks their SHA-256, and runs
the installed `modewise` command with the thesis MVMD settings. Needs network:
    MODEWISE_SYSTEM_TESTS=1 pytest tests/system
"""

import hashlib
import json
import os
import subprocess
import sys
import sysconfig
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import modewise

pytestmark = pytest.mark.skipif(
    os.environ.get("MODEWISE_SYSTEM_TESTS") != "1", reason="set MODEWISE_SYSTEM_TESTS=1 (downloads public data)"
)

BUCKET = "https://s3.amazonaws.com/openneuro.org/ds005237"
PARCELLATED = "fMRI_timeseries_clean_denoised_GSR_parcellated"
FILES = {
    "participants.tsv": "ec2eb0d5457cf80417ce7e12eaae58571613465544caebc13f9937273bf27e83",
    "phenotype/shaps01.tsv": "9d049dc752850f1d59f0ae68f3a5d6381661cc9dca9fc3879a031bcd532dd4b4",
    f"{PARCELLATED}/NDAR_INVAG388HJL/task-restAP_run-01_bold_Atlas_hp2000_clean_GSR_parcellated.h5":
        "e2cc02d8a68c0a3af8da2ec5c8a4810d3aa0372fa1186a62cb8b6d7cd0b80e8b",
    f"{PARCELLATED}/NDAR_INVAG388HJL/task-hammerAP_run-01_bold_Atlas_hp2000_clean_GSR_parcellated.h5":
        "64a27e0aea00cbdc86deac4cc05fdced7606e7f077112017b54775a348bf923b",
    "sub-NDARINVAG388HJL/func/sub-NDARINVAG388HJL_task-restAP_run-01_bold.json":
        "c4fd29eb274359323ad8a464f580eafd2ce71cd00b7a3e88f4cca3d98ba12c32",
    "sub-NDARINVAG388HJL/func/sub-NDARINVAG388HJL_task-hammerAP_run-01_bold.json":
        "91006794418dd6c3c508610b911caaecd60dad58f57327aaf2c0c4d47a4bdf27",
    "sub-NDARINVAG388HJL/func/sub-NDARINVAG388HJL_task-hammerAP_run-01_events.tsv":
        "99b71438838b2d507db38fc771b7273fd82e6c9fc04d98b45f2f90f5bf046405",
}

DATASET = f"""
root = "ds005237"

[[input]]
name = "bold"
pattern = "{PARCELLATED}/NDAR_{{subject}}/task-{{task}}_run-{{run}}_bold_Atlas_hp2000_clean_GSR_parcellated.h5"
format = "hdf5"
dataset = "dataset"
sidecar = "sub-NDAR{{subject}}/func/sub-NDAR{{subject}}_task-{{task}}_run-{{run}}_bold.json"

[channels]
tables = [{{ path = "channels.tsv", format = "tsv" }}]
names_for = ["bold"]
"""

EXAMPLES = Path(__file__).parents[2] / "examples" / "tcp"
HAMMER_DATASET = (EXAMPLES / "dataset.toml").read_text().replace('root = "/path/to/ds005237"', 'root = "ds005237"')

THESIS_MVMD = """
[pipeline]
name = "thesis-mvmd"
inputs = ["bold"]

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 10, alpha = 1000.0, init = "exponential", tolerance = 1e-8, tau = 1e-3 }
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("system")
    cache = Path(os.environ.get("MODEWISE_DATA_CACHE", root / "cache"))
    for relative, sha in FILES.items():
        cached = cache / relative
        if not cached.exists() or hashlib.sha256(cached.read_bytes()).hexdigest() != sha:
            cached.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{BUCKET}/{relative}", cached)
        assert hashlib.sha256(cached.read_bytes()).hexdigest() == sha, f"{relative} changed upstream"
        target = root / "ds005237" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(cached.read_bytes())
    (root / "dataset.toml").write_text(DATASET)
    (root / "hammer.toml").write_text(HAMMER_DATASET)
    (root / "channels.tsv").write_text((EXAMPLES / "channels.tsv").read_text())
    (root / "blocks.toml").write_text((EXAMPLES / "blocks.toml").read_text())
    (root / "thesis.toml").write_text(THESIS_MVMD)
    return root


def run(*args, code=0):
    name = "modewise.exe" if sys.platform == "win32" else "modewise"
    script = Path(sysconfig.get_path("scripts")) / name
    finished = subprocess.run([str(script), *map(str, args)], capture_output=True, text=True)
    assert finished.returncode == code, finished.stdout + finished.stderr
    return finished.stdout


def test_discovers_official_layout(project):
    out = run("inspect", project / "dataset.toml", "--load")
    assert "2 recordings" in out
    assert "434 channels" in out and "at 1.25 Hz" in out


@pytest.fixture(scope="module")
def results(project):
    out = project / "results"
    first = run("run", project / "thesis.toml", "--dataset", project / "dataset.toml", "--out", out)
    assert "2 succeeded, 0 failed" in first
    return out


def test_thesis_mvmd_runs_and_caches(project, results):
    rerun = run("run", project / "thesis.toml", "--dataset", project / "dataset.toml", "--out", results)
    assert rerun.count("(1/1 steps cached)") == 2

    results = modewise.Results(results)
    for recording in results.recordings("thesis-mvmd"):
        modes = results.output("thesis-mvmd", recording, "modes")
        freqs = modes.centre_frequencies_hz
        assert modes.modes.shape[:2] == (10, 434)
        assert np.all(np.isfinite(modes.modes))
        assert np.all(np.diff(freqs) >= 0) and 0 <= freqs[0] and freqs[-1] <= 1.25 / 2
        manifest = results.manifest("thesis-mvmd", recording)
        assert manifest["steps"][0]["params"]["init"] == "exponential"


def test_matches_direct_python_call(project, results):
    ds = modewise.Dataset.from_file(project / "dataset.toml")
    recording = ds.recordings[0]
    bold = ds.load(recording)["bold"]
    direct = modewise.mvmd(bold.data, bold.sampling_rate, 10, 1000.0, init="exponential", tolerance=1e-8, tau=1e-3)
    stored = modewise.Results(results).output("thesis-mvmd", recording, "modes")
    np.testing.assert_array_equal(stored.modes, direct.modes)


def test_task_blocks_are_cut_from_the_full_run(project, results):
    inspected = run("inspect", project / "hammer.toml", "--load")
    assert "1 recordings" in inspected and "trials: 108 events, labels: face, shape" in inspected

    out = run("run", project / "blocks.toml", "--dataset", project / "hammer.toml", "--out", results)
    assert "(1/2 steps cached)" in out, "the full-run MVMD is reused from the earlier run"

    saved = modewise.Results(results)
    recording = saved.recordings("tcp-blocks")[0]
    epochs = saved.output("tcp-blocks", recording, "blocks")
    modes = saved.output("tcp-blocks", recording, "modes")
    assert len(epochs) == 18 and sorted(set(epochs.labels)) == ["face", "shape"]
    assert epochs.labels.count("face") == 9 and not epochs.dropped
    assert np.all((epochs.durations_s > 16) & (epochs.durations_s < 18))

    expected = modes.centre_frequencies_hz[None, :] * epochs.durations_s[:, None] >= 1.0
    np.testing.assert_array_equal(epochs.resolvable, expected)
    assert not epochs.resolvable[:, 0].any(), "the slowest thesis mode cannot fit in a ~17 s block"
    first = int(epochs.start_sample[0])
    np.testing.assert_array_equal(epochs.data(0), modes.modes[:, :, first:int(epochs.end_sample[0])])


def test_hilbert_on_real_thesis_modes(project, results):
    saved = modewise.Results(results)
    modes = saved.output("thesis-mvmd", saved.recordings("thesis-mvmd")[0], "modes")
    attrs = modewise.hilbert(modes)
    assert np.all(np.isfinite(attrs.frequency_hz)) and np.all(attrs.amplitude >= 0)
    # Away from the run edges, each mode's median instantaneous frequency sits near its centre frequency.
    inner = attrs.frequency_hz[:, :, 50:-50]
    medians = np.median(inner, axis=(1, 2))
    assert np.all(np.abs(medians - modes.centre_frequencies_hz) < 0.05), (medians, modes.centre_frequencies_hz)
    negative = float(np.mean(inner < 0))
    assert negative < 0.05, f"{negative:.1%} negative instantaneous frequency in the run interior"


def test_cwt_on_real_bold_flags_slow_frequencies_in_blocks(project):
    ds = modewise.Dataset.from_file(project / "hammer.toml")
    recording = ds.recordings[0]
    loaded = ds.load(recording)
    bold, trials = loaded["bold"], loaded["trials"]
    spec = {
        "pipeline": {"name": "cwt-blocks", "inputs": {"bold": "series", "trials": "events"}},
        "step": [
            {"id": "scalogram", "op": "cwt", "input": "bold", "params": {"f_min": 0.01, "f_max": 0.6, "n_bins": 24}},
            {"id": "blocks", "op": "epoch", "input": {"data": "scalogram", "events": "trials"},
             "params": {"lag_s": 5.0, "merge_gap_s": 2.0}},
        ],
    }
    run = modewise.Pipeline.from_dict(spec).run({"bold": bold, "trials": trials})
    scalogram, blocks = run.output("scalogram"), run.output("blocks")
    assert scalogram.power.shape == (434, 24, bold.data.shape[1])
    assert np.all(np.isfinite(scalogram.power)) and np.all(scalogram.power >= 0)
    expected_cycles = scalogram.frequencies_hz[None, :] * blocks.durations_s[:, None] >= 1.0
    assert not np.any(blocks.resolvable & ~expected_cycles), "nothing below one cycle is marked resolvable"
    assert not blocks.resolvable[:, 0].any() and blocks.resolvable[:, -1].any()


def test_vlmd_on_real_bold_gives_bounded_loadings_and_consistent_modes(project):
    ds = modewise.Dataset.from_file(project / "hammer.toml")
    bold = ds.load(ds.recordings[0])["bold"]
    x = bold.data
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    modes, latent, loadings = modewise.vlmd(x, bold.sampling_rate, 4, num_latents=20, alpha=1000.0)
    assert modes.modes.shape == (4, 434, x.shape[1]) and latent.modes.shape == (4, 20, x.shape[1])
    assert np.all(np.isfinite(modes.modes)) and np.all(np.abs(loadings.values) <= 1)
    centres = modes.centre_frequencies_hz
    assert np.all(np.diff(centres) >= 0) and centres[0] >= 0 and centres[-1] <= bold.sampling_rate / 2
    assert np.allclose(np.einsum("klt,lc->kct", latent.modes, loadings.values), modes.modes)
    fit = modes.modes.sum(axis=0)
    explained = 1 - np.sum((x - fit) ** 2) / np.sum(x**2)
    # Four narrow modes cannot cover broadband BOLD; this run measured 0.26. A sanity bound only.
    assert explained > 0.1, f"the latent fit explains {explained:.2f} of the variance"


def test_block_connectivity_of_real_modes_is_symmetric_with_cycle_flags(project):
    ds = modewise.Dataset.from_file(project / "hammer.toml")
    loaded = ds.load(ds.recordings[0])
    spec = {
        "pipeline": {"name": "block-fc", "inputs": {"bold": "series", "trials": "events"}},
        "step": [
            {"id": "modes", "op": "mvmd", "input": "bold", "params": {"num_modes": 3, "alpha": 1000.0}},
            {"id": "blocks", "op": "epoch", "input": {"data": "modes", "events": "trials"},
             "params": {"lag_s": 5.0, "merge_gap_s": 2.0}},
            {"id": "fc", "op": "functional_connectivity", "input": "blocks", "params": {"measure": "pearson"}},
            {"id": "plv", "op": "functional_connectivity", "input": "blocks", "params": {"measure": "plv"}},
        ],
    }
    run = modewise.Pipeline.from_dict(spec).run({"bold": loaded["bold"], "trials": loaded["trials"]})
    blocks, fc, plv = run.output("blocks"), run.output("fc"), run.output("plv")
    assert fc.values.shape == (len(blocks), 3, 434, 434)
    np.testing.assert_array_equal(fc.resolvable, blocks.resolvable)
    for values, low in ((fc.values, -1.0), (plv.values, 0.0)):
        finite = values[np.isfinite(values)]
        assert finite.size > 0.99 * values.size and finite.min() >= low and finite.max() <= 1.0
        np.testing.assert_allclose(values, np.swapaxes(values, 2, 3), equal_nan=True)


ROIS = [{"region": "PFCv"}, {"region": "PFCm"}, {"region": "*AMY*"}]


def roi_order_comparison(bold, channels, num_modes=10):
    """MVMD on all channels then the ROIs, versus MVMD on the ROIs only (thesis settings)."""
    params = {"num_modes": num_modes, "alpha": 1000.0, "init": "exponential", "tolerance": 1e-8, "tau": 1e-3}
    spec = {
        "pipeline": {"name": "roi-order", "inputs": {"bold": "series", "channels": "channels"}},
        "step": [
            {"id": "all_modes", "op": "mvmd", "input": "bold", "params": params},
            {"id": "after", "op": "select", "input": {"data": "all_modes", "channels": "channels"}, "params": {"rules": ROIS}},
            {"id": "rois", "op": "select", "input": {"data": "bold", "channels": "channels"}, "params": {"rules": ROIS}},
            {"id": "before", "op": "mvmd", "input": "rois", "params": params},
        ],
    }
    run = modewise.Pipeline.from_dict(spec).run({"bold": bold, "channels": channels})
    return run.output("after"), run.output("before")


def test_roi_selection_works_before_and_after_decomposition(project):
    ds = modewise.Dataset.from_file(project / "hammer.toml")
    loaded = ds.load(ds.recordings[0])
    bold, channels = loaded["bold"], loaded["channels"]
    assert len(channels) == 434 and bold.labels[402] == "lAMY-rh"
    after, before = roi_order_comparison(bold, channels)
    assert after.labels == before.labels and len(after.labels) == len(channels.select(ROIS))
    a, b = after.centre_frequencies_hz, before.centre_frequencies_hz
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    r = [np.median([np.corrcoef(after.modes[k, c], before.modes[k, c])[0, 1] for c in range(len(after.labels))]) for k in range(len(a))]
    print("\nROI-order comparison, one hammerAP run, thesis MVMD settings")
    print("centre frequencies, all channels first:", np.round(a, 4))
    print("centre frequencies, ROIs first:        ", np.round(b, 4))
    print("median correlation of matching modes:  ", np.round(r, 2))


def test_participants_and_block_features_form_one_table(project, results):
    ds = modewise.Dataset.from_file(project / "hammer.toml")
    p = ds.participants
    assert "Group" in p.columns and "shaps_total" in p.columns and len(p) >= 244
    subject = ds.entities(ds.recordings[0])["subject"]
    assert p.value(subject, "Group") in {"Patient", "GenPop"}

    out = project / "fc-results"
    spec = (EXAMPLES / "blocks.toml").read_text()
    pipeline = modewise.Pipeline.from_toml(spec + """
[[step]]
id = "fc"
op = "functional_connectivity"
input = "blocks"
params = { measure = "pearson" }
""")
    pipeline.run_dataset(ds, out)
    saved = modewise.Results(out)
    table = saved.feature_table(pipeline.name, "fc", participants=p)
    assert sorted(table.conditions) == ["face", "shape"], "one row per condition, in order of first appearance"
    assert table.metadata["Group"][0] == p.value(subject, "Group")
    n_modes = saved.output(pipeline.name, saved.recordings(pipeline.name)[0], "modes").n_modes
    assert table.values.shape[1] == n_modes * 434 * 433 // 2
    finite = np.isfinite(table.values)
    assert finite.any(), "resolvable components give finite edges"
