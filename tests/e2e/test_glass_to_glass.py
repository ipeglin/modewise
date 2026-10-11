"""The user's path through the installed package, checked with independent readers.

Runs against the installed `modewise` wheel, not the source tree:
    pytest tests/e2e
"""

import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import h5py
import numpy as np
import pytest

import modewise

FS = 10.0
TONES_HZ = (0.5, 2.0)
SAMPLES = 400

PIPELINE = """
[pipeline]
name = "glass"
inputs = ["bold"]

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 2000.0, init = "linear", tau = 0.0, tolerance = 1e-7 }

[[step]]
id = "power"
op = "stft"
input = "bold"
params = { window_len = 64, hop = 32 }
"""

DATASET = """
root = "bids"

[[input]]
name = "bold"
pattern = "sub-{subject}/func/sub-{subject}_task-{task}_bold.h5"
format = "hdf5"
dataset = "timeseries"
sidecar = "sub-{subject}/func/sub-{subject}_task-{task}_bold.json"
"""

SUBJECTS = ("01", "02", "03")


def command():
    """The console script pip installed for this interpreter (Scripts\\modewise.exe on Windows)."""
    name = "modewise.exe" if sys.platform == "win32" else "modewise"
    script = Path(sysconfig.get_path("scripts")) / name
    assert script.exists(), f"modewise command not installed at {script}"
    return str(script)


def modewise_cli(*args, check_code=0):
    finished = subprocess.run([command(), *map(str, args)], capture_output=True, text=True)
    assert finished.returncode == check_code, finished.stdout + finished.stderr
    return finished.stdout


def signal(subject):
    t = np.arange(SAMPLES) / FS
    weight = 1.0 + int(subject) / 10
    slow, fast = (np.sin(2 * np.pi * f * t) for f in TONES_HZ)
    return np.vstack([weight * slow + 0.5 * fast, 0.4 * slow + weight * fast])


@pytest.fixture
def project(tmp_path):
    for subject in SUBJECTS:
        func = tmp_path / "bids" / f"sub-{subject}" / "func"
        func.mkdir(parents=True)
        with h5py.File(func / f"sub-{subject}_task-rest_bold.h5", "w") as f:
            f["timeseries"] = signal(subject).astype(np.float32)
        (func / f"sub-{subject}_task-rest_bold.json").write_text(json.dumps({"RepetitionTime": 1 / FS}))
    (tmp_path / "pipeline.toml").write_text(PIPELINE)
    (tmp_path / "dataset.toml").write_text(DATASET)
    return tmp_path


def test_smoke():
    assert modewise.__version__ in modewise_cli("--version")
    ops = modewise_cli("ops")
    for name in ("mvmd", "stft", "reconstruct"):
        assert name in ops
    assert "run" in modewise_cli("--help")


def test_command_line_run_matches_direct_python_call(project):
    inspected = modewise_cli("inspect", project / "dataset.toml", "--load")
    assert "3 recordings" in inspected and "at 10 Hz" in inspected

    out = project / "results"
    first = modewise_cli("run", project / "pipeline.toml", "--dataset", project / "dataset.toml", "--out", out)
    assert "3 succeeded, 0 failed" in first
    second = modewise_cli("run", project / "pipeline.toml", "--dataset", project / "dataset.toml", "--out", out)
    assert second.count("(2/2 steps cached)") == 3

    modewise_cli("export", out, "--to", project / "export")
    for subject in SUBJECTS:
        recording = f"subject-{subject}_task-rest"
        with h5py.File(project / "export" / "glass" / f"{recording}.h5") as f:
            exported_modes = f["modes/modes/modes"][()]
            exported_freqs = f["modes/modes/centre_frequencies_hz"][()]
            exported_power = f["power/power/power"][()]
            manifest = json.loads(f.attrs["manifest"])

        x = signal(subject).astype(np.float32).astype(np.float64)
        direct = modewise.mvmd(x, FS, 2, 2000.0, init="linear", tau=0.0, tolerance=1e-7)
        np.testing.assert_allclose(exported_modes, direct.modes, rtol=0, atol=1e-12)
        np.testing.assert_allclose(exported_freqs, TONES_HZ, atol=0.05)
        np.testing.assert_allclose(exported_power, modewise.stft(x, FS, 64, 32).power, rtol=1e-12)
        assert [s["id"] for s in manifest["steps"]] == ["modes", "power"]

        reopened = modewise.Results(out).output("glass", recording, "modes")
        np.testing.assert_array_equal(reopened.modes, exported_modes)


def test_zarr_store_is_readable_by_zarr_python(project):
    zarr = pytest.importorskip("zarr", minversion="3")
    out = project / "results"
    modewise_cli("run", project / "pipeline.toml", "--dataset", project / "dataset.toml", "--out", out)
    recording = "subject-01_task-rest"
    manifest = json.loads((out / "runs" / "glass" / f"{recording}.json").read_text())
    key = manifest["steps"][0]["outputs"]["modes"]
    group = zarr.open_group(out / "store" / key[:2] / key, mode="r")
    assert group.attrs["modewise"]["kind"] == "decomposition"
    np.testing.assert_array_equal(group["modes"][:], modewise.Results(out).output("glass", recording, "modes").modes)


def test_python_api_and_command_line_agree(project):
    pipe = modewise.Pipeline.from_file(project / "pipeline.toml")
    ds = modewise.Dataset.from_file(project / "dataset.toml")
    batch = pipe.run_dataset(ds, project / "py-results", progress=False)
    modewise_cli("run", project / "pipeline.toml", "--dataset", project / "dataset.toml", "--out", project / "cli-results")
    cli = modewise.Results(project / "cli-results")
    for recording in batch.succeeded:
        py_modes = batch.output(recording, "modes").modes
        np.testing.assert_array_equal(py_modes, cli.output("glass", recording, "modes").modes)


def test_shards_cover_the_cohort_and_failures_are_reported(project):
    out = project / "results"
    args = ("run", project / "pipeline.toml", "--dataset", project / "dataset.toml", "--out", out, "--shard")
    modewise_cli(*args, "1/2")
    modewise_cli(*args, "2/2")
    summaries = out / "runs" / "glass" / "summaries"
    done = [r for n in ("shard-1-of-2", "shard-2-of-2") for r in json.loads((summaries / f"{n}.json").read_text())["succeeded"]]
    assert sorted(done) == [f"subject-{s}_task-rest" for s in SUBJECTS]

    (project / "bids/sub-02/func/sub-02_task-rest_bold.h5").write_bytes(b"corrupt")
    output = modewise_cli("run", project / "pipeline.toml", "--dataset", project / "dataset.toml", "--out", project / "r2", check_code=2)
    assert "FAILED  subject-02_task-rest" in output and "2 succeeded, 1 failed" in output


EPOCH_PIPELINE = """
[pipeline]
name = "glass-blocks"
inputs = { bold = "series", trials = "events" }

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 2000.0, init = "linear", tau = 0.0, tolerance = 1e-7 }

[[step]]
id = "blocks"
op = "epoch"
input = { data = "modes", events = "trials" }
params = { lag_s = 2.0 }
"""

EPOCH_DATASET = DATASET + """
[[input]]
name = "trials"
pattern = "sub-{subject}/func/sub-{subject}_task-{task}_events.tsv"
format = "events_tsv"
"""


def test_epochs_through_command_line_export_and_independent_reader(project):
    for subject in SUBJECTS:
        rows = "".join(f"{onset}\t1.5\t{'a' if i % 2 else 'b'}\n" for i, onset in enumerate(range(2, 38, 6)))
        (project / "bids" / f"sub-{subject}" / "func" / f"sub-{subject}_task-rest_events.tsv").write_text("onset\tduration\ttrial_type\n" + rows)
    (project / "epochs.toml").write_text(EPOCH_PIPELINE)
    (project / "epoch_dataset.toml").write_text(EPOCH_DATASET)

    out = project / "epoch-results"
    assert "3 succeeded, 0 failed" in modewise_cli("run", project / "epochs.toml", "--dataset", project / "epoch_dataset.toml", "--out", out)
    modewise_cli("export", out, "--to", project / "epoch-export")

    recording = "subject-01_task-rest"
    with h5py.File(project / "epoch-export" / "glass-blocks" / f"{recording}.h5") as f:
        group = f["blocks/epochs"]
        starts, ends = group["start_sample"][()], group["end_sample"][()]
        resolvable = group["resolvable"][()]
        source_modes = group["source/modes"][()]
        assert group.attrs["lag_s"] == 2.0 and group.attrs["kind"] == b"epochs"
    epochs = modewise.Results(out).output("glass-blocks", recording, "blocks")
    np.testing.assert_array_equal(starts, epochs.start_sample)
    np.testing.assert_array_equal(ends, epochs.end_sample)
    np.testing.assert_array_equal(resolvable.astype(bool), epochs.resolvable)
    assert starts[0] == int(np.ceil((2 + 2.0) * FS)), "first event at 2 s plus 2 s lag"
    np.testing.assert_array_equal(source_modes[:, :, starts[0]:ends[0]], epochs.data(0))
    assert epochs.resolvable[:, 1].all(), "2 Hz fits 3 cycles in a 1.5 s block"
    assert not epochs.resolvable[:, 0].any(), "0.5 Hz has a 2 s period, longer than a 1.5 s block"


def test_quickstart_example_runs():
    script = Path(__file__).parents[2] / "examples" / "quickstart.py"
    finished = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert finished.returncode == 0, finished.stderr
    assert "10 epochs" in finished.stdout
    assert "'modes': True" in finished.stdout, "the lag sweep must reuse the decomposition"
