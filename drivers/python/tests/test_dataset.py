import subprocess
import sys

import numpy as np
import pytest

import modewise

h5py = pytest.importorskip("h5py")

PIPELINE = """
[pipeline]
name = "demo"
inputs = ["bold"]

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 1000.0, init = "linear" }
"""

DATASET = """
root = "data"

[[input]]
name = "bold"
pattern = "sub-{subject}/sub-{subject}_task-{task}_bold.h5"
format = "hdf5"
dataset = "signal"
sidecar = "sub-{subject}/sub-{subject}_task-{task}_bold.json"
"""


@pytest.fixture
def project(tmp_path):
    rng = np.random.default_rng(0)
    for subject in ["01", "02", "03"]:
        folder = tmp_path / "data" / f"sub-{subject}"
        folder.mkdir(parents=True)
        with h5py.File(folder / f"sub-{subject}_task-rest_bold.h5", "w") as f:
            f["signal"] = rng.standard_normal((2, 64))
        (folder / f"sub-{subject}_task-rest_bold.json").write_text('{"RepetitionTime": 0.8}')
    (tmp_path / "pipeline.toml").write_text(PIPELINE)
    (tmp_path / "dataset.toml").write_text(DATASET)
    return tmp_path


def test_dataset_discovers_and_loads(project):
    ds = modewise.Dataset.from_file(project / "dataset.toml")
    assert len(ds) == 3 and ds.inputs == ["bold"]
    first = ds.recordings[0]
    assert ds.entities(first) == {"subject": "01", "task": "rest"}
    bold = ds.load(first)["bold"]
    assert isinstance(bold, modewise.Series) and bold.sampling_rate == 1.25
    with pytest.raises(KeyError):
        ds.load("nope")


def test_run_dataset_saves_and_reopens(project, capsys):
    pipe = modewise.Pipeline.from_file(project / "pipeline.toml")
    ds = modewise.Dataset.from_file(project / "dataset.toml")
    batch = pipe.run_dataset(ds, project / "out")
    assert len(batch.succeeded) == 3 and batch.failed == {}
    assert "0/1 steps cached" in capsys.readouterr().out
    assert isinstance(batch.output(batch.succeeded[0], "modes"), modewise.Decomposition)

    again = pipe.run_dataset(ds, project / "out", progress=False)
    assert again.succeeded == batch.succeeded

    results = modewise.Results(project / "out")
    recording = results.recordings("demo")[1]
    assert results.manifest("demo", recording)["steps"][0]["cached"] is True
    modes = results.output("demo", recording, "modes")
    assert modes.modes.shape == (2, 2, 64)
    with pytest.raises(KeyError):
        results.output("demo", recording, "missing")


def test_shards_and_export(project):
    pipe = modewise.Pipeline.from_file(project / "pipeline.toml")
    ds = modewise.Dataset.from_file(project / "dataset.toml")
    first = pipe.run_dataset(ds, project / "out", shard=(1, 2), progress=False)
    second = pipe.run_dataset(ds, project / "out", shard=(2, 2), progress=False)
    assert sorted(first.succeeded + second.succeeded) == ds.recordings
    with pytest.raises(ValueError):
        pipe.run_dataset(ds, project / "out", shard=(3, 2))

    files = modewise.Results(project / "out").export_hdf5("demo", project / "export")
    with h5py.File(files[0]) as f:
        assert f["modes/modes/modes"].shape == (2, 2, 64)
        assert [label.decode() for label in f["modes/modes/labels"][()]] == ["ch0", "ch1"]


def test_input_mismatch_is_a_value_error(project):
    pipe = modewise.Pipeline.from_toml(PIPELINE.replace('inputs = ["bold"]', 'inputs = ["eeg"]').replace('input = "bold"', 'input = "eeg"'))
    with pytest.raises(ValueError, match="not all among the inputs"):
        pipe.run_dataset(modewise.Dataset.from_file(project / "dataset.toml"), project / "out")


def test_cli_entry_point(project):
    from modewise._core import cli

    code = cli(["modewise", "inspect", str(project / "dataset.toml")])
    assert code == 0
    run = [sys.executable, "-c", "import modewise._cli as c; c.main()", "run", str(project / "pipeline.toml"),
           "--dataset", str(project / "dataset.toml"), "--out", str(project / "cli-out")]
    finished = subprocess.run(run, capture_output=True, text=True)
    assert finished.returncode == 0, finished.stderr
    assert "3 succeeded, 0 failed" in finished.stdout
