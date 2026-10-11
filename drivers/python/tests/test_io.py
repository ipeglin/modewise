import numpy as np
import pytest

import modewise

h5py = pytest.importorskip("h5py")


@pytest.fixture
def h5_file(tmp_path):
    path = tmp_path / "signals.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("group/bold", data=np.arange(6, dtype=np.float32).reshape(2, 3))
        f.create_dataset("flat", data=np.arange(3.0))
    return path


def test_read_hdf5(h5_file):
    series = modewise.read_hdf5(h5_file, "group/bold", 1.25, labels=["a", "b"])
    assert isinstance(series, modewise.Series)
    np.testing.assert_array_equal(series.data, [[0, 1, 2], [3, 4, 5]])
    assert series.labels == ["a", "b"] and series.sampling_rate == 1.25


def test_read_hdf5_time_by_channels(h5_file):
    series = modewise.read_hdf5(h5_file, "group/bold", 1.0, layout="time_by_channels")
    assert series.data.shape == (3, 2)


def test_read_hdf5_errors(h5_file):
    with pytest.raises(OSError):
        modewise.read_hdf5(h5_file.with_name("missing.h5"), "group/bold", 1.0)
    with pytest.raises(ValueError, match="2D"):
        modewise.read_hdf5(h5_file, "flat", 1.0)
    with pytest.raises(ValueError):
        modewise.read_hdf5(h5_file, "group/bold", 1.0, layout="sideways")


def test_series_input_carries_rate_and_labels():
    spec = {
        "pipeline": {"name": "s", "inputs": ["x"]},
        "step": [{"id": "power", "op": "stft", "input": "x", "params": {"window_len": 16, "hop": 4}}],
    }
    series = modewise.Series(np.ones((2, 64)), 10.0, ["left", "right"])
    power = modewise.Pipeline.from_dict(spec).run({"x": series}).output("power")
    assert power.labels == ["left", "right"]
    assert power.frequencies_hz[1] == pytest.approx(10.0 / 16)


def test_plain_array_needs_sampling_rate():
    spec = {"pipeline": {"name": "s", "inputs": ["x"]}, "step": []}
    with pytest.raises(ValueError, match="sampling_rate"):
        modewise.Pipeline.from_dict(spec).run({"x": np.ones((1, 8))})


def test_series_rejects_invalid_data():
    with pytest.raises(ValueError):
        modewise.Series([[1.0, np.nan]], 1.0)
