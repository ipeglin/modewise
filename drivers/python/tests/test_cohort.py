import numpy as np
import pytest

import modewise

h5py = pytest.importorskip("h5py")

PIPELINE = """
[pipeline]
name = "fc"
inputs = ["bold"]

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 1000.0, init = "linear" }

[[step]]
id = "fc"
op = "functional_connectivity"
input = "modes"
params = { measure = "pearson" }
"""

DATASET = """
root = "data"

[[input]]
name = "bold"
pattern = "sub-{subject}/sub-{subject}_task-{task}_bold.h5"
format = "hdf5"
dataset = "signal"
sampling_rate_hz = 1.25

[participants]
tables = [{ path = "participants.tsv", id_pattern = "sub-{subject}", missing = ["-9"] }]
"""


@pytest.fixture
def results(tmp_path):
    rng = np.random.default_rng(0)
    for subject in ["01", "02", "03"]:
        folder = tmp_path / "data" / f"sub-{subject}"
        folder.mkdir(parents=True)
        with h5py.File(folder / f"sub-{subject}_task-rest_bold.h5", "w") as f:
            f["signal"] = rng.standard_normal((3, 80))
    (tmp_path / "data" / "participants.tsv").write_text(
        "participant_id\tgroup\tage\nsub-01\tPatient\t30\nsub-02\tGenPop\t-9\nsub-03\tPatient\t41\n"
    )
    ds = modewise.Dataset.from_toml(DATASET, tmp_path)
    modewise.Pipeline.from_toml(PIPELINE).run_dataset(ds, tmp_path / "results")
    return modewise.Results(tmp_path / "results"), ds


def test_participants_come_from_the_dataset(results):
    _, ds = results
    p = ds.participants
    assert p.columns == ["group", "age"] and p.subjects == ["01", "02", "03"]
    assert p.value("02", "age") is None and p.value("03", "group") == "Patient"
    pd = pytest.importorskip("pandas")
    frame = p.to_pandas()
    assert isinstance(frame, pd.DataFrame) and frame.loc["01", "age"] == "30"


def test_feature_tables_join_participants_and_export(results, tmp_path):
    saved, ds = results
    table = saved.feature_table("fc", "fc", participants=ds.participants)
    assert table.values.shape == (3, 2 * 3) and table.feature_names[0] == "m0:ch0~ch1"
    assert table.subjects == ["01", "02", "03"] and table.conditions == [None, None, None]
    assert table.metadata["group"] == ["Patient", "GenPop", "Patient"]
    direct = saved.output("fc", saved.recordings("fc")[0], "fc").values[0, 0, 0, 1]
    assert table.values[0, 0] == pytest.approx(np.arctanh(direct), abs=1e-12), "Pearson edges are Fisher z"

    table.to_tsv(tmp_path / "features.tsv")
    lines = (tmp_path / "features.tsv").read_text().splitlines()
    assert lines[0].startswith("recording\tsubject\tcondition\tage\tgroup\tm0:ch0~ch1")
    assert lines[2].split("\t")[3] == "n/a"
    pd = pytest.importorskip("pandas")
    frame = table.to_pandas()
    assert list(frame.columns[:5]) == ["recording", "subject", "condition", "age", "group"]
    np.testing.assert_array_equal(frame[table.feature_names].to_numpy(), table.values)

    power = saved.feature_table("fc", "modes", "mode_power")
    assert power.feature_names[:4] == ["m0:centre_hz", "m0:ch0", "m0:ch1", "m0:ch2"]
    with pytest.raises(ValueError, match="features must be"):
        saved.feature_table("fc", "fc", "coherence")
    with pytest.raises(ValueError, match="mode power needs"):
        saved.feature_table("fc", "fc", "mode_power")


def test_stack_puts_recordings_first(results):
    saved, _ = results
    recordings, modes = saved.stack("fc", "modes")
    assert recordings == saved.recordings("fc") and modes.shape == (3, 2, 3, 80)
    np.testing.assert_array_equal(modes[1], saved.output("fc", recordings[1], "modes").modes)
