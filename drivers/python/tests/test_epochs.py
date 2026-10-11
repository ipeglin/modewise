import numpy as np
import pytest

import modewise

FS = 1.25
PIPELINE = {
    "pipeline": {"name": "blocks", "inputs": {"bold": "series", "trials": "events"}},
    "step": [
        {"id": "modes", "op": "mvmd", "input": "bold",
         "params": {"num_modes": 2, "alpha": 2000.0, "init": "linear", "tau": 0.0, "tolerance": 1e-7}},
        {"id": "blocks", "op": "epoch", "input": {"data": "modes", "events": "trials"}, "params": {"lag_s": 4.0}},
    ],
}


def recording():
    t = np.arange(600) / FS
    data = np.vstack([np.sin(2 * np.pi * 0.02 * t) + np.sin(2 * np.pi * 0.2 * t)] * 2)
    onsets = [20.0 + 40.0 * b for b in range(10)] + [590.0]
    labels = ["face" if b % 2 == 0 else "shape" for b in range(10)] + ["late"]
    return modewise.Series(data, FS, ["l", "r"]), modewise.Events(onsets, [18.0] * 11, labels)


def test_events_validate_their_input():
    events = modewise.Events([5.0, 1.0], [1.0, 0.0], ["b", "a"])
    assert events.labels == ["a", "b"] and len(events) == 2
    np.testing.assert_array_equal(events.onsets_s, [1.0, 5.0])
    for bad in ([[np.nan], [1.0], ["a"]], [[1.0], [-1.0], ["a"]], [[1.0], [1.0], [""]], [[1.0, 2.0], [1.0], ["a", "b"]]):
        with pytest.raises(ValueError):
            modewise.Events(*bad)


def test_pipeline_cuts_full_run_modes_into_flagged_epochs():
    bold, trials = recording()
    run = modewise.Pipeline.from_dict(PIPELINE).run({"bold": bold, "trials": trials})
    epochs = run.output("blocks")
    modes = run.output("modes")
    assert isinstance(epochs, modewise.Epochs) and len(epochs) == 10
    assert epochs.lag_s == 4.0 and epochs.min_cycles == 1.0
    assert epochs.start_sample[0] == 30, "20 s + 4 s lag = 24 s = sample 30"
    assert epochs.resolvable.shape == (10, 2)
    assert not epochs.resolvable[:, 0].any() and epochs.resolvable[:, 1].all()
    assert epochs.dropped == [{"label": "late", "event_onset_s": 590.0, "reason": epochs.dropped[0]["reason"]}]
    assert "after the recording" in epochs.dropped[0]["reason"]
    s0, e0 = int(epochs.start_sample[0]), int(epochs.end_sample[0])
    np.testing.assert_array_equal(epochs.data(0), modes.modes[:, :, s0:e0])
    assert isinstance(epochs.source, modewise.Decomposition)
    with pytest.raises(IndexError):
        epochs.data(10)


def test_lag_is_required_and_events_must_be_events():
    spec = {**PIPELINE, "step": [PIPELINE["step"][0], {**PIPELINE["step"][1], "params": {}}]}
    with pytest.raises(ValueError, match="lag_s"):
        modewise.Pipeline.from_dict(spec)
    bold, _ = recording()
    with pytest.raises(ValueError, match="trials"):
        modewise.Pipeline.from_dict(PIPELINE).run({"bold": bold, "trials": bold})


def test_read_events_tsv(tmp_path):
    path = tmp_path / "events.tsv"
    path.write_text("onset\tduration\ttrial_type\textra\n2.0\t1.0\tface\tn/a\n0.5\t1.0\tshape\t3\n")
    events = modewise.read_events_tsv(path)
    assert events.labels == ["shape", "face"]
    assert modewise.read_events_tsv(path, label_column="extra").labels == ["3", "n/a"]
    with pytest.raises(OSError):
        modewise.read_events_tsv(tmp_path / "missing.tsv")
    path.write_text("onset\tduration\ttrial_type\nn/a\t1.0\tface\n")
    with pytest.raises(ValueError, match="n/a"):
        modewise.read_events_tsv(path)
