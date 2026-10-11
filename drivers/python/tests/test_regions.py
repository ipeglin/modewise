from pathlib import Path

import numpy as np
import pytest

import modewise

TCP_CHANNELS = Path(__file__).parents[3] / "examples" / "tcp" / "channels.tsv"


def table():
    return modewise.ChannelTable.from_records([
        {"name": "L_PFCm", "region": "PFCm", "hemisphere": "left", "tags": {"yeo17": "DefaultB"}},
        {"name": "R_PFCm", "region": "PFCm", "hemisphere": "right", "tags": {"yeo17": "DefaultB"}},
        {"name": "lAMY-lh", "region": "lAMY", "hemisphere": "left", "tags": {"yeo17": "Limbic"}},
        {"name": "vermis"},
    ])


def test_tables_describe_and_select_channels():
    t = table()
    assert len(t) == 4 and t.hemispheres == ["left", "right", "left", None]
    assert t.regions[3] is None and t.tags[2] == {"yeo17": "Limbic"}
    assert t.select([{"region": "PFCm", "hemisphere": "left"}, {"region": "*AMY*"}]) == ["L_PFCm", "lAMY-lh"]
    assert t.select([{"tags": {"yeo17": "Default*"}}]) == ["L_PFCm", "R_PFCm"]
    assert "vermis" not in t.select([{"hemisphere": "left"}, {"hemisphere": "right"}])
    both = t + modewise.ChannelTable.from_records([{"name": "extra"}])
    assert both.names[-1] == "extra"
    with pytest.raises(ValueError):
        t + t
    with pytest.raises(ValueError):
        modewise.ChannelTable.from_records([{"region": "x"}])
    with pytest.raises(ValueError):
        modewise.ChannelTable.from_records([{"name": "x", "hemisphere": "up"}])


def test_pipelines_select_and_average_regions():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 200))
    spec = {
        "pipeline": {"name": "regions", "inputs": {"bold": "series", "channels": "channels"}},
        "step": [
            {"id": "pfc", "op": "select", "input": {"data": "bold", "channels": "channels"},
             "params": {"rules": [{"region": "PFCm"}]}},
            {"id": "sides", "op": "region_average", "input": {"data": "bold", "channels": "channels"},
             "params": {"by": "hemisphere"}},
        ],
    }
    run = modewise.Pipeline.from_dict(spec).run({"bold": modewise.Series(x, 1.0, labels=table().names), "channels": table()})
    pfc, sides = run.output("pfc"), run.output("sides")
    assert pfc.labels == ["L_PFCm", "R_PFCm"]
    np.testing.assert_array_equal(pfc.data, x[:2])
    assert sides.labels == ["left", "right"]
    np.testing.assert_allclose(sides.data[0], x[[0, 2]].mean(axis=0))


def test_the_tcp_table_reads_from_disk():
    t = modewise.read_channels(TCP_CHANNELS)
    assert len(t) == 434 and t.tags[0]["yeo17"] == "TempPar"
    assert t.select([{"region": "*AMY*"}]) == ["lAMY-rh", "mAMY-rh", "lAMY-lh", "mAMY-lh"]
    with pytest.raises(ValueError, match="format"):
        modewise.read_channels(TCP_CHANNELS, "aal")
    with pytest.raises(OSError):
        modewise.read_channels("missing.tsv")
