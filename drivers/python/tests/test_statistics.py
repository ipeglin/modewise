import numpy as np
import pytest

import modewise

h5py = pytest.importorskip("h5py")

PIPELINE = """
[pipeline]
name = "fc"
inputs = ["bold"]

[[step]]
id = "fc"
op = "functional_connectivity"
input = "bold"
params = { measure = "pearson" }
"""

DATASET = """
root = "data"

[[input]]
name = "bold"
pattern = "sub-{subject}_bold.h5"
format = "hdf5"
dataset = "signal"
sampling_rate_hz = 1.0

[participants]
tables = [{ path = "participants.tsv", id_pattern = "sub-{subject}" }]
"""


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    root = tmp_path_factory.mktemp("stats")
    rng = np.random.default_rng(1)
    lines = ["participant_id\tgroup\tage\tshaps"]
    for s in range(16):
        patient = s < 8
        shared = rng.standard_normal(120)
        signal = rng.standard_normal((3, 120))
        signal[1] += (1.5 if patient else 0.2) * shared
        signal[0] += (1.5 if patient else 0.2) * shared
        (root / "data").mkdir(exist_ok=True)
        with h5py.File(root / "data" / f"sub-{s:02d}_bold.h5", "w") as f:
            f["signal"] = signal
        lines.append(f"sub-{s:02d}\t{'Patient' if patient else 'GenPop'}\t{20 + s}\t{s % 5}")
    (root / "data" / "participants.tsv").write_text("\n".join(lines) + "\n")
    ds = modewise.Dataset.from_toml(DATASET, root)
    modewise.Pipeline.from_toml(PIPELINE).run_dataset(ds, root / "results")
    return modewise.Results(root / "results").feature_table("fc", "fc", participants=ds.participants)


def ols_t(y, design, column):
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    sigma2 = (residuals**2).sum(0) / (len(y) - design.shape[1])
    se = np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[column, column])
    return beta[column] / se


def test_group_glm_matches_numpy_and_finds_the_effect(table):
    report = modewise.test_groups(table, "group", ("Patient", "GenPop"), covariates=["age"], n_permutations=999)
    patient = np.array([g == "Patient" for g in table.metadata["group"]], float)
    age = np.array([float(a) for a in table.metadata["age"]])
    design = np.column_stack([np.ones(16), patient, age - age.mean()])
    np.testing.assert_allclose(report.t, ols_t(table.values, design, 1), rtol=1e-10)
    assert report.regressors == ["intercept", "group=Patient", "age"]
    assert report.feature_names[0] == "m0:ch0~ch1" and report.p_fwer[0] < 0.05
    assert np.all(report.p_fwer >= report.p_permutation - 1e-12)
    pd = pytest.importorskip("pandas")
    frame = report.to_pandas()
    assert list(frame.columns) == ["feature", "t", "df", "p_parametric", "p_permutation", "p_fwer", "q_fdr"]
    assert (report.statistic, report.rank) == ("t", 1)


def test_welch_correlation_nbs_and_errors(table):
    welch = modewise.test_groups(table, "group", ("Patient", "GenPop"), welch=True, n_permutations=199, seed=4)
    assert welch.t[0] > 3 and welch.df[0] < 14
    again = modewise.test_groups(table, "group", ("Patient", "GenPop"), welch=True, n_permutations=199, seed=4)
    np.testing.assert_array_equal(welch.p_permutation, again.p_permutation)
    corr = modewise.test_correlation(table, "shaps", n_permutations=199)
    assert corr.regressors == ["intercept", "shaps"] and corr.n_observations == 16
    nbs = modewise.test_groups(table, "group", ("Patient", "GenPop"), tail="greater", nbs_threshold=3.0, n_permutations=199)
    assert nbs.components and nbs.components[0]["edges"] == ["m0:ch0~ch1"]
    with pytest.raises(ValueError, match="tail"):
        modewise.test_groups(table, "group", ("Patient", "GenPop"), tail="up")
    with pytest.raises(ValueError, match="fdr"):
        modewise.test_correlation(table, "shaps", fdr="storey")
    v = modewise.test_groups(table, "group", ("Patient", "GenPop"), welch=True, covariates=["age"])
    assert v.statistic == "v" and "v" in v.to_pandas().columns
    with pytest.raises(ValueError, match="no recording has both"):
        modewise.test_paired(table, ("face", "shape"))


def test_feature_tables_from_your_own_values_go_to_the_tests():
    rng = np.random.default_rng(2)
    n = 12
    groups = ["patient"] * 6 + ["control"] * 6
    values = rng.standard_normal((n, 2))
    values[:6, 0] += 2.0                                       # a group effect in the first feature
    table = modewise.FeatureTable(values, ["ratio", "noise"], [f"sub-{i:02d}" for i in range(n)],
                                  subjects=[f"{i:02d}" for i in range(n)], metadata={"group": groups})
    report = modewise.test_groups(table, "group", ("patient", "control"), n_permutations=2000, seed=0)
    assert report.p_permutation[0] < 0.01 and report.p_permutation[1] > 0.05
    assert table.metadata["group"][0] == "patient" and table.conditions == [None] * n

    paired = modewise.FeatureTable(
        np.concatenate([values[:6], values[:6] - 1.0]), ["ratio", "noise"],
        [f"sub-{i:02d}" for i in range(6)] * 2, conditions=["a"] * 6 + ["b"] * 6)
    assert modewise.test_paired(paired, ("a", "b"), n_permutations=64, seed=0).t[0] > 10
    with pytest.raises(ValueError, match="conditions has 1 values"):
        modewise.FeatureTable(values, ["ratio", "noise"], ["r"] * n, conditions=["a"])
    with pytest.raises(ValueError, match="column \"group\""):
        modewise.FeatureTable(values, ["ratio", "noise"], ["r"] * n, metadata={"group": ["x"]})


def test_glm_takes_a_design_matrix_with_contrasts_blocks_and_variance_groups():
    rng = np.random.default_rng(3)
    n = 12
    site = ["a", "b", "c"] * 4
    values = rng.standard_normal((n, 2))
    values[:, 0] += np.array(["abc".index(s) for s in site], float)
    table = modewise.FeatureTable(values, ["x", "y"], [f"sub-{i:02d}" for i in range(n)],
                                  subjects=[f"{i:02d}" for i in range(n)], metadata={"site": site})
    groups = modewise.test_groups(table, "site", ["a", "b", "c"], n_permutations=199)
    assert (groups.statistic, groups.rank) == ("F", 2)
    assert groups.regressors == ["intercept", "site=a", "site=b"]

    cells = np.array([[s == level for level in "abc"] for s in site], float)
    f = modewise.test_glm(table, cells, [[1, -1, 0], [0, 1, -1]], regressors=["a", "b", "c"], n_permutations=199)
    np.testing.assert_allclose(f.t, groups.t, rtol=1e-10)
    assert f.regressors == ["a", "b", "c"] and f.to_pandas().columns[1] == "F"
    g = modewise.test_glm(table, cells, [[1, -1, 0], [0, 1, -1]], variance_groups="site", n_permutations=99)
    assert g.statistic == "G" and g.regressors == ["x0", "x1", "x2"]
    t = modewise.test_glm(table, cells, [1, -1, 0], n_permutations=99)
    assert t.statistic == "t" and np.all(np.isfinite(t.p_parametric))

    # Two runs per subject: whole subjects move together.
    runs = modewise.FeatureTable(np.concatenate([values, values + 0.1]), ["x", "y"],
                                 [f"sub-{i:02d}_run-{r}" for r in (1, 2) for i in range(n)],
                                 subjects=[f"{i:02d}" for _ in (1, 2) for i in range(n)])
    design = np.column_stack([np.ones(2 * n), np.tile(np.arange(n) < 6, 2)])
    with pytest.raises(ValueError, match="test_glm with blocks"):
        modewise.test_glm(runs, design, [0, 1])
    whole = modewise.test_glm(runs, design, [0, 1], blocks="subject", whole_blocks=True, n_permutations=99)
    assert whole.n_observations == 24
    flips = modewise.test_glm(table, np.ones((n, 1)), [1], exchange="signflip", n_permutations=5000)
    assert flips.exhaustive and flips.n_rearrangements == 2**n - 1
    with pytest.raises(ValueError, match="exchange"):
        modewise.test_glm(table, cells, [1, -1, 0], exchange="rotate")
    with pytest.raises(ValueError, match="1-D or 2-D"):
        modewise.test_glm(table, cells, np.ones((1, 1, 3)))
    with pytest.raises(ValueError, match="no direction"):
        modewise.test_glm(table, cells, [[1, -1, 0], [0, 1, -1]], tail="greater")
