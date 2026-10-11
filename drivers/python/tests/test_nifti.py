import json
from pathlib import Path

import numpy as np
import pytest

import modewise

FIXTURES = Path(__file__).parents[3] / "methods" / "parcellate" / "tests" / "fixtures"


def test_read_nifti_matches_nilearn_and_concatenates_atlases():
    expected = json.loads((FIXTURES / "nilearn.json").read_text())
    s = modewise.read_nifti(FIXTURES / "bold.nii.gz", [FIXTURES / "atlas.nii.gz"])
    np.testing.assert_allclose(s.data, expected["signals"], atol=1e-5)
    assert s.sampling_rate == pytest.approx(1 / 0.8)
    z = modewise.read_nifti(FIXTURES / "bold.nii.gz", [FIXTURES / "atlas.nii.gz"], standardize="zscore_sample", sampling_rate=2.0)
    np.testing.assert_allclose(z.data, expected["zscored"], atol=1e-5)
    assert z.sampling_rate == 2.0
    both = modewise.read_nifti(FIXTURES / "bold.nii.gz", [FIXTURES / "atlas.nii.gz", FIXTURES / "atlas_qform.nii.gz"],
                               labels=[f"r{i}" for i in range(10)])
    assert both.data.shape == (10, 20) and both.labels[-1] == "r9"
    np.testing.assert_array_equal(both.data[:5], both.data[5:])


def test_read_nifti_errors():
    with pytest.raises(ValueError, match="standardize"):
        modewise.read_nifti(FIXTURES / "bold.nii.gz", [FIXTURES / "atlas.nii.gz"], standardize="robust")
    with pytest.raises(ValueError, match="at least one atlas"):
        modewise.read_nifti(FIXTURES / "bold.nii.gz", [])
    with pytest.raises(ValueError):
        modewise.read_nifti(FIXTURES / "missing.nii.gz", [FIXTURES / "atlas.nii.gz"])
