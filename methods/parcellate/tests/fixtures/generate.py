"""Regenerate parcellation fixtures with nilearn's NiftiLabelsMasker (resampling_target="data",
strategy="mean"), on synthetic volumes whose atlas has a different grid and affine than the data.

Usage: uv run --no-project -p 3.12 --with nilearn --with nibabel --with numpy python generate.py
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiLabelsMasker

here = Path(__file__).parent
rng = np.random.default_rng(4)

# Data: 6×7×5 voxels of 2 mm, 20 volumes, TR 0.8 s.
data_affine = np.array([[2.0, 0, 0, -6.0], [0, 2.0, 0, -7.0], [0, 0, 2.0, -4.0], [0, 0, 0, 1]])
bold = rng.standard_normal((6, 7, 5, 20)).astype(np.float32) + np.arange(20, dtype=np.float32) * 0.1
img = nib.Nifti1Image(bold, data_affine)
img.header.set_xyzt_units("mm", "sec")
img.header["pixdim"][4] = 0.8
nib.save(img, here / "bold.nii.gz")

# Atlas: 4×5×4 voxels of 3 mm, shifted, labels 0 (background) and 1..5 in blocks.
atlas_affine = np.array([[3.0, 0, 0, -5.0], [0, 3.0, 0, -6.5], [0, 0, 3.0, -3.0], [0, 0, 0, 1]])
labels = np.zeros((4, 5, 4), dtype=np.int16)
labels[:2, :2, :] = 1
labels[2:, :2, :] = 2
labels[:2, 2:, :2] = 3
labels[2:, 2:, :2] = 4
labels[:, 2:, 2:] = 5
labels[0, 0, 0] = 0
nib.save(nib.Nifti1Image(labels, atlas_affine), here / "atlas.nii.gz")

masker = NiftiLabelsMasker(here / "atlas.nii.gz", resampling_target="data", strategy="mean", standardize=None)
signals = masker.fit_transform(here / "bold.nii.gz")
z = NiftiLabelsMasker(here / "atlas.nii.gz", resampling_target="data", strategy="mean", standardize="zscore_sample").fit_transform(here / "bold.nii.gz")

# The same atlas stored with only a qform (sform code 0).
qform_only = nib.Nifti1Image(labels, None)
qform_only.set_qform(atlas_affine, code=1)
qform_only.set_sform(None, code=0)
nib.save(qform_only, here / "atlas_qform.nii.gz")

# Headers for the NIfTI reader: a qform with rotation and qfac = -1, a TR in milliseconds,
# and an atlas stored as 4D with one volume. nibabel gives the expected affines.
def axis_rotation(axis, degrees):
    t = np.deg2rad(degrees)
    c, s_ = np.cos(t), np.sin(t)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s_, s_, c
    return r


# An oblique rotation, so every quaternion term matters.
rotation = axis_rotation(0, 20) @ axis_rotation(1, -35) @ axis_rotation(2, 30)
rotated = np.eye(4)
rotated[:3, :3] = rotation @ np.diag([2.0, 3.0, -4.0])
rotated[:3, 3] = [10.0, -20.0, 30.0]
img = nib.Nifti1Image(labels, None)
img.set_qform(rotated, code=1)
img.set_sform(None, code=0)
nib.save(img, here / "rotated_qform.nii.gz")
ms = nib.Nifti1Image(np.ones((2, 2, 2, 3), np.float32), np.eye(4))
ms.header.set_xyzt_units("mm", "msec")
ms.header["pixdim"][4] = 800.0
nib.save(ms, here / "tr_ms.nii.gz")
nib.save(nib.Nifti1Image(labels[..., None], atlas_affine), here / "labels_4d.nii.gz")
both = nib.Nifti1Image(labels, None)
both.set_qform(rotated, code=1)
both.set_sform(atlas_affine, code=2)
nib.save(both, here / "sform_wins.nii.gz")
for name, units, tr in (("tr_us", "usec", 800000.0), ("tr_zero", "sec", 0.0)):
    img = nib.Nifti1Image(np.ones((2, 2, 2, 3), np.float32), np.eye(4))
    img.header.set_xyzt_units("mm", units)
    img.header["pixdim"][4] = tr
    nib.save(img, here / f"{name}.nii.gz")
headers = {
    "rotated_qform": nib.load(here / "rotated_qform.nii.gz").affine.tolist(),
    "atlas": nib.load(here / "atlas.nii.gz").affine.tolist(),
    "sform_wins": nib.load(here / "sform_wins.nii.gz").affine.tolist(),
}

out = {
    "headers": headers,
    "labels": [int(v) for v in masker.labels_[1:]] if len(masker.labels_) > signals.shape[1] else [int(v) for v in masker.labels_],
    "signals": signals.T.tolist(),
    "zscored": z.T.tolist(),
    "tr": 0.8,
}
(here / "nilearn.json").write_text(json.dumps(out))
print("labels", out["labels"], "signals", signals.shape)
