# Volumes

How modewise turns a 4D NIfTI run into region signals, so pipelines can start from preprocessed volumes (for example fMRIPrep output) instead of parcellated files.

## Parcellation

Each region's signal is the mean over its atlas label's voxels, at every volume.

| Choice | Sources | Strength |
|---|---|---|
| The atlas is resampled onto the data grid by nearest neighbour, through both voxel-to-world affines | nilearn `NiftiLabelsMasker` (`resampling_target="data"`); the thesis masker | consensus |
| A data voxel whose atlas coordinate lies outside [0, n − 1] on any axis is background; inside, the nearest atlas voxel is ⌊p + ½⌋ | scipy `affine_transform` (order 0), which nilearn and nibabel both use | consensus. The thesis masker rounded first and then checked bounds. That differs only for data voxels within half a voxel outside the atlas's outer edge, which is background in MNI atlases. |
| Label 0 is background; labels come out in ascending order | nilearn; the thesis masker | consensus |
| Mean over voxels | nilearn's default strategy; the thesis masker; the TCP parcellation | consensus |
| The affine is the sform when its code is above 0, otherwise the qform | the NIfTI-1 standard; nibabel | consensus |
| `standardize = "zscore_sample"`: minus the mean, divided by the sample SD (ddof = 1) | nilearn `zscore_sample`; the thesis default | consensus. Off by default, as in nilearn. |

**Labels with no voxel on the data grid** (an atlas that does not cover the field of view) give a NaN channel and a warning. Keeping the channel keeps names from a label table aligned. Recent nilearn versions drop such labels, which would shift every later name.

**Several atlases** (for example Schaefer cortex and Tian subcortex) are parcellated one after the other and concatenated in the order given. Name the channels with a `[channels]` section that lists the matching label tables in the same order (`docs/regions.md`).

**Volumes are not stored.** A run is parcellated when it is loaded, as maskers do; a whole BOLD run would be gigabytes per recording.

## Use

In a dataset spec (`docs/datasets.md`):

```toml
[[input]]
name = "bold"
pattern = "derivatives/fmriprep/sub-{subject}/func/sub-{subject}_task-{task}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
format = "nifti"
atlases = ["atlases/Schaefer400_17Networks.nii.gz", "atlases/Tian_S2.nii"]
standardize = "zscore_sample"
sidecar = "sub-{subject}/func/sub-{subject}_task-{task}_bold.json"   # optional; else the header's TR
```

In Python: `modewise.read_nifti(path, [atlas, ...], standardize="zscore_sample")`.

## Evidence

| Check | Result |
|---|---|
| Region signals vs nilearn `NiftiLabelsMasker`, on an atlas with a coarser, shifted grid (3 mm on 2 mm data), raw and z-scored | within 1e-5 (float32 data) |
| The same atlas stored with only a qform | identical result |
| Repetition time from the header's time units | read correctly |
| An atlas entirely off the data grid | every label NaN, with a warning |
