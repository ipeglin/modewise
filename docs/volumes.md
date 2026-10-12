# Volumes

How modewise turns a 4D NIfTI run into region signals, so pipelines can start from preprocessed volumes (for example fMRIPrep output) instead of parcellated files.

## Parcellation

Each region's signal is the mean over its atlas label's voxels, at every volume, as nilearn's
`NiftiLabelsMasker` computes it (Abraham et al., 2014):

- **Resampling:** the atlas is resampled onto the data grid by nearest neighbour, through both
  images' voxel-to-world affines. A data voxel that falls outside the atlas is background.
- **Affines:** each image's affine is its sform when the sform code is set, otherwise its qform,
  as the NIfTI-1 standard specifies.
- **Labels:** 0 is background, and regions come out in ascending label order.
- **Standardising:** `standardize = "zscore_sample"` subtracts each region's mean and divides by
  its sample standard deviation. It is off by default.

**Labels with no voxel on the data grid** (an atlas that does not cover the field of view) give a NaN channel and a warning. Keeping the channel keeps names from a label table aligned. Recent nilearn versions drop such labels, which would shift every later name.

**Several atlases** (for example Schaefer cortex and Tian subcortex) are parcellated one after the other and concatenated in the order given. Name the channels with a `[channels]` section that lists the matching label tables in the same order ([regions](regions.md)).

**Volumes are not stored.** A run is parcellated when it is loaded, as maskers do; a whole BOLD run would be gigabytes per recording.

## Use

In a dataset spec ([datasets](datasets.md)):

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

## References

- Abraham, A., Pedregosa, F., Eickenberg, M., et al. (2014). Machine learning for neuroimaging with scikit-learn. *Frontiers in Neuroinformatics*, 8, 14. https://doi.org/10.3389/fninf.2014.00014
- Cox, R. W., et al. (2004). A (sort of) new image data format standard: NIfTI-1. *NeuroImage*, 22(Supplement 1), e1440.
