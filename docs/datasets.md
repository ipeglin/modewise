# Datasets

A dataset spec tells modewise where recordings are and how to read them. It is separate from the pipeline, so one pipeline runs on many datasets. It works for BIDS and for other layouts, and for any modality.

```toml
root = "/data/ds005237"              # relative paths resolve against this file's folder
filter = { task = ["hammerAP"] }     # optional: keep only these entity values

[[input]]                            # the first input decides which recordings exist
name = "bold"                        # must match a pipeline input
pattern = "sub-{subject}/func/sub-{subject}_task-{task}_run-{run}_bold.h5"
format = "hdf5"
dataset = "timeseries"               # path of the 2D dataset inside the file
layout = "channels_by_time"          # or "time_by_channels"
sidecar = "sub-{subject}/func/sub-{subject}_task-{task}_run-{run}_bold.json"
# sampling_rate_hz = 1.25            # instead of `sidecar`
# labels_file = "channels.txt"       # one channel label per line

[[input]]                            # later inputs must exist for every recording
name = "trials"
pattern = "sub-{subject}/func/sub-{subject}_task-{task}_run-{run}_events.tsv"
format = "events_tsv"
# label_column = "trial_type"        # default
```

## Patterns

- **Placeholders.** `{name}` stands for an entity value, and the same name must match the same value everywhere in the pattern.
- **Recording IDs** are made from the entities, for example `subject-01_task-rest_run-01`.
- **Labels.** Use BIDS-style labels (letters and digits). On case-insensitive disks, the macOS default, `sub-A` and `sub-a` are the same folder.

## Formats

| `format` | Gives | Needs |
|---|---|---|
| `hdf5` | series | `dataset`, plus `sidecar` or `sampling_rate_hz` |
| `nifti` | series: one channel per atlas label, parcellated at load time (`docs/volumes.md`) | `atlases` (paths relative to the spec's folder); optional `standardize`; `sidecar` or `sampling_rate_hz`, else the header's TR |
| `events_tsv` | events | nothing else |

## Sampling rate and events

- **Sidecar:** `SamplingFrequency` (Hz) is read first, otherwise `RepetitionTime` (s), so the same rule serves EEG, MEG, fNIRS and fMRI.
- **`events_tsv`** follows BIDS. `onset` and `duration` are in seconds from the first stored sample. Rows with `n/a` onset or duration are rejected.

## Channels

A `[channels]` section says what each channel is, so pipelines can select and summarise regions (`docs/regions.md`):

```toml
[channels]
name = "channels"                                   # pipeline input name (default)
tables = [                                           # in channel order; concatenated
  { path = "atlas/schaefer400_info.txt", format = "schaefer" },
  { path = "atlas/tian_s2.txt", format = "tian" },
]
names_for = ["bold"]                                 # these inputs take the table's names
```

- **Paths:** relative to the dataset spec's folder, because atlases usually live outside the data.
- **Formats:** `tsv` (covers BIDS `dseg.tsv`), `schaefer`, `tian` and `names`.
- **Names:** inputs in `names_for` get the table's names instead of `ch0`, `ch1`, …. The number of names must match the channels.
- **Pipeline input:** the table becomes one more input (`channels = "channels"` in a pipeline's `inputs`). Pipelines that do not use it are unaffected, because unused dataset inputs are left out.

## Participants

A `[participants]` section reads subject information for group statistics and machine learning (`docs/features.md`):

```toml
[participants]
entity = "subject"                       # the recording entity tables are keyed by (default)
tables = [
  { path = "participants.tsv", id_pattern = "sub-{subject}" },
  { path = "phenotype/shaps01.tsv", id_column = "subjectkey", id_pattern = "NDAR_{subject}",
    columns = ["shaps_total"], missing = ["999"] },
]
```

- **Paths:** relative to the dataset root, as BIDS places them.
- **Ids:** `id_column` defaults to `participant_id`. `id_pattern` says how ids relate to the entity, in the same `{entity}` syntax as file patterns; by default the id is the entity value.
- **Columns and missing values:** `columns` keeps only the listed columns. `missing` adds codes meaning "no value", besides empty cells, `n/a` and `NA`.
- **Joins:** tables are joined on the subject. A column name that appears in two tables is an error unless one table sets `prefix`, for example `prefix = "shaps."`.

## Checking a spec

`modewise inspect dataset.toml --load` lists the entities found, and loads the first recording to show shapes, sampling rates and event counts.
