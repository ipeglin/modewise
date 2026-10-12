# Regions

How modewise describes channels, selects regions and summarises them. The design works for any atlas, and for any modality: an fMRI parcel, an EEG electrode and an fNIRS optode are all channels.

## Describing channels

Each channel has:

- a **name**, which is what the data's labels refer to
- a **region**: an open vocabulary, as the atlas states it (`PFCm`, `lAMY`, `frontopolar`). Atlases share almost no region names, so there is no fixed list.
- a **hemisphere**: left, right, or none. Midline structures such as the vermis, the brainstem and EEG Cz have none.
- **tags**: open key–value pairs, so several region definitions can sit side by side. Examples: `yeo17 = DefaultB`, `yeo7 = Default`, `lobe = frontal`, `tcp_network = striatum`.

## Atlas label files

| `format` | Reads | Gives |
|---|---|---|
| `tsv` | a tab-separated table with a header: `name` (required), plus optional `region` and `hemisphere` columns. `index` and `color` columns are ignored; every other column becomes a tag. This covers BIDS `dseg.tsv`. | whatever the columns say |
| `schaefer` | Schaefer and Yan (homotopic) labels such as `17Networks_LH_DefaultB_PFCm_1`, one per line or as a column of a lookup table (the `_info.txt` and LUT files both work) | hemisphere, a `yeo7`/`yeo17` tag, and the region where the label has one (somatomotor parcels often have only a network) |
| `tian` | Tian subcortex labels such as `NAc-shell-rh`, one per line | region and hemisphere |
| `names` | one name per line | name only |

A dataset spec lists tables in channel order and concatenates them, for example cortex, then subcortex, then cerebellum (see `docs/datasets.md`). In Python, use `modewise.read_channels(path, format)`, `ChannelTable.from_records([...])`, and `+` to concatenate.

## Selecting

The `select` step keeps the channels that match any of its rules. Within a rule, every field given must match:

```toml
[[step]]
id = "roi"
op = "select"
input = { data = "modes", channels = "channels" }
params = { rules = [
  { region = "PFCv" },
  { region = "PFCm", hemisphere = "left" },
  { region = "*AMY*" },                      # lAMY, mAMY: * matches any text
  { tags = { yeo17 = "Limbic*" } },
] }
```

A rule that names a hemisphere leaves out midline channels. A rule on a region or tag leaves out channels that lack that description. An empty rule (`{}`) selects every channel.

`select` works on series, decompositions, instantaneous attributes, time–frequency maps, epochs, functional connectivity (rows and columns), loadings and channel tables. The output has the same kind as the input.

In Python, `ChannelTable.select` returns the names a set of rules keeps:

```python
import modewise

table = modewise.ChannelTable.from_records([
    {"name": "PFCm_L", "region": "PFCm", "hemisphere": "left", "tags": {"yeo17": "DefaultA"}},
    {"name": "PFCm_R", "region": "PFCm", "hemisphere": "right", "tags": {"yeo17": "DefaultA"}},
    {"name": "lAMY-lh", "region": "lAMY", "hemisphere": "left"},
    {"name": "V1_L", "region": "Striate", "hemisphere": "left", "tags": {"yeo17": "VisCent"}},
])
print(table.select([{"region": "*AMY*"}, {"tags": {"yeo17": "Default*"}, "hemisphere": "left"}]))
# ['PFCm_L', 'lAMY-lh']
```

## Selecting before or after decomposition

Both orders are supported; where `select` sits in the pipeline decides which:

- **Decompose all channels, then select** (`mvmd` → `select`). Modes are fitted to the whole brain, so they are the same whichever regions you study.
- **Select, then decompose** (`select` → `mvmd`). Modes are fitted to the selected regions only.

For slow bands the order changes little. On one ds005237 run with the MVMD settings of `examples/tcp/thesis.toml` (K = 10), decomposing 434 channels and decomposing 17 selected channels gave centre frequencies within about 0.02 Hz, and matching modes of the selected regions correlated at a median of 0.99–1.00 up to 0.2 Hz (0.86–0.89 between 0.30 and 0.46 Hz). Prefer whole-brain-first when results from different region sets must be comparable.

## Summarising regions

The `region_average` step averages channels within each group:

| `by` | Groups by | Example output labels |
|---|---|---|
| `region` | region name | `PFCm`, `lAMY` |
| `hemisphere` | left and right | `left`, `right` |
| `tag` (with `tag = "yeo17"`) | a tag | `DefaultB`, `LimbicA` |

`split_hemispheres = true` keeps the sides apart (`PFCm_left`, `PFCm_right`). Channels without the grouping description are left out.

- **Signals, modes, time–frequency maps and epochs:** the mean over the group's channels. This is the "mean" strategy of nilearn's labels masker and of the TCP parcellation.
- **Functional connectivity:** group × group matrices, each the mean over all pairs of distinct channels, one from each group. The diagonal gives within-group connectivity; the rest gives between-group connectivity. Pearson r is averaged as Fisher z, and the result is reported as z (see `docs/connectivity.md`). Fisher z and PLV are averaged as they are. Pairs with NaN are skipped.
- **Instantaneous attributes:** not averaged, because a mean of instantaneous frequencies across channels has no clear meaning.

Averaging signals and then computing connectivity is not the same as computing connectivity and then averaging it. The first gives the coupling of region-mean signals; the second gives the mean coupling between the regions' channels. Choose by the question.

## Statistical contrasts

Comparing regions or groups statistically (Welch t, permutation tests, FDR, the network-based statistic) belongs to S8.

## ds005237

`examples/tcp/channels.tsv` describes the 434 parcellated channels of the Transdiagnostic Connectome Project:

| Channels | Atlas |
|---|---|
| 0–399 | homotopic Schaefer 400 with Yeo 17 networks (Yan et al. 2023, NeuroImage 273:120010): left 0–199, right 200–399 |
| 400–431 | Tian subcortex, scale II (right hemisphere first) |
| 432–433 | cerebellum, left then right (Buckner atlas) |

The layout comes from the TCP data descriptor (Scientific Data, 2025), and the names and order from the TCP processing repository's label files. `examples/tcp/make_channels.py` rebuilds the table. It reads the pickled name list with `pickletools`, so nothing from the download is executed.
