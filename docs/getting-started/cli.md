# Command line

Installing the Python package also installs `modewise`; `cargo install --path drivers/cli` builds
it from source. It runs the same pipelines and dataset specs as Python, which suits shell scripts
and HPC clusters.

```bash
modewise inspect dataset.toml --load        # what the dataset spec finds, and the first recording
modewise run pipeline.toml --dataset dataset.toml --out results/
modewise run pipeline.toml --dataset dataset.toml --out results/ --shard 3/8   # every 8th recording from #3
modewise export results/ --to export/       # one HDF5 file per recording, for MATLAB
modewise ops                                # operations with their ports
modewise ops --json                         # the same with parameter schemas, for tools and agents
```

**Exit codes:** `0` when every recording succeeded, `2` when some failed (the summary is in
`results/runs/<pipeline>/summaries/`), and `1` for invalid input.

**Slurm:** run one shard per array task, all writing to the same results folder. Steps already in
the folder are reused, so a rerun after a failure only computes what is missing.

```bash
#SBATCH --array=1-8
modewise run pipeline.toml --dataset dataset.toml --out results/ --shard ${SLURM_ARRAY_TASK_ID}/8
```

[Pipelines](../pipelines.md) and [datasets](../datasets.md) describe the two files, and the
[operation reference](../reference/operations.md) every step you can use.
