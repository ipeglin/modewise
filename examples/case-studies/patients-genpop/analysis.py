"""Patients versus the general population: resting-state connectivity per band (exploratory).

The Transdiagnostic Connectome Project (TCP) recruited patients with a range of psychiatric
diagnoses and a general-population comparison group. This compares their resting-state functional
connectivity edge by edge, for the whole signal and for three octave bands of the MODWT MRA, with
the network-based statistic (Zalesky et al. 2010): edges whose group difference passes a t
threshold form connected components, and each component is tested against the largest component
in permuted data. Age, sex, scanning site and head motion (mean framewise displacement) are
covariates. There is no hypothesis here; the report states whatever comes out.

Usage: python examples/case-studies/fetch_tcp.py restAP
       python examples/case-studies/patients-genpop/analysis.py [--workers N] [--permutations N]

Writes REPORT.md next to this file, stamped with the modewise version and commit that made it.
"""

import argparse
import csv
import datetime
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

import modewise

HERE = Path(__file__).parent
BANDS = {
    "full": "full band",
    "d6_fc": "D6, about 0.010–0.020 Hz",
    "d5_fc": "D5, about 0.020–0.039 Hz",
    "d4_fc": "D4, about 0.039–0.078 Hz",
}
THRESHOLDS = (3.0, 3.5)
DIRECTIONS = {"less": "Patient < GenPop", "greater": "Patient > GenPop"}
COVARIATES = ["age", "sex", "Site", "mean_fd"]
FD_TABLE = HERE.parent / "data" / "ds005237" / "motion_FD" / "TCP_FD_rest_AP_1.csv"


def stamp(subjects: int) -> str:
    """The provenance line tests/docs reads to flag a report made by older code."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    today = datetime.date.today().isoformat()
    return f"<!-- stamp: modewise {modewise.__version__}, commit {commit or 'unknown'}, {today}, {subjects} subjects -->"


def networks() -> dict[str, str]:
    """Each channel's network: its Yeo 17-network label for cortex, its TCP network otherwise."""
    with open(HERE.parent.parent / "tcp" / "channels.tsv", newline="") as f:
        return {row["name"]: row["yeo17"] or row["tcp_network"] for row in csv.DictReader(f, delimiter="\t")}


def mean_fd() -> dict[str, float]:
    """Each subject's mean framewise displacement (mm) in the first restAP run. The table has one
    row per subject (NDAR_<subject>) and one value per volume. Its first three values are 0 for
    every subject, which scales every mean by the same factor and leaves the tests unchanged."""
    with open(FD_TABLE, newline="", encoding="utf-8-sig") as f:
        return {row[0].removeprefix("NDAR_"): float(np.mean([float(v) for v in row[1:]])) for row in csv.reader(f)}


def with_mean_fd(table: modewise.FeatureTable, fd: dict[str, float]) -> modewise.FeatureTable:
    """The table with a mean_fd column for its covariates."""
    metadata = {**table.metadata, "mean_fd": [str(fd[s]) for s in table.subjects]}
    return modewise.FeatureTable(table.values, table.feature_names, table.recordings, subjects=table.subjects,
                                 conditions=table.conditions, metadata=metadata)


def finite_edges(table: modewise.FeatureTable) -> tuple[modewise.FeatureTable, list[str]]:
    """Drops edges that are not finite in some recording (Fisher z of r = ±1)."""
    keep = np.isfinite(table.values).all(axis=0)
    dropped = [n for n, k in zip(table.feature_names, keep) if not k]
    kept = modewise.FeatureTable(
        table.values[:, keep], [n for n, k in zip(table.feature_names, keep) if k], table.recordings,
        subjects=table.subjects, conditions=table.conditions, metadata=table.metadata)
    return kept, dropped


def describe(edges: list[str], network_of: dict[str, str]) -> str:
    """The component's size, its most connected regions and its most common network pairs."""
    pairs = [name.split(":", 1)[1].split("~") for name in edges]
    degree = Counter(node for pair in pairs for node in pair)
    hubs = ", ".join(f"{node} ({count})" for node, count in degree.most_common(3))
    blocks = Counter(" – ".join(sorted((network_of[a], network_of[b]))) for a, b in pairs)
    common = ", ".join(f"{block} ({count})" for block, count in blocks.most_common(3))
    return f"{len(edges)} edges, {len(degree)} regions; hubs {hubs}; network pairs {common}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    dataset = modewise.Dataset.from_file(HERE / "dataset.toml")
    pipeline = modewise.Pipeline.from_file(HERE / "pipeline.toml")
    pipeline.run_dataset(dataset, HERE / "results", workers=args.workers, progress=False)
    results = modewise.Results(HERE / "results")
    network_of = networks()
    fd = mean_fd()

    full = results.feature_table("rest_fc", "full", "edges", participants=dataset.participants)
    groups = Counter(full.metadata["Group"])
    fd_by_group = {g: np.median([fd[s] for s, m in zip(full.subjects, full.metadata["Group"]) if m == g])
                   for g in ("Patient", "GenPop")}
    lines = ["# Patients versus the general population: report", "", stamp(len(full.recordings)), "",
             f"{len(full.recordings)} subjects of TCP (ds005237), first restAP run: "
             f"{groups['Patient']} patients and {groups['GenPop']} from the general population. "
             f"Network-based statistic on Fisher-z edges, covariates {', '.join(COVARIATES)}, "
             f"{args.permutations} permutations. Median mean framewise displacement: "
             f"{fd_by_group['Patient']:.3f} mm in patients, {fd_by_group['GenPop']:.3f} mm in the general "
             "population. Method: `pipeline.toml` and `analysis.py`.", "",
             "| band | direction | t threshold | components | largest component (p) |",
             "|---|---|---|---|---|"]
    found: list[str] = []
    dropped_edges: set[str] = set()
    for output, label in BANDS.items():
        table, dropped = finite_edges(results.feature_table("rest_fc", output, "edges", participants=dataset.participants))
        table = with_mean_fd(table, fd)
        dropped_edges.update(name.split(":", 1)[1] for name in dropped)
        for tail, direction in DIRECTIONS.items():
            for threshold in THRESHOLDS:
                report = modewise.test_groups(table, "Group", ("Patient", "GenPop"), covariates=COVARIATES,
                                              tail=tail, n_permutations=args.permutations, nbs_threshold=threshold)
                components = sorted(report.components, key=lambda c: (c["p"], -len(c["edges"])))
                largest = f"{len(components[0]['edges'])} edges ({components[0]['p']:.3f})" if components else "none"
                lines.append(f"| {label} | {direction} | {threshold} | {len(components)} | {largest} |")
                for c in (c for c in components if c["p"] < 0.05):
                    found.append(f"- {label}, {direction}, t > {threshold}: p = {c['p']:.3f}; {describe(c['edges'], network_of)}")
    lines += ["", "## Components with p < 0.05", ""] + (found or ["None."])
    lines += [
        "",
        "## Notes",
        "",
        "- Each component's p is family-wise over all edges of that band, for one direction and one "
        "threshold. Sixteen tests are reported (four bands, two directions, two thresholds), "
        "without correction across them, so treat a single component near p = 0.05 with care.",
        "- The t threshold is a free choice of the network-based statistic; results at both "
        "thresholds are shown so the dependence is visible.",
        f"- Dropped edges, not finite in some recording: {', '.join(sorted(dropped_edges)) or 'none'}. "
        "In the TCP parcellated files the two cerebellum channels hold identical signals, so their "
        "correlation is 1 and its Fisher z infinite.",
        "- Head motion changes connectivity and often differs between groups. The files are "
        "denoised with global signal regression, and each subject's mean framewise displacement, "
        "from the dataset's motion_FD table, is a covariate.",
        "- Patients in the TCP have a range of diagnoses; a difference between the groups need not "
        "hold for any single diagnosis.",
        "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
