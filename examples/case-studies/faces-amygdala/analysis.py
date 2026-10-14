"""Faces against shapes in the amygdala (replicating Hariri et al. 2002).

In the face-matching task, people match faces with fearful or angry expressions, or match
shapes. Faces drive the amygdala more than shapes, one of the most replicated results in
affective neuroimaging, along with the ventral visual stream that processes faces. This tests
it on every TCP (ds005237) subject's hammerAP run with block averages and with a task GLM, each
with paired permutation tests, then looks, exploratorily, at whether the amygdala response tracks anhedonia (SHAPS) or
differs between patients and the general population.

Usage: python examples/case-studies/fetch_tcp.py hammerAP
       python examples/case-studies/faces-amygdala/analysis.py [--workers N]

Writes REPORT.md next to this file, stamped with the modewise version and commit that made it.
"""

import argparse
import datetime
import subprocess
from pathlib import Path

import numpy as np

import modewise

HERE = Path(__file__).parent
AMYGDALA = ["lAMY-lh", "mAMY-lh", "lAMY-rh", "mAMY-rh"]


def stamp(subjects: int) -> str:
    """The provenance line tests/docs reads to flag a report made by older code."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    today = datetime.date.today().isoformat()
    return f"<!-- stamp: modewise {modewise.__version__}, commit {commit or 'unknown'}, {today}, {subjects} subjects -->"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    dataset = modewise.Dataset.from_file(HERE / "dataset.toml")
    pipeline = modewise.Pipeline.from_file(HERE / "pipeline.toml")
    batch = pipeline.run_dataset(dataset, HERE / "results", workers=args.workers, progress=False)
    results = modewise.Results(HERE / "results")
    table = results.feature_table("faces", "blocks", "mean", participants=dataset.participants)
    n = len(set(table.recordings))
    lines = ["# Faces against shapes: report", "", stamp(n), "",
             f"{n} subjects of TCP (ds005237), hammerAP run. Method: `pipeline.toml` and `analysis.py`.", ""]
    if batch.failed:
        lines += [
            f"Not analysed: {len(batch.failed)} runs that have parcellated data but no BIDS events or "
            "sidecar on OpenNeuro, so their blocks cannot be defined.",
            "",
        ]

    # 1. Faces against shapes in every region, paired within subject.
    report = modewise.test_paired(table, ("face", "shape"), n_permutations=10000, seed=1)
    significant = report.p_fwer < 0.05
    lines += [
        "## Faces − shapes per region (paired sign-flip test, FWER over 434 regions)",
        "",
        f"- faces > shapes: {np.sum(significant & (report.t > 0))} regions",
        f"- shapes > faces: {np.sum(significant & (report.t < 0))} regions",
        "",
        "| region | t (face − shape) | FWER p |",
        "|---|---|---|",
    ]
    for name in AMYGDALA:
        i = report.feature_names.index(name)
        lines.append(f"| **{name}** | {report.t[i]:+.1f} | {report.p_fwer[i]:.4f} |")
    lines += ["", "Strongest faces > shapes:", "", "| region | t | FWER p |", "|---|---|---|"]
    for i in np.argsort(report.t)[::-1][:10]:
        lines.append(f"| {report.feature_names[i]} | {report.t[i]:+.1f} | {report.p_fwer[i]:.4f} |")
    lines.append("")

    # 2. The same contrast from a task GLM: each subject's face and shape betas, paired.
    betas = results.feature_table("faces", "glm.betas", "coefficients", participants=dataset.participants)
    glm = modewise.test_paired(betas, ("face", "shape"), n_permutations=10000, seed=1)
    assert glm.feature_names == report.feature_names
    glm_significant = glm.p_fwer < 0.05
    lines += [
        "## Faces − shapes from a task GLM (AR(1) noise, paired sign-flip test on betas, FWER over 434 regions)",
        "",
        f"- faces > shapes: {np.sum(glm_significant & (glm.t > 0))} regions",
        f"- shapes > faces: {np.sum(glm_significant & (glm.t < 0))} regions",
        f"- Across regions, the t values of the GLM and of the block averages correlate at r = {np.corrcoef(report.t, glm.t)[0, 1]:.3f}.",
        "",
        "| region | t (face − shape) | FWER p |",
        "|---|---|---|",
    ]
    for name in AMYGDALA:
        i = glm.feature_names.index(name)
        lines.append(f"| **{name}** | {glm.t[i]:+.1f} | {glm.p_fwer[i]:.4f} |")
    lines.append("")

    # 3. Exploratory: does each subject's amygdala response track anhedonia, or differ by group?
    columns = [table.feature_names.index(a) for a in AMYGDALA]
    face = {r: v for r, v, c in zip(table.recordings, table.values[:, columns].mean(1), table.conditions) if c == "face"}
    shape = {r: v for r, v, c in zip(table.recordings, table.values[:, columns].mean(1), table.conditions) if c == "shape"}
    recordings = sorted(face)
    subjects = [s for r, s, c in zip(table.recordings, table.subjects, table.conditions) if c == "face"]
    subject_of = dict(zip([r for r, c in zip(table.recordings, table.conditions) if c == "face"], subjects))
    contrast = modewise.FeatureTable(
        np.array([[face[r] - shape[r]] for r in recordings]), ["amygdala_face_minus_shape"], recordings,
        subjects=[subject_of[r] for r in recordings], participants=dataset.participants)
    shaps = modewise.test_correlation(contrast, "shaps_total", covariates=["age", "sex"], n_permutations=10000, seed=1)
    groups = modewise.test_groups(contrast, "Group", ("Patient", "GenPop"), covariates=["age", "sex"], n_permutations=10000, seed=1)
    lines += [
        "## Exploratory: the amygdala response (mean of four amygdala parcels, face − shape)",
        "",
        f"- Mean response: {contrast.values.mean():+.1f} (SD {contrast.values.std(ddof=1):.1f}, in the TCP files' signal units), positive in {np.mean(contrast.values > 0):.0%} of subjects.",
        f"- Against SHAPS anhedonia (age and sex as covariates): t = {shaps.t[0]:+.2f}, permutation p = {shaps.p_permutation[0]:.3f}, {shaps.n_observations} subjects ({shaps.dropped} without a score).",
        f"- Patients against the general population (age and sex as covariates): t = {groups.t[0]:+.2f}, permutation p = {groups.p_permutation[0]:.3f}, {groups.n_observations} subjects.",
        "",
    ]
    lines += [
        "## Notes",
        "",
        "- The TCP series had the global signal regressed out. That shifts every region's signal by the "
        "brain-wide mean, so regions with no face response tend to show shapes > faces; the count above "
        "reflects this, not a shape response.",
        "- Blocks are averages of the BOLD signal 5 s after each block's events (a fixed lag). The GLM "
        "instead models every trial with the canonical haemodynamic response, so the two methods share "
        "only the data.",
        "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
