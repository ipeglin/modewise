"""Connectome fingerprinting (replicating Finn et al. 2015).

Finn et al. showed that a person's functional connectivity is distinctive enough to pick them out
of a group: correlate a target scan's edges with every scan in a database from another session,
and the best match is usually the same person. Here the database is each TCP subject's first
restAP run and the target their first restPA run (and the reverse), for the full band and for two
octave bands from the MODWT MRA.

Usage: python examples/case-studies/fetch_tcp.py restAP restPA
       python examples/case-studies/fingerprinting/analysis.py [--workers N]

Writes REPORT.md next to this file, stamped with the modewise version and commit that made it.
"""

import argparse
import datetime
import subprocess
from pathlib import Path

import numpy as np

import modewise

HERE = Path(__file__).parent
BANDS = {"full": "full band", "slow_fc": "D6, about 0.010–0.020 Hz", "fast_fc": "D4, about 0.039–0.078 Hz"}


def stamp(subjects: int) -> str:
    """The provenance line tests/docs reads to flag a report made by older code."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    today = datetime.date.today().isoformat()
    return f"<!-- stamp: modewise {modewise.__version__}, commit {commit or 'unknown'}, {today}, {subjects} subjects -->"


def edges(results: modewise.Results, recording: str, output: str) -> np.ndarray:
    """The upper triangle of a recording's connectivity matrix, Fisher z, as float32."""
    r = results.output("connectomes", recording, output).values[0, 0]
    upper = np.triu_indices(r.shape[0], 1)
    return np.arctanh(np.clip(r[upper], -0.999999, 0.999999)).astype(np.float32)


def best_matches(database: np.ndarray, target: np.ndarray) -> np.ndarray:
    """For each target row, the database row it correlates with most."""
    z = lambda m: (m - m.mean(1, keepdims=True)) / m.std(1, keepdims=True)
    return np.argmax(z(target) @ z(database).T, axis=1)


def null_accuracy(best: np.ndarray, rng: np.random.Generator, shuffles: int) -> float:
    """The highest accuracy among shuffles of which database row belongs to whom. Shuffling rows
    only relabels the best matches, so they need not be recomputed."""
    n = len(best)
    return max(float(np.mean(np.argsort(rng.permutation(n))[best] == np.arange(n))) for _ in range(shuffles))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    dataset = modewise.Dataset.from_file(HERE / "dataset.toml")
    pipeline = modewise.Pipeline.from_file(HERE / "pipeline.toml")
    batch = pipeline.run_dataset(dataset, HERE / "results", workers=args.workers, progress=False)
    results = modewise.Results(HERE / "results")

    runs: dict[str, dict[str, str]] = {}
    for recording in sorted(batch.succeeded):
        entities = dataset.entities(recording)
        runs.setdefault(entities["subject"], {})[entities["task"]] = recording
    subjects = sorted(s for s, r in runs.items() if {"restAP", "restPA"} <= set(r))
    n = len(subjects)
    rng = np.random.default_rng(1)
    lines = ["# Connectome fingerprinting: report", "", stamp(n), "",
             f"{n} subjects of TCP (ds005237): first restAP run against first restPA run. "
             f"Chance is 1/{n} = {1 / n:.1%}. Method: `pipeline.toml` and `analysis.py`.", "",
             "| band | restAP → restPA | restPA → restAP | best chance-level accuracy in 1000 shuffles |",
             "|---|---|---|---|"]
    for output, label in BANDS.items():
        ap = np.stack([edges(results, runs[s]["restAP"], output) for s in subjects])
        pa = np.stack([edges(results, runs[s]["restPA"], output) for s in subjects])
        to_pa, to_ap = best_matches(ap, pa), best_matches(pa, ap)
        forward, backward = np.mean(to_pa == np.arange(n)), np.mean(to_ap == np.arange(n))
        null = null_accuracy(to_pa, rng, 1000)
        lines.append(f"| {label} | {forward:.1%} | {backward:.1%} | {null:.1%} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Identification correlates Fisher-z edge vectors (434 regions, 93,961 edges); a target is "
        "identified when its best match is the same subject, as in Finn et al. (2015).",
        "- Finn et al. reported 92.9% and 94.4% between two 14-minute resting runs recorded on "
        "different days, with 268 regions and no global signal regression. Here the runs are 6.5 "
        "minutes from one session with opposite phase encoding (AP and PA), with 434 regions and "
        "global signal regression, so the numbers are not directly comparable.",
        "- A 6.5-minute run holds only about 4 to 8 cycles at 0.010–0.020 Hz, so connectivity in "
        "that band is estimated from little data; its lower accuracy says as much about estimation "
        "as about how individual the band is.",
        "",
    ]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
