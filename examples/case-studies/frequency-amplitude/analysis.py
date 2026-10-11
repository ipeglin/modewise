"""Frequency-specific amplitude in resting-state BOLD (replicating Zuo et al. 2010).

Zuo et al. found that the slow-4 (0.027–0.073 Hz) and slow-5 (0.01–0.027 Hz) bands weigh
differently across the brain: relatively more slow-4 in the basal ganglia, more slow-5 in
cortex, most in default-mode regions. This runs the same comparison on the first resting-state
run of every TCP (ds005237) subject.

Usage: python examples/case-studies/fetch_tcp.py restAP
       python examples/case-studies/frequency-amplitude/analysis.py [--workers N]

Writes REPORT.md next to this file, stamped with the modewise version and commit that made it.
"""

import argparse
import datetime
import subprocess
from pathlib import Path

import numpy as np

import modewise

HERE = Path(__file__).parent
BANDS = {"slow4": (0.027, 0.073), "slow5": (0.01, 0.027)}


def groups(table: modewise.ChannelTable) -> dict[str, np.ndarray]:
    network = np.array([t.get("tcp_network", "") for t in table.tags])
    cortex = np.array([n.startswith("17networks") for n in table.names])
    out = {"cortex": cortex}
    for g in ("striatum", "thalamus", "medial_temporal_lobe"):
        out[g] = network == g
    out["cerebellum"] = np.char.startswith(network.astype(str), "cerebellum")
    return out


def stamp(subjects: int) -> str:
    """The provenance line tests/docs reads to flag a report made by older code."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    today = datetime.date.today().isoformat()
    return f"<!-- stamp: modewise {modewise.__version__}, commit {commit or 'unknown'}, {today}, {subjects} subjects -->"


def contrast_test(ratio: np.ndarray, region: dict[str, np.ndarray], subjects: list[str]) -> str:
    """Striatum against cortex, each subject's mean log ratio, as a paired sign-flip test."""
    rows = np.concatenate([ratio[:, region["striatum"]].mean(1), ratio[:, region["cortex"]].mean(1)])[:, None]
    table = modewise.FeatureTable(rows, ["log_ratio"], subjects * 2, subjects=subjects * 2,
                                  conditions=["striatum"] * len(subjects) + ["cortex"] * len(subjects))
    report = modewise.test_paired(table, ("striatum", "cortex"), n_permutations=10000, seed=1)
    return f"paired t = {report.t[0]:.1f}, permutation p = {report.p_permutation[0]:.4f}"


def mean_se(x: np.ndarray) -> str:
    return f"{x.mean():+.3f} ± {x.std(ddof=1) / np.sqrt(len(x)):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    dataset = modewise.Dataset.from_file(HERE / "dataset.toml")
    pipeline = modewise.Pipeline.from_file(HERE / "pipeline.toml")
    out = HERE / "results"
    batch = pipeline.run_dataset(dataset, out, workers=args.workers, progress=False)
    results = modewise.Results(out)
    channels = modewise.read_channels(HERE.parent.parent / "tcp" / "channels.tsv", "tsv")
    region = groups(channels)
    lines = [
        "# Frequency-specific amplitude: report",
        "",
        stamp(len(dataset.recordings)),
        "",
        f"{len(dataset.recordings)} subjects of TCP (ds005237), first restAP run. Method: `pipeline.toml` and `analysis.py`.",
        "",
    ]
    if batch.failed:
        lines += [f"Failed recordings: {batch.failed}", ""]

    # 1. Zuo et al.'s test: slow-4 against slow-5 ALFF in every region, paired across subjects.
    alff = results.feature_table("amplitude", "psd", "band_amplitude", bands=BANDS, participants=dataset.participants)
    report = modewise.test_paired(alff, ("slow4", "slow5"), n_permutations=5000, seed=1)
    significant = report.p_fwer < 0.05
    lines += [
        "## Slow-4 vs slow-5 ALFF per region (paired sign-flip test, FWER over 434 regions)",
        "",
        f"- slow-5 > slow-4: {np.sum(significant & (report.t < 0))} regions",
        f"- slow-4 > slow-5: {np.sum(significant & (report.t > 0))} regions",
        f"- no FWER-significant difference: {np.sum(~significant)} regions",
        "",
        "| region | t (slow4 − slow5) | FWER p |",
        "|---|---|---|",
    ]
    for i in np.argsort(report.t)[::-1][:8]:
        lines.append(f"| {report.feature_names[i]} | {report.t[i]:+.1f} | {report.p_fwer[i]:.3f} |")
    lines.append("")

    # 2. The balance between bands, by structure (descriptive).
    slow4 = alff.values[[i for i, c in enumerate(alff.conditions) if c == "slow4"]]
    slow5 = alff.values[[i for i, c in enumerate(alff.conditions) if c == "slow5"]]
    ratio = np.log(slow4 / slow5)
    lines += ["## log(slow-4 / slow-5 ALFF) by structure (mean ± SE over subjects)", "", "| structure | regions | ALFF ratio |", "|---|---|---|"]
    for name, mask in region.items():
        lines.append(f"| {name} | {mask.sum()} | {mean_se(ratio[:, mask].mean(axis=1))} |")
    contrast = ratio[:, region["striatum"]].mean(1) - ratio[:, region["cortex"]].mean(1)
    subjects = [r for r, c in zip(alff.recordings, alff.conditions) if c == "slow4"]
    lines += ["", f"Striatum − cortex: {mean_se(contrast)}, positive in {np.mean(contrast > 0):.0%} of subjects; {contrast_test(ratio, region, subjects)}.", ""]

    # 3. The same contrast from the wavelet MRA: variance of D4 (≈ slow-4) over D6 (≈ slow-5).
    power = results.feature_table("amplitude", "octaves", "mode_power")
    names = power.feature_names
    d6 = power.values[:, [names.index(f"m1:{c}") for c in channels.names]]
    d4 = power.values[:, [names.index(f"m3:{c}") for c in channels.names]]
    wavelet = np.log(np.sqrt(d4 / d6))
    w_contrast = wavelet[:, region["striatum"]].mean(1) - wavelet[:, region["cortex"]].mean(1)
    agreement = np.corrcoef(ratio.mean(0), wavelet.mean(0))[0, 1]
    lines += [
        "## Second method: MODWT MRA (sym4), log(SD of D4 / SD of D6)",
        "",
        f"Striatum − cortex: {mean_se(w_contrast)}, positive in {np.mean(w_contrast > 0):.0%} of subjects; {contrast_test(wavelet, region, power.recordings)}.",
        f"Across regions, the wavelet ratio correlates with the ALFF ratio at r = {agreement:.2f}.",
        "",
    ]

    # 4. Networks: which cortical networks lean most to slow-5.
    yeo = np.array([t.get("yeo17", "") for t in channels.tags])
    by_net = sorted((ratio[:, yeo == n].mean(), n) for n in set(yeo) - {""})
    lines += ["## Cortical networks, most slow-5 first", "", " · ".join(f"{n} {v:+.2f}" for v, n in by_net), ""]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
