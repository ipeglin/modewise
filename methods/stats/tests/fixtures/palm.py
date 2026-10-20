"""Regenerate palm.json from FSL PALM (Winkler et al. 2014; 2015), run in GNU Octave, with an
independent numpy enumeration of every rearrangement for the exhaustive cases.

PALM is GPL-3: it is run here to make fixtures, never copied. Its CSV writer rounds to four
decimals, so this script runs a scratch copy with the precision raised.

Usage:
    git clone https://github.com/andersonwinkler/PALM <palm>
    printf 'FROM debian:trixie-slim\\nRUN apt-get update && apt-get install -y --no-install-recommends octave octave-statistics\\n' \\
        | docker build -t modewise-octave -
    uv run --no-project -p 3.12 --with numpy python palm.py <palm>
"""

import glob
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

IMAGE = "modewise-octave"
rng = np.random.default_rng(14)


def palm(src: Path, work: Path, data, design, contrast, flags, *, ftest=False, eb=None, vg=None, exhaustive=True):
    """Runs PALM on one case and returns its statistic, p-values and degrees of freedom. Without
    `exhaustive`, only the statistic and its parametric p are kept."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    np.savetxt(work / "y.csv", data, delimiter=",", fmt="%.17g")
    np.savetxt(work / "d.csv", design, delimiter=",", fmt="%.17g")
    np.savetxt(work / "t.csv", np.atleast_2d(contrast), delimiter=",", fmt="%.17g")
    args = ["-i y.csv", "-d d.csv", "-t t.csv", "-o out", f"-n {1000000 if exhaustive else 10}", "-saveparametric", "-savedof", "-quiet"]
    if ftest:
        np.savetxt(work / "f.csv", np.ones((1, len(contrast))), delimiter=",", fmt="%d")
        args += ["-f f.csv", "-fonly"]
    if eb is not None:
        np.savetxt(work / "eb.csv", eb, fmt="%d")
        args.append("-eb eb.csv")
    if vg is not None:
        np.savetxt(work / "vg.csv", vg, fmt="%d")
        args.append("-vg vg.csv")
    command = f"addpath('/palm'); palm {' '.join(args + flags)}"
    result = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{src}:/palm:ro", "-v", f"{work}:/w", "-w", "/w",
         IMAGE, "octave-cli", "--eval", command],
        capture_output=True, text=True,
    )
    if result.returncode:
        sys.exit(result.stdout + result.stderr)

    def read(suffix):
        (path,) = [p for p in glob.glob(str(work / f"out_dat_*{suffix}")) if "_max" not in p]
        return Path(path)

    stat = read("stat.csv")
    # Denominator degrees of freedom: last value of a text file, or per feature (v, G) in a CSV.
    per_feature = glob.glob(str(work / "out_dat_*stat_dof.csv"))
    if per_feature:
        dof = {"df2": np.loadtxt(per_feature[0], delimiter=",", ndmin=1).tolist()}
    else:
        dof = {"df2": float(read("stat_dof.txt").read_text().split()[-1])}
    out = {
        "statistic": stat.name.removeprefix("out_dat_").removesuffix(".csv"),
        "value": np.loadtxt(stat, delimiter=",", ndmin=1).tolist(),
        "p_parametric": np.loadtxt(read("stat_uncparap.csv"), delimiter=",", ndmin=1).tolist(),
        "p_permutation": np.loadtxt(read("stat_uncp.csv"), delimiter=",", ndmin=1).tolist(),
        "p_fwer": np.loadtxt(read("stat_fwep.csv"), delimiter=",", ndmin=1).tolist(),
        "dof": dof,
    }
    if not exhaustive:
        del out["p_permutation"], out["p_fwer"]
    return out


def enumerate_exactly(y, design, contrast, rearrangements):
    """Two-sided Freedman–Lane p-values over every rearrangement (rows taken from `source`, then
    multiplied by `signs`), counting ties within a relative 1e-12, in plain numpy."""
    c = np.asarray(contrast, dtype=float)
    null = np.linalg.svd(c[None, :])[2][1:].T
    z = design @ null
    fitted = z @ np.linalg.lstsq(z, y, rcond=None)[0] if z.shape[1] else np.zeros_like(y)
    residual = y - fitted

    def t(data):
        beta, rss = np.linalg.lstsq(design, data, rcond=None)[:2]
        scale = c @ np.linalg.inv(design.T @ design) @ c
        return np.abs(c @ beta) / np.sqrt(rss / (len(y) - design.shape[1]) * scale)

    observed = t(y)
    stats = np.array([t(residual[source] * signs[:, None] + fitted) for source, signs in rearrangements])
    tie = 1e-12 * np.maximum(observed, 1.0)
    return {
        "p_enumerated": ((stats >= observed - tie).sum(0) / len(stats)).tolist(),
        "p_fwer_enumerated": [float((stats.max(1) >= o - tie[k]).sum() / len(stats))
                              for k, o in enumerate(observed)],
    }


def permutations_within(blocks):
    """Every permutation within blocks (labels in order of first appearance)."""
    members = [np.flatnonzero(blocks == b) for b in dict.fromkeys(blocks)]
    for orders in itertools.product(*(itertools.permutations(m) for m in members)):
        source = np.arange(len(blocks))
        for m, order in zip(members, orders):
            source[m] = order
        yield source, np.ones(len(blocks))


def permutations_of_whole(blocks):
    members = [np.flatnonzero(blocks == b) for b in dict.fromkeys(blocks)]
    for order in itertools.permutations(range(len(members))):
        yield np.concatenate([members[k] for k in order]), np.ones(len(blocks))


def flips_of_whole(blocks):
    members = [np.flatnonzero(blocks == b) for b in dict.fromkeys(blocks)]
    for signs in itertools.product([1.0, -1.0], repeat=len(members)):
        flips = np.ones(len(blocks))
        for m, sign in zip(members, signs):
            flips[m] = sign
        yield np.arange(len(blocks)), flips


def case(src, work, name, data, design, contrast, flags, **kwargs):
    out = palm(src, work / name, data, design, contrast, flags, **kwargs)
    out.update(
        y=data.tolist(),
        design=design.tolist(),
        contrast=np.atleast_2d(contrast).tolist(),
        eb=None if kwargs.get("eb") is None else kwargs["eb"].tolist(),
        vg=None if kwargs.get("vg") is None else kwargs["vg"].tolist(),
    )
    print(name, out["statistic"], out["value"], out["dof"])
    return out


def main(palm_dir: str):
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "palm"
        shutil.copytree(palm_dir, src, ignore=shutil.ignore_patterns(".git"))
        writer = src / "palm_miscwrite.m"
        writer.write_text(writer.read_text().replace("'%0.4f'", "'%0.17g'"))

        # A contrast of two regressors, and the F test of both, with an intercept.
        n = 12
        design = np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
        y = rng.standard_normal((n, 3))
        y[:, 0] += 0.9 * design[:, 1]
        y[:, 2] += 0.6 * design[:, 2]
        out["contrast_t"] = case(src, tmp, "contrast_t", y, design, [0.0, 1.0, -1.0], ["-twotail"],
                                 exhaustive=False)
        out["f"] = case(src, tmp, "f", y, design, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [], ftest=True,
                    exhaustive=False)

        # Exhaustive permutation of two groups with only the intercept as nuisance: PALM's unique
        # permutations of the design give the same p-values as all n! orderings.
        n = 8
        design = np.column_stack([np.ones(n), np.repeat([1.0, 0.0], 4)])
        y = rng.standard_normal((n, 2))
        y[:, 0] += 1.5 * design[:, 1]
        out["exhaustive"] = case(src, tmp, "exhaustive", y, design, [0.0, 1.0], ["-twotail"])
        out["exhaustive"].update(enumerate_exactly(y, design, [0.0, 1.0], permutations_within(np.zeros(n))))

        # Aspin–Welch v: two groups with unequal sizes and variances, and a covariate.
        sizes = [5, 9]
        group = np.repeat([1.0, 0.0], sizes)
        cov = rng.normal(0, 1, sum(sizes))
        design = np.column_stack([np.ones(sum(sizes)), group, cov])
        y = rng.standard_normal((sum(sizes), 2)) * np.where(group == 1, 3.0, 0.7)[:, None]
        y[:, 0] += 1.2 * group + 0.5 * cov
        out["v"] = case(src, tmp, "v", y, design, [0.0, 1.0, 0.0], ["-twotail"],
                        vg=(2 - group).astype(int), exhaustive=False)

        # The same without the covariate: v is Welch's t.
        out["welch"] = case(src, tmp, "welch", y, design[:, :2], [0.0, 1.0], ["-twotail"],
                            vg=(2 - group).astype(int), exhaustive=False)

        # G: three groups (cell means) and a covariate, equality of all three means.
        sizes = [4, 5, 7]
        labels = np.repeat([0, 1, 2], sizes)
        cells = (labels[:, None] == np.arange(3)).astype(float)
        cov = rng.normal(0, 1, sum(sizes))
        design = np.column_stack([cells, cov])
        y = rng.standard_normal((sum(sizes), 2)) * np.array([0.5, 1.0, 2.5])[labels][:, None]
        y[:, 0] += np.array([0.0, 0.8, 1.6])[labels] + 0.3 * cov
        out["g"] = case(src, tmp, "g", y, design, [[1.0, -1.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0]], [],
                        ftest=True, vg=labels + 1, exhaustive=False)

        # Within-block permutation: 3 blocks of 3, a regressor that varies within each block
        # (distinct values, so PALM's unique permutations are all of them), block means as nuisance.
        blocks = np.repeat([1, 2, 3], 3)
        x = np.tile([-1.0, 0.0, 1.0], 3) + rng.normal(0, 0.1, 9)
        design = np.column_stack([x, (blocks[:, None] == np.array([1, 2, 3])).astype(float)])
        y = rng.standard_normal((9, 2)) + 2.0 * (blocks[:, None] - 2)
        y[:, 0] += 0.8 * x
        out["within"] = case(src, tmp, "within", y, design, [1.0, 0.0, 0.0, 0.0], ["-twotail", "-within"],
                             eb=blocks)
        out["within"].update(enumerate_exactly(y, design, [1.0, 0.0, 0.0, 0.0], permutations_within(blocks)))

        # Whole-block permutation: 4 blocks of 2 move as units; the group varies between blocks.
        blocks = np.repeat([1, 2, 3, 4], 2)
        group = np.repeat([1.0, 1.0, 0.0, 0.0], 2)
        design = np.column_stack([np.ones(8), group])
        y = rng.standard_normal((8, 2)) + rng.normal(0, 1, 4)[blocks - 1][:, None]
        y[:, 0] += 1.0 * group
        out["whole"] = case(src, tmp, "whole", y, design, [0.0, 1.0], ["-twotail", "-whole"], eb=blocks)
        out["whole"].update(enumerate_exactly(y, design, [0.0, 1.0], permutations_of_whole(blocks)))

        # Whole-block sign flipping: 5 blocks of 2 flip together, one-sample test.
        blocks = np.repeat([1, 2, 3, 4, 5], 2)
        y = rng.normal(0.4, 1.0, (10, 2)) + rng.normal(0, 1, 5)[blocks - 1][:, None]
        out["whole_flip"] = case(src, tmp, "whole_flip", y, np.ones((10, 1)), [1.0],
                                 ["-twotail", "-ise", "-whole"], eb=blocks)
        out["whole_flip"].update(enumerate_exactly(y, np.ones((10, 1)), [1.0], flips_of_whole(blocks)))

    meta = {"palm_commit": subprocess.run(["git", "-C", palm_dir, "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()}
    Path(__file__).with_name("palm.json").write_text(json.dumps({"meta": meta, **out}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
