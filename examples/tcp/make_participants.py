"""Write the thesis anhedonia grouping of ds005237 as a participants table modewise can read.

Reproduces the legacy subject selection (stage 00) rules:
- SHAPS: items shaps1a–shaps14a (item 8 is spelled `shaps_8a` in the file); only 0 and 1 count,
  anything else (-9 means unanswered) counts as 0; subjects whose `shaps_total` is 999 are
  excluded. Anhedonic when the recomputed total is 3 or more.
- TEPS: anticipatory items 1, 3, 7, 11, 12, 14, 15, 16, 17, 18 and consummatory items 2, 4, 5, 6,
  8, 9, 10, 13, each averaged over the answered items. Anhedonic when a mean is more than 2
  sample SD below the mean over all respondents.
- Anhedonic: flagged by any of the three. Control: Group GenPop (participants.tsv) and not
  flagged. Everyone else: excluded.

`thesis_group` applies these rules exactly. The legacy code averaged every TEPS value as a
number, but the NDA definitions (teps01_definitions.tsv) mark 99, -900 and -300 as codes, not
answers: 8 respondents have -300 items, which pulls their means far below the rest (flagging
them) and inflates the SD so no one else can be flagged by TEPS. `corrected_group` treats those
codes as unanswered.

These are dataset rules from one thesis, kept here rather than in modewise (docs/features.md).

Usage: python make_participants.py <ds005237 root> <output.tsv>
"""

import csv
import statistics
import sys
from pathlib import Path

root, output = Path(sys.argv[1]), Path(sys.argv[2])


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


shaps = {}
for row in rows(root / "phenotype" / "shaps01.tsv"):
    if number(row.get("shaps_total")) == 999 or row["subjectkey"] in shaps:
        continue
    items = [row.get(f"shaps{i}a", row.get(f"shaps_{i}a")) for i in range(1, 15)]
    shaps[row["subjectkey"]] = sum(v for v in map(number, items) if v in (0.0, 1.0))

ANTICIPATORY = [1, 3, 7, 11, 12, 14, 15, 16, 17, 18]
CONSUMMATORY = [2, 4, 5, 6, 8, 9, 10, 13]
CODES = {99.0, -900.0, -300.0}


def teps_means(skip_codes):
    means = {}
    for row in rows(root / "phenotype" / "teps01.tsv"):
        if row["subjectkey"] in means:
            continue
        pair = []
        for items in (ANTICIPATORY, CONSUMMATORY):
            values = [v for v in (number(row.get(f"teps{i}")) for i in items) if v is not None and not (skip_codes and v in CODES)]
            pair.append(sum(values) / len(values) if values else None)
        if any(m is not None for m in pair):
            means[row["subjectkey"]] = pair
    thresholds = []
    for k in range(2):
        values = [m[k] for m in means.values() if m[k] is not None]
        thresholds.append(statistics.mean(values) - 2 * statistics.stdev(values))
    return means, thresholds


groups = {r["participant_id"].replace("sub-NDAR", "NDAR_"): r["Group"].strip() for r in rows(root / "participants.tsv")}
legacy, legacy_thresholds = teps_means(skip_codes=False)
corrected, corrected_thresholds = teps_means(skip_codes=True)
subjects = sorted(set(groups) | set(shaps) | set(legacy))


def group(subject, teps, thresholds):
    ant, con = teps.get(subject, (None, None))
    flags = [subject in shaps and shaps[subject] >= 3, ant is not None and ant < thresholds[0], con is not None and con < thresholds[1]]
    return flags, "anhedonic" if any(flags) else ("control" if groups.get(subject) == "GenPop" else "excluded")


with open(output, "w", newline="") as f:
    out = csv.writer(f, delimiter="\t", lineterminator="\n")
    out.writerow(["subjectkey", "shaps_recomputed", "teps_ant_mean", "teps_con_mean", "anhedonic_shaps", "anhedonic_teps_ant", "anhedonic_teps_con", "thesis_group", "corrected_group"])
    for s in subjects:
        flags, thesis = group(s, legacy, legacy_thresholds)
        _, fixed = group(s, corrected, corrected_thresholds)
        ant, con = legacy.get(s, (None, None))
        fmt = lambda v: "n/a" if v is None else f"{v:g}"
        out.writerow([s, fmt(shaps.get(s)), fmt(ant), fmt(con), *("yes" if x else "no" for x in flags), thesis, fixed])
print(f"{len(subjects)} subjects; TEPS thresholds as in the thesis {legacy_thresholds[0]:.3f}, {legacy_thresholds[1]:.3f}; "
      f"with codes treated as missing {corrected_thresholds[0]:.3f}, {corrected_thresholds[1]:.3f}")
