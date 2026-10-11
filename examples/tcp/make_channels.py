"""Build channels.tsv for ds005237's 434 parcellated channels from the TCP processing repository.

Source: github.com/HolmesLab/TransdiagnosticConnectomeProject (MIT), scripts/atlas_files,
"unordered" files, which follow the channel order of the released timeseries:
  0-399    homotopic Schaefer 400, Yeo 17 networks (Yan et al. 2023); left 0-199, right 200-399
  400-431  Tian subcortex, scale II
  432-433  cerebellum, left and right (Buckner atlas)

The region names are a pickled object array. This script reads it with pickletools, which only
disassembles the pickle and never runs it, so no code from the download is executed.

Usage: uv run --no-project --with numpy python make_channels.py <path to TCP repo clone>
"""

import io
import pickletools
import re
import sys
from pathlib import Path

import numpy as np

atlas = Path(sys.argv[1]) / "scripts" / "atlas_files"


def pickled_strings(path):
    raw = path.read_bytes()
    major = raw[6]
    header_len = int.from_bytes(raw[8:10] if major == 1 else raw[8:12], "little")
    start = (10 if major == 1 else 12) + header_len
    listing = io.StringIO()
    pickletools.dis(raw[start:], out=listing)
    texts = re.findall(r"(?:SHORT_BINUNICODE|BINUNICODE|UNICODE)\s+'([^']*)'", listing.getvalue())
    return [t for t in texts if t not in {"numpy", "ndarray", "dtype", "O8", "|", "_reconstruct"} and "numpy" not in t]


names = pickled_strings(atlas / "whole_brain_region_labels_unordered_Cerebellum_Seperate.npy")
networks = np.load(atlas / "whole_brain_network_labels_by_region_unordered_Cerebellum_Seperate.npy", allow_pickle=False)
assert len(names) == len(networks) == 434, (len(names), len(networks))

rows = ["name\tregion\themisphere\tyeo17\ttcp_network"]
for name, network in zip(names, networks):
    region, hemisphere, yeo17 = "", "", ""
    if m := re.fullmatch(r"17networks_(LH|RH)_([A-Za-z]+)((?:_[A-Za-z][A-Za-z0-9]*)*)(?:_\d+)?", name):
        hemisphere = {"LH": "left", "RH": "right"}[m[1]]
        yeo17, region = m[2], m[3].lstrip("_")
    elif m := re.fullmatch(r"(.+)-(lh|rh)", name):
        region, hemisphere = m[1], {"lh": "left", "rh": "right"}[m[2]]
    elif m := re.fullmatch(r"cerebellum_(left|right)", name):
        region, hemisphere = "cerebellum", m[1]
    else:
        raise SystemExit(f"unrecognised label {name!r}")
    rows.append("\t".join([name, region, hemisphere, yeo17, str(network)]))

(Path(__file__).parent / "channels.tsv").write_text("\n".join(rows) + "\n")
print(f"wrote {len(rows) - 1} channels")
