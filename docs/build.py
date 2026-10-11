"""Build the documentation site: Sphinx (warnings are errors) plus rustdoc under rust/.

Usage: python docs/build.py [--no-rust]
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
OUT = DOCS / "_build" / "html"

subprocess.run([sys.executable, "-m", "sphinx", "-W", "--keep-going", "-n", "-b", "html", str(DOCS), str(OUT)], check=True)


def full_text() -> str:
    """Every page llms.txt links to, in its order, plus the Python stubs: llms-full.txt."""
    index = (DOCS / "_extra" / "llms.txt").read_text()
    parts = [index]
    for target in re.findall(r"\]\(_sources/(.+?)\.txt\)", index):
        page = OUT / "_sources" / f"{target}.txt"
        parts.append(f"\n\n---\n\n<!-- {target} -->\n\n{page.read_text()}")
    stubs = ROOT / "drivers" / "python" / "python" / "modewise" / "_core.pyi"
    parts.append(f"\n\n---\n\n<!-- Python type stubs -->\n\n```python\n{stubs.read_text()}```\n")
    return "".join(parts)


(OUT / "llms-full.txt").write_text(full_text())
if "--no-rust" not in sys.argv:
    env = {**os.environ, "RUSTDOCFLAGS": "-D warnings"}
    subprocess.run(["cargo", "doc", "--workspace", "--no-deps", "--exclude", "modewise_python", "--exclude", "modewise_docs"], cwd=ROOT, env=env, check=True)
    shutil.copytree(ROOT / "target" / "doc", OUT / "rust", dirs_exist_ok=True)
print(f"Built {OUT / 'index.html'}")
