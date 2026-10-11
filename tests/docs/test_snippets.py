"""Every ```python block in docs/ runs. Blocks of one page run in order and share variables, in a
temporary folder. Blocks that need data on disk use ```{code-block} python, which renders the same
but is not run."""

import os
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
PAGES = sorted(p for p in DOCS.rglob("*.md") if "_build" not in p.parts and FENCE.search(p.read_text()))


@pytest.mark.parametrize("page", PAGES, ids=[str(p.relative_to(DOCS)) for p in PAGES])
def test_python_snippets_run(page, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__docs__"}
    for i, block in enumerate(FENCE.findall(page.read_text()), start=1):
        try:
            exec(compile(block, f"{page.relative_to(DOCS)} block {i}", "exec"), namespace)
        except Exception as error:
            raise AssertionError(f"{page.relative_to(DOCS)}, python block {i} failed: {error!r}") from error


def test_pages_with_python_were_found():
    assert len(PAGES) >= 6, [str(p) for p in PAGES]


def test_llms_txt_lists_every_page_of_the_site():
    """llms.txt links each page's markdown source; it must cover every page in the toctrees."""
    llms = (DOCS / "_extra" / "llms.txt").read_text()
    linked = set(re.findall(r"\]\(_sources/(.+?)\.(?:md|rst)\.txt\)", llms))
    for page in linked:
        assert (DOCS / f"{page}.md").exists() or page.startswith(("reference/python/", "auto_examples/")), page
    index = (DOCS / "index.md").read_text()
    pages = {line.strip() for block in re.findall(r"```\{toctree\}\n(.*?)```", index, re.S) for line in block.splitlines() if line.strip() and not line.startswith(":")}
    missing = {p for p in pages if p not in linked and p != "reference/python" and p != "reference/rust"}
    assert not missing, f"pages missing from docs/_extra/llms.txt: {sorted(missing)}"


CASE_STUDIES = DOCS.parent / "examples" / "case-studies"
STAMP = re.compile(r"<!-- stamp: modewise (\S+), commit (\S+), (\S+), (\d+) subjects -->")


@pytest.mark.parametrize("report", sorted(CASE_STUDIES.glob("*/REPORT.md")), ids=lambda p: p.parent.name)
def test_case_study_reports_are_from_this_version(report):
    """A case study is rerun by hand (it needs gigabytes of data). Its report must come from the
    current modewise version, so a release cannot ship results made by older code."""
    import modewise

    stamp = STAMP.search(report.read_text())
    assert stamp, f"{report} has no provenance stamp; rerun its analysis.py"
    assert stamp.group(1) == modewise.__version__, (
        f"{report.parent.name} was made with modewise {stamp.group(1)}, this is {modewise.__version__}: rerun it"
    )
