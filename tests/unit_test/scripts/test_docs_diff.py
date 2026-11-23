# SPDX-License-Identifier: Apache-2.0
"""The rendered docs diff reports changed, added and removed pages from two HTML builds."""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIFF = REPO_ROOT / ".github/scripts/docs_diff.py"


def write_page(root: Path, rel: str, title: str, paragraphs: list[str]) -> None:
    body = "".join(f"<p>{text}</p>" for text in paragraphs)
    page = root / rel
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        "<html><body><nav>Sidebar text that changes between builds</nav>"
        f'<article class="bd-article"><h1>{title}<a class="headerlink" href="#x">#</a></h1>'
        '<img src="a.png" alt="Figure"><audio><source src="x.wav"></audio>'
        f"{body}<script>var noise = 1;</script></article>"
        "<footer>Last updated on a date that differs per build</footer></body></html>",
        encoding="utf-8",
    )


def run_docs_diff(base: Path, pr: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(DOCS_DIFF), str(base), str(pr), str(out)],
        capture_output=True,
        text=True,
        check=False,
    )


def visible(html_text: str) -> str:
    # difflib writes spaces as &nbsp; and wraps changed words in spans.
    return re.sub(r"<[^>]+>", "", html_text).replace("&nbsp;", " ")


def test_reports_changed_added_and_removed_pages(tmp_path: Path) -> None:
    base, pr, out = tmp_path / "base", tmp_path / "pr", tmp_path / "out"
    write_page(base, "index.html", "Welcome", ["Old intro."])
    write_page(pr, "index.html", "Welcome", ["New intro."])
    write_page(
        base, "cookbook/model.html", "Model", ["Old sentence.", "Kept sentence."]
    )
    write_page(
        pr,
        "cookbook/model.html",
        "Model",
        ["New sentence.", "Kept sentence.", "Extra."],
    )
    write_page(base, "cookbook/gone.html", "Gone", ["Bye."])
    write_page(pr, "cookbook/fresh.html", "Fresh", ["Hello."])
    write_page(base, "genindex.html", "Index", ["a"])
    write_page(pr, "genindex.html", "Index", ["b"])

    result = run_docs_diff(base, pr, out)

    assert result.returncode == 0, result.stderr
    pages = json.loads((out / "pages.json").read_text(encoding="utf-8"))
    assert [entry["page"] for entry in pages["changed"]] == [
        "cookbook/model.html",
        "index.html",
    ]
    assert pages["changed"][0]["title"] == "Model"
    assert pages["changed"][0]["added"] == 2
    assert pages["changed"][0]["removed"] == 1
    assert pages["added"] == ["cookbook/fresh.html"]
    assert pages["removed"] == ["cookbook/gone.html"]

    diff_page = (out / "pages/cookbook/model.html").read_text(encoding="utf-8")
    diff_text = visible(diff_page)
    assert "New sentence." in diff_text
    assert "Old sentence." in diff_text
    assert "[image: Figure]" in diff_text
    assert 'href="../../../cookbook/model.html"' in diff_page
    assert 'href="../../index.html"' in diff_page
    assert "Sidebar text" not in diff_text
    assert "Last updated" not in diff_text
    assert "noise" not in diff_text
    # The site's own index.html diff must not collide with the listing.
    assert "New intro." in visible(
        (out / "pages/index.html").read_text(encoding="utf-8")
    )
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'href="pages/cookbook/model.html"' in index
    assert 'href="pages/index.html"' in index
    assert "new page" in index
    assert "removed" in index
    assert (out / "pages/cookbook/fresh.html").exists()


def test_identical_builds_report_nothing(tmp_path: Path) -> None:
    base, pr, out = tmp_path / "base", tmp_path / "pr", tmp_path / "out"
    write_page(base, "index.html", "Welcome", ["Same intro."])
    write_page(pr, "index.html", "Welcome", ["Same intro."])

    result = run_docs_diff(base, pr, out)

    assert result.returncode == 0, result.stderr
    pages = json.loads((out / "pages.json").read_text(encoding="utf-8"))
    assert pages == {"changed": [], "added": [], "removed": []}
    assert "No rendered page changed." in (out / "index.html").read_text(
        encoding="utf-8"
    )
