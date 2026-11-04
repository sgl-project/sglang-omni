# SPDX-License-Identifier: Apache-2.0
"""Exercise RST navigation coverage for Markdown documentation."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("emoji")

from scripts.check_doc_style import check_docs

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/check_doc_style.py"


def write_document(docs_root: Path, relative_path: str, source: str) -> Path:
    """Write a document in an isolated documentation tree."""
    path = docs_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "rst_path, entry, markdown_path",
    [
        ("index.rst", "guide.md", "guide.md"),
        ("index.rst", "guide", "guide.md"),
        ("index.rst", "Guide title <guide>", "guide.md"),
        ("index.rst", "Guide<guide>", "guide.md"),
        ("index.rst", "Guide<guide.md>", "guide.md"),
        ("navigation/index.rst", "Guide</guide>", "guide.md"),
        ("navigation/index.rst", "/guide", "guide.md"),
        ("navigation/index.rst", "/guide.md", "guide.md"),
        ("navigation/index.rst", "child/guide", "navigation/child/guide.md"),
        ("navigation/index.rst", "../guide.md", "guide.md"),
        (
            "navigation/index.rst",
            "Guide title <child/guide.md>",
            "navigation/child/guide.md",
        ),
    ],
)
def test_check_docs_resolves_explicit_toctree_paths(
    tmp_path: Path, rst_path: str, entry: str, markdown_path: str
) -> None:
    docs_root = tmp_path / "docs"
    write_document(docs_root, markdown_path, "# Guide\n")
    write_document(docs_root, rst_path, f".. toctree::\n\n   {entry}\n")

    assert check_docs(docs_root) == []


@pytest.mark.parametrize(
    "inactive_source",
    [
        "::\n\n   .. toctree::\n\n      orphan\n",
        "Example::\n\n   .. toctree::\n\n      orphan\n",
        "..\n   .. toctree::\n\n      orphan\n",
        ".. A navigation example\n\n   .. toctree::\n\n      orphan\n",
        ".. code-block:: rst\n\n   .. toctree::\n\n      orphan\n",
        ".. code:: rst\n\n   .. toctree::\n\n      orphan\n",
        ".. sourcecode:: rst\n\n   .. toctree::\n\n      orphan\n",
        ".. parsed-literal::\n\n   .. toctree::\n\n      orphan\n",
        ".. raw:: html\n\n   .. toctree::\n\n      orphan\n",
        (".. note::\n\n   Example::\n\n" "      .. toctree::\n\n         orphan\n"),
    ],
)
def test_check_docs_ignores_inactive_toctrees(
    tmp_path: Path, inactive_source: str
) -> None:
    docs_root = tmp_path / "docs"
    orphan_path = write_document(docs_root, "orphan.md", "# Orphan\n")
    write_document(docs_root, "included.md", "# Included\n")
    write_document(
        docs_root,
        "index.rst",
        f"{inactive_source}\n.. toctree::\n\n   included\n",
    )

    violations = check_docs(docs_root)

    assert [(violation.path, violation.rule) for violation in violations] == [
        (orphan_path, "missing-toctree")
    ]


def test_check_docs_stops_at_directive_boundaries_and_ignores_prose(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    for relative_path in (
        "included.md",
        "second.md",
        "outside.md",
        "after_note.md",
        "nested/child.md",
        "nested/after_nested.md",
    ):
        write_document(docs_root, relative_path, "# Guide\n")
    write_document(
        docs_root,
        "index.rst",
        """An ordinary reference to outside.md must not include that page.

.. toctree::
   :maxdepth: 1

   included
   nested/index.rst

Outside the directive:

   outside

.. note::

   after_note

.. toctree::

   second
""",
    )
    write_document(
        docs_root,
        "nested/index.rst",
        """.. note::

   .. toctree::

      child

   The nested toctree has ended.

      after_nested
""",
    )

    violations = check_docs(docs_root)

    assert [
        (violation.path.relative_to(docs_root).as_posix(), violation.line_number)
        for violation in violations
        if violation.rule == "missing-toctree"
    ] == [("after_note.md", 1), ("nested/after_nested.md", 1), ("outside.md", 1)]


def test_check_docs_ignores_options_external_urls_and_self(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    for relative_path in (
        ":caption: option_page.md",
        "continued_caption.md",
        "https:/example.test/remote.md",
        "self.md",
        "cookbook/guide.md",
    ):
        write_document(docs_root, relative_path, "# Guide\n")
    write_document(
        docs_root,
        "index.rst",
        """.. toctree::
   :caption: option_page
      continued_caption
   :glob:

   Remote page <https://example.test/remote>
   self
   cookbook/*
""",
    )

    violations = check_docs(docs_root)

    assert {
        violation.path.relative_to(docs_root).as_posix()
        for violation in violations
        if violation.rule == "missing-toctree"
    } == {
        ":caption: option_page.md",
        "continued_caption.md",
        "https:/example.test/remote.md",
        "self.md",
    }


@pytest.mark.parametrize(
    "rst_path, entry, included_pages",
    [
        ("index.rst", "cookbook/*", {"guide", "guide_two", "guide1"}),
        ("navigation/index.rst", "../cookbook/*", {"guide", "guide_two", "guide1"}),
        ("navigation/index.rst", "/cookbook/*", {"guide", "guide_two", "guide1"}),
        (
            "index.rst",
            "cookbook/**",
            {"guide", "guide_two", "guide1", "nested/guide", "nested/deeper/guide"},
        ),
        ("index.rst", "cookbook/*/guide", {"nested/guide"}),
        ("index.rst", "cookbook/**/guide", {"nested/guide", "nested/deeper/guide"}),
        ("index.rst", "cookbook/guide?", {"guide1"}),
        ("index.rst", "cookbook/guide[0-9]", {"guide1"}),
        ("index.rst", "cookbook/guide[!a-z]", {"guide1"}),
        ("index.rst", "cookbook[!x]*", set()),
        ("index.rst", "cookbook[/]*", {"guide", "guide_two", "guide1"}),
        ("index.rst", "cookbook?guide", set()),
        ("index.rst", "cookbook/*.md", set()),
        ("index.rst", "absent/*", set()),
        ("index.rst", "cookbook/guide", {"guide"}),
        ("index.rst", "Guide <cookbook/guide.md>", {"guide"}),
        ("index.rst", "Guides <cookbook/*>", set()),
    ],
)
def test_toctree_glob_includes_matching_markdown(
    tmp_path: Path, rst_path: str, entry: str, included_pages: set[str]
) -> None:
    docs_root = tmp_path / "docs[2026]"
    cookbook_pages = {
        "guide",
        "guide_two",
        "guide1",
        "nested/guide",
        "nested/deeper/guide",
    }
    for page in cookbook_pages:
        write_document(docs_root, f"cookbook/{page}.md", "# Guide\n")
    write_document(docs_root, "outside.md", "# Outside\n")
    write_document(
        docs_root,
        rst_path,
        f".. toctree::\n   :glob:\n   :maxdepth: 2\n\n   {entry}\n",
    )

    violations = check_docs(docs_root)

    assert {
        violation.path.relative_to(docs_root).as_posix()
        for violation in violations
        if violation.rule == "missing-toctree"
    } == {"outside.md"} | {
        f"cookbook/{page}.md" for page in cookbook_pages - included_pages
    }


@pytest.mark.parametrize(
    "rst_source",
    [
        ".. toctree::\n\n   cookbook/*\n",
        ".. toctree::\n   :glob:\n\n   included\n\n" ".. toctree::\n\n   cookbook/*\n",
        ".. toctree::\n   :caption: Examples\n      :glob:\n\n   cookbook/*\n",
        ".. code-block:: rst\n\n   .. toctree::\n      :glob:\n\n      cookbook/*\n",
    ],
)
def test_check_docs_limits_glob_to_its_active_directive(
    tmp_path: Path, rst_source: str
) -> None:
    docs_root = tmp_path / "docs"
    write_document(docs_root, "cookbook/guide.md", "# Guide\n")
    write_document(docs_root, "included.md", "# Included\n")
    write_document(docs_root, "navigation.rst", rst_source)
    write_document(docs_root, "index.rst", ".. toctree::\n\n   included\n")

    assert [
        violation.path.relative_to(docs_root).as_posix()
        for violation in check_docs(docs_root)
        if violation.rule == "missing-toctree"
    ] == ["cookbook/guide.md"]


@pytest.mark.parametrize(
    "navigation_source, is_included",
    [
        ("```{toctree}\n---\nmaxdepth: 1\n---\n\n../guide.md\n```", True),
        ("```{toctree}\n:maxdepth: 1\n\n../guide\n```", True),
        ("~~~{toctree}\n\nGuide <../guide.md>\n~~~", True),
        ("```{toctree}\n---\nglob: true\n---\n\n../*\n```", True),
        ("```{toctree}\n:glob:\n\n../*\n```", True),
        ("```{toctree}\n---\nglob: false\n---\n\n../*\n```", False),
        ("````rst\n```{toctree}\n\n../guide\n```\n````", False),
        ("<!--\n```{toctree}\n\n../guide\n```\n-->", False),
        ("```{toctree}\n---\ncaption: ../guide.md\n---\n```", False),
        ("See ../guide.md for an example.", False),
    ],
)
def test_check_docs_recognizes_active_markdown_toctrees(
    tmp_path: Path, navigation_source: str, is_included: bool
) -> None:
    docs_root = tmp_path / "docs"
    guide_path = write_document(docs_root, "cookbook/guide.md", "# Guide\n")
    write_document(
        docs_root, "cookbook/companies/qwen.md", f"# Qwen\n\n{navigation_source}\n"
    )
    write_document(
        docs_root, "index.rst", ".. toctree::\n\n   cookbook/companies/qwen\n"
    )

    assert [
        violation.path
        for violation in check_docs(docs_root)
        if violation.rule == "missing-toctree"
    ] == ([] if is_included else [guide_path])


def test_markdown_toctree_glob_does_not_include_its_own_document(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    guide_path = write_document(
        docs_root, "guide.md", "# Guide\n\n```{toctree}\n:glob:\n\n*\n```\n"
    )

    assert [
        (violation.path, violation.rule) for violation in check_docs(docs_root)
    ] == [(guide_path, "missing-toctree")]


def test_check_docs_reports_new_orphans_until_explicitly_included(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    path = write_document(docs_root, "design/new_page.md", "# New page\n")

    violations = check_docs(docs_root)

    assert [
        (violation.path, violation.line_number, violation.rule, violation.detail)
        for violation in violations
    ] == [
        (path, 1, "missing-toctree", "add this page to an explicit RST toctree entry")
    ]
    write_document(docs_root, "index.rst", ".. toctree::\n\n   design/new_page\n")
    assert check_docs(docs_root) == []


@pytest.mark.parametrize(
    "relative_path",
    [
        "README.md",
        "design/gpu_radix_hash.md",
        "design/refactor_rfc.md",
        "developer_reference/mlx.md",
        "get_started/apiserver_quickstart.md",
        "basic_usage/python_router.md",
    ],
)
def test_navigation_exemptions_keep_emoji_and_fence_checks(
    tmp_path: Path, relative_path: str
) -> None:
    docs_root = tmp_path / "docs"
    write_document(docs_root, relative_path, "# Notes\n\n😀\n\n```\nbody\n```\n")

    violations = check_docs(docs_root)

    assert [(violation.rule, violation.line_number) for violation in violations] == [
        ("no-emoji", 3),
        ("fence-language", 5),
    ]


def test_navigation_exemptions_do_not_apply_to_whole_directories(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    write_document(docs_root, "design/Bad-Note.md", "# Note\n")

    violations = check_docs(docs_root)

    assert [(violation.rule, violation.line_number) for violation in violations] == [
        ("filename-conventions", 1),
        ("missing-toctree", 1),
    ]


def test_cli_reports_missing_toctree_with_path_and_line_one(tmp_path: Path) -> None:
    script_path = tmp_path / "scripts" / "check_doc_style.py"
    script_path.parent.mkdir()
    shutil.copy2(SCRIPT, script_path)
    write_document(tmp_path / "docs", "orphan.md", "# Orphan\n")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == (
        "docs/orphan.md:1: missing-toctree: "
        "add this page to an explicit RST toctree entry\n"
    )
