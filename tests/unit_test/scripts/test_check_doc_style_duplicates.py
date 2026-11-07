# SPDX-License-Identifier: Apache-2.0
"""Exercise cross-file duplicate fenced-block diagnostics and boundaries."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("emoji")

from scripts.check_doc_style import check_docs

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/check_doc_style.py"
SHARED_BODY = "\n".join(["shared command with enough content"] * 3)


def make_fenced_source(
    body_text: str, fence_info: str = "text", fence_marker: str = "```"
) -> str:
    """Wrap a body in a completed top-level code fence."""
    return f"{fence_marker}{fence_info}\n{body_text}\n{fence_marker}\n"


def write_documents(docs_root: Path, documents: dict[str, str]) -> None:
    """Publish synthetic Markdown documents in a minimal documentation tree."""
    docs_root.mkdir(parents=True)
    (docs_root / "index.rst").write_text(
        ".. toctree::\n\n" + "".join(f"   {filename}\n" for filename in documents),
        encoding="utf-8",
    )
    for filename, source in documents.items():
        (docs_root / filename).write_text(source, encoding="utf-8")


@pytest.mark.parametrize(
    "body_text, should_report",
    [
        ("\n".join(["a" * 23, "a" * 23, "a" * 22]), True),
        ("\n".join(["b" * 35, "b" * 34]), False),
        ("\n".join(["c" * 23, "c" * 22, "c" * 22]), False),
        ("\n".join(["d" * 33, "d" * 34]), False),
        ("\n".join(["汉" * 23, "汉" * 22, "汉" * 22]), False),
        ("\n".join(["汉" * 23, "汉" * 23, "汉" * 22]), True),
    ],
    ids=[
        "exact_3_lines_70_characters",
        "2_lines_70_characters",
        "3_lines_69_characters",
        "both_below_boundary",
        "69_unicode_characters_not_bytes",
        "70_unicode_characters",
    ],
)
def test_duplicate_size_boundaries(
    tmp_path: Path, body_text: str, should_report: bool
) -> None:
    docs_root = tmp_path / "docs"
    source = make_fenced_source(body_text)
    write_documents(docs_root, {"first.md": source, "second.md": source})

    violations = check_docs(docs_root)

    if should_report:
        assert [
            (violation.path.name, violation.line_number, violation.rule)
            for violation in violations
        ] == [("first.md", 1, "duplicated-block"), ("second.md", 1, "duplicated-block")]
    else:
        assert violations == []


def test_reports_every_location_after_trailing_whitespace_normalization(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    padded_body = "\n".join(f"{line} \t" for line in SHARED_BODY.split("\n"))
    write_documents(
        docs_root,
        {
            "third.md": make_fenced_source(SHARED_BODY, "bash", "~~~"),
            "second.md": make_fenced_source(SHARED_BODY, "python"),
            "first.md": make_fenced_source(padded_body) * 2,
        },
    )

    violations = check_docs(docs_root)

    assert [
        (violation.path.name, violation.line_number, violation.rule)
        for violation in violations
    ] == [
        ("first.md", 1, "duplicated-block"),
        ("first.md", 6, "duplicated-block"),
        ("second.md", 1, "duplicated-block"),
        ("third.md", 1, "duplicated-block"),
    ]
    assert {violation.detail for violation in violations} == {
        "identical fenced body at first.md:1, first.md:6, second.md:1, third.md:1"
    }


def test_same_file_repetition_is_not_a_cross_file_duplicate(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    write_documents(docs_root, {"guide.md": make_fenced_source(SHARED_BODY) * 2})

    assert check_docs(docs_root) == []


@pytest.mark.parametrize(
    "different_body",
    [
        SHARED_BODY.replace("\n", "\n    ", 1),
        SHARED_BODY.replace("\n", "\n\n", 1),
    ],
    ids=["internal_indentation", "internal_blank_line"],
)
def test_internal_whitespace_remains_significant(
    tmp_path: Path, different_body: str
) -> None:
    docs_root = tmp_path / "docs"
    write_documents(
        docs_root,
        {
            "first.md": make_fenced_source(SHARED_BODY),
            "second.md": make_fenced_source(different_body),
        },
    )

    assert check_docs(docs_root) == []


def test_unclosed_body_is_not_collected_as_a_duplicate(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    write_documents(
        docs_root,
        {
            "first.md": make_fenced_source(SHARED_BODY),
            "second.md": f"```text\n{SHARED_BODY}\n",
        },
    )

    violations = check_docs(docs_root)

    assert [
        (violation.path.name, violation.line_number, violation.rule)
        for violation in violations
    ] == [("second.md", 1, "unclosed-fence")]


def test_shorter_or_different_markers_remain_in_the_body(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    body_text = f"```python\n{SHARED_BODY}\n```"
    write_documents(
        docs_root,
        {
            "first.md": make_fenced_source(body_text, "text", "````"),
            "second.md": make_fenced_source(body_text, "text", "~~~"),
        },
    )

    assert [violation.rule for violation in check_docs(docs_root)] == [
        "duplicated-block",
        "duplicated-block",
    ]


def test_groups_follow_first_occurrence_order(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    first_body = SHARED_BODY.replace("shared", "first")
    second_body = SHARED_BODY.replace("shared", "second")
    write_documents(
        docs_root,
        {
            "third.md": make_fenced_source(second_body),
            "second.md": make_fenced_source(first_body)
            + make_fenced_source(second_body),
            "first.md": make_fenced_source(second_body)
            + make_fenced_source(first_body),
        },
    )

    assert [
        (violation.path.name, violation.line_number)
        for violation in check_docs(docs_root)
    ] == [
        ("first.md", 1),
        ("second.md", 6),
        ("third.md", 1),
        ("first.md", 6),
        ("second.md", 1),
    ]


@pytest.mark.parametrize("has_duplicate", [False, True])
def test_cli_reports_locations_and_exit_status(
    tmp_path: Path, has_duplicate: bool
) -> None:
    repository_root = tmp_path / "fixture"
    script_path = repository_root / "scripts" / "check_doc_style.py"
    script_path.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script_path)
    second_body = (
        SHARED_BODY if has_duplicate else SHARED_BODY.replace("shared", "other")
    )
    write_documents(
        repository_root / "docs",
        {
            "first.md": make_fenced_source(SHARED_BODY),
            "second.md": make_fenced_source(second_body),
        },
    )

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.stdout == ""
    if has_duplicate:
        assert result.returncode == 1
        assert result.stderr == (
            "docs/first.md:1: duplicated-block: identical fenced body at first.md:1, second.md:1\n"
            "docs/second.md:1: duplicated-block: identical fenced body at first.md:1, second.md:1\n"
        )
    else:
        assert result.returncode == 0
        assert result.stderr == ""
