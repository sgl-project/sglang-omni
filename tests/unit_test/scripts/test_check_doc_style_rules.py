# SPDX-License-Identifier: Apache-2.0
"""Exercise the non-emoji documentation style rules."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("emoji")

from scripts.check_doc_style import check_docs, check_source


def make_fenced_source(body_text: str, info_string: str = "text") -> str:
    """Wrap body text in a named Markdown fenced block."""
    return f"```{info_string}\n{body_text}\n```\n"


def test_check_source_accepts_named_fences_without_a_language_whitelist() -> None:
    source = (
        "```text\ntext body\n```\n"
        "```{raw}\nraw body\n```\n"
        "~~~mermaid\ngraph TD\n~~~\n"
        "```arbitrary-info\narbitrary body\n```\n"
    )

    violations = check_source(Path("guide.md"), source)

    assert violations == []


def test_check_source_reports_unlabeled_fences() -> None:
    violations = check_source(Path("guide.md"), "```\nbody\n```\n")

    assert [(violation.rule, violation.line_number) for violation in violations] == [
        ("fence-language", 1)
    ]


def test_check_source_masks_fences_inline_code_and_urls_for_proper_nouns() -> None:
    source = (
        "```python\nCuda Mlx Openai Sglang Huggingface\n```\n"
        "`Cuda` ``Mlx`` ```Openai``` ````Sglang````\n"
        "See https://example.test/Cuda/Openai and www.example.test/Sglang.\n"
        "Cuda Mlx Openai Sglang Huggingface\n"
    )

    violations = check_source(Path("guide.md"), source)

    assert [
        (violation.rule, violation.line_number, violation.detail)
        for violation in violations
    ] == [
        ("proper-noun", 6, "use CUDA instead of Cuda"),
        ("proper-noun", 6, "use MLX instead of Mlx"),
        ("proper-noun", 6, "use OpenAI instead of Openai"),
        ("proper-noun", 6, "use SGLang instead of Sglang"),
        ("proper-noun", 6, "use Hugging Face instead of Huggingface"),
    ]


def test_check_source_matches_proper_nouns_as_whole_words() -> None:
    source = "CUDA MLX OpenAI SGLang Hugging Face CudaX Mlx2 Sglang_name\n"

    violations = check_source(Path("guide.md"), source)

    assert violations == []


def test_check_source_masks_multiline_inline_code_for_proper_nouns() -> None:
    source = "Inline ````Mlx\nHuggingface````\nCuda\n"

    violations = check_source(Path("guide.md"), source)

    assert [(violation.line_number, violation.detail) for violation in violations] == [
        (3, "use CUDA instead of Cuda")
    ]


def test_check_source_handles_mismatched_inline_code_delimiters() -> None:
    malformed_ending = "`Cuda``\n"
    valid_double_delimiters = "``Cuda ` literal``\n"

    malformed_violations = check_source(Path("guide.md"), malformed_ending)
    valid_violations = check_source(Path("guide.md"), valid_double_delimiters)

    assert [
        (violation.rule, violation.line_number) for violation in malformed_violations
    ] == [("proper-noun", 1)]
    assert valid_violations == []


def test_check_docs_enforces_basename_pattern_except_root_readme(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    nested_root = docs_root / "nested"
    nested_root.mkdir(parents=True)
    (docs_root / "README.md").write_text("# Root\n", encoding="utf-8")
    (docs_root / "bad-name.md").write_text("# Bad\n", encoding="utf-8")
    (nested_root / "README.md").write_text("# Nested\n", encoding="utf-8")
    (docs_root / "index.rst").write_text(
        ".. toctree::\n\n   bad-name.md\n   nested/README.md\n", encoding="utf-8"
    )

    violations = check_docs(docs_root)

    assert [
        violation.path.relative_to(docs_root).as_posix()
        for violation in violations
        if violation.rule == "filename-separators"
    ] == ["bad-name.md", "nested/README.md"]
    assert [
        violation.path.relative_to(docs_root).as_posix()
        for violation in violations
        if violation.rule == "missing-toctree"
    ] == []


def test_check_docs_resolves_root_relative_titled_and_extensionless_toctree_entries(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "index.rst").write_text(
        """An ordinary reference to outside.md must not publish that page.

.. toctree::
   :maxdepth: 1

   root_page
   navigation/index.rst
   Nested page <navigation/titled_page>
   /root_absolute.md
   https://example.test/external
   self
""",
        encoding="utf-8",
    )
    navigation_root = docs_root / "navigation"
    navigation_root.mkdir()
    (navigation_root / "index.rst").write_text(
        """.. toctree::

   child/relative_page
""",
        encoding="utf-8",
    )
    (navigation_root / "child").mkdir()
    (navigation_root / "child" / "relative_page.md").write_text(
        "# Relative\n", encoding="utf-8"
    )
    (docs_root / "root_page.md").write_text("# Root\n", encoding="utf-8")
    (navigation_root / "titled_page.md").write_text("# Titled\n", encoding="utf-8")
    (docs_root / "root_absolute.md").write_text("# Absolute\n", encoding="utf-8")
    orphan_path = docs_root / "orphan.md"
    orphan_path.write_text("# Orphan\n", encoding="utf-8")
    (docs_root / "outside.md").write_text("# Outside\n", encoding="utf-8")

    violations = check_docs(docs_root)

    assert [
        violation.path.relative_to(docs_root).as_posix()
        for violation in violations
        if violation.rule == "missing-toctree"
    ] == ["orphan.md", "outside.md"]


def test_check_docs_preserves_known_toctree_exemptions(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    (docs_root / "design").mkdir(parents=True)
    (docs_root / "developer_reference").mkdir()
    exempt_paths = (
        "README.md",
        "design/gpu_radix_hash.md",
        "design/refactor_rfc.md",
        "developer_reference/mlx.md",
        "get_started/apiserver_quickstart.md",
        "basic_usage/python_router.md",
    )
    for relative_path in exempt_paths:
        path = docs_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "# Exempt\n"
        if relative_path == "design/gpu_radix_hash.md":
            content = "Cuda 😀\n\n```\nbody\n```\n"
        else:
            pass
        path.write_text(content, encoding="utf-8")

    violations = check_docs(docs_root)

    assert [
        violation for violation in violations if violation.rule == "missing-toctree"
    ] == []
    assert {
        violation.rule
        for violation in violations
        if violation.path.relative_to(docs_root).as_posix()
        == "design/gpu_radix_hash.md"
    } == {"no-emoji", "proper-noun", "fence-language"}


def test_check_docs_reports_every_cross_file_duplicate_location(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    body_lines = ["duplicated content line " * 2 for _ in range(6)]
    body_with_trailing_spaces = "\n".join(f"{line}   " for line in body_lines)
    body_without_trailing_spaces = "\n".join(body_lines)
    (docs_root / "index.rst").write_text(
        ".. toctree::\n\n   first\n   second\n", encoding="utf-8"
    )
    (docs_root / "first.md").write_text(
        make_fenced_source(body_with_trailing_spaces)
        + make_fenced_source(body_with_trailing_spaces),
        encoding="utf-8",
    )
    (docs_root / "second.md").write_text(
        make_fenced_source(body_without_trailing_spaces), encoding="utf-8"
    )

    violations = check_docs(docs_root)

    duplicate_violations = [
        violation for violation in violations if violation.rule == "duplicated-block"
    ]
    assert [
        (violation.path.relative_to(docs_root).as_posix(), violation.line_number)
        for violation in duplicate_violations
    ] == [("first.md", 1), ("first.md", 9), ("second.md", 1)]
    assert all(
        "first.md:1" in violation.detail
        and "first.md:9" in violation.detail
        and "second.md:1" in violation.detail
        for violation in duplicate_violations
    )


def test_check_docs_does_not_report_duplicates_within_one_file(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    body_text = "\n".join("repeatable line with enough content " * 2 for _ in range(6))
    (docs_root / "README.md").write_text(
        make_fenced_source(body_text) + make_fenced_source(body_text),
        encoding="utf-8",
    )

    violations = check_docs(docs_root)

    assert [
        violation for violation in violations if violation.rule == "duplicated-block"
    ] == []


def test_check_docs_applies_duplicate_size_boundaries(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "index.rst").write_text(
        ".. toctree::\n\n"
        "   exact_one\n   exact_two\n"
        "   short_one\n   short_two\n"
        "   characters_one\n   characters_two\n",
        encoding="utf-8",
    )
    exact_boundary_lines = ["a" * 33] * 5 + ["a" * 30]
    exact_boundary_body = "\n".join(exact_boundary_lines)
    short_boundary_body = "\n".join(["b" * 40] * 5)
    character_boundary_body = "\n".join(["c" * 33] * 5 + ["c" * 29])
    for stem, body_text in (
        ("exact_one", exact_boundary_body),
        ("exact_two", exact_boundary_body),
        ("short_one", short_boundary_body),
        ("short_two", short_boundary_body),
        ("characters_one", character_boundary_body),
        ("characters_two", character_boundary_body),
    ):
        (docs_root / f"{stem}.md").write_text(
            make_fenced_source(body_text), encoding="utf-8"
        )

    violations = check_docs(docs_root)

    assert [
        violation.path.relative_to(docs_root).as_posix()
        for violation in violations
        if violation.rule == "duplicated-block"
    ] == ["exact_one.md", "exact_two.md"]
