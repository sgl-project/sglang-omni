# SPDX-License-Identifier: Apache-2.0
"""Exercise the documentation style checker against small synthetic inputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_doc_style import check_docs, check_source

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/check_doc_style.py"


def make_fenced_source(body_text: str, info_string: str = "text") -> str:
    """Wrap body text in a Markdown fenced block."""
    return f"```{info_string}\n{body_text}\n```\n"


def test_check_source_ignores_emoji_and_casing_inside_fences() -> None:
    source = """```python
print("😀 Cuda")
```

Prose contains 😀 and Cuda.
"""

    violations, blocks = check_source(Path("guide.md"), source)

    assert {(violation.rule, violation.line_number) for violation in violations} == {
        ("no-emoji", 5),
        ("proper-noun", 5),
    }
    assert len(blocks) == 1
    assert blocks[0].body == 'print("😀 Cuda")'


def test_check_source_masks_inline_code_with_one_to_four_backticks_and_newlines() -> (
    None
):
    source = (
        "`Cuda` ``Openai`` ```Sglang``` "
        "````Mlx\nHuggingface````\n"
        "A prose line without forbidden casing.\n"
    )

    violations, blocks = check_source(Path("guide.md"), source)

    assert violations == []
    assert blocks == []


def test_check_source_masks_inline_url_before_url_scanning() -> None:
    source = "`Cuda https://example.test/Openai`\n"

    violations, blocks = check_source(Path("guide.md"), source)

    assert violations == []
    assert blocks == []


def test_check_source_handles_inline_code_delimiter_edges() -> None:
    malformed_ending = "`Cuda``\n"
    valid_double_delimiters = "``Cuda ` literal``\n"

    malformed_violations, malformed_blocks = check_source(
        Path("guide.md"), malformed_ending
    )
    valid_violations, valid_blocks = check_source(
        Path("guide.md"), valid_double_delimiters
    )

    assert [
        (violation.rule, violation.line_number) for violation in malformed_violations
    ] == [("proper-noun", 1)]
    assert malformed_blocks == []
    assert valid_violations == []
    assert valid_blocks == []


def test_check_source_masks_urls_in_links_and_autolinks() -> None:
    source = (
        "See [the guide](https://example.test/Cuda/Openai) and "
        "<https://example.test/Openai/Cuda>.\n"
        "The mirror is www.example.test/Cuda/Openai.\n"
    )

    violations, blocks = check_source(Path("guide.md"), source)

    assert violations == []
    assert blocks == []


def test_check_source_rejects_an_unlabeled_fence() -> None:
    source = "```\nbody\n```\n"

    violations, blocks = check_source(Path("guide.md"), source)

    assert [(violation.rule, violation.line_number) for violation in violations] == [
        ("fence-language", 1)
    ]
    assert [block.body for block in blocks] == ["body"]


def test_check_source_checks_all_five_proper_nouns_as_whole_words() -> None:
    source = "Cuda Mlx Openai Sglang Huggingface\n"

    violations, _ = check_source(Path("guide.md"), source)

    assert [violation.detail for violation in violations] == [
        "use CUDA instead of Cuda",
        "use MLX instead of Mlx",
        "use OpenAI instead of Openai",
        "use SGLang instead of Sglang",
        "use Hugging Face instead of Huggingface",
    ]


@pytest.mark.parametrize(
    "source",
    [
        "CUDA MLX OpenAI SGLang Hugging Face\n",
        "CudaX Mlx2 Openaish Sglang_name Huggingfacex\n",
    ],
)
def test_check_source_accepts_correct_casing_and_nonmatching_words(source: str) -> None:
    violations, _ = check_source(Path("guide.md"), source)

    assert violations == []


def test_check_source_requires_named_fences_and_valid_closures() -> None:
    source = (
        "```text\n"
        "same character closes this block\n"
        "~~~\n"
        "Cuda 😀\n"
        "```   \n"
        "```raw\n"
        "raw body\n"
        "``` trailing text\n"
        "```\n"
        "```mermaid\n"
        "graph TD\n"
        "```\n"
        "```   \n"
        "invalid opening info\n"
        "```\n"
        "``\n"
        "Cuda\n"
        "plain prose\n"
    )

    violations, blocks = check_source(Path("guide.md"), source)

    assert [(violation.rule, violation.line_number) for violation in violations] == [
        ("fence-language", 13),
        ("proper-noun", 17),
    ]
    assert [block.body for block in blocks] == [
        "same character closes this block\n~~~\nCuda 😀",
        "raw body\n``` trailing text",
        "graph TD",
        "invalid opening info",
    ]


def test_check_source_accepts_named_text_raw_and_mermaid_fences() -> None:
    source = (
        "```text\ntext body\n```\n```{raw}\nraw body\n```\n```mermaid\ngraph TD\n```\n"
    )

    violations, blocks = check_source(Path("guide.md"), source)

    assert violations == []
    assert [block.body for block in blocks] == [
        "text body",
        "raw body",
        "graph TD",
    ]


@pytest.mark.parametrize("basename", ["README.md", "bad-name.md"])
def test_check_docs_exempts_only_root_readme(tmp_path: Path, basename: str) -> None:
    docs_root = tmp_path / "docs"
    (docs_root / "nested").mkdir(parents=True)
    (docs_root / "README.md").write_text("# Root\n", encoding="utf-8")
    (docs_root / "bad-name.md").write_text("# Hyphen\n", encoding="utf-8")
    (docs_root / "nested" / basename).write_text("# Nested\n", encoding="utf-8")

    violations = check_docs(docs_root)

    assert {
        (violation.path.relative_to(docs_root).as_posix(), violation.rule)
        for violation in violations
    } == {
        (f"nested/{basename}", "filename-separators"),
        (f"nested/{basename}", "missing-toctree"),
        ("bad-name.md", "filename-separators"),
        ("bad-name.md", "missing-toctree"),
    }


def test_check_docs_resolves_only_toctree_entries_from_nested_rst_files(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    navigation_root = docs_root / "navigation"
    navigation_root.mkdir(parents=True)
    (docs_root / "index.rst").write_text(
        """An ordinary reference to outside.md must not publish that page.

.. toctree::
   :maxdepth: 1

   navigation/index.rst
   plain
   A titled page <titled>
""",
        encoding="utf-8",
    )
    (navigation_root / "index.rst").write_text(
        """.. toctree::

   child/relative
""",
        encoding="utf-8",
    )
    (navigation_root / "child").mkdir()
    (navigation_root / "child" / "relative.md").write_text(
        "# Relative\n", encoding="utf-8"
    )
    (docs_root / "plain.md").write_text("# Plain\n", encoding="utf-8")
    (docs_root / "titled.md").write_text("# Titled\n", encoding="utf-8")
    (docs_root / "outside.md").write_text("# Outside\n", encoding="utf-8")

    violations = check_docs(docs_root)

    assert [
        violation.path.relative_to(docs_root).as_posix()
        for violation in violations
        if violation.rule == "missing-toctree"
    ] == ["outside.md"]


def test_check_docs_allows_known_navigation_pages_but_rejects_new_design_pages(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    (docs_root / "design").mkdir(parents=True)
    (docs_root / "developer_reference").mkdir()
    for relative_path in (
        "README.md",
        "design/gpu_radix_hash.md",
        "design/refactor_rfc.md",
        "developer_reference/mlx.md",
        "design/new_page.md",
    ):
        path = docs_root / relative_path
        contents = (
            "Cuda 😀\n"
            if relative_path == "design/gpu_radix_hash.md"
            else f"# {path.stem}\n"
        )
        path.write_text(contents, encoding="utf-8")

    violations = check_docs(docs_root)

    assert [
        violation.path.relative_to(docs_root).as_posix()
        for violation in violations
        if violation.rule == "missing-toctree"
    ] == ["design/new_page.md"]
    assert {
        violation.rule
        for violation in violations
        if violation.path.relative_to(docs_root).as_posix()
        == "design/gpu_radix_hash.md"
    } == {"no-emoji", "proper-noun"}


def test_check_docs_exempts_sidebar_omissions_but_keeps_style_rules(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    (docs_root / "get_started").mkdir(parents=True)
    (docs_root / "basic_usage").mkdir()
    relative_paths = (
        "get_started/apiserver_quickstart.md",
        "basic_usage/python_router.md",
    )
    for relative_path in relative_paths:
        (docs_root / relative_path).write_text(
            "# Title\n\nCuda 😀\n\n```\nbody\n```\n",
            encoding="utf-8",
        )

    violations = check_docs(docs_root)

    assert [
        violation.path.relative_to(docs_root).as_posix()
        for violation in violations
        if violation.rule == "missing-toctree"
    ] == []
    for relative_path in relative_paths:
        assert {
            violation.rule
            for violation in violations
            if violation.path.relative_to(docs_root).as_posix() == relative_path
        } == {"no-emoji", "proper-noun", "fence-language"}


def test_check_docs_does_not_report_duplicate_blocks_within_one_file(
    tmp_path: Path,
) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    body_text = "\n".join("repeatable line with enough content " * 2 for _ in range(6))
    (docs_root / "README.md").write_text(
        make_fenced_source(body_text) + make_fenced_source(body_text),
        encoding="utf-8",
    )

    violations = check_docs(docs_root)

    assert violations == []


def test_check_docs_reports_every_duplicate_location_and_strips_trailing_spaces(
    tmp_path: Path,
) -> None:
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

    assert [
        (
            violation.path.relative_to(docs_root).as_posix(),
            violation.line_number,
        )
        for violation in violations
        if violation.rule == "duplicated-block"
    ] == [("first.md", 1), ("first.md", 9), ("second.md", 1)]
    assert all(
        "first.md:1" in violation.detail
        and "first.md:9" in violation.detail
        and "second.md:1" in violation.detail
        for violation in violations
        if violation.rule == "duplicated-block"
    )


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
    character_boundary_body = "\n".join(["c" * 32] * 5 + ["c" * 29])
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


@pytest.mark.parametrize("arguments, exit_code", [([], 0), (["--fix"], 2)])
def test_check_doc_style_cli(arguments: list[str], exit_code: int) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=SCRIPT.parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == exit_code, result.stderr


def test_shorter_fence_does_not_close_longer_opening() -> None:
    source = "````text\n```\nCuda 😀\n````\nOpenai\n"
    violations, _ = check_source(Path("guide.md"), source)
    assert [(entry.rule, entry.line_number) for entry in violations] == [
        ("proper-noun", 5)
    ]
