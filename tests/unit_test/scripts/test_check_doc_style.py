# SPDX-License-Identifier: Apache-2.0
"""Exercise documentation emoji and fence checks with synthetic Markdown inputs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("emoji")

from scripts.check_doc_style import check_docs, check_source

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/check_doc_style.py"


@pytest.mark.parametrize(
    "sequence",
    [
        "©",
        "®",
        "™",
        "☀",
        "♟",
        "↔",
        "0",
        "9",
        "©\ufe0e",
        "↔\ufe0e",
        "⌚\ufe0e",
        "#\ufe0e",
        "*\ufe0e",
    ],
)
def test_check_source_allows_text_symbols_and_text_presentation(
    sequence: str,
) -> None:
    assert check_source(Path("guide.md"), f"Allowed {sequence}.\n") == []


@pytest.mark.parametrize(
    "sequence, reported_sequence",
    [
        ("©\ufe0f", "©\ufe0f"),
        ("☀\ufe0f", "☀\ufe0f"),
        ("⌚\ufe0f", "⌚\ufe0f"),
        ("⌚", "⌚"),
        ("😀\ufe0e", "😀"),
    ],
)
def test_check_source_reports_emoji_presentation(
    sequence: str, reported_sequence: str
) -> None:
    violations = check_source(Path("guide.md"), f"Emoji: {sequence}\n")

    assert len(violations) == 1
    assert violations[0].rule == "no-emoji"
    assert violations[0].line_number == 1
    assert violations[0].detail == f"remove emoji sequence {reported_sequence!r}"


@pytest.mark.parametrize(
    "sequence",
    [
        "🏽",
        "1\u20e3",
        "#\ufe0f\u20e3",
        "#\u20e3",
        "*\ufe0f\u20e3",
        "*\u20e3",
        "👍🏽",
        "👩‍💻",
        "👩🏻‍💻",
        "🏳‍🌈",
        "👁‍🗨",
        "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
        "\U0001faea",
    ],
)
def test_check_source_reports_complete_emoji_sequences(sequence: str) -> None:
    violations = check_source(Path("guide.md"), f"Sequence: {sequence}\n")

    assert len(violations) == 1
    assert violations[0].rule == "no-emoji"
    assert violations[0].line_number == 1
    assert violations[0].detail == f"remove emoji sequence {sequence!r}"


def test_check_source_reports_non_rgi_emoji_bases_separately() -> None:
    violations = check_source(Path("guide.md"), "Non-RGI 😀‍😀 sequence.\n")

    assert [
        (violation.rule, violation.line_number, violation.detail)
        for violation in violations
    ] == [
        ("no-emoji", 1, "remove emoji sequence '😀'"),
        ("no-emoji", 1, "remove emoji sequence '😀'"),
    ]


@pytest.mark.parametrize(
    "sequence, expected_sequences",
    [
        ("\U0001f1e8", ["\U0001f1e8"]),
        ("\U0001f1e8\U0001f1f3", ["\U0001f1e8\U0001f1f3"]),
        ("\U0001f1ff\U0001f1ff", ["\U0001f1ff", "\U0001f1ff"]),
        ("1", []),
        ("#", []),
        ("*", []),
        ("1\ufe0f", ["1\ufe0f"]),
        ("#\ufe0f", ["#\ufe0f"]),
        ("*\ufe0f", ["*\ufe0f"]),
        ("1\ufe0f\u20e3", ["1\ufe0f\u20e3"]),
        ("1\ufe0e", []),
        ("A\ufe0f", []),
    ],
)
def test_check_source_matches_supplement_codepoint_table(
    sequence: str, expected_sequences: list[str]
) -> None:
    violations = check_source(Path("guide.md"), f"Codepoint {sequence}.\n")

    assert [
        (violation.rule, violation.line_number, violation.detail)
        for violation in violations
    ] == [
        ("no-emoji", 1, f"remove emoji sequence {expected_sequence!r}")
        for expected_sequence in expected_sequences
    ]


def test_check_source_deduplicates_supplements_already_in_sequences() -> None:
    source = "\U0001f1e8\U0001f1f3\U0001f1ff\U0001f1ff"
    source += "1\ufe0f\u20e31\u20e31\ufe0f\n"

    violations = check_source(Path("guide.md"), source)

    assert [violation.detail for violation in violations] == [
        "remove emoji sequence '🇨🇳'",
        "remove emoji sequence '1️⃣'",
        "remove emoji sequence '1⃣'",
        "remove emoji sequence '🇿'",
        "remove emoji sequence '🇿'",
        "remove emoji sequence '1️'",
    ]


def test_check_source_ignores_emoji_inside_matching_fences() -> None:
    source = (
        "  ````python\n"
        "😀 inside a longer backtick fence\n"
        "   ```\n"
        "😀 a shorter marker does not close it\n"
        "  ```` \t\n"
        "😀 outside the backtick fence\n"
        "   ~~~text\n"
        "😀 inside a tilde fence\n"
        "  ```\n"
        "😀 a different marker does not close it\n"
        "  ~~~\t\n"
        "😀 outside the tilde fence\n"
    )

    violations = check_source(Path("guide.md"), source)

    assert [(violation.line_number, violation.detail) for violation in violations] == [
        (6, "remove emoji sequence '😀'"),
        (12, "remove emoji sequence '😀'"),
    ]


def test_check_source_does_not_treat_nested_fences_as_top_level() -> None:
    source = "    ```python\n    😀 nested fence content\n    ```\n> ```\n> 😀\n> ```\n"

    violations = check_source(Path("guide.md"), source)

    assert [
        (violation.rule, violation.line_number, violation.detail)
        for violation in violations
    ] == [
        ("no-emoji", 2, "remove emoji sequence '😀'"),
        ("no-emoji", 5, "remove emoji sequence '😀'"),
    ]


def test_check_source_requires_closing_fence_trailing_text_to_be_absent() -> None:
    source = (
        "```python\n😀 inside\n``` trailing text\n😀 still inside\n```\n😀 outside\n"
    )

    violations = check_source(Path("guide.md"), source)

    assert [(violation.line_number, violation.detail) for violation in violations] == [
        (6, "remove emoji sequence '😀'"),
    ]


def test_check_source_reports_unclosed_fence_at_opening_line() -> None:
    violations = check_source(Path("guide.md"), "# Guide\n\n```python\n😀 inside\n")

    assert len(violations) == 1
    assert (
        violations[0].line_number,
        violations[0].rule,
        violations[0].detail,
    ) == (3, "unclosed-fence", "unclosed code fence")


@pytest.mark.parametrize(
    "opening_fence, closing_fence",
    [("```", "```"), ("~~~", "~~~"), ("   ``` \t", "   ```")],
)
def test_check_source_reports_empty_fence_info_at_opening_line(
    opening_fence: str, closing_fence: str
) -> None:
    source = f"# Guide\n\n{opening_fence}\n😀 inside\n{closing_fence}\n"

    violations = check_source(Path("guide.md"), source)

    assert [
        (violation.rule, violation.line_number, violation.detail)
        for violation in violations
    ] == [("fence-language", 3, "opening fence needs a non-empty info string")]


@pytest.mark.parametrize(
    "opening_fence, closing_fence",
    [
        ("```text", "```"),
        ("~~~mermaid", "~~~"),
        ("```{raw}", "```"),
        ("   ````custom-format option=value", "   ````\t"),
    ],
)
def test_check_source_accepts_any_nonempty_fence_info(
    opening_fence: str, closing_fence: str
) -> None:
    source = f"{opening_fence}\n😀 inside\n{closing_fence}\n"

    assert check_source(Path("guide.md"), source) == []


def test_check_source_checks_inline_code_and_urls() -> None:
    source = (
        "Inline `😀` code.\nhttps://example.test/😀\n"
        "[link](https://example.test/😀)\n"
    )

    violations = check_source(Path("guide.md"), source)

    assert [(violation.line_number, violation.detail) for violation in violations] == [
        (1, "remove emoji sequence '😀'"),
        (2, "remove emoji sequence '😀'"),
        (3, "remove emoji sequence '😀'"),
    ]


def test_check_docs_scans_all_markdown_files_recursively(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    nested_root = docs_root / "nested" / "deeper"
    nested_root.mkdir(parents=True)
    (docs_root / "clean.md").write_text("# Clean\n", encoding="utf-8")
    nested_path = nested_root / "guide.md"
    nested_path.write_text("# Guide\n\nNested 😀.\n", encoding="utf-8")
    (docs_root / "ignored.txt").write_text("😀\n", encoding="utf-8")

    violations = check_docs(docs_root)

    assert len(violations) == 1
    assert violations[0].path == nested_path
    assert violations[0].line_number == 3
    assert violations[0].rule == "no-emoji"
    assert violations[0].detail == "remove emoji sequence '😀'"


def copy_script_fixture(tmp_path: Path, document_text: str) -> tuple[Path, Path]:
    """Create an isolated scripts/docs tree for CLI checks."""
    repository_root = tmp_path / "fixture"
    script_path = repository_root / "scripts" / "check_doc_style.py"
    document_path = repository_root / "docs" / "guide.md"
    script_path.parent.mkdir(parents=True)
    document_path.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script_path)
    document_path.write_text(document_text, encoding="utf-8")
    return repository_root, document_path


@pytest.mark.parametrize(
    "document_text, expected_exit_code",
    [("# Clean\n", 0), ("# Bad\n\n😀\n", 1), ("# Bad\n\n```\n😀\n", 1)],
)
def test_cli_scans_the_copied_script_fixture(
    tmp_path: Path, document_text: str, expected_exit_code: int
) -> None:
    repository_root, document_path = copy_script_fixture(tmp_path, document_text)
    script_path = repository_root / "scripts" / "check_doc_style.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == expected_exit_code
    if expected_exit_code == 0:
        assert result.stderr == ""
    else:
        if "```" in document_text:
            expected_stderr = (
                f"docs/{document_path.name}:3: fence-language: "
                "opening fence needs a non-empty info string\n"
                f"docs/{document_path.name}:3: unclosed-fence: unclosed code fence\n"
            )
        else:
            expected_stderr = (
                f"docs/{document_path.name}:3: no-emoji: "
                "remove emoji sequence '😀'\n"
            )
        assert result.stderr == expected_stderr


def test_cli_rejects_arguments_with_exit_status_two(tmp_path: Path) -> None:
    repository_root, _ = copy_script_fixture(tmp_path, "# Clean\n")
    script_path = repository_root / "scripts" / "check_doc_style.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--unexpected"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "unrecognized arguments" in result.stderr
