# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the if/else lint hook."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_if_else import (
    InvalidRewriteError,
    apply_else_blocks,
    fix_file,
    run_fix,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_if_else.py"
PROBE_PACKAGE = "lint_if_else_probe"


def run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER_PATH), *args],
        check=False,
        capture_output=True,
        text=True,
    )


@contextmanager
def probe_model_file(source: str) -> Iterator[Path]:
    probe_dir = REPO_ROOT / "sglang_omni" / "models" / PROBE_PACKAGE
    probe = probe_dir / "runner.py"
    probe_dir.mkdir(parents=True)
    probe.write_text(source, encoding="utf-8")
    try:
        yield probe
    finally:
        probe.unlink(missing_ok=True)
        if probe_dir.exists():
            probe_dir.rmdir()
        else:
            pass


def test_current_sglang_omni_tree_is_clean() -> None:
    result = run_checker()
    assert result.returncode == 0, result.stderr


def test_if_elif_else_and_expressions_are_clean() -> None:
    source = """
def choose(flag, items):
    if flag:
        return 1
    elif items:
        return 2
    else:
        return 3

value = 1 if flag else 0
kept = [item for item in items if item]
"""
    with probe_model_file(source) as probe:
        result = run_checker(str(probe))
        assert result.returncode == 0, result.stderr


def test_noqa_comment_does_not_exempt_a_bare_if() -> None:
    source = "def close(session):\n    if session is not None:  # noqa: if-else\n        session.close()\n"
    with probe_model_file(source) as probe:
        result = run_checker(str(probe))
        assert result.returncode == 1
        assert "At least use `else: pass` to fix this lint" in result.stderr
        assert "python scripts/check_if_else.py --fix" in result.stderr


def test_fix_fills_nested_one_line_and_elif() -> None:
    source = (
        "def run(flag, nested):\n"
        "    if flag:\n"
        "        if nested:\n"
        "            return 1\n"
        "    elif nested:\n"
        "        return 2\n"
        "    if flag: return 3\n"
    )
    expected = (
        "def run(flag, nested):\n"
        "    if flag:\n"
        "        if nested:\n"
        "            return 1\n"
        "        else:\n"
        "            pass\n"
        "    elif nested:\n"
        "        return 2\n"
        "    else:\n"
        "        pass\n"
        "    if flag: return 3\n"
        "    else:\n"
        "        pass\n"
    )
    with probe_model_file(source) as probe:
        result = run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        assert probe.read_text(encoding="utf-8") == expected
        again = run_checker(str(probe))
        assert again.returncode == 0, again.stderr


def test_new_bare_if_fails_the_default_scan() -> None:
    with probe_model_file("def load():\n    if True:\n        return 1\n") as probe:
        result = run_checker()
        assert result.returncode == 1
        assert f"{PROBE_PACKAGE}/runner.py" in result.stderr
        assert probe.is_file()


def check_without_rewriting(source: str) -> int:
    with probe_model_file(source) as probe:
        original_source = probe.read_bytes()
        result = run_checker(str(probe))
        assert probe.read_bytes() == original_source
        return result.returncode


def fix_and_run(source: str) -> str:
    """Fix a valid probe, verify idempotence, and return its execution output."""
    compile(source, "probe.py", "exec")
    with probe_model_file(source) as probe:
        result = run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        rewritten_source = probe.read_bytes()
        execution = subprocess.run(
            [sys.executable, str(probe)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert execution.stderr == ""

        check_result = run_checker(str(probe))
        assert check_result.returncode == 0, check_result.stderr
        second_fix_result = run_checker("--fix", str(probe))
        assert second_fix_result.returncode == 0, second_fix_result.stderr
        assert probe.read_bytes() == rewritten_source
        return execution.stdout


def test_inner_else_does_not_complete_outer_if() -> None:
    source = """
if outer:
    if inner:
        pass
    else:
        pass
"""
    assert check_without_rewriting(source) == 1


def test_outer_else_does_not_complete_inner_if() -> None:
    source = """
if outer:
    if inner:
        pass
else:
    pass
"""
    assert check_without_rewriting(source) == 1


def test_loop_else_does_not_complete_if() -> None:
    source = """
if flag:
    for value in values:
        pass
    else:
        pass
"""
    assert check_without_rewriting(source) == 1


def test_elif_chain_requires_final_else() -> None:
    source = """
if first:
    pass
elif second:
    pass
elif third:
    pass
"""
    assert check_without_rewriting(source) == 1


def test_raise_still_requires_else() -> None:
    source = """
if flag:
    raise ValueError("invalid")
"""
    assert check_without_rewriting(source) == 1


def test_complete_nested_branches_are_allowed() -> None:
    source = """
if outer:
    if inner:
        pass
    else:
        pass
else:
    pass
"""
    assert check_without_rewriting(source) == 0


def test_multiline_condition_with_inline_suites_is_allowed() -> None:
    source = """
if (
    flag
): value = 1
else: value = 2
"""
    assert check_without_rewriting(source) == 0


def test_conditional_expressions_and_comprehensions_are_allowed() -> None:
    source = """
value = 1 if flag else 2
values = [value for value in candidates if value]
"""
    assert check_without_rewriting(source) == 0


def test_fix_preserves_multiline_condition_with_inline_suite() -> None:
    source = """
if (
    True
): print("value")
"""
    assert fix_and_run(source) == "value\n"


def test_fix_preserves_unicode_line_separator_in_string() -> None:
    source = """
value = "a\u2028b"
if True:
    print(value)
"""
    assert fix_and_run(source) == "a\u2028b\n"


def test_fix_preserves_form_feed_indentation() -> None:
    source = "if True:\n\f    print('value')\n"
    assert fix_and_run(source) == "value\n"


def test_fix_handles_nested_ifs_sharing_end_line() -> None:
    source = """
if True:
    if True:
        print("value")
"""
    assert fix_and_run(source) == "value\n"


def test_fix_handles_missing_final_newline() -> None:
    source = "if True: print('value')"
    assert fix_and_run(source) == "value\n"


def test_fix_preserves_crlf_newlines() -> None:
    with probe_model_file("") as probe:
        probe.write_bytes(b"if True:\r\n    print('value')\r\n")
        result = run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        rewritten_source = probe.read_bytes()
        assert b"\r\n" in rewritten_source
        assert b"\n" not in rewritten_source.replace(b"\r\n", b"")


def test_fix_never_leaves_invalid_python_on_disk() -> None:
    source = """
if (
    True
): print("value")
"""
    compile(source, "probe.py", "exec")
    with probe_model_file(source) as probe:
        original_source = probe.read_bytes()
        result = run_checker("--fix", str(probe))
        assert result.returncode in (0, 2), result.stderr
        if result.returncode == 2:
            assert probe.read_bytes() == original_source
        else:
            compile(probe.read_bytes(), str(probe), "exec")
            check_result = run_checker(str(probe))
            assert check_result.returncode == 0, check_result.stderr


def test_fix_rejects_invalid_output_before_writing() -> None:
    source = "if True: print('value')\n"
    with probe_model_file(source) as probe:
        original_source = probe.read_bytes()
        with patch(
            "scripts.check_if_else.apply_else_blocks", return_value="if True:\nelse:\n"
        ):
            with pytest.raises(InvalidRewriteError):
                fix_file(probe)
        assert probe.read_bytes() == original_source


def test_fix_leaves_an_invalid_rewrite_unchanged_and_fixes_the_rest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    unsafe = tmp_path / "unsafe.py"
    unsafe.write_text("unsafe = True\nif unsafe: pass\n", encoding="utf-8")
    safe = tmp_path / "safe.py"
    safe.write_text("if True:\n    pass\n", encoding="utf-8")
    original_unsafe = unsafe.read_bytes()

    def rewrite(source: str, nodes: list[ast.If]) -> str:
        if source.startswith("unsafe"):
            return "if True:\nelse:\n"
        else:
            return apply_else_blocks(source, nodes)

    with patch("scripts.check_if_else.apply_else_blocks", side_effect=rewrite):
        exit_code = run_fix([unsafe, safe])
    assert exit_code == 1
    assert unsafe.read_bytes() == original_unsafe
    assert "else:" in safe.read_text(encoding="utf-8")
    error_output = capsys.readouterr().err
    assert "left unchanged" in error_output
    assert "failed to parse" not in error_output


def test_fix_handles_inline_suite_with_backslash_continuation() -> None:
    source = "def f(x):\n    if x: y = 1; \\\n  z = 2\n    return x\n"
    with probe_model_file(source) as probe:
        result = run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        compile(probe.read_text(encoding="utf-8"), str(probe), "exec")


def test_fix_preserves_crlf_without_final_newline() -> None:
    with probe_model_file("") as probe:
        probe.write_bytes(b"value = 1\r\nif True: print(value)")
        result = run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        rewritten_source = probe.read_bytes()
        assert b"\r\n" in rewritten_source
        assert b"\n" not in rewritten_source.replace(b"\r\n", b"")
        compile(rewritten_source, str(probe), "exec")


def test_fix_preserves_carriage_return_newlines() -> None:
    with probe_model_file("") as probe:
        probe.write_bytes(b"if True:\r    print('value')\r")
        result = run_checker("--fix", str(probe))
        assert result.returncode == 0, result.stderr
        rewritten_source = probe.read_bytes()
        assert b"\r" in rewritten_source
        assert b"\n" not in rewritten_source
        compile(rewritten_source, str(probe), "exec")


def test_fix_handles_multiline_elif_inside_tab_indentation() -> None:
    # note (haiyang): Explicit tab escapes keep the tested indentation visible.
    source = (
        "def choose():\n"
        "\tif False: return 'first'\n"
        "\telif (\n"
        "\t\tTrue\n"
        "\t): return 'second'\n"
        "print(choose())\n"
    )
    assert fix_and_run(source) == "second\n"
