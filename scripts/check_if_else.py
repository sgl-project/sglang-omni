#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lint if statements that have no else. Checking does not rewrite files.

Every Python file under sglang_omni/ is in scope, including files added later.
Vendor copies are ignored. An if/elif chain is legal only when it ends in else.
A bare if, including one that returns or raises, is a violation. There is no
noqa exemption.

The default run only reports violations. Prefer a real else branch. At least
write else: pass. --fix is the same insertion, kept for when that is the
branch you want. Pre-commit does not pass --fix.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "sglang_omni"
VENDOR_ROOT = SOURCE_ROOT / "vendor"
PASS_INDENT = "    "


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    col: int
    end_lineno: int


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()
    else:
        pass


def is_in_scope(path: Path) -> bool:
    resolved = path.resolve()
    if not resolved.is_relative_to(SOURCE_ROOT):
        return False
    elif resolved.is_relative_to(VENDOR_ROOT):
        return False
    else:
        return resolved.suffix == ".py"


def iter_default_files() -> list[Path]:
    return sorted(path for path in SOURCE_ROOT.rglob("*.py") if is_in_scope(path))


def resolve_targets(raw_paths: list[str]) -> list[Path]:
    if not raw_paths:
        return iter_default_files()
    else:
        return [Path(item) for item in raw_paths if is_in_scope(Path(item))]


def missing_else_nodes(tree: ast.AST) -> list[ast.If]:
    found: list[ast.If] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and not node.orelse:
            found.append(node)
        else:
            pass
    return found


def check_source(path: Path, source: str) -> list[Violation]:
    tree = ast.parse(source, filename=str(path))
    violations: list[Violation] = []
    for node in missing_else_nodes(tree):
        end_lineno = node.end_lineno or node.lineno
        violations.append(Violation(path, node.lineno, node.col_offset, end_lineno))
    return violations


def check_file(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    return check_source(path, source)


def line_newline(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    else:
        return "\n"


def terminated_line(line: str) -> tuple[str, str]:
    if line.endswith("\r\n") or line.endswith("\n"):
        return line, line_newline(line)
    else:
        return line + "\n", "\n"


def if_indent(lines: list[str], node: ast.If) -> str:
    line = lines[node.lineno - 1]
    return line[: node.col_offset]


def pass_indent(lines: list[str], node: ast.If, indent: str) -> str:
    for statement in node.body:
        if statement.lineno > node.lineno:
            body_line = lines[statement.lineno - 1]
            return body_line[: statement.col_offset]
        else:
            pass
    return indent + PASS_INDENT


def else_block(indent: str, body_indent: str, newline: str) -> list[str]:
    return [f"{indent}else:{newline}", f"{body_indent}pass{newline}"]


def apply_else_blocks(source: str, nodes: list[ast.If]) -> str:
    if not nodes:
        return source
    else:
        pass
    lines = source.splitlines(keepends=True)
    ordered = sorted(
        nodes,
        key=lambda node: (-(node.end_lineno or node.lineno), node.col_offset),
    )
    for node in ordered:
        end_lineno = node.end_lineno or node.lineno
        indent = if_indent(lines, node)
        body_indent = pass_indent(lines, node, indent)
        index = end_lineno - 1
        lines[index], newline = terminated_line(lines[index])
        insert_at = index + 1
        lines[insert_at:insert_at] = else_block(indent, body_indent, newline)
    return "".join(lines)


def fix_file(path: Path) -> tuple[int, list[Violation]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    nodes = missing_else_nodes(tree)
    if not nodes:
        return 0, []
    else:
        pass
    rewritten = apply_else_blocks(source, nodes)
    if rewritten == source:
        return 0, check_source(path, source)
    else:
        path.write_text(rewritten, encoding="utf-8")
    leftover = check_file(path)
    return max(len(nodes) - len(leftover), 0), leftover


def format_violation(violation: Violation) -> str:
    rel = repo_relative(violation.path)
    return (
        f"{rel}:{violation.lineno}:{violation.col}: "
        "if without else. At least use `else: pass` to fix this lint"
    )


def report_violations(violations: list[Violation]) -> int:
    if not violations:
        return 0
    else:
        pass
    for violation in violations:
        print(format_violation(violation), file=sys.stderr)
    print(
        f"{len(violations)} if statement(s) without else in sglang_omni/. "
        "At least use `else: pass` to fix this lint, "
        "or run `python scripts/check_if_else.py --fix`.",
        file=sys.stderr,
    )
    return 1


def parse_error(path: Path, exc: SyntaxError) -> int:
    print(f"{path}: failed to parse: {exc}", file=sys.stderr)
    return 2


def run_check(paths: list[Path]) -> int:
    violations: list[Violation] = []
    for path in paths:
        try:
            violations.extend(check_file(path))
        except SyntaxError as exc:
            return parse_error(path, exc)
        else:
            pass
    return report_violations(violations)


def run_fix(paths: list[Path]) -> int:
    remaining: list[Violation] = []
    filled = 0
    files = 0
    for path in paths:
        try:
            count, leftover = fix_file(path)
        except SyntaxError as exc:
            return parse_error(path, exc)
        else:
            pass
        if count:
            files += 1
            filled += count
        else:
            pass
        remaining.extend(leftover)
    if filled:
        print(f"added else to {filled} if statement(s) in {files} file(s)")
    else:
        pass
    return report_violations(remaining)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", help="Python files (defaults to sglang_omni/)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Last resort: insert else: pass after every if that has no else",
    )
    args = parser.parse_args(argv)
    targets = resolve_targets(args.paths)
    if args.fix:
        return run_fix(targets)
    else:
        return run_check(targets)


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
