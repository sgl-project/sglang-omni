#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lint Markdown document heading structure. Checking does not rewrite files.

Every Markdown file under docs/ is in scope, including files added later.

Rules, enforced on ATX headings outside fenced code blocks:

  1. A file must have exactly one level-1 heading (``#``). Zero or more than
     one H1 is a violation.
  2. Headings may not go deeper than level 3 (``###``). Any ``####`` (or deeper)
     is an immediate error.

There is no noqa exemption. Run without arguments to check every doc file, or
pass explicit paths to check a subset.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"

# ATX heading: up to three leading spaces (CommonMark), 1-6 '#', a required
# space, then the title, optionally closed by trailing '#' characters.
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*#*[ \t]*$")
# Fence opener/closer: a run of at least three backticks or tildes, up to
# three leading spaces. The opener may carry an info string after the run.
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    col: int
    rule: str
    detail: str


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def is_in_scope(path: Path) -> bool:
    resolved = path.resolve()
    if not resolved.is_relative_to(DOCS_ROOT):
        return False
    else:
        return resolved.suffix == ".md"


def iter_default_files() -> list[Path]:
    return sorted(path for path in DOCS_ROOT.rglob("*.md") if is_in_scope(path))


def resolve_targets(raw_paths: list[str]) -> list[Path]:
    if not raw_paths:
        return iter_default_files()
    else:
        return [Path(item) for item in raw_paths if is_in_scope(Path(item))]


def iter_headings(source: str):
    """Yield (lineno, level, col) for each ATX heading outside code fences."""
    fence_char: str | None = None
    for lineno, line in enumerate(source.splitlines(), start=1):
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)[0]
            if fence_char is None:
                fence_char = marker
            elif marker == fence_char:
                fence_char = None
            else:
                pass
            continue
        if fence_char is not None:
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            col = len(line) - len(line.lstrip()) + 1
            yield lineno, level, col
        else:
            pass


def check_source(path: Path, source: str) -> list[Violation]:
    violations: list[Violation] = []
    h1_lines: list[tuple[int, int]] = []
    for lineno, level, col in iter_headings(source):
        if level == 1:
            h1_lines.append((lineno, col))
        elif level >= 4:
            violations.append(
                Violation(
                    path,
                    lineno,
                    col,
                    "heading-too-deep",
                    f"heading level {level} (####) is not allowed; max is level 3 (###)",
                )
            )
        else:
            pass
    if len(h1_lines) == 0:
        violations.append(
            Violation(
                path,
                1,
                1,
                "missing-h1",
                "document must have exactly one level-1 heading (#)",
            )
        )
    elif len(h1_lines) > 1:
        first = h1_lines[0]
        others = ", ".join(f"line {n}" for n, _ in h1_lines[1:])
        violations.append(
            Violation(
                path,
                first[0],
                first[1],
                "too-many-h1",
                f"document must have exactly one level-1 heading (#); "
                f"found {len(h1_lines)} (also at {others})",
            )
        )
    else:
        pass
    return violations


def check_file(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    return check_source(path, source)


def format_violation(violation: Violation) -> str:
    rel = repo_relative(violation.path)
    return f"{rel}:{violation.lineno}:{violation.col}: {violation.detail}"


def report_violations(violations: list[Violation]) -> int:
    if not violations:
        return 0
    else:
        pass
    for violation in violations:
        print(format_violation(violation), file=sys.stderr)
    print(
        f"{len(violations)} heading violation(s) in docs/. "
        "Each doc must have exactly one '#', and headings may not go deeper than '###'.",
        file=sys.stderr,
    )
    return 1


def run_check(paths: list[Path]) -> int:
    violations: list[Violation] = []
    for path in paths:
        violations.extend(check_file(path))
    return report_violations(violations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Markdown files (defaults to docs/)")
    args = parser.parse_args(argv)
    targets = resolve_targets(args.paths)
    return run_check(targets)


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
