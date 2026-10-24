#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lint the Cookbook sidebar order in docs/index.rst. Checking does not rewrite files.

The Cookbook toctree in docs/index.rst must follow the model order defined for
PR #2876 (sgl-project/sglang-omni#2876), which mirrors the SGLang-Omni Roadmap
#1081:

  1. Roadmap part: the models listed under **Focus -> Supporting** in #1081,
     in that exact order (ROADMAP_ORDER in scripts/cookbook_order_data.py).
  2. Downloads part: every remaining cookbook entry, sorted by Hugging Face
     download counts in descending order (DOWNLOADS_DESC in
     scripts/cookbook_order_data.py).

All rules live in scripts/cookbook_order_data.py, so a roadmap change or a
download-count change never requires touching this file's logic:

  * Roadmap changed in #1081: update ROADMAP_ORDER there.
  * Download counts moved: update DOWNLOADS_DESC by hand (re-query
    https://huggingface.co/api/models/{repo} for the affected models), then
    update docs/index.rst if ranks changed.
  * New model: add it to ROADMAP_ORDER (roadmap model) or HF_REPO +
    DOWNLOADS_DESC (non-roadmap model). An unregistered cookbook entry fails
    with a message pointing at exactly what to update.

Missing, extra, duplicate, unknown, or misplaced entries are violations and
fail CI. There is no noqa exemption.

Run without arguments to check docs/index.rst, or pass explicit paths.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from cookbook_order_data import DOWNLOADS_DESC, ROADMAP_ORDER

# Paths are resolved against the repository root so every reported location is
# a repo-relative path, which stays stable and clickable in CI logs regardless
# of where the runner checks out the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "docs" / "index.rst"

# Fast membership lookups used throughout the checks. ROADMAP_ORDER keeps the
# fixed positional order; DOWNLOADS_DESC keeps the per-model download counts.
# Models in one set never belong to the other.
ROADMAP_SET = frozenset(ROADMAP_ORDER)
DOWNLOAD_SET = frozenset(DOWNLOADS_DESC)

# A toctree entry is either a bare path ("   cookbook/x.md") or a Sphinx
# explicit-title form ("   Display Name <cookbook/x.md>"). Both are accepted;
# group(1) always captures the target cookbook path, so the two forms are
# handled identically from here on. This matters because a contributor can
# otherwise bypass the checks by hiding a duplicate/unknown model behind an
# explicit title.
_ENTRY_RE = re.compile(r"^   (cookbook/[a-z0-9_.-]+\.md)$")
_EXPLICIT_TITLE_RE = re.compile(r"^   .+ <(cookbook/[a-z0-9_.-]+\.md)>$")


@dataclass(frozen=True)
class Violation:
    """One cookbook-order problem found in a checked file.

    Carries the source location (path + line + column) and a human-readable
    description so the reported message can point the contributor at the
    exact line to fix.
    """

    path: Path
    lineno: int
    col: int
    rule: str
    detail: str


def repo_relative(path: Path) -> str:
    """Render a path relative to the repo root for user-facing messages."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_targets(raw_paths: list[str]) -> list[Path]:
    """Resolve CLI paths; check docs/index.rst when nothing is passed."""
    if not raw_paths:
        return [DEFAULT_TARGET]
    else:
        return [Path(item) for item in raw_paths]


def extract_cookbook_entries(source: str) -> tuple[list[str], dict[str, int]]:
    """Return (entries, lineno_by_entry) for the Cookbook toctree block.

    Walks the reStructuredText source and isolates the toctree whose caption is
    "Cookbook": once that block is entered, every non-blank line that matches a
    cookbook entry (bare path or explicit title) is collected; the first
    non-blank line that matches neither terminates the block. The line number
    of each entry is kept so violations can point at the exact source line.
    """
    lines = source.splitlines()
    entries: list[str] = []
    lineno_by_entry: dict[str, int] = {}
    in_cookbook = False
    for lineno, line in enumerate(lines, start=1):
        # A new ".. toctree::" directive ends any previous block.
        if line.startswith(".. toctree::"):
            in_cookbook = False
            continue
        # Enter the Cookbook block only when its caption is seen.
        if line.strip() == ":caption: Cookbook":
            in_cookbook = True
            continue
        if not in_cookbook:
            continue
        # Skip blank lines inside the block; they do not terminate it.
        if not line.strip():
            continue
        # Bare path or explicit-title form -> keep the entry and its line.
        match = _ENTRY_RE.match(line) or _EXPLICIT_TITLE_RE.match(line)
        if match:
            entry = match.group(1)
            entries.append(entry)
            lineno_by_entry[entry] = lineno
        else:
            # First non-entry line closes the Cookbook block.
            break
    return entries, lineno_by_entry


def check_source(path: Path, source: str) -> list[Violation]:
    """Validate one file's Cookbook block against the data file."""
    violations: list[Violation] = []
    entries, lineno_by_entry = extract_cookbook_entries(source)

    # If the block is absent (or empty), nothing else can be checked.
    if not entries:
        violations.append(
            Violation(path, 1, 1, "cookbook-order", "Cookbook toctree block not found")
        )
        return violations

    # 1. Duplicates: the same cookbook file listed more than once.
    seen: set[str] = set()
    for entry in entries:
        if entry in seen:
            lineno = lineno_by_entry.get(entry, 1)
            violations.append(
                Violation(
                    path, lineno, 1, "cookbook-order", f"duplicate entry '{entry}'"
                )
            )
        else:
            seen.add(entry)

    # 2. Unknown entries: cookbook pages that are not registered in the data
    #    file. The message tells the contributor exactly where to register.
    expected_set = ROADMAP_SET | DOWNLOAD_SET
    actual_set = set(entries)
    for entry in sorted(actual_set - expected_set):
        lineno = lineno_by_entry.get(entry, 1)
        violations.append(
            Violation(
                path,
                lineno,
                1,
                "cookbook-order",
                f"unknown entry '{entry}': add it to ROADMAP_ORDER (roadmap model) or "
                f"to HF_REPO + DOWNLOADS_DESC (non-roadmap model) in "
                f"scripts/cookbook_order_data.py",
            )
        )

    # 3. Roadmap models must stay inside the roadmap part (the first N
    #    entries); a roadmap model moved into the downloads part is invalid
    #    even if the overall sequence is otherwise correct.
    for index, entry in enumerate(entries):
        if entry in ROADMAP_SET and index >= len(ROADMAP_ORDER):
            lineno = lineno_by_entry.get(entry, 1)
            violations.append(
                Violation(
                    path,
                    lineno,
                    1,
                    "cookbook-order",
                    f"'{entry}' is a roadmap (#1081) model and must be in the roadmap part "
                    f"(first {len(ROADMAP_ORDER)} entries), not in the downloads part",
                )
            )

    # 4. Missing entries: registered models that are absent from the toctree.
    for entry in sorted(expected_set - actual_set):
        violations.append(
            Violation(
                path,
                1,
                1,
                "cookbook-order",
                f"missing entry '{entry}' (removed from the cookbook toctree, or not yet added)",
            )
        )

    # 5. Roadmap part: exact positional match against ROADMAP_ORDER. Report
    #    only the first mismatch to keep the message actionable.
    roadmap_part = entries[: len(ROADMAP_ORDER)]
    for index, (actual, expected) in enumerate(zip(roadmap_part, ROADMAP_ORDER)):
        if actual != expected:
            lineno = lineno_by_entry.get(actual, 1)
            violations.append(
                Violation(
                    path,
                    lineno,
                    1,
                    "cookbook-order",
                    f"roadmap position {index + 1}: expected '{expected}', got '{actual}'",
                )
            )
            break

    # 6. Downloads part: only the descending-count constraint is enforced, not
    #    a fixed sequence. This lets a new model be inserted at whatever rank
    #    its download count implies, without pinning an explicit order here.
    downloads_part = entries[len(ROADMAP_ORDER) :]
    for index, (first, second) in enumerate(zip(downloads_part, downloads_part[1:])):
        # Entries not in DOWNLOADS_DESC were already reported as unknown above;
        # skip them here to avoid noise and never fabricate a count.
        if first not in DOWNLOADS_DESC or second not in DOWNLOADS_DESC:
            continue
        first_count = DOWNLOADS_DESC[first]
        second_count = DOWNLOADS_DESC[second]
        if first_count < second_count:
            lineno = lineno_by_entry.get(first, 1)
            violations.append(
                Violation(
                    path,
                    lineno,
                    1,
                    "cookbook-order",
                    f"downloads not in descending order at downloads position {index + 1}: "
                    f"'{first}' ({first_count:,}) must come after '{second}' ({second_count:,})",
                )
            )
            break
    return violations


def check_file(path: Path) -> list[Violation]:
    """Read a file and run all checks against it."""
    source = path.read_text(encoding="utf-8")
    return check_source(path, source)


def report_violations(violations: list[Violation]) -> int:
    """Print the violations (or OK) and return the exit code for CI.

    0 means the order matches; 1 means at least one violation, which makes the
    pre-commit hook and any CI step fail.
    """
    if not violations:
        print(
            f"OK: cookbook order matches PR #2876 "
            f"({len(ROADMAP_ORDER)} roadmap + {len(DOWNLOADS_DESC)} downloads entries)"
        )
        return 0
    else:
        for v in violations:
            print(f"{repo_relative(v.path)}:{v.lineno}: {v.rule}: {v.detail}")
        print(
            f"cookbook order check failed with {len(violations)} violation(s)",
            file=sys.stderr,
        )
        return 1


def print_expected() -> int:
    """Print the expected cookbook order (roadmap, then downloads by count).

    Useful for maintainers and for generating docs/index.rst programmatically.
    """
    downloads_sorted = sorted(
        DOWNLOADS_DESC, key=DOWNLOADS_DESC.__getitem__, reverse=True
    )
    for entry in ROADMAP_ORDER + downloads_sorted:
        print(entry)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse arguments, check the targets, exit with the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", help="Files to check (defaults to docs/index.rst)"
    )
    parser.add_argument(
        "--print-expected",
        action="store_true",
        help="Print the expected cookbook order (roadmap part, then downloads part "
        "sorted by HF downloads) and exit without checking",
    )
    args = parser.parse_args(argv)
    if args.print_expected:
        return print_expected()
    targets = resolve_targets(args.paths)
    violations: list[Violation] = []
    for path in targets:
        # A missing target is a violation, not a skip: silently reporting
        # success without checking anything would defeat the lint.
        if not path.is_file():
            violations.append(Violation(path, 1, 1, "cookbook-order", "file not found"))
        else:
            violations.extend(check_file(path))
    return report_violations(violations)


if __name__ == "__main__":
    raise SystemExit(main())
