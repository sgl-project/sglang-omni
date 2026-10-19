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
  * Download counts moved: run `--refresh-downloads` (local, queries the HF
    API and rewrites DOWNLOADS_DESC), then update docs/index.rst if ranks
    changed.
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

from cookbook_order_data import DOWNLOADS_DESC, HF_REPO, ROADMAP_ORDER

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "scripts" / "cookbook_order_data.py"
DEFAULT_TARGET = REPO_ROOT / "docs" / "index.rst"

ROADMAP_SET = frozenset(ROADMAP_ORDER)
DOWNLOAD_SET = frozenset(DOWNLOADS_DESC)

_ENTRY_RE = re.compile(r"^   (cookbook/[a-z0-9_.-]+\.md)$")


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


def resolve_targets(raw_paths: list[str]) -> list[Path]:
    if not raw_paths:
        return [DEFAULT_TARGET]
    else:
        return [Path(item) for item in raw_paths]


def extract_cookbook_entries(source: str) -> tuple[list[str], dict[str, int]]:
    """Return (entries, lineno_by_entry) for the Cookbook toctree block."""
    lines = source.splitlines()
    entries: list[str] = []
    lineno_by_entry: dict[str, int] = {}
    in_cookbook = False
    for lineno, line in enumerate(lines, start=1):
        if line.startswith(".. toctree::"):
            in_cookbook = False
            continue
        if line.strip() == ":caption: Cookbook":
            in_cookbook = True
            continue
        if not in_cookbook:
            continue
        if not line.strip():
            continue
        match = _ENTRY_RE.match(line)
        if match:
            entry = match.group(1)
            entries.append(entry)
            lineno_by_entry[entry] = lineno
        else:
            break
    return entries, lineno_by_entry


def check_source(path: Path, source: str) -> list[Violation]:
    violations: list[Violation] = []
    entries, lineno_by_entry = extract_cookbook_entries(source)

    if not entries:
        violations.append(
            Violation(path, 1, 1, "cookbook-order", "Cookbook toctree block not found")
        )
        return violations

    # Duplicates.
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

    # Unknown entries.
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
                f"{repo_relative(DATA_FILE)}",
            )
        )

    # Roadmap models must stay inside the roadmap part.
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

    # Missing entries.
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

    # Roadmap part: exact positional match.
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

    # Downloads part: descending-order constraint instead of a fixed sequence,
    # so a new model just has to be inserted where its download count implies.
    downloads_part = entries[len(ROADMAP_ORDER) :]
    for index, (first, second) in enumerate(zip(downloads_part, downloads_part[1:])):
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
                    f"'{first}' ({first_count:,}) must come before '{second}' ({second_count:,})",
                )
            )
            break
    return violations


def check_file(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    return check_source(path, source)


def report_violations(violations: list[Violation]) -> int:
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
    downloads_sorted = sorted(
        DOWNLOADS_DESC, key=DOWNLOADS_DESC.__getitem__, reverse=True
    )
    for entry in ROADMAP_ORDER + downloads_sorted:
        print(entry)
    return 0


def _format_counts_block(counts: dict[str, int]) -> str:
    lines = ["DOWNLOADS_DESC = {"]
    for page in sorted(counts, key=counts.get, reverse=True):
        lines.append(f'    "{page}": {counts[page]:_},')
    lines.append("}")
    return "\n".join(lines)


def refresh_downloads() -> int:
    """Query the HF API for every HF_REPO entry and rewrite DOWNLOADS_DESC.

    Local maintenance command; CI never runs it (lint must stay deterministic
    and offline). Reports value changes and the new expected order.
    """
    import json
    import urllib.error
    import urllib.request
    from datetime import date

    new_counts: dict[str, int] = {}
    failures: list[str] = []
    for page, repo in sorted(HF_REPO.items()):
        url = f"https://huggingface.co/api/models/{repo}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "sglang-omni-cookbook-lint"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            new_counts[page] = int(data.get("downloads", 0))
            print(f"  {page}: {DOWNLOADS_DESC.get(page, 0):,} -> {new_counts[page]:,}")
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            ValueError,
            KeyError,
        ) as exc:
            failures.append(f"{page} ({repo}): {exc}")
    if not new_counts:
        print("refresh failed for every repo; data file unchanged", file=sys.stderr)
        return 1

    old_order = sorted(DOWNLOADS_DESC, key=DOWNLOADS_DESC.get, reverse=True)
    new_order = sorted(new_counts, key=new_counts.get, reverse=True)
    moved = [
        (i, page, old_order[i], new_order[i])
        for i, page in enumerate(new_order)
        if i < len(old_order) and page != old_order[i]
    ]
    if moved:
        print("rank changes vs current table:")
        for index, page, old_page, new_page in moved:
            print(f"  position {index + 1}: was '{old_page}', now '{new_page}'")
    else:
        print("no rank changes vs current table")

    text = DATA_FILE.read_text(encoding="utf-8")
    text = re.sub(
        r"DOWNLOADS_DESC = \{.*?\n\}",
        _format_counts_block(new_counts),
        text,
        count=1,
        flags=re.S,
    )
    queried_at = date.today().strftime("%Y-%m")
    text = re.sub(
        r'DOWNLOADS_QUERIED_AT = "[0-9]{4}-[0-9]{2}"',
        f'DOWNLOADS_QUERIED_AT = "{queried_at}"',
        text,
        count=1,
    )
    DATA_FILE.write_text(text, encoding="utf-8")
    print(f"wrote {repo_relative(DATA_FILE)} (queried {queried_at})")

    for failure in failures:
        print(f"  skipped: {failure}", file=sys.stderr)
    if failures:
        print(
            f"{len(failures)} repo(s) failed; their previous counts were dropped. "
            "Re-run --refresh-downloads after fixing connectivity.",
            file=sys.stderr,
        )
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument(
        "--refresh-downloads",
        action="store_true",
        help="Local maintenance: query the HF API, rewrite DOWNLOADS_DESC in "
        "scripts/cookbook_order_data.py, and report rank changes. Never run in CI.",
    )
    args = parser.parse_args(argv)
    if args.print_expected:
        return print_expected()
    if args.refresh_downloads:
        return refresh_downloads()
    targets = resolve_targets(args.paths)
    violations: list[Violation] = []
    for path in targets:
        if not path.is_file():
            print(f"{path}: not a file; skipped", file=sys.stderr)
        else:
            violations.extend(check_file(path))
    return report_violations(violations)


if __name__ == "__main__":
    raise SystemExit(main())
