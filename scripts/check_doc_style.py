#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Lint Markdown document style. Checking does not rewrite files.

Every Markdown file under docs/ is in scope, including files added later.

Rules, enforced on text outside fenced code blocks:

  1. No emoji anywhere in the document. Emoji used as decorative prefixes
     (rockets, checkmarks, stars, flags, ...) or inline body decoration are
     rejected; only plain text is allowed. Typographic arrows (->, <-, etc.)
     and math-style symbols are not emoji and are accepted.
  2. A line break may only follow a meaningful sentence-ending punctuation
     mark (``.`` ``!`` ``?`` ``;`` ``:`` and their full-width forms). Breaking
     in the middle of a sentence -- a hard wrap after a comma, after a word,
     or anywhere without a sentence-ending mark -- is a violation. A "natural
     paragraph" is a maximal run of consecutive non-blank lines of the same
     kind (prose, one list item with its continuation lines, or one
     blockquote); inside such a run every line except the last must end with a
     sentence-ending punctuation mark.

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

# Emoji: pictographic blocks + variation selector-16 (emoji presentation).
# Typographic arrows (U+2190-U+21FF) are deliberately not included.
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF"
    "\uFE0F]"
)
# Fence opener/closer: a run of at least three backticks or tildes.
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
# Structural lines that end a natural paragraph run: ATX headings, table rows,
# HTML tags, and thematic breaks.
_STRUCT_RE = re.compile(
    r"^ {0,3}(#{1,6}\s|#{1,6}$|\||<|(?:-{3,}|\*{3,}|_{3,})$)"
)
# List item marker: "- x", "* x", "+ x", or "1. x", up to 3 leading spaces.
_LIST_RE = re.compile(r"^ {0,3}([-*+]\s|\d+[.)]\s)")
# Blockquote marker. A line that is only ">" is an empty quote line that
# separates two quoted paragraphs.
_QUOTE_RE = re.compile(r"^ {0,3}>[ \t]?")
_EMPTY_QUOTE_RE = re.compile(r"^ {0,3}>[ \t]*$")
# Sentence-ending punctuation (half- and full-width). The ellipsis is a
# sentence-ending mark; ASCII "..." also ends on ".".
_SENTENCE_END = "。！？；：.!?;:…"

# Closers stripped from the line end before checking for a sentence-ending
# mark: closing quotes/brackets, bold/italic markers, inline code, links.
_CLOSERS = (
    '"', "'", "」", "』", "）", "】", "》", "〉", "〕", "›", "»",
    ")", "]", "}", "*", "_", "`",
)
_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)$")


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


def ends_with_sentence_punctuation(text: str) -> bool:
    """Return True when ``text`` (after trailing closers) ends with a
    sentence-ending punctuation mark."""
    t = text.rstrip()
    while t:
        if t.endswith(")"):
            m = _LINK_RE.search(t)
            if m:
                t = t[: m.start()]
                continue
            t = t[:-1]
            continue
        if t[-1] in _CLOSERS:
            t = t[:-1]
            continue
        break
    return bool(t) and t[-1] in _SENTENCE_END


def check_emoji(path: Path, source: str) -> list[Violation]:
    violations: list[Violation] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        m = _EMOJI_RE.search(line)
        if m:
            violations.append(
                Violation(
                    path,
                    lineno,
                    m.start() + 1,
                    "no-emoji",
                    "emoji characters are not allowed in docs; "
                    f"remove {m.group(0)!r} and use plain text",
                )
            )
        else:
            pass
    return violations


def check_line_breaks(path: Path, source: str) -> list[Violation]:
    """Flag hard wraps that occur inside a natural paragraph.

    A "natural paragraph" is a maximal run of consecutive lines that forms a
    single logical unit: a prose paragraph, one list item together with its
    continuation lines, or one blockquote paragraph. Inside such a run every
    line except the last must end with sentence-ending punctuation.
    """
    violations: list[Violation] = []
    lines = source.splitlines()
    in_fence = False
    run: list[tuple[int, str]] = []  # (lineno, text)

    def flush() -> None:
        nonlocal run
        if len(run) > 1:
            for lineno, text in run[:-1]:
                if not ends_with_sentence_punctuation(text):
                    violations.append(
                        Violation(
                            path,
                            lineno,
                            1,
                            "mid-sentence-break",
                            "line breaks only after sentence-ending punctuation; "
                            "do not hard-wrap inside a sentence",
                        )
                    )
        run = []

    for lineno, line in enumerate(lines, start=1):
        if _FENCE_RE.match(line):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.strip():
            flush()
            continue
        if _STRUCT_RE.match(line):
            flush()
            continue
        if _EMPTY_QUOTE_RE.match(line):
            flush()
            continue
        if _QUOTE_RE.match(line):
            # A quote line continues an existing blockquote paragraph when the
            # previous line is also a quote; otherwise it starts a new run.
            if run and _QUOTE_RE.match(run[-1][1]):
                run.append((lineno, line))
            else:
                flush()
                run.append((lineno, line))
            continue
        if _LIST_RE.match(line):
            # A new list item always ends the previous run.
            flush()
            run.append((lineno, line))
            continue
        # Plain text: continues the current run (lazy continuation of a list
        # item or quote, or more prose of the same paragraph).
        run.append((lineno, line))
    flush()
    return violations


def check_file(path: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8")
    return check_emoji(path, source) + check_line_breaks(path, source)


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
        f"{len(violations)} doc style violation(s) in docs/. "
        "No emoji allowed; line breaks only after sentence-ending punctuation.",
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
