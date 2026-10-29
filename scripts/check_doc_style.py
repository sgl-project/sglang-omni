# SPDX-License-Identifier: Apache-2.0
"""Check emoji and code fence labels in Markdown documentation."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import emoji

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"
FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
TEXT_PRESENTATION_SELECTOR = "\ufe0e"
EMOJI_PRESENTATION_SELECTOR = "\ufe0f"
# note (fengxiaohu): Unicode 17.0 includes these two forms missing from emoji.
EMOJI_SUPPLEMENT = re.compile(r"[\U0001f1e6-\U0001f1ff]|[#*0-9]\ufe0f")


@dataclass(frozen=True, kw_only=True)
class Violation:
    path: Path
    line_number: int
    rule: str
    detail: str


def check_source(path: Path, source: str) -> list[Violation]:
    """Check fence labels and emoji presentation with complete sequence matches."""
    violations: list[Violation] = []
    fence_marker = ""
    fence_start_line_number = 0
    for line_number, line in enumerate(source.splitlines(), start=1):
        fence = FENCE_PATTERN.match(line)
        if fence_marker:
            if (
                fence
                and fence.group(1)[0] == fence_marker[0]
                and len(fence.group(1)) >= len(fence_marker)
                and not fence.group(2).strip()
            ):
                fence_marker = ""
            else:
                pass
        elif fence:
            fence_marker = fence.group(1)
            fence_start_line_number = line_number
            if not fence.group(2).strip():
                violations.append(
                    Violation(
                        path=path,
                        line_number=line_number,
                        rule="fence-language",
                        detail="opening fence needs a non-empty info string",
                    )
                )
            else:
                pass
        else:
            emoji_matches = emoji.emoji_list(line)
            sequences: list[str] = []
            for sequence_match in emoji_matches:
                sequence = sequence_match["emoji"]
                match_end = sequence_match["match_end"]
                variation_selector = line[match_end : match_end + 1]
                emoji_properties = emoji.EMOJI_DATA[sequence]
                # note (fengxiaohu): matches include text-default symbols and omit VS15.
                is_text_style = len(sequence) == 1 and (
                    emoji_properties["status"] == emoji.STATUS["unqualified"]
                    or (
                        emoji_properties.get("variant", False)
                        and variation_selector == TEXT_PRESENTATION_SELECTOR
                    )
                )
                if is_text_style:
                    pass
                else:
                    if (
                        len(sequence) == 1
                        and emoji_properties.get("variant", False)
                        and variation_selector == EMOJI_PRESENTATION_SELECTOR
                    ):
                        sequence += EMOJI_PRESENTATION_SELECTOR
                    else:
                        pass
                    sequences.append(sequence)
            covered_positions = {
                position
                for sequence_match in emoji_matches
                for position in range(
                    sequence_match["match_start"], sequence_match["match_end"]
                )
            }
            for supplement_match in EMOJI_SUPPLEMENT.finditer(line):
                if supplement_match.start() in covered_positions:
                    pass
                else:
                    sequences.append(supplement_match.group())
            for sequence in sequences:
                violations.append(
                    Violation(
                        path=path,
                        line_number=line_number,
                        rule="no-emoji",
                        detail=f"remove emoji sequence {sequence!r}",
                    )
                )
    if fence_marker:
        violations.append(
            Violation(
                path=path,
                line_number=fence_start_line_number,
                rule="unclosed-fence",
                detail="unclosed code fence",
            )
        )
    else:
        pass
    return violations


def check_docs(docs_root: Path = DOCS_ROOT) -> list[Violation]:
    """Scan all Markdown documentation without rewriting files."""
    violations: list[Violation] = []
    for path in sorted(docs_root.rglob("*.md")):
        violations.extend(check_source(path, path.read_text(encoding="utf-8")))
    return violations


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    violations = check_docs()
    for violation in violations:
        print(
            f"{violation.path.relative_to(DOCS_ROOT.parent)}:"
            f"{violation.line_number}: {violation.rule}: {violation.detail}",
            file=sys.stderr,
        )
    return int(bool(violations))


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
