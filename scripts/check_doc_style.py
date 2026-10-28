# SPDX-License-Identifier: Apache-2.0
"""Check Markdown documentation style without rewriting source files."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import emoji

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"
FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
INLINE_CODE_PATTERN = re.compile(r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)", re.DOTALL)
URL_PATTERN = re.compile(r"(?:[a-zA-Z][a-zA-Z0-9+.-]*://|www\.)[^\s<>]+")
PROPER_NOUN_PATTERN = re.compile(r"\b(Cuda|Mlx|Openai|Sglang|Huggingface)\b")
EXPECTED_PROPER_NOUN_FORMS = {
    "Cuda": "CUDA",
    "Mlx": "MLX",
    "Openai": "OpenAI",
    "Sglang": "SGLang",
    "Huggingface": "Hugging Face",
}
TEXT_PRESENTATION_SELECTOR = "\ufe0e"
EMOJI_PRESENTATION_SELECTOR = "\ufe0f"
# note (fengxiaohu): Unicode 17.0 includes these two forms missing from emoji.
EMOJI_SUPPLEMENT = re.compile(r"[\U0001f1e6-\U0001f1ff]|[#*0-9]\ufe0f")
MINIMUM_DUPLICATE_BLOCK_LINES = 6
MINIMUM_DUPLICATE_BLOCK_CHARACTERS = 200
TOCTREE_EXEMPTIONS = {
    "README.md": "Repository documentation overview, separate from the published manual.",
    "design/gpu_radix_hash.md": "Internal design note, not part of the user manual.",
    "design/refactor_rfc.md": "Internal refactoring proposal, not part of the user manual.",
    "developer_reference/mlx.md": (
        "Redirect placeholder excluded from the Sphinx build; the published page is mlx/index.md."
    ),
    "get_started/apiserver_quickstart.md": (
        "Linked from installation and API server design; not a sidebar entry."
    ),
    "basic_usage/python_router.md": (
        "Kept until a separate change removes the Python router; not a sidebar entry."
    ),
}


@dataclass(frozen=True, kw_only=True)
class Violation:
    path: Path
    line_number: int
    rule: str
    detail: str


@dataclass(frozen=True, kw_only=True)
class FencedBlock:
    path: Path
    line_number: int
    body: str


def check_source(
    path: Path,
    source: str,
    fenced_blocks: list[FencedBlock] | None = None,
) -> list[Violation]:
    """Check one Markdown source and optionally collect completed fenced blocks."""
    violations: list[Violation] = []
    prose_lines: list[str] = []
    fence_marker = ""
    fence_start_line_number = 0
    body_lines: list[str] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        fence = FENCE_PATTERN.match(line)
        if fence_marker:
            prose_lines.append("")
            if (
                fence
                and fence.group(1)[0] == fence_marker[0]
                and len(fence.group(1)) >= len(fence_marker)
                and not fence.group(2).strip()
            ):
                if fenced_blocks is not None:
                    fenced_blocks.append(
                        FencedBlock(
                            path=path,
                            line_number=fence_start_line_number,
                            body="\n".join(body_lines),
                        )
                    )
                else:
                    pass
                fence_marker = ""
                body_lines = []
            else:
                body_lines.append(line.rstrip())
        elif fence:
            prose_lines.append("")
            fence_marker = fence.group(1)
            fence_start_line_number = line_number
            body_lines = []
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
            prose_lines.append(line)
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
    prose = "\n".join(prose_lines)
    for pattern in (INLINE_CODE_PATTERN, URL_PATTERN):
        prose = pattern.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), prose)
    for line_number, line in enumerate(prose.splitlines(), start=1):
        for noun_match in PROPER_NOUN_PATTERN.finditer(line):
            noun = noun_match.group()
            violations.append(
                Violation(
                    path=path,
                    line_number=line_number,
                    rule="proper-noun",
                    detail=f"use {EXPECTED_PROPER_NOUN_FORMS[noun]} instead of {noun}",
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


def toctree_targets(docs_root: Path) -> set[Path]:
    """Resolve explicit Markdown targets from every RST toctree."""
    targets: set[Path] = set()
    for rst_path in sorted(docs_root.rglob("*.rst")):
        directive_indent: int | None = None
        for line in rst_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            indentation = len(line) - len(line.lstrip())
            if directive_indent is not None and stripped:
                if indentation <= directive_indent:
                    directive_indent = None
                elif stripped.startswith(":"):
                    continue
                else:
                    titled_target = re.fullmatch(r".+\s+<([^<>]+)>", stripped)
                    target = titled_target.group(1) if titled_target else stripped
                    if URL_PATTERN.match(target) or target == "self":
                        continue
                    else:
                        target_path = (
                            docs_root / target.lstrip("/")
                            if target.startswith("/")
                            else rst_path.parent / target
                        )
                        if not target_path.suffix:
                            target_path = target_path.with_suffix(".md")
                        else:
                            pass
                        targets.add(target_path.resolve())
            else:
                pass
            if stripped == ".. toctree::":
                directive_indent = indentation
            else:
                pass
    return targets


def check_docs(docs_root: Path = DOCS_ROOT) -> list[Violation]:
    """Scan Markdown documents for local rules, navigation, and duplicate blocks."""
    violations: list[Violation] = []
    blocks_by_body: dict[str, list[FencedBlock]] = defaultdict(list)
    published_paths = toctree_targets(docs_root)
    for path in sorted(docs_root.rglob("*.md")):
        relative_path = path.relative_to(docs_root).as_posix()
        if relative_path != "README.md" and not re.fullmatch(
            r"[a-z0-9_]+\.md", path.name
        ):
            violations.append(
                Violation(
                    path=path,
                    line_number=1,
                    rule="filename-separators",
                    detail="basename must match [a-z0-9_]+.md",
                )
            )
        else:
            pass
        if (
            relative_path not in TOCTREE_EXEMPTIONS
            and path.resolve() not in published_paths
        ):
            violations.append(
                Violation(
                    path=path,
                    line_number=1,
                    rule="missing-toctree",
                    detail="add this page to an RST toctree",
                )
            )
        else:
            pass
        fenced_blocks_for_source: list[FencedBlock] = []
        source_violations = check_source(
            path,
            path.read_text(encoding="utf-8"),
            fenced_blocks_for_source,
        )
        violations.extend(source_violations)
        for block in fenced_blocks_for_source:
            if (
                len(block.body.split("\n")) >= MINIMUM_DUPLICATE_BLOCK_LINES
                and len(block.body) >= MINIMUM_DUPLICATE_BLOCK_CHARACTERS
            ):
                blocks_by_body[block.body].append(block)
            else:
                pass
    for blocks in blocks_by_body.values():
        if len({block.path for block in blocks}) >= 2:
            locations = ", ".join(
                f"{block.path.relative_to(docs_root)}:{block.line_number}"
                for block in blocks
            )
            for block in blocks:
                violations.append(
                    Violation(
                        path=block.path,
                        line_number=block.line_number,
                        rule="duplicated-block",
                        detail=f"identical fenced body at {locations}",
                    )
                )
        else:
            pass
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
