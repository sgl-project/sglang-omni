# SPDX-License-Identifier: Apache-2.0
"""Check documentation style without rewriting files or allowing suppressions."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
EMOJI = re.compile("[\u2600-\u27bf\U0001f300-\U0001faff]")
INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)", re.DOTALL)
URL = re.compile(r"(?:[a-zA-Z][a-zA-Z0-9+.-]*://|www\.)[^\s<>]+")
PROPER_NOUN = re.compile(r"\b(Cuda|Mlx|Openai|Sglang|Huggingface)\b")
EXPECTED_FORMS = {
    "Cuda": "CUDA",
    "Mlx": "MLX",
    "Openai": "OpenAI",
    "Sglang": "SGLang",
    "Huggingface": "Hugging Face",
}
MIN_BLOCK_LINES = 6
MIN_BLOCK_CHARACTERS = 200
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


def check_source(path: Path, source: str) -> tuple[list[Violation], list[FencedBlock]]:
    """Check local rules and collect completed fenced bodies for comparison."""
    violations: list[Violation] = []
    blocks: list[FencedBlock] = []
    prose_lines: list[str] = []
    fence_marker = ""
    fence_line_number = 0
    body_lines: list[str] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        fence = FENCE.match(line)
        if fence_marker:
            prose_lines.append("")
            if (
                fence
                and fence.group(1)[0] == fence_marker[0]
                and len(fence.group(1)) >= len(fence_marker)
                and not fence.group(2).strip()
            ):
                blocks.append(
                    FencedBlock(
                        path=path,
                        line_number=fence_line_number,
                        body="\n".join(body_lines),
                    )
                )
                fence_marker = ""
                body_lines = []
            else:
                body_lines.append(line.rstrip())
        elif fence:
            prose_lines.append("")
            fence_marker = fence.group(1)
            fence_line_number = line_number
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
            for emoji in EMOJI.finditer(line):
                violations.append(
                    Violation(
                        path=path,
                        line_number=line_number,
                        rule="no-emoji",
                        detail=f"remove pictographic character U+{ord(emoji.group()):04X}",
                    )
                )
    prose = "\n".join(prose_lines)
    for pattern in (INLINE_CODE, URL):
        prose = pattern.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), prose)
    for line_number, line in enumerate(prose.splitlines(), start=1):
        for noun in PROPER_NOUN.finditer(line):
            violations.append(
                Violation(
                    path=path,
                    line_number=line_number,
                    rule="proper-noun",
                    detail=f"use {EXPECTED_FORMS[noun.group()]} instead of {noun.group()}",
                )
            )
    return violations, blocks


def toctree_targets(docs_root: Path) -> set[Path]:
    """Resolve explicit toctree entries relative to their containing RST file."""
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
                    if URL.match(target) or target == "self":
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
    """Scan all Markdown documents, navigation entries, and cross-file copies."""
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
        source_violations, blocks = check_source(path, path.read_text(encoding="utf-8"))
        violations.extend(source_violations)
        for block in blocks:
            if (
                len(block.body.split("\n")) >= MIN_BLOCK_LINES
                and len(block.body) >= MIN_BLOCK_CHARACTERS
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
