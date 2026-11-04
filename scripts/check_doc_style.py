# SPDX-License-Identifier: Apache-2.0
"""Check Markdown documentation navigation, filenames, emoji, and fence labels."""

from __future__ import annotations

import argparse
import fnmatch
import posixpath
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import emoji

DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"
FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
EXTERNAL_TARGET_PATTERN = re.compile(r"(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//|www\.)")
RST_DIRECTIVE_PATTERN = re.compile(r"\.\.\s+([\w-]+)::(?:\s+(.*))?$")
GLOB_TOKEN_PATTERN = re.compile(r"\[[!]?\]?[^]]*\]|\*\*|\*|\?|.")
LITERAL_DIRECTIVE_NAMES = {"code", "code-block", "sourcecode", "parsed-literal", "raw"}
TOCTREE_EXEMPTIONS = {
    "README.md": "Repository overview, separate from the published manual.",
    "design/gpu_radix_hash.md": "Internal design note, outside the user manual.",
    "design/refactor_rfc.md": "Internal refactoring proposal, outside the user manual.",
    "developer_reference/mlx.md": "Excluded redirect; the published page is mlx/index.md.",
    "get_started/apiserver_quickstart.md": "Linked from other pages, outside the sidebar.",
    "basic_usage/python_router.md": "Retained until the Python router is removed separately.",
}
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


def toctree_glob_pattern(target: str) -> re.Pattern[str]:
    """Match document names with single wildcards confined to one directory."""
    pattern_parts: list[str] = []
    for pattern_token in GLOB_TOKEN_PATTERN.findall(target):
        if pattern_token == "**":
            pattern_parts.append(".*")
        elif pattern_token == "*":
            pattern_parts.append("[^/]*")
        elif pattern_token == "?":
            pattern_parts.append("[^/]")
        elif pattern_token.startswith("[") and len(pattern_token) > 1:
            character_pattern = fnmatch.translate(pattern_token)
            pattern_parts.append(
                ("(?!/)" if pattern_token.startswith("[!") else "")
                + character_pattern.removesuffix(r"\Z").removesuffix(r"\z")
            )
        else:
            pattern_parts.append(re.escape(pattern_token))
    return re.compile("".join(pattern_parts))


def markdown_toctree_source(source: str) -> str:
    """Expose fenced MyST toctrees to the RST navigation parser."""
    navigation_lines: list[str] = []
    fence_marker = ""
    is_toctree = False
    is_yaml_options = False
    is_comment = False
    for line in source.splitlines():
        fence = FENCE_PATTERN.match(line)
        stripped = line.strip()
        if fence_marker:
            if (
                fence
                and fence.group(1)[0] == fence_marker[0]
                and len(fence.group(1)) >= len(fence_marker)
                and not fence.group(2).strip()
            ):
                fence_marker = ""
            elif not is_toctree:
                pass
            elif stripped == "---":
                is_yaml_options = not is_yaml_options
            elif is_yaml_options:
                if re.fullmatch(r"glob:\s*(?:true|yes|on)?", stripped, re.IGNORECASE):
                    navigation_lines.append("   :glob:")
                else:
                    pass
            else:
                navigation_lines.append(f"   {line}")
        elif is_comment:
            is_comment = "-->" not in line
        elif stripped.startswith("<!--"):
            is_comment = "-->" not in line
        elif fence:
            fence_marker = fence.group(1)
            is_toctree = fence.group(2).strip() == "{toctree}"
            is_yaml_options = False
            if is_toctree:
                navigation_lines.append(".. toctree::")
            else:
                pass
        else:
            pass
    return "\n".join(navigation_lines)


def toctree_targets(docs_root: Path) -> set[Path]:
    """Resolve explicit and glob entries in RST and Markdown toctrees."""
    targets: set[Path] = set()
    markdown_paths = sorted(docs_root.rglob("*.md"))
    markdown_documents = {
        path.relative_to(docs_root).with_suffix("").as_posix(): path.resolve()
        for path in markdown_paths
    }
    for navigation_path in sorted([*docs_root.rglob("*.rst"), *markdown_paths]):
        directive_indentation: int | None = None
        option_indentation: int | None = None
        inactive_indentation: int | None = None
        has_glob = False
        source = navigation_path.read_text(encoding="utf-8")
        if navigation_path.suffix == ".md":
            source = markdown_toctree_source(source)
        else:
            pass
        for line in source.splitlines():
            expanded_line = line.expandtabs()
            stripped = expanded_line.strip()
            indentation = len(expanded_line) - len(expanded_line.lstrip())
            if not stripped:
                continue
            else:
                pass
            if inactive_indentation is not None and indentation > inactive_indentation:
                continue
            else:
                inactive_indentation = None
            if directive_indentation is None or indentation <= directive_indentation:
                directive_indentation = None
                option_indentation = None
                has_glob = False
                rst_directive = RST_DIRECTIVE_PATTERN.fullmatch(stripped)
                directive_name = rst_directive.group(1).lower() if rst_directive else ""
                if (
                    rst_directive
                    and directive_name == "toctree"
                    and not rst_directive.group(2)
                ):
                    directive_indentation = indentation
                elif directive_name in LITERAL_DIRECTIVE_NAMES or (
                    not rst_directive
                    and (
                        stripped == ".."
                        or stripped.startswith(".. ")
                        or stripped.endswith("::")
                    )
                ):
                    inactive_indentation = indentation
                else:
                    pass
            elif option_indentation is not None and indentation > option_indentation:
                pass
            elif stripped.startswith(":"):
                option_indentation = indentation
                if stripped == ":glob:":
                    has_glob = True
                else:
                    pass
            else:
                option_indentation = None
                titled_target = re.fullmatch(r".+?\s*<([^<>]+)>", stripped)
                target = titled_target.group(1) if titled_target else stripped
                if target == "self" or EXTERNAL_TARGET_PATTERN.match(target):
                    pass
                else:
                    target_path = (
                        docs_root / target.lstrip("/")
                        if target.startswith("/")
                        else navigation_path.parent / target
                    )
                    if has_glob and not titled_target and re.search(r"[*?\[]", target):
                        glob_pattern = toctree_glob_pattern(
                            posixpath.normpath(
                                target_path.relative_to(docs_root).as_posix()
                            )
                        )
                        targets.update(
                            path
                            for document_name, path in markdown_documents.items()
                            if glob_pattern.fullmatch(document_name)
                            and path != navigation_path.resolve()
                        )
                    else:
                        if not target_path.suffix:
                            target_path = target_path.with_suffix(".md")
                        else:
                            pass
                        targets.add(target_path.resolve())
    return targets


def check_docs(docs_root: Path = DOCS_ROOT) -> list[Violation]:
    """Scan all Markdown documentation without rewriting files."""
    violations: list[Violation] = []
    published_paths = toctree_targets(docs_root)
    for path in sorted(docs_root.rglob("*.md")):
        if path != docs_root / "README.md" and not re.fullmatch(
            r"[a-z0-9_]+\.md", path.name
        ):
            violations.append(
                Violation(
                    path=path,
                    line_number=1,
                    rule="filename-conventions",
                    detail=r"Markdown basename must match [a-z0-9_]+\.md",
                )
            )
        else:
            pass
        if (
            path.relative_to(docs_root).as_posix() not in TOCTREE_EXEMPTIONS
            and path.resolve() not in published_paths
        ):
            violations.append(
                Violation(
                    path=path,
                    line_number=1,
                    rule="missing-toctree",
                    detail="add this page to an explicit RST toctree entry",
                )
            )
        else:
            pass
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
