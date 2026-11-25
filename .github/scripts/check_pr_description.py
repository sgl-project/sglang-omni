# SPDX-License-Identifier: Apache-2.0
"""Fail a pull request whose description leaves Motivation or Modifications unfilled."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TEMPLATE_HEADING_LEVEL = 2
SECTION_NAMES = ("Motivation", "Modifications")
SECTION_NAMES_BY_CASEFOLD = {name.casefold(): name for name in SECTION_NAMES}
SECTION_GOALS = {
    "Motivation": "explain why this change is needed",
    "Modifications": "describe what changed",
}
# The template's own prompt lines, for authors who keep them outside the comment.
TEMPLATE_PLACEHOLDERS = frozenset(
    {
        "Explain the purpose of this PR and the goals it aims to achieve.",
        "Describe the changes made in this PR.",
    }
)
HEADING_PATTERN = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)[ \t]*$")
FENCE_PATTERN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")

SectionName = Literal["Motivation", "Modifications"]
ProblemReason = Literal["missing", "empty"]


@dataclass(frozen=True, kw_only=True)
class SectionProblem:
    """A template section that still needs the author's own text."""

    section: SectionName
    reason: ProblemReason


@dataclass(frozen=True, kw_only=True)
class BodyLine:
    """One description line, flagged when Markdown renders it as example content."""

    text: str
    is_example: bool


def read_body_lines(body: str) -> list[BodyLine]:
    """Drop HTML comments and flag fenced code and blockquote lines as example content."""
    body_lines: list[BodyLine] = []
    fence_marker: str | None = None
    in_comment = False
    for line in body.splitlines():
        fence_match = FENCE_PATTERN.match(line)
        if fence_marker is not None:
            if (
                fence_match
                and fence_match[1].startswith(fence_marker)
                and not fence_match[2].strip()
            ):
                fence_marker = None
            else:
                pass
            is_example = True
        elif (
            not in_comment
            and fence_match
            and (fence_match[1].startswith("~") or "`" not in fence_match[2])
        ):
            fence_marker = fence_match[1]
            is_example = True
        elif not in_comment and line.lstrip().startswith(">"):
            is_example = True
        else:
            is_example = False
            visible = []
            while line:
                prefix, delimiter, line = line.partition(
                    "-->" if in_comment else "<!--"
                )
                if not in_comment:
                    visible.append(prefix)
                else:
                    pass
                if delimiter:
                    in_comment = not in_comment
                else:
                    break
            line = "".join(visible)
        body_lines.append(BodyLine(text=line, is_example=is_example))
    return body_lines


def read_sections(body: str) -> dict[SectionName, list[str]]:
    """Collect the lines under each template heading, up to the next heading of the same or higher level."""
    sections: dict[SectionName, list[str]] = {}
    active_section: SectionName | None = None
    for body_line in read_body_lines(body):
        heading_match = (
            None if body_line.is_example else HEADING_PATTERN.match(body_line.text)
        )
        if heading_match is None:
            if active_section is not None:
                sections[active_section].append(body_line.text)
            else:
                pass
            continue
        heading_level = len(heading_match.group(1))
        if heading_level > TEMPLATE_HEADING_LEVEL:
            # A deeper heading documents the section it sits in.
            if active_section is not None:
                sections[active_section].append(body_line.text)
            else:
                pass
            continue
        section_name = (
            SECTION_NAMES_BY_CASEFOLD.get(heading_match.group(2).casefold())
            if heading_level == TEMPLATE_HEADING_LEVEL
            else None
        )
        if section_name is None or section_name in sections:
            # An unknown heading ends the section; a repeated template heading keeps the first one.
            active_section = None
        else:
            sections[section_name] = []
            active_section = section_name
    return sections


def has_author_text(content_lines: list[str]) -> bool:
    """Report whether a section holds text beyond whitespace and template placeholders."""
    for content_line in content_lines:
        stripped_line = content_line.strip()
        if not stripped_line or stripped_line in TEMPLATE_PLACEHOLDERS:
            continue
        else:
            return True
    else:
        return False


def find_problems(body: str) -> list[SectionProblem]:
    """Report every template section that is missing or still holds no author text."""
    sections = read_sections(body)
    problems: list[SectionProblem] = []
    for section_name in SECTION_NAMES:
        content_lines = sections.get(section_name)
        if content_lines is None:
            problems.append(SectionProblem(section=section_name, reason="missing"))
        elif not has_author_text(content_lines):
            problems.append(SectionProblem(section=section_name, reason="empty"))
        else:
            pass
    return problems


def format_summary(problems: list[SectionProblem]) -> str:
    """Render the problems as the Markdown step summary of one check run."""
    summary_lines = ["### PR description", ""]
    for problem in problems:
        section_goal = SECTION_GOALS[problem.section]
        if problem.reason == "missing":
            summary_lines.append(
                f"- `## {problem.section}` is missing. Add that section and {section_goal}."
            )
        else:
            summary_lines.append(
                f"- `## {problem.section}` is empty. Keep the template comment and {section_goal}."
            )
    summary_lines.extend(
        [
            "",
            "Edit the pull request description and this check runs again; no new commit is needed.",
        ]
    )
    return "\n".join(summary_lines) + "\n"


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-path",
        default=os.environ.get("GITHUB_EVENT_PATH"),
        help="GitHub event JSON holding the pull request description.",
    )
    parser.add_argument(
        "--body-file",
        help="Read a raw pull request description from this file instead.",
    )
    parser.add_argument(
        "--summary-path",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="Append the check summary here; defaults to GITHUB_STEP_SUMMARY.",
    )
    return parser.parse_args(arguments)


def read_pull_request_body(event_path: str | None, body_file: str | None) -> str:
    if body_file is not None:
        return Path(body_file).read_text(encoding="utf-8")
    elif event_path is None:
        raise RuntimeError(
            "Set GITHUB_EVENT_PATH, or pass --event-path or --body-file."
        )
    else:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        return event["pull_request"]["body"] or ""


def main(arguments: list[str] | None = None) -> int:
    parsed_arguments = parse_arguments(arguments)
    body = read_pull_request_body(
        event_path=parsed_arguments.event_path,
        body_file=parsed_arguments.body_file,
    )
    problems = find_problems(body)
    if not problems:
        print("PR description is complete.")
        return 0
    else:
        summary = format_summary(problems)
        print(summary)
        for problem in problems:
            print(
                f"::error::{problem.section} is {problem.reason} in the pull request description."
            )
        if parsed_arguments.summary_path is not None:
            with Path(parsed_arguments.summary_path).open(
                "a", encoding="utf-8"
            ) as summary_file:
                summary_file.write(summary)
        else:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
