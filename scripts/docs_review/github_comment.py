# SPDX-License-Identifier: Apache-2.0
"""Format advisory review comments on pull requests."""

from __future__ import annotations

import html
from urllib.parse import quote

from scripts.docs_review.models import Finding, ReviewArtifact
from scripts.docs_review.policy import STYLE_GUIDE_PATH
from scripts.docs_review.reporting import render_markdown

ADVISORY_REVIEW_MARKER = "<!-- sglang-omni-advisory-review -->"
ADVISORY_BOT_LOGIN = "github-actions[bot]"
SUPPLEMENTAL_EVIDENCE_PATHS = {
    "comments.md",
    "evidence.md",
    "checks.json",
    "scope.txt",
    "documentation.diff",
}


def safe_text(value: str) -> str:
    escaped = html.escape(" ".join(value.split())[:2000], quote=True)
    for character in "@[]()*_`~\\":
        escaped = escaped.replace(character, f"&#{ord(character)};")
    return escaped


def format_citation(
    repository_url: str,
    artifact: ReviewArtifact,
    file_path: str,
    line_start: int,
    line_end: int,
) -> str:
    if file_path == "pull_request.md":
        return f"[PR description]({repository_url}/pull/{artifact.pull_request_number})"
    elif file_path in SUPPLEMENTAL_EVIDENCE_PATHS:
        return safe_text(file_path)
    else:
        commit_sha = (
            artifact.report.base_commit_sha
            if file_path == STYLE_GUIDE_PATH
            else artifact.report.head_commit_sha
        )
        encoded_path = quote(file_path, safe="/")
        fragment = (
            f"L{line_start}" if line_end == line_start else f"L{line_start}-L{line_end}"
        )
        return (
            f"[{safe_text(file_path)}:{line_start}]"
            f"({repository_url}/blob/{commit_sha}/{encoded_path}#{fragment})"
        )


def format_docs_finding(
    repository_url: str, artifact: ReviewArtifact, finding: Finding
) -> str:
    document = format_citation(
        repository_url,
        artifact,
        finding.file_path,
        finding.line_start,
        finding.line_end,
    )
    evidence = format_citation(
        repository_url,
        artifact,
        finding.evidence_file_path,
        finding.evidence_line_start,
        finding.evidence_line_end,
    )
    return "\n".join(
        [
            f"- **{safe_text(finding.rule_id)}** — claim under review: {safe_text(finding.claim)}",
            f"  - Document: {document}",
            f"  - Evidence: {evidence} — {safe_text(finding.evidence_quote)}",
            f"  - Impact: {safe_text(finding.impact)}",
            f"  - Suggested fix: {safe_text(finding.suggested_fix)}",
        ]
    )


def render_advisory_comment(repository_url: str, artifact: ReviewArtifact) -> str:
    report = artifact.report
    sections = [ADVISORY_REVIEW_MARKER, "### Advisory documentation review"]
    if report.findings:
        sections.append(
            f"Advisory findings for commit `{report.head_commit_sha}`. Check the current head before acting."
        )
        sections.extend(
            format_docs_finding(repository_url, artifact, finding)
            for finding in report.findings
        )
    else:
        sections.append(f"No advisory findings for commit `{report.head_commit_sha}`.")
    sections.append(
        f"Review run: {repository_url}/actions/runs/{artifact.run_id}/attempts/{artifact.run_attempt}"
    )
    return "\n\n".join(sections)


def render_artifact(artifact: ReviewArtifact, *, offline: bool, automatic: bool) -> str:
    notices = [
        f"Pull request #{artifact.pull_request_number}",
        f"Review run {artifact.run_id}, attempt {artifact.run_attempt}.",
        (
            "Findings post automatically when complete."
            if automatic
            else "Manual advisory review."
        ),
    ]
    if offline:
        notices.append("Collection-only run without a model call.")
    else:
        pass
    if artifact.stale:
        notices.append(
            "Stale report. The pull request changed or closed during review."
        )
    else:
        pass
    return "\n\n".join(
        [*notices, render_markdown(artifact.report, number_findings=True)]
    )
