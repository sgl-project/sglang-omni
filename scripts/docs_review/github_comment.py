# SPDX-License-Identifier: Apache-2.0
"""Format and post advisory review comments on pull requests."""

from __future__ import annotations

import html
import json
import re
from urllib.parse import quote

from scripts.docs_review.github_pilot import PilotError, github_api
from scripts.docs_review.models import Finding, Report, ReviewArtifact
from scripts.docs_review.policy import STYLE_GUIDE_PATH

ADVISORY_REVIEW_MARKER = "<!-- sglang-omni-advisory-review -->"
SUPPLEMENTAL_EVIDENCE_PATHS = {"comments.md", "evidence.md", "checks.json", "scope.txt"}


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


def render_advisory_comment(
    repository_url: str,
    artifact: ReviewArtifact,
    section_title: str,
    findings: list[Finding],
    run_url: str,
) -> str:
    report = artifact.report
    if not findings:
        return ""
    else:
        pass
    return "\n\n".join(
        [
            ADVISORY_REVIEW_MARKER,
            section_title,
            f"Advisory findings for commit `{report.head_commit_sha}`. "
            "They apply only to this revision; check the current head before acting.",
            *[
                format_docs_finding(repository_url, artifact, finding)
                for finding in findings
            ],
            f"Review run: {run_url}",
        ]
    )


def list_issue_comments(
    repository_name: str, issue_number: int
) -> list[dict[str, object]]:
    comments: list[dict[str, object]] = []
    page = 1
    while True:
        response = github_api(
            f"repos/{repository_name}/issues/{issue_number}/comments"
            f"?per_page=100&page={page}",
            method="GET",
        )
        batch = json.loads(response)
        if not batch:
            break
        else:
            comments.extend(batch)
            page += 1
    return comments


def find_advisory_comment_id(
    comments: list[dict[str, object]],
) -> int | None:
    for comment in comments:
        body = comment.get("body")
        comment_id = comment.get("id")
        if (
            isinstance(body, str)
            and ADVISORY_REVIEW_MARKER in body
            and isinstance(comment_id, int)
        ):
            return comment_id
        else:
            pass
    return None


def upsert_advisory_comment(
    repository_name: str,
    pull_request_number: int,
    body: str,
) -> None:
    if ADVISORY_REVIEW_MARKER not in body:
        raise PilotError("The advisory comment body is missing its marker")
    else:
        pass
    comments = list_issue_comments(repository_name, pull_request_number)
    comment_id = find_advisory_comment_id(comments)
    if comment_id is None:
        github_api(
            f"repos/{repository_name}/issues/{pull_request_number}/comments",
            body=json.dumps({"body": body}),
            method="POST",
        )
    else:
        github_api(
            f"repos/{repository_name}/issues/comments/{comment_id}",
            body=json.dumps({"body": body}),
            method="PATCH",
        )


def publish_reports(
    repository_name: str,
    repository_url: str,
    run_id: str,
    run_attempt: int,
    artifacts: list[ReviewArtifact],
) -> None:
    run_url = f"{repository_url}/actions/runs/{run_id}/attempts/{run_attempt}"
    sections: list[str] = []
    for artifact in artifacts:
        if not artifact.publishable or artifact.stale:
            continue
        else:
            pass
        report = artifact.report
        if report.execution != "complete" or not report.findings:
            continue
        else:
            pass
        title = (
            "### Advisory documentation review"
            if not sections
            else "### Advisory documentation review (continued)"
        )
        section = render_advisory_comment(
            repository_url,
            artifact,
            title,
            report.findings,
            run_url,
        )
        if section:
            sections.append(section)
        else:
            pass
    if not sections:
        return
    else:
        pass
    body = "\n\n".join(sections)
    if len(artifacts) == 1:
        pull_request_number = artifacts[0].pull_request_number
    else:
        pull_request_number = artifacts[0].pull_request_number
    upsert_advisory_comment(repository_name, pull_request_number, body)
