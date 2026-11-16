# SPDX-License-Identifier: Apache-2.0
"""Format and post advisory review comments on pull requests."""

from __future__ import annotations

import html
import json
import re
from urllib.parse import quote

from scripts.docs_review.github_pilot import PilotError, github_api
from scripts.docs_review.models import Finding, ReviewArtifact
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


def format_unit_test_finding(
    repository_url: str, artifact: ReviewArtifact, finding: Finding
) -> str:
    test_citation = format_citation(
        repository_url,
        artifact,
        finding.file_path,
        finding.line_start,
        finding.line_end,
    )
    contract = format_citation(
        repository_url,
        artifact,
        finding.evidence_file_path,
        finding.evidence_line_start,
        finding.evidence_line_end,
    )
    lines = [
        f"- **{safe_text(finding.rule_id)}** — {safe_text(finding.claim)}",
        f"  - Test: {test_citation} — {safe_text(finding.quote)}",
    ]
    if finding.change_evidence is not None:
        change = finding.change_evidence
        lines.append(
            f"  - Changed dependency: {format_citation(repository_url, artifact, change.file_path, change.line_start, change.line_end)}"
            f" — {safe_text(change.quote)}"
        )
    else:
        pass
    lines.extend(
        [
            f"  - Contract: {contract} — {safe_text(finding.public_contract)}",
            f"  - Regression: {safe_text(finding.regression)}",
            f"  - Impact: {safe_text(finding.impact)}",
            f"  - Suggested assertion: {safe_text(finding.suggested_fix)}",
        ]
    )
    return "\n".join(lines)


def format_finding(
    repository_url: str, artifact: ReviewArtifact, finding: Finding
) -> str:
    if artifact.review_scope == "unit_tests":
        return format_unit_test_finding(repository_url, artifact, finding)
    else:
        return format_docs_finding(repository_url, artifact, finding)


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


def section_title_for_scope(review_scope: str) -> str:
    if review_scope == "unit_tests":
        return "### Advisory unit-test contract review"
    else:
        return "### Advisory documentation review"


def list_issue_comments(
    repository_name: str, issue_number: int
) -> list[dict[str, object]]:
    comments: list[dict[str, object]] = []
    page = 1
    while True:
        response = github_api(
            f"repos/{repository_name}/issues/{issue_number}/comments"
            f"?per_page=100&page={page}"
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
        author = comment.get("user")
        if (
            isinstance(body, str)
            and ADVISORY_REVIEW_MARKER in body
            and isinstance(comment_id, int)
            and isinstance(author, dict)
            and author.get("login") == "github-actions[bot]"
        ):
            return comment_id
        else:
            pass
    return None


def upsert_advisory_comment(
    repository_name: str,
    pull_request_number: int,
    body: str,
    *,
    expected_base_sha: str = "",
    expected_head_sha: str = "",
    run_id: str = "",
    run_attempt: int = 0,
) -> bool:
    if ADVISORY_REVIEW_MARKER not in body:
        raise PilotError("The advisory comment body is missing its marker")
    else:
        pass
    comments = list_issue_comments(repository_name, pull_request_number)
    comment_id = find_advisory_comment_id(comments)
    previous = next(
        (
            comment.get("body")
            for comment in comments
            if comment.get("id") == comment_id
        ),
        "",
    )
    if isinstance(previous, str) and run_id:
        newer_run = re.search(r"<!-- advisory-run:(\d+):(\d+) -->", previous)
        if newer_run and (int(newer_run.group(1)), int(newer_run.group(2))) > (
            int(run_id),
            run_attempt,
        ):
            return False
        else:
            pass
    else:
        pass
    if expected_base_sha and expected_head_sha:
        current = json.loads(
            github_api(f"repos/{repository_name}/pulls/{pull_request_number}")
        )
        if (
            current["state"] != "open"
            or current["base"]["sha"] != expected_base_sha
            or current["head"]["sha"] != expected_head_sha
        ):
            return False
        else:
            pass
    else:
        pass
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
    return True


def publish_reports(
    repository_name: str,
    repository_url: str,
    run_id: str,
    run_attempt: int,
    artifacts: list[ReviewArtifact],
    *,
    expected_base_sha: str = "",
    expected_head_sha: str = "",
) -> bool:
    run_url = f"{repository_url}/actions/runs/{run_id}/attempts/{run_attempt}"
    if any(artifact.stale for artifact in artifacts):
        return False
    else:
        pass
    publishable = [artifact for artifact in artifacts if artifact.publishable]
    if not publishable:
        return True
    else:
        pass
    head_commit_sha = publishable[0].report.head_commit_sha
    if any(
        artifact.report.head_commit_sha != head_commit_sha for artifact in artifacts
    ):
        return False
    else:
        pass
    comments = list_issue_comments(repository_name, publishable[0].pull_request_number)
    comment_id = find_advisory_comment_id(comments)
    previous = next(
        (comment["body"] for comment in comments if comment.get("id") == comment_id), ""
    )
    sections: dict[str, str] = {}
    if (
        isinstance(previous, str)
        and f"<!-- advisory-head:{head_commit_sha} -->" in previous
        and (
            not expected_base_sha
            or f"<!-- advisory-base:{expected_base_sha} -->" in previous
        )
    ):
        for scope in ("docs", "unit_tests"):
            start = f"<!-- advisory-scope:{scope}:start -->"
            end = f"<!-- advisory-scope:{scope}:end -->"
            if start in previous and end in previous:
                sections[scope] = previous.split(start, 1)[1].split(end, 1)[0].strip()
            else:
                pass
    else:
        pass
    parts = [
        ADVISORY_REVIEW_MARKER,
        f"<!-- advisory-head:{head_commit_sha} -->",
        f"<!-- advisory-base:{expected_base_sha} -->",
        f"<!-- advisory-run:{run_id}:{run_attempt} -->",
        f"Advisory findings for commit `{head_commit_sha}`. "
        "They apply only to this revision; check the current head before acting.",
    ]
    for artifact in artifacts:
        scope_parts = [section_title_for_scope(artifact.review_scope)]
        if artifact.report.execution != "complete":
            scope_parts.append(
                "Review incomplete. See the report for coverage limitations."
            )
            sections[artifact.review_scope] = "\n\n".join(scope_parts)
            continue
        else:
            pass
        if not artifact.publishable:
            continue
        else:
            pass
        if artifact.report.findings:
            scope_parts.extend(
                format_finding(repository_url, artifact, finding)
                for finding in artifact.report.findings
            )
        else:
            scope_parts.append("No findings.")
        if artifact.report.test_hygiene_notes:
            scope_parts.append("### Unit-test hygiene")
            for note in artifact.report.test_hygiene_notes:
                rule_id, separator, detail = note.partition(": ")
                if separator:
                    scope_parts.append(f"- **{rule_id}** — {safe_text(detail)}")
                else:
                    scope_parts.append(f"- {safe_text(note)}")
        else:
            pass
        sections[artifact.review_scope] = "\n\n".join(scope_parts)
    for scope in ("docs", "unit_tests"):
        if scope in sections:
            parts.append(
                f"<!-- advisory-scope:{scope}:start -->\n{sections[scope]}\n"
                f"<!-- advisory-scope:{scope}:end -->"
            )
        else:
            pass
    parts.append(f"Review run: {run_url}")
    return upsert_advisory_comment(
        repository_name,
        publishable[0].pull_request_number,
        "\n\n".join(parts),
        expected_base_sha=expected_base_sha,
        expected_head_sha=expected_head_sha,
        run_id=run_id,
        run_attempt=run_attempt,
    )
