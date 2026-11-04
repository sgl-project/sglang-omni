# SPDX-License-Identifier: Apache-2.0
"""Run one advisory documentation review and validate its citations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import ValidationError

from scripts.docs_review.audit import (
    audit_bundle,
    author_confirmation_statement,
    interpret_checks,
)
from scripts.docs_review.collect import (
    bundle_payload,
    collect_review_bundle,
    render_bundle,
    review_input_sha256,
)
from scripts.docs_review.ledger import ReviewLedger
from scripts.docs_review.models import (
    BundlePayload,
    CheckView,
    Clock,
    DocumentationReviewer,
    Finding,
    ModelSettings,
    Report,
    ReviewBundle,
    ReviewRequest,
)
from scripts.docs_review.policy import KNOWN_RULE_IDS, read_prompt

MODEL_TIMEOUT_SECONDS = 60


class ReviewerError(Exception):
    """The model call or its response could not be used."""


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(frozen=True, kw_only=True)
class ChatMessage:
    role: str
    content: str


class ChatRequest:
    """JSON body sent to an OpenAI-compatible chat endpoint."""

    def __init__(
        self, model_name: str, messages: list[ChatMessage], max_output_tokens: int
    ) -> None:
        self.model_name = model_name
        self.messages = messages
        self.max_output_tokens = max_output_tokens

    def to_json(self) -> str:
        return json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in self.messages
                ],
                "temperature": 0,
                "max_tokens": self.max_output_tokens,
            }
        )


@dataclass(kw_only=True)
class ReviewRun:
    bundle: ReviewBundle
    payload: BundlePayload
    report: Report


class OpenAICompatibleReviewer:
    """Read-only model call. No shell, web search, or GitHub writes."""

    def __init__(self, settings: ModelSettings, prompt_text: str) -> None:
        self.settings = settings
        self.prompt_text = prompt_text

    def review(self, bundle_text: str) -> str:
        request = build_chat_request(bundle_text, self.settings, self.prompt_text)
        http_request = urllib.request.Request(
            self.settings.endpoint_url,
            data=request.to_json().encode(),
            headers={
                "Authorization": authorization_header(self.settings),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request, timeout=MODEL_TIMEOUT_SECONDS
            ) as response:
                payload_bytes = response.read()
        except urllib.error.URLError as error:
            raise ReviewerError("model request failed") from error
        else:
            return parse_chat_content(payload_bytes)


def build_chat_request(
    bundle_text: str, settings: ModelSettings, prompt_text: str
) -> ChatRequest:
    return ChatRequest(
        model_name=settings.model_name,
        messages=[
            ChatMessage(role="system", content=prompt_text),
            ChatMessage(role="user", content=bundle_text),
        ],
        max_output_tokens=settings.max_output_tokens,
    )


def authorization_header(settings: ModelSettings) -> str:
    variable_name = settings.api_key_environment_variable
    try:
        secret = os.environ[variable_name]
    except KeyError as error:
        raise ReviewerError(
            f"Missing API key environment variable {variable_name}"
        ) from error
    else:
        return f"Bearer {secret}"


def parse_chat_content(payload_bytes: bytes) -> str:
    try:
        payload: object = json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise ReviewerError("model response was not JSON") from error
    else:
        pass
    if not isinstance(payload, dict):
        raise ReviewerError("model response was not an object")
    else:
        pass
    if "choices" not in payload:
        raise ReviewerError("model response has no choices")
    else:
        choices: object = payload["choices"]
    if not isinstance(choices, list) or not choices:
        raise ReviewerError("model response has no choices")
    else:
        first: object = choices[0]
    if not isinstance(first, dict) or "message" not in first:
        raise ReviewerError("model response has no message")
    else:
        message: object = first["message"]
    if not isinstance(message, dict) or "content" not in message:
        raise ReviewerError("model response has no content")
    else:
        content: object = message["content"]
    if not isinstance(content, str):
        raise ReviewerError("model response content was not text")
    else:
        return content


def run_review(
    request: ReviewRequest,
    reviewer: DocumentationReviewer | None,
    clock: Clock,
) -> ReviewRun:
    if request.retry_count < 0:
        raise ValueError("retry_count must be zero or greater")
    else:
        pass
    bundle = collect_review_bundle(request)
    prompt_text = read_prompt()
    prompt_version = hashlib.sha256(prompt_text.encode()).hexdigest()
    if request.model_settings is None:
        model_name = ""
    else:
        model_name = request.model_settings.model_name
    input_hash = review_input_sha256(bundle, model_name, prompt_version)
    payload = bundle_payload(bundle, prompt_version, input_hash)
    ledger = ReviewLedger(request.ledger_path)
    try:
        cached = ledger.fetch(input_hash)
        if cached is not None:
            return ReviewRun(bundle=bundle, payload=payload, report=cached)
        else:
            pass
        report = review_uncached(
            request,
            reviewer,
            clock,
            bundle,
            prompt_text,
            prompt_version,
            input_hash,
            ledger,
        )
        return ReviewRun(bundle=bundle, payload=payload, report=report)
    finally:
        ledger.close()


def review_uncached(
    request: ReviewRequest,
    reviewer: DocumentationReviewer | None,
    clock: Clock,
    bundle: ReviewBundle,
    prompt_text: str,
    prompt_version: str,
    input_hash: str,
    ledger: ReviewLedger,
) -> Report:
    reviewed_at = clock.now()
    if reviewer is None and request.model_settings is None:
        return assemble_report(
            bundle=bundle,
            prompt_version=prompt_version,
            input_hash=input_hash,
            model_findings=None,
            model_failed=True,
            incomplete_reason="model is not configured",
            usage_cost_cents=0,
        )
    else:
        pass
    if request.budget is None:
        return assemble_report(
            bundle=bundle,
            prompt_version=prompt_version,
            input_hash=input_hash,
            model_findings=None,
            model_failed=True,
            incomplete_reason="quotas are not configured",
            usage_cost_cents=0,
        )
    else:
        pass
    authorization = ledger.authorize(
        pull_request_number=request.pull_request_number,
        budget=request.budget,
        follow_up_authorized=request.follow_up_authorized,
        retry_count=request.retry_count,
        reviewed_at=reviewed_at,
    )
    if not authorization.allowed:
        return assemble_report(
            bundle=bundle,
            prompt_version=prompt_version,
            input_hash=input_hash,
            model_findings=None,
            model_failed=True,
            incomplete_reason=authorization.reason,
            usage_cost_cents=0,
        )
    else:
        pass
    settings = request.model_settings
    if reviewer is None:
        if settings is None:
            raise RuntimeError("model settings missing after authorization")
        else:
            active_reviewer: DocumentationReviewer = OpenAICompatibleReviewer(
                settings, prompt_text
            )
    else:
        active_reviewer = reviewer
    raw_output, attempts, failed = call_with_retries(
        active_reviewer, render_bundle(bundle), request.retry_count
    )
    usage_cost_cents = attempts * request.budget.call_cost_cents
    if failed or raw_output is None:
        report = assemble_report(
            bundle=bundle,
            prompt_version=prompt_version,
            input_hash=input_hash,
            model_findings=None,
            model_failed=True,
            incomplete_reason="model request failed",
            usage_cost_cents=usage_cost_cents,
        )
    else:
        parsed = parse_model_findings(raw_output)
        if parsed is None:
            report = assemble_report(
                bundle=bundle,
                prompt_version=prompt_version,
                input_hash=input_hash,
                model_findings=None,
                model_failed=True,
                incomplete_reason="model output was not a finding list",
                usage_cost_cents=usage_cost_cents,
            )
        else:
            report = assemble_report(
                bundle=bundle,
                prompt_version=prompt_version,
                input_hash=input_hash,
                model_findings=parsed,
                model_failed=False,
                incomplete_reason="",
                usage_cost_cents=usage_cost_cents,
            )
    ledger.store(request.pull_request_number, report, reviewed_at)
    return report


def call_with_retries(
    reviewer: DocumentationReviewer, bundle_text: str, retry_count: int
) -> tuple[str | None, int, bool]:
    attempts = 0
    last_error = False
    raw_output: str | None = None
    while attempts <= retry_count:
        attempts += 1
        try:
            raw_output = reviewer.review(bundle_text)
        except ReviewerError:
            last_error = True
            raw_output = None
        else:
            last_error = False
            break
    return raw_output, attempts, last_error


def parse_model_findings(raw: str) -> list[Finding] | None:
    try:
        payload: object = json.loads(raw)
    except json.JSONDecodeError:
        return None
    else:
        pass
    if not isinstance(payload, list):
        return None
    else:
        findings: list[Finding] = []
        for entry in payload:
            if not isinstance(entry, dict):
                return None
            else:
                try:
                    findings.append(Finding.model_validate(entry))
                except ValidationError:
                    return None
        return findings


def assemble_report(
    bundle: ReviewBundle,
    prompt_version: str,
    input_hash: str,
    model_findings: list[Finding] | None,
    model_failed: bool,
    incomplete_reason: str,
    usage_cost_cents: int,
) -> Report:
    deterministic = audit_bundle(bundle)
    combined = list(deterministic)
    if model_findings is None:
        pass
    else:
        combined.extend(model_findings)
    kept, pre_existing, dropped_citations, dropped_absence = validate_findings(
        combined, bundle
    )
    check_results = interpret_checks(bundle.checks, bundle.head_commit_sha)
    if model_failed:
        execution = "incomplete"
        outcome = "incomplete"
    elif kept or pre_existing:
        execution = "complete"
        outcome = "findings"
    else:
        execution = "complete"
        outcome = "no_findings"
    return Report(
        advisory=True,
        execution=execution,
        outcome=outcome,
        review_input_sha256=input_hash,
        policy_version=bundle.policy_version,
        prompt_version=prompt_version,
        repository_identity=bundle.repository_identity,
        base_commit_sha=bundle.base_commit_sha,
        head_commit_sha=bundle.head_commit_sha,
        merge_base_commit_sha=bundle.merge_base_commit_sha,
        reviewed_paths=list(bundle.reviewed_paths),
        style_guide_limitation=bundle.style_guide_limitation,
        findings=kept,
        pre_existing_findings=pre_existing,
        check_results=check_results,
        author_confirmation=author_confirmation_statement(bundle.pull_request_text),
        usage_cost_cents=usage_cost_cents,
        incomplete_reason=incomplete_reason,
        dropped_citation_count=dropped_citations,
        dropped_unsupported_absence_count=dropped_absence,
    )


def validate_findings(
    findings: list[Finding], bundle: ReviewBundle
) -> tuple[list[Finding], list[Finding], int, int]:
    kept: list[Finding] = []
    pre_existing: list[Finding] = []
    dropped_citations = 0
    dropped_absence = 0
    for finding in findings:
        if finding.rule_id not in KNOWN_RULE_IDS:
            dropped_citations += 1
        elif not citation_matches(
            bundle,
            finding.file_path,
            finding.line_start,
            finding.line_end,
            finding.quote,
        ):
            dropped_citations += 1
        elif finding.evidence_quote == "" or not citation_matches(
            bundle,
            finding.evidence_file_path,
            finding.evidence_line_start,
            finding.evidence_line_end,
            finding.evidence_quote,
        ):
            dropped_citations += 1
        elif finding.parameter_absence_from_text_search and parameter_defined(
            bundle, finding.parameter_name
        ):
            dropped_absence += 1
        else:
            accepted = finding
            if finding.parameter_absence_from_text_search:
                accepted = finding.model_copy(update={"certainty": "uncertain"})
            else:
                pass
            if accepted.change_relation == "pre_existing":
                pre_existing.append(accepted)
            else:
                kept.append(accepted)
    return kept, pre_existing, dropped_citations, dropped_absence


def citation_matches(
    bundle: ReviewBundle, file_path: str, line_start: int, line_end: int, quote: str
) -> bool:
    if quote == "":
        return False
    else:
        pass
    for collected in bundle.files:
        if collected.file_path == file_path:
            lines = collected.content.splitlines()
            if line_start < 1 or line_end > len(lines) or line_end < line_start:
                return False
            else:
                excerpt = "\n".join(lines[line_start - 1 : line_end])
                return quote in excerpt
        else:
            pass
    return False


def parameter_defined(bundle: ReviewBundle, parameter_name: str) -> bool:
    if parameter_name == "":
        return False
    else:
        pattern = re.compile(rf"\b{re.escape(parameter_name)}\b")
        for collected in bundle.files:
            suffix_is_code = collected.file_path.endswith(
                (".py", ".yaml", ".yml", ".toml", ".json")
            )
            if (
                suffix_is_code
                and collected.change != "review_input"
                and pattern.search(collected.content)
            ):
                return True
            else:
                pass
        return False


def render_markdown(report: Report) -> str:
    lines = [
        "# Documentation review",
        "",
        "This report is advisory. It does not approve, merge, or close the pull request.",
        "",
        f"Input hash: `{report.review_input_sha256}`",
        f"Policy: `{report.policy_version}`",
        f"Base: `{report.base_commit_sha}`",
        f"Head: `{report.head_commit_sha}`",
        f"Author confirmation: {report.author_confirmation}",
        "",
        "## Reviewed paths",
        "",
    ]
    if report.reviewed_paths:
        for file_path in report.reviewed_paths:
            lines.append(f"- `{file_path}`")
    else:
        lines.append("- None.")
    if report.style_guide_limitation:
        lines.extend(["", "## Style guide", "", report.style_guide_limitation])
    else:
        pass
    lines.extend(["", "## Checks", ""])
    if report.check_results:
        for check in report.check_results:
            lines.append(format_check(check))
    else:
        lines.append("No check results were selected. That is not a pass.")
    lines.extend(["", "## Findings", ""])
    if report.execution == "incomplete":
        lines.append(f"This review is incomplete. {report.incomplete_reason}")
        lines.append("")
    else:
        pass
    lines.extend(format_finding_section(report.findings))
    lines.extend(["", "## Pre-existing", ""])
    lines.extend(format_finding_section(report.pre_existing_findings))
    if (
        report.execution == "complete"
        and not report.findings
        and not report.pre_existing_findings
    ):
        lines.extend(["", "No findings."])
    else:
        pass
    if report.dropped_citation_count or report.dropped_unsupported_absence_count:
        lines.extend(
            [
                "",
                "## Dropped",
                "",
                f"Dropped citations: {report.dropped_citation_count}.",
                f"Dropped unsupported parameter-absence claims: {report.dropped_unsupported_absence_count}.",
            ]
        )
    else:
        pass
    return "\n".join(lines) + "\n"


def format_check(check: CheckView) -> str:
    passed = "counted as passed" if check.counted_as_passed else "not counted as passed"
    return (
        f"- {check.name}: {check.status.value}, {check.freshness.value}, "
        f"commit {check.execution_commit_sha}, coverage {check.coverage}, {passed}. {check.explanation}"
    )


def format_finding_section(findings: list[Finding]) -> list[str]:
    if not findings:
        return ["None."]
    else:
        lines: list[str] = []
        for finding in findings:
            lines.append(
                f"- `{finding.rule_id}` ({finding.certainty}, {finding.change_relation}) "
                f"`{finding.file_path}`:{finding.line_start}: {finding.claim}"
            )
            lines.append(
                f"  Evidence: `{finding.evidence_file_path}` `{finding.evidence_quote}`"
            )
            lines.append(f"  Impact: {finding.impact}")
            lines.append(f"  Fix: {finding.suggested_fix}")
        return lines
