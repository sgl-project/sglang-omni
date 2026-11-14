# SPDX-License-Identifier: Apache-2.0
"""Run one advisory documentation review and validate its citations."""

from __future__ import annotations

import hashlib
import json
import re
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
    review_input_sha256,
)
from scripts.docs_review.ledger import ReviewLedger
from scripts.docs_review.models import (
    BundlePayload,
    Clock,
    DocumentationReviewer,
    Finding,
    Report,
    ReviewBundle,
    ReviewRequest,
)
from scripts.docs_review.policy import KNOWN_RULE_IDS, read_prompt
from scripts.docs_review.reporting import render_bundle
from scripts.docs_review.transport import OpenAICompatibleReviewer, ReviewerError


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(kw_only=True)
class ReviewRun:
    bundle: ReviewBundle
    payload: BundlePayload
    report: Report


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
    input_hash = review_input_sha256(bundle, request.model_settings, prompt_version)
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
    coverage_limitations = [
        f"{collected.file_path}: {'missing' if collected.missing else 'truncated'}"
        for collected in bundle.files
        if collected.missing or collected.truncated
    ]
    if coverage_limitations:
        incomplete_reason = "; ".join(
            filter(None, [incomplete_reason, *coverage_limitations])
        )
    else:
        pass
    if model_failed or coverage_limitations:
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
        document_scopes=list(bundle.document_scopes),
        coverage_limitations=coverage_limitations,
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
        elif any(
            scope.file_path == finding.file_path
            and finding.rule_id not in scope.rule_ids
            for scope in bundle.document_scopes
        ):
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
