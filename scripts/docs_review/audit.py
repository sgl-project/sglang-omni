# SPDX-License-Identifier: Apache-2.0
"""Deterministic checklist and check-status audit.

These checks report missing evidence. They do not reject a pull request.
"""

from __future__ import annotations

import re

from scripts.docs_review.models import (
    CheckFreshness,
    CheckRecord,
    CheckStatus,
    CheckView,
    Finding,
    ReviewBundle,
)
from scripts.docs_review.policy import CONTRIBUTOR_CHECKLIST, PERFORMANCE_EVIDENCE

CHECKLIST_HEADING = "## Contributor checklist"
PERFORMANCE_CLAIM = re.compile(
    r"(faster|speedup|speed-up|\d+(?:\.\d+)?\s*x|\d+(?:\.\d+)?\s*%)",
    re.IGNORECASE,
)
SPEECH_WORKLOAD = re.compile(r"\b(tts|asr|speech|seedtts)\b", re.IGNORECASE)
REQUIRED_CHECKLIST_FIELDS = (
    "Purpose",
    "Reviewed the final diff",
    "Checks",
    "Untested",
    "Performance",
)
PERFORMANCE_DETAIL_FIELDS = (
    "Baseline commit",
    "Head commit",
    "Before",
    "After",
    "Unit",
    "Command",
    "Hardware",
    "Software",
    "Model",
    "Dataset",
    "Config",
    "Load",
    "Repeats",
    "Raw results",
    "Quality",
)


def audit_bundle(bundle: ReviewBundle) -> list[Finding]:
    findings: list[Finding] = []
    if bundle.reviewed_paths:
        findings.extend(audit_checklist(bundle))
    else:
        pass
    return findings


def interpret_checks(
    checks: list[CheckRecord], head_commit_sha: str
) -> list[CheckView]:
    return [interpret_check(check, head_commit_sha) for check in checks]


def author_confirmation_statement(pull_request_text: str) -> str:
    fields = parse_checklist(pull_request_text)
    if fields is None or "Reviewed the final diff" not in fields:
        return "not provided"
    else:
        statement = fields["Reviewed the final diff"]
        return f"author_reported: {statement}"


def parse_checklist(pull_request_text: str) -> dict[str, str] | None:
    if CHECKLIST_HEADING not in pull_request_text:
        return None
    else:
        pass
    after = pull_request_text.split(CHECKLIST_HEADING, maxsplit=1)[1]
    fields: dict[str, str] = {}
    for line in after.splitlines():
        if line.startswith("## "):
            break
        elif line.startswith("- ") and ": " in line:
            field_name, value = line[2:].split(": ", maxsplit=1)
            fields[field_name.strip()] = value.strip()
        else:
            pass
    return fields


def audit_checklist(bundle: ReviewBundle) -> list[Finding]:
    fields = parse_checklist(bundle.pull_request_text)
    findings: list[Finding] = []
    if fields is None:
        findings.append(
            checklist_finding(
                bundle,
                "Contributor checklist is missing.",
                "A maintainer cannot see what the author reviewed or ran.",
                "Paste the contributor checklist from CONTRIBUTING.md. This is a follow-up, not a rejection.",
            )
        )
        return findings
    else:
        pass
    missing = [
        field_name
        for field_name in REQUIRED_CHECKLIST_FIELDS
        if field_missing(fields, field_name)
    ]
    if missing:
        findings.append(
            checklist_finding(
                bundle,
                f"Contributor checklist is missing {', '.join(missing)}.",
                "A maintainer cannot tell what was reviewed or measured.",
                "Fill in the missing checklist fields. This is a follow-up, not a rejection.",
            )
        )
    else:
        pass
    reviewed = fields.get("Reviewed the final diff", "")
    if reviewed and reviewed.lower() != "yes":
        findings.append(
            checklist_finding(
                bundle,
                "The author has not confirmed that they reviewed the final diff.",
                "The confirmation is self-reported, and this one is not yes.",
                "Read the final diff and set Reviewed the final diff to yes. This is not a rejection.",
            )
        )
    else:
        pass
    if performance_details_required(fields, bundle.added_documentation_text):
        findings.extend(audit_performance(bundle, fields))
    else:
        pass
    return findings


def field_missing(fields: dict[str, str], field_name: str) -> bool:
    if field_name == "Performance" and any(
        fields.get(name) for name in PERFORMANCE_DETAIL_FIELDS
    ):
        return False
    else:
        return field_name not in fields or fields[field_name] == ""


def performance_details_required(fields: dict[str, str], added_text: str) -> bool:
    if PERFORMANCE_CLAIM.search(added_text):
        return True
    elif fields.get("Performance", "").lower() == "not applicable":
        return False
    elif "Performance" not in fields:
        return False
    else:
        return True


def audit_performance(bundle: ReviewBundle, fields: dict[str, str]) -> list[Finding]:
    findings: list[Finding] = []
    missing = [
        field_name
        for field_name in PERFORMANCE_DETAIL_FIELDS
        if field_name not in fields or fields[field_name] == ""
    ]
    if missing:
        findings.append(
            performance_finding(
                bundle,
                f"Performance claim is missing comparable evidence: {', '.join(missing)}.",
                "A reader cannot tell whether the numbers are comparable.",
                "Add the missing before/after measurements. This is a follow-up, not a rejection.",
            )
        )
    else:
        pass
    comparable = fields.get("Conditions comparable", "yes")
    if comparable.lower() == "no":
        findings.append(
            performance_finding(
                bundle,
                "Performance conditions are not comparable.",
                "Before and after numbers from different conditions are not a speedup.",
                "Rerun both sides on the same hardware, model, dataset, and load.",
            )
        )
    else:
        pass
    if SPEECH_WORKLOAD.search(bundle.added_documentation_text):
        dataset = fields.get("Dataset", "")
        baseline_choice = fields.get("Baseline choice", "")
        if "seedtts" in dataset.lower() or baseline_choice:
            pass
        else:
            findings.append(
                performance_finding(
                    bundle,
                    "Speech performance evidence does not use the SeedTTS baseline and does not explain the choice.",
                    "Speech numbers are not comparable to the repository baseline.",
                    "Use SeedTTS or fill in Baseline choice.",
                )
            )
    else:
        pass
    return findings


def interpret_check(check: CheckRecord, head_commit_sha: str) -> CheckView:
    freshness = check.freshness
    if (
        check.execution_commit_sha != head_commit_sha
        and freshness == CheckFreshness.CURRENT
    ):
        freshness = CheckFreshness.STALE
    else:
        pass
    counted_as_passed = (
        check.status == CheckStatus.PASSED and freshness == CheckFreshness.CURRENT
    )
    if check.coverage == "docs_build" and counted_as_passed:
        explanation = "Docs build passed. This does not validate launch commands or hardware claims."
    elif freshness == CheckFreshness.STALE:
        explanation = "Stale check result is not a current pass."
    elif check.status == CheckStatus.SKIPPED:
        explanation = "Skipped is not a pass."
    elif check.status == CheckStatus.NOT_RUN:
        explanation = "Not run is not a pass."
    elif check.status == CheckStatus.UNKNOWN:
        explanation = "Unknown is not a pass."
    elif counted_as_passed:
        explanation = "Passed for the recorded coverage only."
    else:
        explanation = f"{check.status.value} is not a pass."
    return CheckView(
        name=check.name,
        status=check.status,
        freshness=freshness,
        execution_commit_sha=check.execution_commit_sha,
        coverage=check.coverage,
        counted_as_passed=counted_as_passed,
        explanation=explanation,
    )


def checklist_finding(
    bundle: ReviewBundle, claim: str, impact: str, suggested_fix: str
) -> Finding:
    return anchored_finding(bundle, CONTRIBUTOR_CHECKLIST, claim, impact, suggested_fix)


def performance_finding(
    bundle: ReviewBundle, claim: str, impact: str, suggested_fix: str
) -> Finding:
    return anchored_finding(bundle, PERFORMANCE_EVIDENCE, claim, impact, suggested_fix)


def anchored_finding(
    bundle: ReviewBundle,
    rule_id: str,
    claim: str,
    impact: str,
    suggested_fix: str,
) -> Finding:
    file_path, quote = citation_anchor(bundle)
    return Finding(
        rule_id=rule_id,
        file_path=file_path,
        line_start=1,
        line_end=1,
        quote=quote,
        claim=claim,
        evidence_file_path=file_path,
        evidence_line_start=1,
        evidence_line_end=1,
        evidence_quote=quote,
        impact=impact,
        suggested_fix=suggested_fix,
        change_relation="introduced",
        certainty="supported",
    )


def citation_anchor(bundle: ReviewBundle) -> tuple[str, str]:
    for collected in bundle.files:
        if (
            collected.file_path in bundle.reviewed_paths
            and collected.content.splitlines()
        ):
            return collected.file_path, collected.content.splitlines()[0]
        else:
            pass
    for collected in bundle.files:
        if collected.file_path == "scope.txt" and collected.content.splitlines():
            return collected.file_path, collected.content.splitlines()[0]
        else:
            pass
    return "scope.txt", "No documentation files in scope."
