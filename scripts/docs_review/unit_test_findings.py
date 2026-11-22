# SPDX-License-Identifier: Apache-2.0
"""Resolve model citations against the frozen unit-test review input."""

from __future__ import annotations

import ast
import re

from scripts.docs_review.models import (
    Finding,
    ReviewBundle,
    SourceCitation,
    UnitTestCase,
)
from scripts.docs_review.snapshot import citation_matches
from scripts.docs_review.unit_tests import (
    UNIT_TEST_ROOT,
    case_start,
    is_context_path,
    is_unit_test_file,
    test_definitions,
)

SELF_DERIVED_EXPECTATION = "UNIT_TEST_SELF_DERIVED_EXPECTATION"
DUPLICATE_RULE_ID = "UNIT_TEST_DUPLICATE"
UNIT_TEST_RULE_IDS = {
    SELF_DERIVED_EXPECTATION,
    "UNIT_TEST_INTERNAL_ASSERTION",
    DUPLICATE_RULE_ID,
    "UNIT_TEST_INSUFFICIENT_EVIDENCE",
}
INSUFFICIENT_EVIDENCE = "UNIT_TEST_INSUFFICIENT_EVIDENCE"


def locate_quoted_lines(source: str, quote: str) -> tuple[int, int] | None:
    quote = quote.strip()
    if quote.startswith("```") and quote.endswith("```") and "\n" in quote:
        quote = quote.split("\n", 1)[1][:-3].strip()
    else:
        pass
    quote_lines = []
    for line in quote.splitlines():
        quote_lines.append(re.sub(r"^\s*\d+: ", "", line).strip())
    if not quote_lines or not any(quote_lines):
        return None
    else:
        pass
    if (
        len(quote_lines) == 1
        and quote_lines[0].startswith("`")
        and quote_lines[0].endswith("`")
    ):
        quote_lines[0] = quote_lines[0][1:-1].strip()
    else:
        pass
    if not any(quote_lines):
        return None
    else:
        pass

    source_lines = source.splitlines()
    matches = []
    last_start = len(source_lines) - len(quote_lines)
    for start in range(last_start + 1):
        lines = source_lines[start : start + len(quote_lines)]
        if len(quote_lines) == 1 and quote_lines[0] in lines[0]:
            matches.append((start + 1, start + 1))
        elif [line.strip() for line in lines] == quote_lines:
            matches.append((start + 1, start + len(quote_lines)))
        else:
            pass
    return matches[0] if len(matches) == 1 else None


def locate_unit_test_citation(
    bundle: ReviewBundle, file_path: str, quote: str
) -> SourceCitation | None:
    source = next(
        (file.content for file in bundle.files if file.file_path == file_path),
        None,
    )
    if source is None:
        return None
    else:
        positions = locate_quoted_lines(source, quote)
        if positions is None:
            return None
        else:
            line_start, line_end = positions
            return SourceCitation(
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                quote="\n".join(source.splitlines()[line_start - 1 : line_end]),
            )


def resolve_unit_test_citation(
    bundle: ReviewBundle, citation: SourceCitation
) -> SourceCitation | None:
    if not citation.quote.strip():
        return None
    elif citation_matches(
        bundle,
        citation.file_path,
        citation.line_start,
        citation.line_end,
        citation.quote,
    ):
        return citation
    else:
        return locate_unit_test_citation(bundle, citation.file_path, citation.quote)


def complete_citation_payload(
    bundle: ReviewBundle, citation_payload: dict[str, object], prefix: str
) -> None:
    start_field = f"{prefix}line_start"
    end_field = f"{prefix}line_end"
    quote_field = f"{prefix}quote"
    file_path = citation_payload.get(f"{prefix}file_path")
    quote = citation_payload.get(quote_field)
    start = citation_payload.get(start_field)
    end = citation_payload.get(end_field)
    if not isinstance(file_path, str):
        return
    elif isinstance(quote, str) and quote.strip():
        if start is not None and end is not None:
            return
        elif (
            type(start) is int
            and end is None
            and citation_matches(bundle, file_path, start, start, quote)
        ):
            citation_payload[end_field] = start
            return
        else:
            pass
        citation = locate_unit_test_citation(bundle, file_path, quote)
        if citation is not None:
            citation_payload[start_field] = citation.line_start
            citation_payload[end_field] = citation.line_end
        else:
            pass
    elif (
        type(start) is int
        and start > 0
        and (end is None or (type(end) is int and end == start))
    ):
        source = next(
            (file.content for file in bundle.files if file.file_path == file_path), ""
        )
        lines = source.splitlines()
        if start <= len(lines) and lines[start - 1].strip():
            citation_payload[end_field] = start
            citation_payload[quote_field] = lines[start - 1]
        else:
            pass
    else:
        pass


def normalize_unit_test_citations(
    bundle: ReviewBundle, finding_payload: dict[str, object]
) -> None:
    complete_citation_payload(bundle, finding_payload, "")
    if finding_payload.get("rule_id") == INSUFFICIENT_EVIDENCE:
        finding_payload["evidence_file_path"] = finding_payload.get("file_path")
        finding_payload["evidence_line_start"] = finding_payload.get("line_start")
        finding_payload["evidence_line_end"] = finding_payload.get("line_end")
        finding_payload["evidence_quote"] = ""
    else:
        evidence = finding_payload.get("evidence")
        if isinstance(evidence, dict):
            for field in ("file_path", "line_start", "line_end", "quote"):
                if field in evidence:
                    finding_payload.setdefault(f"evidence_{field}", evidence[field])
                else:
                    pass
        else:
            pass
        complete_citation_payload(bundle, finding_payload, "evidence_")
    covering_test = finding_payload.get("duplicate_test")
    if isinstance(covering_test, dict):
        complete_citation_payload(bundle, covering_test, "")
    else:
        pass
    changed_dependency = finding_payload.get("change_evidence")
    if isinstance(changed_dependency, dict):
        complete_citation_payload(bundle, changed_dependency, "")
    else:
        pass


def covers_other_test(
    bundle: ReviewBundle, case: UnitTestCase, duplicate: SourceCitation
) -> bool:
    if (
        not is_unit_test_file(duplicate.file_path)
        or not duplicate.quote.strip()
        or not citation_matches(
            bundle,
            duplicate.file_path,
            duplicate.line_start,
            duplicate.line_end,
            duplicate.quote,
        )
        or (
            duplicate.file_path == case.file_path
            and duplicate.line_start <= case.line_end
            and case.line_start <= duplicate.line_end
        )
    ):
        return False
    else:
        pass
    source = next(
        file.content for file in bundle.files if file.file_path == duplicate.file_path
    )
    try:
        definitions = test_definitions(ast.parse(source))
    except (SyntaxError, ValueError):
        return False
    else:
        return any(
            definition.end_lineno is not None
            and case_start(definition)
            <= duplicate.line_start
            <= duplicate.line_end
            <= definition.end_lineno
            for definition in definitions.values()
        )


def validate_unit_test_findings(
    findings: list[Finding], bundle: ReviewBundle, *, check_contracts: bool
) -> tuple[list[Finding], list[Finding], int, list[str]]:
    accepted: list[Finding] = []
    pre_existing: list[Finding] = []
    limitations: list[str] = []
    dropped = 0
    contract_paths = {
        file.file_path
        for file in bundle.files
        if is_context_path(file.file_path)
        and not file.file_path.startswith(UNIT_TEST_ROOT)
    }
    candidates = findings
    for index, finding in enumerate(candidates, 1):
        test_citation = resolve_unit_test_citation(
            bundle,
            SourceCitation(
                file_path=finding.file_path,
                line_start=finding.line_start,
                line_end=finding.line_end,
                quote=finding.quote,
            ),
        )
        changed_dependency = (
            resolve_unit_test_citation(bundle, finding.change_evidence)
            if finding.change_evidence is not None
            else None
        )
        case = next(
            (
                case
                for case in bundle.unit_test_cases
                if test_citation is not None
                and not case.evidence_limitations
                and case.file_path == test_citation.file_path
                and case.line_start
                <= test_citation.line_start
                <= test_citation.line_end
                <= case.line_end
                and (
                    any(
                        start <= test_citation.line_end
                        and test_citation.line_start <= end
                        for start, end in case.changed_line_ranges
                    )
                    or (
                        changed_dependency is not None
                        and any(
                            changed_dependency.file_path == citation.file_path
                            and changed_dependency.line_start <= citation.line_end
                            and citation.line_start <= changed_dependency.line_end
                            for citation in case.change_evidence
                        )
                    )
                )
            ),
            None,
        )
        evidence_citation = (
            resolve_unit_test_citation(
                bundle,
                SourceCitation(
                    file_path=finding.evidence_file_path,
                    line_start=finding.evidence_line_start,
                    line_end=finding.evidence_line_end,
                    quote=finding.evidence_quote,
                ),
            )
            if finding.evidence_file_path in contract_paths
            else None
        )
        duplicate = (
            resolve_unit_test_citation(bundle, finding.duplicate_test)
            if finding.duplicate_test is not None
            else None
        )
        if test_citation is None or case is None:
            rejection = "test citation does not match a changed case"
        elif (
            finding.rule_id == INSUFFICIENT_EVIDENCE
            and finding.certainty == "uncertain"
        ):
            limitations.append(
                f"{case.file_path}:{case.line_start}: insufficient contract evidence "
                f"for {case.name}. {finding.claim.strip()}"
            )
            continue
        elif (
            finding.rule_id not in UNIT_TEST_RULE_IDS
            or finding.certainty != "supported"
        ):
            rejection = "unsupported rule or certainty"
        elif not all(
            value.strip()
            for value in (
                finding.claim,
                finding.public_contract,
                finding.regression,
                finding.impact,
                finding.suggested_fix,
            )
        ):
            rejection = "missing contract, regression, impact, or suggested fix"
        elif evidence_citation is None:
            rejection = "contract citation does not match the collected source"
        elif finding.rule_id == DUPLICATE_RULE_ID and (
            duplicate is None or not covers_other_test(bundle, case, duplicate)
        ):
            rejection = "covering citation does not match a different test"
        elif (
            finding.rule_id != DUPLICATE_RULE_ID and finding.duplicate_test is not None
        ):
            rejection = "covering citation was supplied for a nonduplicate finding"
        else:
            verified_finding = finding.model_copy(
                update={
                    "line_start": test_citation.line_start,
                    "line_end": test_citation.line_end,
                    "quote": test_citation.quote,
                    "evidence_line_start": evidence_citation.line_start,
                    "evidence_line_end": evidence_citation.line_end,
                    "evidence_quote": evidence_citation.quote,
                    "duplicate_test": duplicate,
                    "change_evidence": changed_dependency,
                }
            )
            if verified_finding.change_relation == "pre_existing":
                pre_existing.append(verified_finding)
            else:
                accepted.append(verified_finding)
            continue
        dropped += 1
        limitations.append(f"Candidate {index}: {rejection}.")
    return accepted, pre_existing, dropped, limitations
