# SPDX-License-Identifier: Apache-2.0
"""Versioned documentation review rules.

The rules ship with this tool. A pull request cannot replace them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from pydantic import TypeAdapter

from scripts.docs_review.models import (
    DocumentScope,
    DocumentType,
    Finding,
    ReviewRule,
    RuleKind,
)

POLICY_VERSION = "2026-11-05.1"
STYLE_GUIDE_PATH = "docs/STYLE_GUIDE.md"
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.md"

DOCS_MATCH_CODE = "DOCS_MATCH_CODE"
CLAIMS_HAVE_EVIDENCE = "CLAIMS_HAVE_EVIDENCE"
NO_DUPLICATED_SHARED_CONTENT = "NO_DUPLICATED_SHARED_CONTENT"
CONTRIBUTOR_CHECKLIST = "CONTRIBUTOR_CHECKLIST"
PERFORMANCE_EVIDENCE = "PERFORMANCE_EVIDENCE"
COOKBOOK_RECIPE = "COOKBOOK_RECIPE"
API_CONTRACT = "API_CONTRACT"
ARCHITECTURE_CONSISTENCY = "ARCHITECTURE_CONSISTENCY"

DOCUMENT_TYPE_PREFIXES = {
    "docs/cookbook/": DocumentType.COOKBOOK,
    "docs/api_reference/": DocumentType.API_REFERENCE,
    "docs/developer_reference/": DocumentType.DEVELOPER_REFERENCE,
    "docs/benchmarks/": DocumentType.BENCHMARK,
}
API_REFERENCE_PATHS = frozenset(
    {
        "docs/user_guide/serving/speech_api.md",
        "docs/user_guide/serving/transcription_api.md",
    }
)
DOCUMENT_TYPE_HEADINGS = {
    DocumentType.COOKBOOK: re.compile(
        r"\b(cookbook|deployment recipe)\b", re.IGNORECASE
    ),
    DocumentType.API_REFERENCE: re.compile(
        r"\b(api reference|api contract|endpoint reference)\b", re.IGNORECASE
    ),
    DocumentType.DEVELOPER_REFERENCE: re.compile(
        r"\b(architecture|developer reference|call flow)\b", re.IGNORECASE
    ),
    DocumentType.BENCHMARK: re.compile(
        r"\b(benchmark|performance results|performance comparison)\b", re.IGNORECASE
    ),
}
PERFORMANCE_SECTION = re.compile(
    r"\b(benchmark|speedup|speed-up|faster|performance results|performance comparison)\b",
    re.IGNORECASE,
)

REVIEW_RULES: tuple[ReviewRule, ...] = (
    ReviewRule(
        rule_id=DOCS_MATCH_CODE,
        kind=RuleKind.IMPLEMENTATION_CONSISTENCY,
        summary="Documentation claims match the code and config at the reviewed commit.",
        required_evidence="A quote from the doc and a quote from the implementation or schema.",
    ),
    ReviewRule(
        rule_id=CLAIMS_HAVE_EVIDENCE,
        kind=RuleKind.MISSING_EVIDENCE,
        summary="Operational and hardware claims cite evidence, or the gap is reported as missing.",
        required_evidence="The claim quote and the evidence quote, or an explicit missing-evidence note.",
    ),
    ReviewRule(
        rule_id=NO_DUPLICATED_SHARED_CONTENT,
        kind=RuleKind.DUPLICATED_CONTENT,
        summary="Shared behavior lives in one page. Other pages point at it.",
        required_evidence="Quotes from both pages and the canonical page the duplicate should link to.",
    ),
    ReviewRule(
        rule_id=CONTRIBUTOR_CHECKLIST,
        kind=RuleKind.CONTRIBUTOR_CHECKLIST,
        summary="The contributor checklist in CONTRIBUTING.md is filled in when docs change.",
        required_evidence="The checklist section in the pull request, recorded as author-reported.",
    ),
    ReviewRule(
        rule_id=PERFORMANCE_EVIDENCE,
        kind=RuleKind.MISSING_EVIDENCE,
        summary="A performance claim has comparable before and after numbers, units, and quality.",
        required_evidence="Baseline and head commits, both measurements, the command, and the setup.",
        document_types=(DocumentType.BENCHMARK,),
    ),
    ReviewRule(
        rule_id=COOKBOOK_RECIPE,
        kind=RuleKind.MISSING_EVIDENCE,
        summary="Deployment steps, model-specific exceptions, and validated configurations follow the doc standards.",
        required_evidence="The recipe claim, checked-in configuration, and validation record or explicit evidence gap.",
        document_types=(DocumentType.COOKBOOK,),
    ),
    ReviewRule(
        rule_id=API_CONTRACT,
        kind=RuleKind.IMPLEMENTATION_CONSISTENCY,
        summary="Parameters, defaults, inputs, outputs, and errors match the implementation.",
        required_evidence="The API claim and the corresponding parser, schema, or response implementation.",
        document_types=(DocumentType.API_REFERENCE,),
    ),
    ReviewRule(
        rule_id=ARCHITECTURE_CONSISTENCY,
        kind=RuleKind.IMPLEMENTATION_CONSISTENCY,
        summary="Module responsibilities and call flow match the reviewed implementation.",
        required_evidence="The architecture claim and the implementation of the relevant call flow.",
        document_types=(DocumentType.DEVELOPER_REFERENCE,),
    ),
)

KNOWN_RULE_IDS = frozenset(rule.rule_id for rule in REVIEW_RULES)


def read_prompt() -> str:
    rules = [
        {
            **asdict(rule),
            "kind": rule.kind.value,
            "document_types": [
                document_type.value for document_type in rule.document_types
            ],
        }
        for rule in REVIEW_RULES
    ]
    contract = {
        "rule_sources": ["CONTRIBUTING.md", STYLE_GUIDE_PATH],
        "rules": rules,
        "output_schema": TypeAdapter(list[Finding]).json_schema(),
    }
    return (
        PROMPT_PATH.read_text(encoding="utf-8")
        + "\nTrusted review contract:\n"
        + json.dumps(contract, sort_keys=True)
    )


def classify_document(file_path: str, content: str) -> DocumentScope:
    document_types = [
        document_type
        for prefix, document_type in DOCUMENT_TYPE_PREFIXES.items()
        if file_path.startswith(prefix)
    ]
    if file_path in API_REFERENCE_PATHS:
        document_types.append(DocumentType.API_REFERENCE)
    else:
        pass
    if document_types:
        classification_basis = "path"
    else:
        heading = next(
            (line.strip("# ") for line in content.splitlines() if line.strip()), ""
        )
        document_types = [
            document_type
            for document_type, pattern in DOCUMENT_TYPE_HEADINGS.items()
            if pattern.search(heading)
        ]
        classification_basis = (
            "content" if document_types else "general fallback; type undetermined"
        )
    if (
        PERFORMANCE_SECTION.search(content)
        and DocumentType.BENCHMARK not in document_types
    ):
        document_types.append(DocumentType.BENCHMARK)
        classification_basis += "; performance content"
    else:
        pass
    if not document_types:
        document_types = [DocumentType.GENERAL]
    else:
        pass
    return DocumentScope(
        file_path=file_path,
        document_types=document_types,
        classification_basis=classification_basis,
        rule_ids=[
            rule.rule_id
            for rule in REVIEW_RULES
            if any(
                document_type in rule.document_types for document_type in document_types
            )
        ],
    )
