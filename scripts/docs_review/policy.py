# SPDX-License-Identifier: Apache-2.0
"""Versioned documentation review rules.

The rules ship with this tool. A pull request cannot replace them.
"""

from __future__ import annotations

from pathlib import Path

from scripts.docs_review.models import ReviewRule, RuleKind

POLICY_VERSION = "2026-11-05"
STYLE_GUIDE_PATH = "docs/STYLE_GUIDE.md"
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.md"

DOCS_MATCH_CODE = "DOCS_MATCH_CODE"
CLAIMS_HAVE_EVIDENCE = "CLAIMS_HAVE_EVIDENCE"
NO_DUPLICATED_SHARED_CONTENT = "NO_DUPLICATED_SHARED_CONTENT"
CONTRIBUTOR_CHECKLIST = "CONTRIBUTOR_CHECKLIST"
PERFORMANCE_EVIDENCE = "PERFORMANCE_EVIDENCE"

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
    ),
)

KNOWN_RULE_IDS = frozenset(rule.rule_id for rule in REVIEW_RULES)


def read_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")
