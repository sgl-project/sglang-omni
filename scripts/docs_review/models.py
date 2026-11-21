# SPDX-License-Identifier: Apache-2.0
"""Types for one local documentation review."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, model_validator


class CheckStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_RUN = "not_run"
    PENDING = "pending"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"


class CheckFreshness(Enum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class RuleKind(Enum):
    IMPLEMENTATION_CONSISTENCY = "implementation_consistency"
    MISSING_EVIDENCE = "missing_evidence"
    DUPLICATED_CONTENT = "duplicated_content"
    CONTRIBUTOR_CHECKLIST = "contributor_checklist"


class DocumentType(Enum):
    GENERAL = "general"
    COOKBOOK = "cookbook"
    API_REFERENCE = "api_reference"
    DEVELOPER_REFERENCE = "developer_reference"
    BENCHMARK = "benchmark"


class DocumentScope(BaseModel):
    file_path: str
    document_types: list[DocumentType]
    classification_basis: str
    rule_ids: list[str]


FileChange = Literal[
    "added",
    "modified",
    "deleted",
    "included",
    "trusted_base",
    "review_input",
]


class CheckRecord(BaseModel):
    name: str
    status: CheckStatus
    freshness: CheckFreshness
    execution_commit_sha: str
    coverage: str


class BudgetConfig(BaseModel):
    max_cost_cents_per_call: int = Field(ge=0)
    max_cost_cents_per_pull_request: int = Field(ge=0)
    max_cost_cents_per_day: int = Field(ge=0)
    call_cost_cents: int = Field(ge=0)


ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]


class ModelSettings(BaseModel):
    model_name: str
    endpoint_url: str
    max_output_tokens: int = Field(gt=0)
    api_key_environment_variable: str
    reasoning_effort: ReasoningEffort | None = None

    @model_validator(mode="after")
    def set_default_reasoning_effort(self) -> ModelSettings:
        if self.model_name == "gpt-6.1-terra" and self.reasoning_effort is None:
            self.reasoning_effort = "medium"
        else:
            pass
        return self


class Finding(BaseModel):
    rule_id: str
    file_path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    quote: str
    claim: str
    evidence_file_path: str
    evidence_line_start: int = Field(ge=1)
    evidence_line_end: int = Field(ge=1)
    evidence_quote: str
    impact: str
    suggested_fix: str
    change_relation: Literal["introduced", "worsened", "pre_existing", "undetermined"]
    certainty: Literal["supported", "uncertain"]
    parameter_name: str = ""
    parameter_absence_from_text_search: bool = False


class CheckView(BaseModel):
    name: str
    status: CheckStatus
    freshness: CheckFreshness
    execution_commit_sha: str
    coverage: str
    counted_as_passed: bool
    explanation: str


class Report(BaseModel):
    advisory: Literal[True]
    execution: Literal["complete", "incomplete"]
    outcome: Literal["findings", "no_findings", "incomplete"]
    review_input_sha256: str
    policy_version: str
    prompt_version: str
    repository_identity: str
    base_commit_sha: str
    head_commit_sha: str
    merge_base_commit_sha: str
    reviewed_paths: list[str]
    style_guide_limitation: str
    findings: list[Finding]
    pre_existing_findings: list[Finding]
    check_results: list[CheckView]
    author_confirmation: str
    usage_cost_cents: int = Field(ge=0)
    incomplete_reason: str
    dropped_citation_count: int = Field(ge=0)
    dropped_unsupported_absence_count: int = Field(ge=0)
    document_scopes: list[DocumentScope] = Field(default_factory=list)
    coverage_limitations: list[str] = Field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class ReviewRule:
    rule_id: str
    kind: RuleKind
    summary: str
    required_evidence: str
    document_types: tuple[DocumentType, ...] = tuple(DocumentType)


@dataclass(frozen=True, kw_only=True)
class CollectedFile:
    file_path: str
    change: FileChange
    truncated: bool
    content: str
    missing: bool = False


@dataclass(frozen=True, kw_only=True)
class ReviewBundle:
    repository_identity: str
    base_commit_sha: str
    head_commit_sha: str
    merge_base_commit_sha: str
    policy_version: str
    pull_request_text: str
    comment_text: str
    evidence_text: str
    checks: list[CheckRecord]
    files: list[CollectedFile]
    reviewed_paths: list[str]
    added_documentation_text: str
    style_guide_limitation: str
    document_scopes: list[DocumentScope] = field(default_factory=list)


@dataclass(kw_only=True)
class ReviewRequest:
    repository: Path
    base_commit_sha: str
    head_commit_sha: str
    pull_request_number: int
    pull_request_text: str
    comment_text: str = ""
    evidence_text: str = ""
    checks: list[CheckRecord] = field(default_factory=list)
    model_settings: ModelSettings | None = None
    budget: BudgetConfig | None = None
    ledger_path: Path
    follow_up_authorized: bool = False
    retry_count: int = 0
    include_paths: list[str] = field(default_factory=list)


class DocumentationReviewer(Protocol):
    def review(self, bundle_text: str) -> str:
        """Return a JSON list of findings. The bundle text is data."""


class Clock(Protocol):
    def now(self) -> datetime:
        """Return an aware UTC time."""


@dataclass(frozen=True, kw_only=True)
class SpendAuthorization:
    allowed: bool
    reason: str


class BundleFileRecord(BaseModel):
    file_path: str
    change: FileChange
    truncated: bool
    content: str
    missing: bool = False


class BundlePayload(BaseModel):
    repository_identity: str
    base_commit_sha: str
    head_commit_sha: str
    merge_base_commit_sha: str
    policy_version: str
    prompt_version: str
    review_input_sha256: str
    reviewed_paths: list[str]
    added_documentation_text: str
    style_guide_limitation: str
    pull_request_text: str
    comment_text: str
    evidence_text: str
    files: list[BundleFileRecord]
    document_scopes: list[DocumentScope] = Field(default_factory=list)
