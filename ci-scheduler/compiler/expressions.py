from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from .loader import CompilerError, UnsupportedFeatureError

ExpressionClass = Literal["copied", "rewritten", "unsupported"]

GITHUB_HOSTED_LABELS = frozenset(
    {
        "ubuntu-latest",
        "ubuntu-24.04",
        "ubuntu-22.04",
        "windows-latest",
        "macos-latest",
        "macos-14",
        "macos-13",
    }
)

NEEDS_RESULT_RE = re.compile(
    r"\bneeds\.([A-Za-z0-9_-]+)\.result\b"
)
NEEDS_OUTPUT_RE = re.compile(
    r"\bneeds\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_.-]+)\b"
)
NEEDS_OUTPUTS_RE = re.compile(
    r"\bneeds\.([A-Za-z0-9_-]+)\.outputs\b"
)
PREFLIGHT_OUTPUT_RE = re.compile(
    r"\bneeds\.preflight\.outputs\.([A-Za-z0-9_.-]+)\b"
)
GITHUB_PR_RE = re.compile(
    r"github\.event\.pull_request\.([A-Za-z0-9_.]+)"
)
GITHUB_EVENT_NAME_RE = re.compile(
    r"github\.event_name\b"
)
GITHUB_EVENT_INPUTS_RE = re.compile(
    r"github\.event\.inputs\.([A-Za-z0-9_.-]+)"
)
MATRIX_RE = re.compile(r"\bmatrix\b|strategy:\s*\n\s*matrix:")


@dataclass(frozen=True)
class RewriteResult:
    expression: str
    classification: ExpressionClass


def classify_runner(runs_on: Any) -> Literal["github-hosted", "self-hosted", "unsupported"]:
    if runs_on is None:
        return "unsupported"
    if isinstance(runs_on, str):
        if "${{" in runs_on:
            return "unsupported"
        if "self-hosted" in runs_on:
            return "self-hosted"
        if runs_on in GITHUB_HOSTED_LABELS:
            return "github-hosted"
        return "unsupported"
    if isinstance(runs_on, list):
        if any(isinstance(item, str) and "${{" in item for item in runs_on):
            return "unsupported"
        labels = [str(item) for item in runs_on]
        if any("self-hosted" in label for label in labels):
            return "self-hosted"
        if all(label in GITHUB_HOSTED_LABELS for label in labels):
            return "github-hosted"
        return "unsupported"
    return "unsupported"


def rewrite_expression(
    expr: str,
    *,
    job_key_prefix: str = "",
    need_key_map: dict[str, str] | None = None,
) -> RewriteResult:
    if not isinstance(expr, str):
        raise UnsupportedFeatureError(f"expression must be a string, got {type(expr)!r}")

    if MATRIX_RE.search(expr):
        raise UnsupportedFeatureError("matrix expressions are not supported")

    original = expr
    rewritten = expr

    def _map_need(job_id: str) -> str:
        if need_key_map and job_id in need_key_map:
            return need_key_map[job_id]
        if job_key_prefix and "::" not in job_id:
            return f"{job_key_prefix}::{job_id}"
        return job_id

    rewritten = NEEDS_RESULT_RE.sub(
        lambda match: (
            "fromJSON(inputs.scheduler_context).needs"
            f"['{_map_need(match.group(1))}'].result"
        ),
        rewritten,
    )
    rewritten = NEEDS_OUTPUT_RE.sub(
        lambda match: (
            "fromJSON(inputs.scheduler_context).needs"
            f"['{_map_need(match.group(1))}'].outputs.{match.group(2)}"
        ),
        rewritten,
    )
    rewritten = NEEDS_OUTPUTS_RE.sub(
        lambda match: (
            "fromJSON(inputs.scheduler_context).needs"
            f"['{_map_need(match.group(1))}'].outputs"
        ),
        rewritten,
    )
    rewritten = PREFLIGHT_OUTPUT_RE.sub(
        lambda match: (
            "fromJSON(inputs.scheduler_context).needs['preflight'].outputs."
            f"{match.group(1)}"
        ),
        rewritten,
    )
    rewritten = GITHUB_PR_RE.sub(
        lambda match: f"fromJSON(inputs.scheduler_context).source_event.{match.group(1)}",
        rewritten,
    )
    rewritten = GITHUB_EVENT_NAME_RE.sub(
        "fromJSON(inputs.scheduler_context).source_event.event_name",
        rewritten,
    )
    rewritten = GITHUB_EVENT_INPUTS_RE.sub(
        lambda match: (
            f"fromJSON(inputs.scheduler_context).source_event.inputs.{match.group(1)}"
        ),
        rewritten,
    )

    if "needs." in rewritten and "fromJSON(inputs.scheduler_context).needs[" not in rewritten:
        raise UnsupportedFeatureError(f"unsupported needs reference in expression: {original!r}")

    if "github.event.pull_request" in rewritten:
        raise UnsupportedFeatureError(
            f"unsupported github.event.pull_request reference: {original!r}"
        )

    classification: ExpressionClass
    if rewritten == original:
        classification = "copied"
    else:
        classification = "rewritten"
    return RewriteResult(expression=rewritten, classification=classification)


def rewrite_if_field(value: Any, *, job_key_prefix: str = "") -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return rewrite_expression(value, job_key_prefix=job_key_prefix).expression
    raise UnsupportedFeatureError(f"unsupported if field type: {type(value)!r}")


def rewrite_job_expressions(
    job: dict[str, Any],
    *,
    job_key: str,
    need_key_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    root_prefix = job_key.split("::", 1)[0]

    def _rewrite(value: Any, *, is_if: bool = False) -> Any:
        if isinstance(value, dict):
            return {
                key: _rewrite(child, is_if=(key == "if"))
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [_rewrite(child) for child in value]
        if isinstance(value, str):
            if is_if or "${{" in value:
                return rewrite_expression(
                    value,
                    job_key_prefix=root_prefix,
                    need_key_map=need_key_map,
                ).expression
        return value

    return _rewrite(job)
