# SPDX-License-Identifier: Apache-2.0
"""Run a local advisory documentation review.

Run from the repository root:

    python -m scripts.review_docs review --repository . --base-commit BASE --head-commit HEAD

The command reads git history and writes a bundle plus a report.
It does not approve, merge, or close a pull request, and it does not run
scripts from the pull request.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from scripts.docs_review.evaluate import (
    EvaluationSample,
    EvaluationThresholds,
    evaluate_samples,
)
from scripts.docs_review.models import (
    BudgetConfig,
    CheckRecord,
    ModelSettings,
    ReviewRequest,
)
from scripts.docs_review.reporting import render_markdown
from scripts.docs_review.review import SystemClock, run_review


def main(argv: list[str] | None = None) -> int:
    try:
        return run_command(argv)
    except (RuntimeError, ValueError, OSError, ValidationError) as error:
        print(str(error), file=sys.stderr)
        return 1


def run_command(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(description="Advisory local documentation review.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    review_parser = subcommands.add_parser("review")
    add_review_arguments(review_parser)
    evaluate_parser = subcommands.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="JSON samples with a stable defect_id for each labeled or predicted defect.",
    )
    evaluate_parser.add_argument("--precision-minimum", required=True, type=float)
    evaluate_parser.add_argument("--recall-minimum", required=True, type=float)
    evaluate_parser.add_argument("--output-directory", required=True, type=Path)
    parsed = parser.parse_args(argv)
    if parsed.command == "review":
        return run_review_command(parsed)
    else:
        return run_evaluate_command(parsed)


def add_review_arguments(review_parser: argparse.ArgumentParser) -> None:
    review_parser.add_argument("--repository", required=True, type=Path)
    review_parser.add_argument("--base-commit", required=True)
    review_parser.add_argument("--head-commit", required=True)
    review_parser.add_argument("--pull-request-number", required=True, type=int)
    review_parser.add_argument("--pull-request-text-file", type=Path)
    review_parser.add_argument("--comments-file", type=Path)
    review_parser.add_argument("--evidence-file", type=Path)
    review_parser.add_argument("--checks-file", type=Path)
    review_parser.add_argument("--budget-file", type=Path)
    review_parser.add_argument("--model-file", type=Path)
    review_parser.add_argument("--ledger", required=True, type=Path)
    review_parser.add_argument("--output-directory", required=True, type=Path)
    review_parser.add_argument("--include-path", action="append", default=[])
    review_parser.add_argument("--follow-up", action="store_true")
    review_parser.add_argument("--retry-count", type=int, default=0)


def run_review_command(parsed: argparse.Namespace) -> int:
    request = ReviewRequest(
        repository=parsed.repository,
        base_commit_sha=parsed.base_commit,
        head_commit_sha=parsed.head_commit,
        pull_request_number=parsed.pull_request_number,
        pull_request_text=read_optional_text(parsed.pull_request_text_file),
        comment_text=read_optional_text(parsed.comments_file),
        evidence_text=read_optional_text(parsed.evidence_file),
        checks=read_checks(parsed.checks_file),
        model_settings=read_model(parsed.model_file),
        budget=read_budget(parsed.budget_file),
        ledger_path=parsed.ledger,
        follow_up_authorized=parsed.follow_up,
        retry_count=parsed.retry_count,
        include_paths=list(parsed.include_path),
    )
    result = run_review(request, None, SystemClock())
    output_directory: Path = parsed.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "input_bundle.json").write_text(
        result.payload.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "report.json").write_text(
        result.report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "report.md").write_text(
        render_markdown(result.report), encoding="utf-8"
    )
    print(output_directory / "report.md")
    return 0


def run_evaluate_command(parsed: argparse.Namespace) -> int:
    raw: object = json.loads(parsed.dataset.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("dataset must be a JSON list")
    else:
        samples = [EvaluationSample.model_validate(entry) for entry in raw]
    thresholds = EvaluationThresholds(
        precision_minimum=parsed.precision_minimum,
        recall_minimum=parsed.recall_minimum,
    )
    report = evaluate_samples(samples, thresholds)
    output_directory: Path = parsed.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "evaluation.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(output_directory / "evaluation.json")
    return 0


def read_optional_text(path: Path | None) -> str:
    if path is None:
        return ""
    else:
        return path.read_text(encoding="utf-8")


def read_checks(path: Path | None) -> list[CheckRecord]:
    if path is None:
        return []
    else:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("checks file must be a JSON list")
        else:
            return [CheckRecord.model_validate(entry) for entry in raw]


def read_budget(path: Path | None) -> BudgetConfig | None:
    if path is None:
        return None
    else:
        return BudgetConfig.model_validate_json(path.read_text(encoding="utf-8"))


def read_model(path: Path | None) -> ModelSettings | None:
    if path is None:
        return None
    else:
        return ModelSettings.model_validate_json(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
