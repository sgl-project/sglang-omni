# SPDX-License-Identifier: Apache-2.0
"""Step 3: LLM behavior judge (plus the semantic A/F/U judge for qwen), the v1.0
reference evaluation, then the per-repeat report. With JUDGE=qwen the judge
server runs only during this step."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from benchmarks.duplex.fdb_v15.common import (
    CUSTOM_JUDGE_API_KEY_ENV,
    ENGINE_LABEL,
    GPT_JUDGE_MODEL,
    JUDGE_MODEL_ID,
    JUDGE_SERVED_MODEL,
    REPO_ROOT,
    Settings,
    log,
    read_json,
    reference_command,
    require_v10_export,
    run_command,
    step_command,
    v10_command,
)
from benchmarks.duplex.fdb_v15.servers import judge_server
from benchmarks.duplex.v10_dataset import SUBSETS as V10_SUBSETS

UNAUTHENTICATED_API_KEY = "EMPTY"


def pending_v10_subsets(tree: Path) -> list[str]:
    """Exported v1.0 subsets that summary.json lacks; the evaluator refuses repeats."""
    counts = read_json(tree / "manifest.json")["counts"]
    summary_path = tree / "summary.json"
    evaluated = read_json(summary_path)["subsets"] if summary_path.is_file() else {}
    return [
        subset
        for subset in V10_SUBSETS
        if counts[subset]["eligible"] and subset not in evaluated
    ]


def evaluate_v10(
    settings: Settings,
    tree: Path,
    subsets: list[str],
    judge_arguments: list[str],
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """Run the pinned evaluator on each pending subset in its own call, so one
    failing subset does not hold back the others; returns the subsets that failed."""
    failed = []
    for subset in subsets:
        log(f"== v1.0 reference evaluation of {subset} -> {tree}")
        is_ok = run_command(
            v10_command(
                settings,
                "reference-evaluate",
                "--tree",
                str(tree),
                "--reference-source",
                str(settings.fdb_source),
                "--subset",
                subset,
                *judge_arguments,
            ),
            extra_env=extra_env,
        )
        if not is_ok:
            failed.append(subset)
        else:
            pass
    return failed


def judge_with_qwen(
    settings: Settings,
    repeat_dir: Path,
    tree_arguments: list[str],
    retry_arguments: list[str],
    v10_tree: Path | None,
    v10_subsets: list[str],
) -> tuple[bool, list[str]]:
    scores = repeat_dir / "scores"
    qwen_scores = repeat_dir / "judge-qwen"
    semantic_scores = repeat_dir / "semantic-qwen"
    judge_config = str(settings.judge_dir / "judge-config.json")
    api_key_env = {
        CUSTOM_JUDGE_API_KEY_ENV: os.environ.get(CUSTOM_JUDGE_API_KEY_ENV)
        or UNAUTHENTICATED_API_KEY
    }
    source_arguments = [
        *tree_arguments,
        "--source-scores",
        str(scores),
        "--out",
        str(qwen_scores),
    ]
    with judge_server(settings, repeat_dir / "logs" / "judge-server.log"):
        log(f"== Qwen judge -> {qwen_scores}")
        is_judge_ok = run_command(
            reference_command(
                settings,
                "custom-judge",
                *source_arguments,
                "--judge-config",
                judge_config,
                "--base-url",
                settings.judge_url,
                *retry_arguments,
            ),
            extra_env=api_key_env,
        )
        is_summary_ok = run_command(
            reference_command(
                settings,
                "custom-summarize",
                *source_arguments,
                "--judge-config",
                judge_config,
            )
        )
        log(f"== Semantic A/F/U judge -> {semantic_scores}")
        is_semantic_ok = run_command(
            [
                str(settings.scoring_python),
                "-m",
                "benchmarks.duplex.semantic_judge",
                "--reference-audio",
                str(repeat_dir / "reference-audio"),
                "--scores",
                str(scores),
                "--engine",
                ENGINE_LABEL,
                "--dataset-root",
                str(settings.dataset),
                "--out",
                str(semantic_scores),
                "--base-url",
                settings.judge_url,
                "--model-id",
                JUDGE_MODEL_ID,
                "--model-revision",
                settings.judge_revision_file.read_text().strip(),
                "--served-model",
                JUDGE_SERVED_MODEL,
            ],
            extra_env=api_key_env,
        )
        failed_v10 = evaluate_v10(
            settings,
            v10_tree,
            v10_subsets,
            [
                "--api-key-env",
                CUSTOM_JUDGE_API_KEY_ENV,
                "--base-url",
                settings.judge_url,
                "--served-model",
                JUDGE_SERVED_MODEL,
            ],
            extra_env=api_key_env,
        )
    return is_judge_ok and is_summary_ok and is_semantic_ok, failed_v10


def judge_with_gpt(
    settings: Settings,
    repeat_dir: Path,
    tree_arguments: list[str],
    retry_arguments: list[str],
    v10_tree: Path | None,
    v10_subsets: list[str],
) -> tuple[bool, list[str]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: export OPENAI_API_KEY before running the GPT judge.")
    else:
        pass
    base_url = os.environ.get("OPENAI_BASE_URL")
    base_url_arguments = ["--base-url", base_url] if base_url else []
    scores_arguments = [*tree_arguments, "--out", str(repeat_dir / "scores")]
    log(f"== GPT judge ({GPT_JUDGE_MODEL}) -> {repeat_dir / 'scores'}")
    is_prepare_ok = run_command(
        reference_command(settings, "prepare-judge", *scores_arguments)
    )
    is_judge_ok = run_command(
        reference_command(
            settings,
            "judge",
            *scores_arguments,
            "--judge",
            GPT_JUDGE_MODEL,
            "--api-key-env",
            "OPENAI_API_KEY",
            *base_url_arguments,
            *retry_arguments,
        )
    )
    failed_v10 = evaluate_v10(
        settings, v10_tree, v10_subsets, ["--api-key-env", "OPENAI_API_KEY"]
    )
    return is_prepare_ok and is_judge_ok, failed_v10


def write_report(
    settings: Settings, repeat_dir: Path, tree_arguments: list[str]
) -> bool:
    scores = repeat_dir / "scores"
    log("== Summarize timing, ASR coverage and behavior")
    is_summary_ok = run_command(
        reference_command(settings, "summarize", *tree_arguments, "--out", str(scores))
    )
    semantic_summary = repeat_dir / "semantic-qwen" / "summary.json"
    semantic_arguments = (
        ["--semantic-summary", str(semantic_summary)]
        if settings.judge == "qwen" and semantic_summary.is_file()
        else []
    )
    report_path = repeat_dir / "report.txt"
    with report_path.open("w") as report_file:
        report = subprocess.run(
            reference_command(
                settings,
                "report",
                "--scores",
                str(scores),
                "--engine",
                ENGINE_LABEL,
                *semantic_arguments,
            ),
            cwd=REPO_ROOT,
            stdout=report_file,
            check=False,
        )
    log(f"Wrote {report_path}")
    return is_summary_ok and report.returncode == 0


def judge(settings: Settings, repeat: int, retry_failed: bool) -> None:
    repeat_dir = settings.repeat_dir(repeat)
    if not (
        repeat_dir / "scores" / "engines" / ENGINE_LABEL / "manifest-receipt.json"
    ).is_file():
        raise SystemExit(f"ERROR: run `{step_command('asr', settings, repeat)}` first.")
    else:
        pass
    v10_tree = require_v10_export(settings, repeat)
    if v10_tree is None:
        v10_subsets = []
    elif not (v10_tree / "asr.json").is_file():
        raise SystemExit(
            f"ERROR: run `{step_command('asr', settings, repeat)}` first; "
            f"the v1.0 ASR receipt {v10_tree / 'asr.json'} is missing."
        )
    else:
        v10_subsets = pending_v10_subsets(v10_tree)
    tree_arguments = [
        "--reference-source",
        str(settings.fdb_source),
        "--tree",
        f"{ENGINE_LABEL}={repeat_dir / 'reference-audio'}",
    ]
    retry_arguments = ["--retry-failed"] if retry_failed else []
    if settings.judge == "qwen":
        is_judge_ok, failed_v10 = judge_with_qwen(
            settings, repeat_dir, tree_arguments, retry_arguments, v10_tree, v10_subsets
        )
    else:
        is_judge_ok, failed_v10 = judge_with_gpt(
            settings, repeat_dir, tree_arguments, retry_arguments, v10_tree, v10_subsets
        )
    is_report_ok = write_report(settings, repeat_dir, tree_arguments)
    problems = []
    if not (is_judge_ok and is_report_ok):
        problems.append(
            "WARNING: a phase reported failures. "
            f"Rerun with: {step_command('judge', settings, repeat)} --retry-failed"
        )
    else:
        pass
    if failed_v10:
        problems.append(
            f"WARNING: the v1.0 evaluation failed for {', '.join(failed_v10)}; logs "
            f"are in {v10_tree / 'logs'}. Rerun `{step_command('judge', settings, repeat)}` "
            "to evaluate them again. The pinned turn-taking and interruption "
            "evaluators divide by the number of takeovers, so a selection in which "
            "no sample takes the turn cannot be scored."
        )
    else:
        pass
    if problems:
        raise SystemExit("\n".join(problems))
    else:
        pass
    log(f"Step 3 done. Aggregate repeats with: {step_command('aggregate', settings)}")
