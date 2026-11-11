# SPDX-License-Identifier: Apache-2.0
"""Step 3: LLM behavior judge (plus the semantic A/F/U judge for qwen), the v1.0
reference evaluation, then the per-repeat report. With JUDGE=qwen the judge
server runs only during this step."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from benchmarks.duplex.fdb.common import (
    CUSTOM_JUDGE_API_KEY_ENV,
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
    selected_v15,
    step_command,
    v10_command,
)
from benchmarks.duplex.fdb.servers import RunningServer, judge_server
from benchmarks.duplex.v10_dataset import SUBSET_TASKS as V10_SUBSET_TASKS
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


def run_judge_phase(
    server: RunningServer,
    what: str,
    command: list[str],
    extra_env: dict[str, str] | None,
) -> bool:
    """Run a phase that calls the judge server; True when it exits 0. A server
    that exited before or during the phase stops the step whatever the phase's
    exit code, since the semantic judge records a refused request as unresolved
    and exits 0."""
    server.ensure_alive(f"before {what}")
    is_ok = run_command(command, extra_env=extra_env)
    server.ensure_alive(f"during {what}")
    return is_ok


def evaluate_v10(
    settings: Settings,
    tree: Path,
    subsets: list[str],
    judge_arguments: list[str],
    server: RunningServer | None,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """Run the pinned evaluator on each pending subset in its own call, so one
    failing subset does not hold back the others; returns the subsets that failed.
    server is the local judge server, None for a hosted judge."""
    failed = []
    for subset in subsets:
        log(f"== v1.0 reference evaluation of {subset} -> {tree}")
        command = v10_command(
            settings,
            "reference-evaluate",
            "--tree",
            str(tree),
            "--reference-source",
            str(settings.fdb_source),
            "--subset",
            subset,
            *judge_arguments,
        )
        is_ok = (
            run_command(command, extra_env=extra_env)
            if server is None
            else run_judge_phase(
                server, f"the v1.0 evaluation of {subset}", command, extra_env
            )
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
    has_v15: bool,
    v10_tree: Path | None,
    v10_subsets: list[str],
    should_run_semantic_judge: bool,
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
    is_v15_ok = True
    with judge_server(settings, repeat_dir / "logs" / "judge-server.log") as server:
        if has_v15:
            log(f"== v1.5 Qwen judge -> {qwen_scores}")
            is_judge_ok = run_judge_phase(
                server,
                "the v1.5 Qwen judge",
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
                api_key_env,
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
            if should_run_semantic_judge:
                log(f"== v1.5 semantic A/F/U judge -> {semantic_scores}")
                is_semantic_ok = run_judge_phase(
                    server,
                    "the v1.5 semantic A/F/U judge",
                    [
                        str(settings.scoring_python),
                        "-m",
                        "benchmarks.duplex.semantic_judge",
                        "--reference-audio",
                        str(repeat_dir / "reference-audio"),
                        "--scores",
                        str(scores),
                        "--engine",
                        settings.engine.name,
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
                    api_key_env,
                )
            else:
                log("== v1.5 semantic A/F/U judge skipped")
                is_semantic_ok = True
            is_v15_ok = is_judge_ok and is_summary_ok and is_semantic_ok
        else:
            pass
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
            server,
            extra_env=api_key_env,
        )
    return is_v15_ok, failed_v10


def judge_with_gpt(
    settings: Settings,
    repeat_dir: Path,
    tree_arguments: list[str],
    retry_arguments: list[str],
    has_v15: bool,
    v10_tree: Path | None,
    v10_subsets: list[str],
) -> tuple[bool, list[str]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: export OPENAI_API_KEY before running the GPT judge.")
    else:
        pass
    is_v15_ok = True
    if has_v15:
        base_url = os.environ.get("OPENAI_BASE_URL")
        base_url_arguments = ["--base-url", base_url] if base_url else []
        scores_arguments = [*tree_arguments, "--out", str(repeat_dir / "scores")]
        log(f"== v1.5 GPT judge ({GPT_JUDGE_MODEL}) -> {repeat_dir / 'scores'}")
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
        is_v15_ok = is_prepare_ok and is_judge_ok
    else:
        pass
    failed_v10 = evaluate_v10(
        settings,
        v10_tree,
        v10_subsets,
        ["--api-key-env", "OPENAI_API_KEY"],
        server=None,
    )
    return is_v15_ok, failed_v10


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
                settings.engine.name,
                *semantic_arguments,
            ),
            cwd=REPO_ROOT,
            stdout=report_file,
            check=False,
        )
    log(f"Wrote {report_path}")
    return is_summary_ok and report.returncode == 0


def judge(
    settings: Settings,
    repeat: int,
    retry_failed: bool,
    should_run_semantic_judge: bool,
) -> None:
    repeat_dir = settings.repeat_dir(repeat)
    has_v15 = selected_v15(settings, repeat)
    if (
        has_v15
        and not (
            repeat_dir
            / "scores"
            / "engines"
            / settings.engine.name
            / "manifest-receipt.json"
        ).is_file()
    ):
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
        f"{settings.engine.name}={repeat_dir / 'reference-audio'}",
    ]
    retry_arguments = ["--retry-failed"] if retry_failed else []
    # note (luojiaxuan): only the v1.0 interruption subset asks an LLM; the other
    # v1.0 subsets are scored without a judge, before any judge is started.
    judged_v10 = [
        subset
        for subset in v10_subsets
        if V10_SUBSET_TASKS[subset] == "user_interruption"
    ]
    failed_v10 = evaluate_v10(
        settings,
        v10_tree,
        [subset for subset in v10_subsets if subset not in judged_v10],
        [],
        server=None,
    )
    if failed_v10:
        # note (luojiaxuan): said here as well, since a judge phase that stops
        # the step would otherwise hide it.
        log(f"WARNING: the v1.0 evaluation failed for {', '.join(failed_v10)}")
    else:
        pass
    if (has_v15 or judged_v10) and settings.judge == "qwen":
        is_judge_ok, failed_judged = judge_with_qwen(
            settings,
            repeat_dir,
            tree_arguments,
            retry_arguments,
            has_v15,
            v10_tree,
            judged_v10,
            should_run_semantic_judge,
        )
        failed_v10 += failed_judged
    elif has_v15 or judged_v10:
        is_judge_ok, failed_judged = judge_with_gpt(
            settings,
            repeat_dir,
            tree_arguments,
            retry_arguments,
            has_v15,
            v10_tree,
            judged_v10,
        )
        failed_v10 += failed_judged
    else:
        is_judge_ok = True
    is_report_ok = (
        write_report(settings, repeat_dir, tree_arguments) if has_v15 else True
    )
    problems = []
    if not (is_judge_ok and is_report_ok):
        problems.append(
            "WARNING: a v1.5 phase reported failures. "
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
