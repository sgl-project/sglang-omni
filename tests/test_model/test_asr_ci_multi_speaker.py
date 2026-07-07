# SPDX-License-Identifier: Apache-2.0
"""Multi-speaker ASR/diarization CI for MOSS-Transcribe-Diarize.

The test reuses the movies800 benchmark path and runs two single-GPU workers
behind the managed router, matching the DP=2 shape used by other ASR/TTS CI
stages.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from benchmarks.eval.eval_transcribe_diarize import (
    AISHELL4_REPO_ID,
    MODEL_PATH,
    run_eval,
)
from benchmarks.metrics.transcribe_diarize_metrics import (
    print_diarization_accuracy_summary,
    print_diarization_speed_summary,
)
from benchmarks.tasks.transcribe_diarize import (
    MOVIES800_REPO_ID,
    build_evaluation_payload,
    load_movies800_samples,
)
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
    router_worker_traffic_guard,
)
from tests.utils import MetricCheckCollector, assert_cer_partitioned

MOSS_TD_CI_MODEL_PATH = os.environ.get(
    "MOSS_TRANSCRIBE_DIARIZE_MODEL_PATH",
    MODEL_PATH,
)
MOSS_TD_CONCURRENCY = 16
MOSS_TD_WARMUP_REQUESTS = 0
MOSS_TD_CI_SAMPLES = 800
MOSS_TD_AISHELL4_LONG_CI_SAMPLES = 20
MOSS_TD_STARTUP_TIMEOUT = 600
MOSS_TD_MEM_FRACTION_STATIC = 0.80
MOSS_TD_LONG_MAX_NEW_TOKENS = 65536

# Worst-of-N reference values calibrated by tune.py.
MOSS_TD_CER_PERCENT_REF = 21.66645481123681
MOSS_TD_CER_NO_SPK_PERCENT_REF = 21.66645481123681
MOSS_TD_CER_NO_SPK_BELOW_50_PERCENT_REF: float | None = 4.7008752910945155
MOSS_TD_N_ABOVE_50_CER_MAX: int | None = 30
MOSS_TD_CP_CER_PERCENT_REF = 29.170903775263763
MOSS_TD_CER_NO_SPK_CP_VALID_PERCENT_REF = 21.66645481123681
MOSS_TD_DELTA_CER_PERCENT_REF = 7.50444896402695
# Speaker-timestamp DER (diarization error rate, already a percentage in the
# result JSON). None until the first DER calibration fills in the reference.
MOSS_TD_SPEAKER_TIMESTAMP_DER_PERCENT_REF: float | None = 21.704891749630097
MOSS_TD_CER_VALID_SAMPLES_MIN: int | None = 784
MOSS_TD_CP_CER_VALID_SAMPLES_MIN: int | None = 784
MOSS_TD_THROUGHPUT_QPS_REF = 29.991
MOSS_TD_LATENCY_MEAN_S_REF = 0.465
MOSS_TD_LATENCY_P95_S_REF = 0.857
MOSS_TD_RTF_MEAN_REF = 0.0465
MOSS_TD_RTF_P95_REF = 0.0612

THRESHOLD_SLACK_HIGHER = 0.9
THRESHOLD_SLACK_LOWER = 1.1

MOSS_TD_CER_PERCENT_MAX: float | None = round(
    MOSS_TD_CER_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4
)
MOSS_TD_CER_NO_SPK_PERCENT_MAX: float | None = round(
    MOSS_TD_CER_NO_SPK_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4
)
MOSS_TD_CER_NO_SPK_BELOW_50_PERCENT_MAX: float | None = (
    round(MOSS_TD_CER_NO_SPK_BELOW_50_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_CER_NO_SPK_BELOW_50_PERCENT_REF is not None
    else None
)
MOSS_TD_CP_CER_PERCENT_MAX: float | None = round(
    MOSS_TD_CP_CER_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4
)
MOSS_TD_CER_NO_SPK_CP_VALID_PERCENT_MAX: float | None = round(
    MOSS_TD_CER_NO_SPK_CP_VALID_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4
)
MOSS_TD_DELTA_CER_PERCENT_MAX: float | None = round(
    MOSS_TD_DELTA_CER_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4
)
MOSS_TD_SPEAKER_TIMESTAMP_DER_PERCENT_MAX: float | None = (
    round(MOSS_TD_SPEAKER_TIMESTAMP_DER_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_SPEAKER_TIMESTAMP_DER_PERCENT_REF is not None
    else None
)
MOSS_TD_THROUGHPUT_QPS_MIN: float | None = round(
    MOSS_TD_THROUGHPUT_QPS_REF * THRESHOLD_SLACK_HIGHER, 3
)
MOSS_TD_LATENCY_MEAN_S_MAX: float | None = round(
    MOSS_TD_LATENCY_MEAN_S_REF * THRESHOLD_SLACK_LOWER, 3
)
MOSS_TD_LATENCY_P95_S_MAX: float | None = round(
    MOSS_TD_LATENCY_P95_S_REF * THRESHOLD_SLACK_LOWER, 3
)
MOSS_TD_RTF_MEAN_MAX: float | None = round(
    MOSS_TD_RTF_MEAN_REF * THRESHOLD_SLACK_LOWER, 4
)
MOSS_TD_RTF_P95_MAX: float | None = round(
    MOSS_TD_RTF_P95_REF * THRESHOLD_SLACK_LOWER, 4
)
AISHELL4_LONG_CER_PERCENT_MAX: float | None = 13.435930414342895
AISHELL4_LONG_CER_NO_SPK_PERCENT_MAX: float | None = 13.435930414342895
AISHELL4_LONG_CP_CER_PERCENT_MAX: float | None = 13.9315452761362
AISHELL4_LONG_DELTA_CER_PERCENT_MAX: float | None = 0.5155725072346462
AISHELL4_LONG_SPEAKER_TIMESTAMP_DER_PERCENT_MAX: float | None = 12.388339964922126
AISHELL4_LONG_THROUGHPUT_QPS_MIN: float | None = 0.059
AISHELL4_LONG_LATENCY_MEAN_S_MAX: float | None = 199.827
AISHELL4_LONG_LATENCY_P95_S_MAX: float | None = 259.394
AISHELL4_LONG_RTF_MEAN_MAX: float | None = 0.0874
AISHELL4_LONG_RTF_P95_MAX: float | None = 0.1164


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MOSS-Transcribe-Diarize CI")


@pytest.fixture(scope="module")
def movies800times_samples():
    return load_movies800_samples(
        repo_id=MOVIES800_REPO_ID,
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        max_samples=MOSS_TD_CI_SAMPLES,
    )


@pytest.fixture(scope="module")
def aishell4_long_samples():
    return load_movies800_samples(
        repo_id=AISHELL4_REPO_ID,
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        max_samples=None,
        expected_sample_count=MOSS_TD_AISHELL4_LONG_CI_SAMPLES,
    )


@pytest.fixture(scope="module")
def moss_td_router_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> ManagedRouterHandle:
    worker_extra_args = " ".join(
        [
            "--max-running-requests",
            str(MOSS_TD_CONCURRENCY),
            "--cuda-graph-max-bs",
            str(MOSS_TD_CONCURRENCY),
            "--mem-fraction-static",
            str(MOSS_TD_MEM_FRACTION_STATIC),
        ]
    )
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=MOSS_TD_CI_MODEL_PATH,
        model_name=MOSS_TD_CI_MODEL_PATH,
        worker_extra_args=worker_extra_args,
        wait_timeout=MOSS_TD_STARTUP_TIMEOUT,
        log_prefix="moss_td_router_logs",
    ) as router:
        yield router


@pytest.mark.benchmark
def test_moss_transcribe_diarize_multi_speaker_datasets(
    movies800times_samples,
    aishell4_long_samples,
    moss_td_router_server: ManagedRouterHandle,
    tmp_path: Path,
) -> None:
    _require_cuda()
    checks = MetricCheckCollector("MOSS-Transcribe-Diarize multi-speaker ASR")
    checks.check(
        len(movies800times_samples) == MOSS_TD_CI_SAMPLES,
        f"Expected {MOSS_TD_CI_SAMPLES} movies800times samples, "
        f"got {len(movies800times_samples)}",
    )
    checks.check(
        len(aishell4_long_samples) == MOSS_TD_AISHELL4_LONG_CI_SAMPLES,
        f"Expected {MOSS_TD_AISHELL4_LONG_CI_SAMPLES} aishell4_long samples, "
        f"got {len(aishell4_long_samples)}",
    )
    if not movies800times_samples or not aishell4_long_samples:
        checks.assert_all()

    with router_worker_traffic_guard(
        moss_td_router_server,
        label="MOSS-Transcribe-Diarize movies800times",
    ) as movies800times_router_guard:
        movies800times_outputs, movies800times_wall_clock_s = _run_transcribe_diarize(
            movies800times_samples,
            moss_td_router_server=moss_td_router_server,
            request_timeout_s=300,
            max_new_tokens=None,
        )
    with ThreadPoolExecutor(max_workers=1) as executor:
        aishell4_future = executor.submit(
            _run_transcribe_diarize,
            aishell4_long_samples,
            moss_td_router_server=moss_td_router_server,
            request_timeout_s=1800,
            max_new_tokens=MOSS_TD_LONG_MAX_NEW_TOKENS,
        )
        movies800times_results = _build_results(
            samples=movies800times_samples,
            outputs=movies800times_outputs,
            wall_clock_s=movies800times_wall_clock_s,
            repo_id=MOVIES800_REPO_ID,
        )
        _print_and_save_results(
            results=movies800times_results,
            tmp_path=tmp_path,
            filename="moss_transcribe_diarize_results.json",
            router_ready_s=moss_td_router_server.router_ready_s,
        )
        _assert_movies800times_results(
            checks,
            movies800times_results,
            movies800times_router_guard,
        )
        aishell4_outputs, aishell4_wall_clock_s = aishell4_future.result()
    aishell4_results = _build_results(
        samples=aishell4_long_samples,
        outputs=aishell4_outputs,
        wall_clock_s=aishell4_wall_clock_s,
        repo_id=AISHELL4_REPO_ID,
    )
    _print_and_save_results(
        results=aishell4_results,
        tmp_path=tmp_path,
        filename="moss_transcribe_diarize_aishell4_long_results.json",
        router_ready_s=moss_td_router_server.router_ready_s,
    )
    _assert_aishell4_long_results(checks, aishell4_results)
    checks.assert_all()


def _run_transcribe_diarize(
    samples,
    *,
    moss_td_router_server: ManagedRouterHandle,
    request_timeout_s: int,
    max_new_tokens: int | None,
):
    return asyncio.run(
        run_eval(
            samples,
            base_url=f"http://127.0.0.1:{moss_td_router_server.port}",
            model_path=MOSS_TD_CI_MODEL_PATH,
            language=None,
            concurrency=MOSS_TD_CONCURRENCY,
            warmup=MOSS_TD_WARMUP_REQUESTS,
            request_rate=float("inf"),
            disable_tqdm=False,
            request_timeout_s=request_timeout_s,
            max_new_tokens=max_new_tokens,
        )
    )


def _build_results(
    *,
    samples,
    outputs,
    wall_clock_s: float,
    repo_id: str,
):
    return build_evaluation_payload(
        samples=samples,
        outputs=outputs,
        wall_clock_s=wall_clock_s,
        model_path=MOSS_TD_CI_MODEL_PATH,
        concurrency=MOSS_TD_CONCURRENCY,
        repo_id=repo_id,
        split="validation",
    )


def _print_and_save_results(
    *,
    results,
    tmp_path: Path,
    filename: str,
    router_ready_s: float,
) -> None:
    summary = results["summary"]
    speed = results["speed"]
    diarization_metrics = results["diarization_metrics"]
    print_diarization_accuracy_summary(
        summary=summary,
        diarization_metrics=diarization_metrics,
        model_name=MOSS_TD_CI_MODEL_PATH,
        concurrency=MOSS_TD_CONCURRENCY,
    )
    print_diarization_speed_summary(
        speed=speed,
        model_name=MOSS_TD_CI_MODEL_PATH,
        concurrency=MOSS_TD_CONCURRENCY,
    )

    results_path = tmp_path / filename
    artifact_payload = dict(results)
    artifact_payload["router_ready_s"] = router_ready_s
    results_path.write_text(json.dumps(artifact_payload, indent=2, ensure_ascii=False))


def _assert_movies800times_results(
    checks: MetricCheckCollector,
    results,
    router_guard,
) -> None:
    summary = results["summary"]
    speed = results["speed"]
    diarization_percent = results["diarization_metrics_percent"]
    total = summary["total_samples"]
    evaluated = summary["evaluated"]
    failed_requests = speed.get("failed_requests")
    checks.check(
        total == MOSS_TD_CI_SAMPLES,
        f"Expected {MOSS_TD_CI_SAMPLES}, got {total}",
    )
    checks.check(
        evaluated == total,
        f"Expected all samples evaluated, got {evaluated}/{total}",
    )
    checks.check(
        failed_requests == 0,
        f"Expected 0 failed requests, got {failed_requests}",
    )
    checks.check(
        diarization_percent.get("count") == total,
        f"Expected diarization count {total}, got {diarization_percent.get('count')}",
    )
    _check_optional_max(
        checks,
        "cer",
        diarization_percent.get("cer"),
        MOSS_TD_CER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "cer_no_spk",
        diarization_percent.get("cer_no_spk"),
        MOSS_TD_CER_NO_SPK_PERCENT_MAX,
        unit="%",
    )
    assert_cer_partitioned(
        diarization_percent,
        max_cer_no_spk_below_50_percent=MOSS_TD_CER_NO_SPK_BELOW_50_PERCENT_MAX,
        max_n_above_50_cer=MOSS_TD_N_ABOVE_50_CER_MAX,
        collector=checks,
    )
    _check_optional_max(
        checks,
        "cp_cer",
        diarization_percent.get("cp_cer"),
        MOSS_TD_CP_CER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "cer_no_spk_cp_valid",
        diarization_percent.get("cer_no_spk_cp_valid"),
        MOSS_TD_CER_NO_SPK_CP_VALID_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "delta_cer",
        diarization_percent.get("delta_cer"),
        MOSS_TD_DELTA_CER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "speaker_timestamp_der",
        diarization_percent.get("speaker_timestamp_der"),
        MOSS_TD_SPEAKER_TIMESTAMP_DER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_min(
        checks,
        "cer_valid_samples",
        diarization_percent.get("cer_valid_samples"),
        MOSS_TD_CER_VALID_SAMPLES_MIN,
    )
    _check_optional_min(
        checks,
        "cp_cer_valid_samples",
        diarization_percent.get("cp_cer_valid_samples"),
        MOSS_TD_CP_CER_VALID_SAMPLES_MIN,
    )
    _check_optional_min(
        checks,
        "throughput_qps",
        speed.get("throughput_qps"),
        MOSS_TD_THROUGHPUT_QPS_MIN,
    )
    _check_optional_max(
        checks,
        "latency_mean_s",
        speed.get("latency_mean_s"),
        MOSS_TD_LATENCY_MEAN_S_MAX,
        unit="s",
    )
    _check_optional_max(
        checks,
        "latency_p95_s",
        speed.get("latency_p95_s"),
        MOSS_TD_LATENCY_P95_S_MAX,
        unit="s",
    )
    _check_optional_max(
        checks,
        "rtf_mean",
        speed.get("rtf_mean"),
        MOSS_TD_RTF_MEAN_MAX,
    )
    _check_optional_max(
        checks,
        "rtf_p95",
        speed.get("rtf_p95"),
        MOSS_TD_RTF_P95_MAX,
    )
    checks.check_assertion(
        "router traffic",
        router_guard.assert_served,
        min_total_requests=total,
        min_worker_share=0.40,
    )


def _assert_aishell4_long_results(checks: MetricCheckCollector, results) -> None:
    summary = results["summary"]
    speed = results["speed"]
    diarization_percent = results["diarization_metrics_percent"]
    total = summary["total_samples"]
    evaluated = summary["evaluated"]
    failed_requests = speed.get("failed_requests")
    checks.check(
        total == MOSS_TD_AISHELL4_LONG_CI_SAMPLES,
        f"Expected {MOSS_TD_AISHELL4_LONG_CI_SAMPLES} aishell4_long samples, got {total}",
    )
    checks.check(
        evaluated == total,
        f"Expected all aishell4_long samples evaluated, got {evaluated}/{total}",
    )
    checks.check(
        failed_requests == 0,
        f"Expected 0 aishell4_long failed requests, got {failed_requests}",
    )
    _check_optional_max(
        checks,
        "aishell4_long cer",
        diarization_percent.get("cer"),
        AISHELL4_LONG_CER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "aishell4_long cer_no_spk",
        diarization_percent.get("cer_no_spk"),
        AISHELL4_LONG_CER_NO_SPK_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "aishell4_long cp_cer",
        diarization_percent.get("cp_cer"),
        AISHELL4_LONG_CP_CER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "aishell4_long delta_cer",
        diarization_percent.get("delta_cer"),
        AISHELL4_LONG_DELTA_CER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "aishell4_long speaker_timestamp_der",
        diarization_percent.get("speaker_timestamp_der"),
        AISHELL4_LONG_SPEAKER_TIMESTAMP_DER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_min(
        checks,
        "aishell4_long throughput_qps",
        speed.get("throughput_qps"),
        AISHELL4_LONG_THROUGHPUT_QPS_MIN,
    )
    _check_optional_max(
        checks,
        "aishell4_long latency_mean_s",
        speed.get("latency_mean_s"),
        AISHELL4_LONG_LATENCY_MEAN_S_MAX,
        unit="s",
    )
    _check_optional_max(
        checks,
        "aishell4_long latency_p95_s",
        speed.get("latency_p95_s"),
        AISHELL4_LONG_LATENCY_P95_S_MAX,
        unit="s",
    )
    _check_optional_max(
        checks,
        "aishell4_long rtf_mean",
        speed.get("rtf_mean"),
        AISHELL4_LONG_RTF_MEAN_MAX,
    )
    _check_optional_max(
        checks,
        "aishell4_long rtf_p95",
        speed.get("rtf_p95"),
        AISHELL4_LONG_RTF_P95_MAX,
    )


def _check_optional_max(
    checks: MetricCheckCollector,
    metric_name: str,
    value: object,
    threshold: float | None,
    *,
    unit: str = "",
) -> None:
    if threshold is None:
        print(f"[threshold pending] {metric_name}={value}{unit}")
        return
    checks.check(
        isinstance(value, int | float) and value <= threshold,
        f"{metric_name} {value}{unit} exceeds {threshold}{unit}",
    )


def _check_optional_min(
    checks: MetricCheckCollector,
    metric_name: str,
    value: object,
    threshold: float | None,
    *,
    unit: str = "",
) -> None:
    if threshold is None:
        print(f"[threshold pending] {metric_name}={value}{unit}")
        return
    checks.check(
        isinstance(value, int | float) and value >= threshold,
        f"{metric_name} {value}{unit} is below {threshold}{unit}",
    )
