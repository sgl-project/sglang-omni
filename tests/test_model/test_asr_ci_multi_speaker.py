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
from pathlib import Path

import pytest

from benchmarks.eval.eval_transcribe_diarize import MODEL_PATH, run_eval
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
from tests.utils import MetricCheckCollector

MOSS_TD_CI_MODEL_PATH = os.environ.get(
    "MOSS_TRANSCRIBE_DIARIZE_MODEL_PATH",
    MODEL_PATH,
)
MOSS_TD_CONCURRENCY = 16
MOSS_TD_WARMUP_REQUESTS = 0
MOSS_TD_CI_SAMPLES = 800
MOSS_TD_STARTUP_TIMEOUT = 600
MOSS_TD_MEM_FRACTION_STATIC = 0.80

# Worst-of-N reference values calibrated by tune.py.
MOSS_TD_CER_PERCENT_REF = 6.612385102255017
MOSS_TD_CER_NO_SPK_PERCENT_REF = 6.612385102255017
MOSS_TD_CP_CER_PERCENT_REF = 13.0
MOSS_TD_CER_NO_SPK_CP_VALID_PERCENT_REF = 6.51
MOSS_TD_DELTA_CER_PERCENT_REF = 6.49
MOSS_TD_CER_VALID_SAMPLES_MIN: int | None = 784
MOSS_TD_CP_CER_VALID_SAMPLES_MIN: int | None = 773
MOSS_TD_THROUGHPUT_QPS_REF = 33.761
MOSS_TD_LATENCY_MEAN_S_REF = 0.467
MOSS_TD_LATENCY_P95_S_REF = 1.083
MOSS_TD_RTF_MEAN_REF = 0.0456
MOSS_TD_RTF_P95_REF = 0.0782

THRESHOLD_SLACK_HIGHER = 0.9
THRESHOLD_SLACK_LOWER = 1.1

MOSS_TD_CER_PERCENT_MAX: float | None = round(
    MOSS_TD_CER_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4
)
MOSS_TD_CER_NO_SPK_PERCENT_MAX: float | None = round(
    MOSS_TD_CER_NO_SPK_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4
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


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MOSS-Transcribe-Diarize CI")


@pytest.fixture(scope="module")
def movies800_samples():
    return load_movies800_samples(
        repo_id=MOVIES800_REPO_ID,
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        max_samples=MOSS_TD_CI_SAMPLES,
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
def test_moss_transcribe_diarize_movies800_multi_speaker(
    movies800_samples,
    moss_td_router_server: ManagedRouterHandle,
    tmp_path: Path,
) -> None:
    _require_cuda()
    checks = MetricCheckCollector("MOSS-Transcribe-Diarize multi-speaker ASR")
    checks.check(
        len(movies800_samples) == MOSS_TD_CI_SAMPLES,
        f"Expected {MOSS_TD_CI_SAMPLES} movies800 samples, "
        f"got {len(movies800_samples)}",
    )
    if not movies800_samples:
        checks.assert_all()

    with router_worker_traffic_guard(
        moss_td_router_server,
        label="MOSS-Transcribe-Diarize movies800",
    ) as router_guard:
        outputs, wall_clock_s = asyncio.run(
            run_eval(
                movies800_samples,
                base_url=f"http://127.0.0.1:{moss_td_router_server.port}",
                model_path=MOSS_TD_CI_MODEL_PATH,
                language=None,
                concurrency=MOSS_TD_CONCURRENCY,
                warmup=MOSS_TD_WARMUP_REQUESTS,
                request_rate=float("inf"),
                disable_tqdm=True,
                request_timeout_s=300,
            )
        )

    results = build_evaluation_payload(
        samples=movies800_samples,
        outputs=outputs,
        wall_clock_s=wall_clock_s,
        model_path=MOSS_TD_CI_MODEL_PATH,
        concurrency=MOSS_TD_CONCURRENCY,
        repo_id=MOVIES800_REPO_ID,
        split="validation",
    )
    summary = results["summary"]
    speed = results["speed"]
    diarization_metrics = results["diarization_metrics"]
    diarization_percent = results["diarization_metrics_percent"]

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

    results_path = tmp_path / "moss_transcribe_diarize_results.json"
    artifact_payload = dict(results)
    artifact_payload["router_ready_s"] = moss_td_router_server.router_ready_s
    results_path.write_text(json.dumps(artifact_payload, indent=2, ensure_ascii=False))

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
    checks.assert_all()


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
