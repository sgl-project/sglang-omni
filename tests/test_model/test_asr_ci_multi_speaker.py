# SPDX-License-Identifier: Apache-2.0
"""Multi-speaker ASR/diarization CI for MOSS-Transcribe-Diarize.

The test reuses the movies800 / aishell4_long / googletime benchmark path and
runs two single-GPU workers behind the managed router, matching the DP=2 shape
used by other ASR/TTS CI stages.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path

import pytest

from benchmarks.eval.benchmark_asr_transcribe_diarize import (
    AISHELL4_REPO_ID,
    GOOGLETIME_REPO_ID,
    MODEL_PATH,
    run_eval,
)
from benchmarks.metrics._format import format_benchmark_dataset_label
from benchmarks.metrics.transcribe_diarize_metrics import (
    print_diarization_accuracy_summary,
    print_diarization_speed_summary,
)
from benchmarks.tasks.transcribe_diarize import (
    MOVIES800_REPO_ID,
    build_evaluation_payload,
    build_long_audio_concat_sample,
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
MOSS_TD_GOOGLETIME_CI_SAMPLES = 25
MOSS_TD_STARTUP_TIMEOUT = 600
MOSS_TD_MEM_FRACTION_STATIC = 0.80
MOSS_TD_LONG_MAX_NEW_TOKENS = 65536
# note (db-ol): clip 2 leads and gets truncated to hit 90 minutes, clips 0
# and 1 stay whole so their transcripts sit at the deepest token positions.
MOSS_TD_LONG90_CLIP_INDICES = (2, 0, 1)
MOSS_TD_LONG90_TARGET_S = 5400.0
# note (db-ol): well above the observed 30k output tokens for this sample
# and below the context remaining after the 72k token input.
MOSS_TD_LONG90_MAX_NEW_TOKENS = 50000


MOSS_TD_CER_PERCENT_REF = 5.812253718062794
MOSS_TD_CER_NO_SPK_PERCENT_REF = 5.812253718062794
MOSS_TD_CER_NO_SPK_BELOW_50_PERCENT_REF: float | None = 4.97942651407998
MOSS_TD_N_ABOVE_50_CER_REF: int | None = 29
MOSS_TD_CP_CER_PERCENT_REF = 13.219778822931232
MOSS_TD_CER_NO_SPK_CP_VALID_PERCENT_REF = 5.812253718062794
MOSS_TD_DELTA_CER_PERCENT_REF = 7.421825346383629
MOSS_TD_SPEAKER_TIMESTAMP_DER_PERCENT_REF: float | None = 21.016722368599407
MOSS_TD_CER_VALID_SAMPLES_MIN: int | None = 784
MOSS_TD_CP_CER_VALID_SAMPLES_MIN: int | None = 784
MOSS_TD_THROUGHPUT_QPS_REF = 45.436
MOSS_TD_LATENCY_MEAN_S_REF = 0.295
MOSS_TD_LATENCY_P95_S_REF = 0.609
MOSS_TD_RTF_MEAN_REF = 0.0324
MOSS_TD_RTF_P95_REF = 0.0417

AISHELL4_LONG_CER_PERCENT_REF = 13.855595347651095
AISHELL4_LONG_CER_NO_SPK_PERCENT_REF = 13.855595347651095
AISHELL4_LONG_CP_CER_PERCENT_REF = 14.20651727999468
AISHELL4_LONG_DELTA_CER_PERCENT_REF = 0.36256389218436436
AISHELL4_LONG_SPEAKER_TIMESTAMP_DER_PERCENT_REF = 9.700722891345189
AISHELL4_LONG_THROUGHPUT_QPS_REF = 0.068
AISHELL4_LONG_LATENCY_MEAN_S_REF = 163.564
AISHELL4_LONG_LATENCY_P95_S_REF = 208.081
AISHELL4_LONG_RTF_MEAN_REF = 0.0715
AISHELL4_LONG_RTF_P95_REF = 0.0934
# note (db-ol): catastrophic bounds for the single 90 minute sample, not
# calibrated thresholds. The healthy greedy run measures cer_no_spk 11.63
# and missed detection ratio 0.039, the known format dropout failure
# measures 80.3 and loses DER validity entirely.
AISHELL4_LONG90_CER_NO_SPK_PERCENT_MAX: float | None = 30.0
AISHELL4_LONG90_MISSED_DETECTION_RATIO_MAX = 0.5


GOOGLETIME_CER_PERCENT_REF = 32.88493971954541
GOOGLETIME_CER_NO_SPK_PERCENT_REF = 32.88493971954541
GOOGLETIME_CER_NO_SPK_BELOW_50_PERCENT_REF: float | None = 31.708566672917655
GOOGLETIME_N_ABOVE_50_CER_REF: int | None = 1
GOOGLETIME_CP_CER_PERCENT_REF = 33.70759062926478
GOOGLETIME_DELTA_CER_PERCENT_REF = 0.8364769754289347
GOOGLETIME_SPEAKER_TIMESTAMP_DER_PERCENT_REF = 31.042260093758223
GOOGLETIME_THROUGHPUT_QPS_REF = 0.046
GOOGLETIME_LATENCY_MEAN_S_REF = 253.922
GOOGLETIME_LATENCY_P95_S_REF = 279.229
GOOGLETIME_RTF_MEAN_REF = 0.0984
GOOGLETIME_RTF_P95_REF = 0.1253

# Note (guozhihao): Streaming emits partial deltas, so keep its refs separate
# from non-streaming thresholds to avoid mixing latency and accuracy baselines.
MOSS_TD_STREAM_CER_PERCENT_REF: float | None = 5.820198296682344
MOSS_TD_STREAM_CER_NO_SPK_PERCENT_REF: float | None = 5.820198296682344
MOSS_TD_STREAM_CER_NO_SPK_BELOW_50_PERCENT_REF: float | None = 4.996223504266636

# note (chenyang): It's quite unstable for the MOSS_TD_STREAM_N_ABOVE_50_CER_MAX
# We keep it fixed to 31 and no need to change it during calibration.
MOSS_TD_STREAM_N_ABOVE_50_CER_MAX: int | None = 31
MOSS_TD_STREAM_CP_CER_PERCENT_REF: float | None = 13.22295665437905
MOSS_TD_STREAM_CER_NO_SPK_CP_VALID_PERCENT_REF: float | None = 5.820198296682344
MOSS_TD_STREAM_DELTA_CER_PERCENT_REF: float | None = 7.421825346383626
MOSS_TD_STREAM_SPEAKER_TIMESTAMP_DER_PERCENT_REF: float | None = 21.034847600690796
MOSS_TD_STREAM_CER_VALID_SAMPLES_MIN: int | None = 784
MOSS_TD_STREAM_CP_CER_VALID_SAMPLES_MIN: int | None = 784
MOSS_TD_STREAM_THROUGHPUT_QPS_REF: float | None = 47.847
MOSS_TD_STREAM_LATENCY_MEAN_S_REF: float | None = 0.274
MOSS_TD_STREAM_LATENCY_P95_S_REF: float | None = 0.572
MOSS_TD_STREAM_RTF_MEAN_REF: float | None = 0.0298
MOSS_TD_STREAM_RTF_P95_REF: float | None = 0.0374
MOSS_TD_STREAM_TEXT_TTFT_P95_S_REF: float | None = 0.0555
MOSS_TD_STREAM_INTER_CHUNK_P95_S_REF: float | None = 0.0559

THRESHOLD_SLACK_HIGHER = 0.9
THRESHOLD_SLACK_LOWER = 1.1

# Note (chenyang): AISHELL4-long runs only 20 samples, so a single straggler
#  or a flipped orderline sample moves the aggregate metrics far more than
# the 800-sample movies800 corpus. Widen its slack accordingly.
AISHELL4_LONG_THRESHOLD_SLACK_HIGHER = 0.8
AISHELL4_LONG_THRESHOLD_SLACK_LOWER = 1.2

# Note (chenyang): GoogleTime runs only 25 long podcast samples, so widen
# slack the same way as AISHELL4-long.
GOOGLETIME_THRESHOLD_SLACK_HIGHER = 0.8
GOOGLETIME_THRESHOLD_SLACK_LOWER = 1.2

MOSS_TD_N_ABOVE_50_CER_MAX: int | None = (
    math.ceil(MOSS_TD_N_ABOVE_50_CER_REF * THRESHOLD_SLACK_LOWER)
    if MOSS_TD_N_ABOVE_50_CER_REF is not None
    else None
)

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
MOSS_TD_STREAM_CER_PERCENT_MAX: float | None = (
    round(MOSS_TD_STREAM_CER_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_STREAM_CER_PERCENT_REF is not None
    else None
)
MOSS_TD_STREAM_CER_NO_SPK_PERCENT_MAX: float | None = (
    round(MOSS_TD_STREAM_CER_NO_SPK_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_STREAM_CER_NO_SPK_PERCENT_REF is not None
    else None
)
MOSS_TD_STREAM_CER_NO_SPK_BELOW_50_PERCENT_MAX: float | None = (
    round(MOSS_TD_STREAM_CER_NO_SPK_BELOW_50_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_STREAM_CER_NO_SPK_BELOW_50_PERCENT_REF is not None
    else None
)
MOSS_TD_STREAM_CP_CER_PERCENT_MAX: float | None = (
    round(MOSS_TD_STREAM_CP_CER_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_STREAM_CP_CER_PERCENT_REF is not None
    else None
)
MOSS_TD_STREAM_CER_NO_SPK_CP_VALID_PERCENT_MAX: float | None = (
    round(MOSS_TD_STREAM_CER_NO_SPK_CP_VALID_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_STREAM_CER_NO_SPK_CP_VALID_PERCENT_REF is not None
    else None
)
MOSS_TD_STREAM_DELTA_CER_PERCENT_MAX: float | None = (
    round(MOSS_TD_STREAM_DELTA_CER_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_STREAM_DELTA_CER_PERCENT_REF is not None
    else None
)
MOSS_TD_STREAM_SPEAKER_TIMESTAMP_DER_PERCENT_MAX: float | None = (
    round(MOSS_TD_STREAM_SPEAKER_TIMESTAMP_DER_PERCENT_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_STREAM_SPEAKER_TIMESTAMP_DER_PERCENT_REF is not None
    else None
)
MOSS_TD_STREAM_THROUGHPUT_QPS_MIN: float | None = (
    round(MOSS_TD_STREAM_THROUGHPUT_QPS_REF * THRESHOLD_SLACK_HIGHER, 3)
    if MOSS_TD_STREAM_THROUGHPUT_QPS_REF is not None
    else None
)
MOSS_TD_STREAM_LATENCY_MEAN_S_MAX: float | None = (
    round(MOSS_TD_STREAM_LATENCY_MEAN_S_REF * THRESHOLD_SLACK_LOWER, 3)
    if MOSS_TD_STREAM_LATENCY_MEAN_S_REF is not None
    else None
)
MOSS_TD_STREAM_LATENCY_P95_S_MAX: float | None = (
    round(MOSS_TD_STREAM_LATENCY_P95_S_REF * THRESHOLD_SLACK_LOWER, 3)
    if MOSS_TD_STREAM_LATENCY_P95_S_REF is not None
    else None
)
MOSS_TD_STREAM_RTF_MEAN_MAX: float | None = (
    round(MOSS_TD_STREAM_RTF_MEAN_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_STREAM_RTF_MEAN_REF is not None
    else None
)
MOSS_TD_STREAM_RTF_P95_MAX: float | None = (
    round(MOSS_TD_STREAM_RTF_P95_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_STREAM_RTF_P95_REF is not None
    else None
)
MOSS_TD_STREAM_TEXT_TTFT_P95_S_MAX: float | None = (
    round(MOSS_TD_STREAM_TEXT_TTFT_P95_S_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_STREAM_TEXT_TTFT_P95_S_REF is not None
    else None
)
MOSS_TD_STREAM_INTER_CHUNK_P95_S_MAX: float | None = (
    round(MOSS_TD_STREAM_INTER_CHUNK_P95_S_REF * THRESHOLD_SLACK_LOWER, 4)
    if MOSS_TD_STREAM_INTER_CHUNK_P95_S_REF is not None
    else None
)
AISHELL4_LONG_CER_PERCENT_MAX: float | None = round(
    AISHELL4_LONG_CER_PERCENT_REF * AISHELL4_LONG_THRESHOLD_SLACK_LOWER, 4
)
AISHELL4_LONG_CER_NO_SPK_PERCENT_MAX: float | None = round(
    AISHELL4_LONG_CER_NO_SPK_PERCENT_REF * AISHELL4_LONG_THRESHOLD_SLACK_LOWER, 4
)
AISHELL4_LONG_CP_CER_PERCENT_MAX: float | None = round(
    AISHELL4_LONG_CP_CER_PERCENT_REF * AISHELL4_LONG_THRESHOLD_SLACK_LOWER, 4
)
AISHELL4_LONG_DELTA_CER_PERCENT_MAX: float | None = None
AISHELL4_LONG_SPEAKER_TIMESTAMP_DER_PERCENT_MAX: float | None = round(
    AISHELL4_LONG_SPEAKER_TIMESTAMP_DER_PERCENT_REF
    * AISHELL4_LONG_THRESHOLD_SLACK_LOWER,
    4,
)
AISHELL4_LONG_THROUGHPUT_QPS_MIN: float | None = round(
    AISHELL4_LONG_THROUGHPUT_QPS_REF * AISHELL4_LONG_THRESHOLD_SLACK_HIGHER, 3
)
AISHELL4_LONG_LATENCY_MEAN_S_MAX: float | None = round(
    AISHELL4_LONG_LATENCY_MEAN_S_REF * AISHELL4_LONG_THRESHOLD_SLACK_LOWER, 3
)
AISHELL4_LONG_LATENCY_P95_S_MAX: float | None = round(
    AISHELL4_LONG_LATENCY_P95_S_REF * AISHELL4_LONG_THRESHOLD_SLACK_LOWER, 3
)
AISHELL4_LONG_RTF_MEAN_MAX: float | None = round(
    AISHELL4_LONG_RTF_MEAN_REF * AISHELL4_LONG_THRESHOLD_SLACK_LOWER, 4
)
AISHELL4_LONG_RTF_P95_MAX: float | None = round(
    AISHELL4_LONG_RTF_P95_REF * AISHELL4_LONG_THRESHOLD_SLACK_LOWER, 4
)

GOOGLETIME_CER_PERCENT_MAX: float | None = round(
    GOOGLETIME_CER_PERCENT_REF * GOOGLETIME_THRESHOLD_SLACK_LOWER, 4
)
GOOGLETIME_CER_NO_SPK_PERCENT_MAX: float | None = round(
    GOOGLETIME_CER_NO_SPK_PERCENT_REF * GOOGLETIME_THRESHOLD_SLACK_LOWER, 4
)
GOOGLETIME_CER_NO_SPK_BELOW_50_PERCENT_MAX: float | None = (
    round(
        GOOGLETIME_CER_NO_SPK_BELOW_50_PERCENT_REF * GOOGLETIME_THRESHOLD_SLACK_LOWER, 4
    )
    if GOOGLETIME_CER_NO_SPK_BELOW_50_PERCENT_REF is not None
    else None
)
GOOGLETIME_N_ABOVE_50_CER_MIN: int | None = 0
GOOGLETIME_N_ABOVE_50_CER_MAX: int | None = (
    math.ceil(GOOGLETIME_N_ABOVE_50_CER_REF * GOOGLETIME_THRESHOLD_SLACK_LOWER)
    if GOOGLETIME_N_ABOVE_50_CER_REF is not None
    else None
)
GOOGLETIME_CP_CER_PERCENT_MAX: float | None = round(
    GOOGLETIME_CP_CER_PERCENT_REF * GOOGLETIME_THRESHOLD_SLACK_LOWER, 4
)
GOOGLETIME_DELTA_CER_PERCENT_MAX: float | None = None
GOOGLETIME_SPEAKER_TIMESTAMP_DER_PERCENT_MAX: float | None = round(
    GOOGLETIME_SPEAKER_TIMESTAMP_DER_PERCENT_REF * GOOGLETIME_THRESHOLD_SLACK_LOWER,
    4,
)
GOOGLETIME_THROUGHPUT_QPS_MIN: float | None = round(
    GOOGLETIME_THROUGHPUT_QPS_REF * GOOGLETIME_THRESHOLD_SLACK_HIGHER, 3
)
GOOGLETIME_LATENCY_MEAN_S_MAX: float | None = round(
    GOOGLETIME_LATENCY_MEAN_S_REF * GOOGLETIME_THRESHOLD_SLACK_LOWER, 3
)
GOOGLETIME_LATENCY_P95_S_MAX: float | None = round(
    GOOGLETIME_LATENCY_P95_S_REF * GOOGLETIME_THRESHOLD_SLACK_LOWER, 3
)
GOOGLETIME_RTF_MEAN_MAX: float | None = round(
    GOOGLETIME_RTF_MEAN_REF * GOOGLETIME_THRESHOLD_SLACK_LOWER, 4
)
GOOGLETIME_RTF_P95_MAX: float | None = round(
    GOOGLETIME_RTF_P95_REF * GOOGLETIME_THRESHOLD_SLACK_LOWER, 4
)


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
def aishell4_long90_sample(aishell4_long_samples):
    return build_long_audio_concat_sample(
        clips=[aishell4_long_samples[i] for i in MOSS_TD_LONG90_CLIP_INDICES],
        target_duration_s=MOSS_TD_LONG90_TARGET_S,
        sample_id="aishell4_long90",
    )


@pytest.fixture(scope="module")
def googletime_samples():
    return load_movies800_samples(
        repo_id=GOOGLETIME_REPO_ID,
        split="validation",
        audio_column="audio",
        expected_column="transcription",
        max_samples=None,
        expected_sample_count=MOSS_TD_GOOGLETIME_CI_SAMPLES,
    )


@pytest.fixture(scope="module")
def moss_td_router_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> ManagedRouterHandle:
    worker_extra_args = " ".join(
        [
            "--asr.engine.max_running_requests",
            str(MOSS_TD_CONCURRENCY),
            "--asr.engine.cuda_graph_max_bs",
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
    aishell4_long90_sample,
    googletime_samples,
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
    checks.check(
        len(googletime_samples) == MOSS_TD_GOOGLETIME_CI_SAMPLES,
        f"Expected {MOSS_TD_GOOGLETIME_CI_SAMPLES} googletime samples, "
        f"got {len(googletime_samples)}",
    )
    if (
        not movies800times_samples
        or not aishell4_long_samples
        or not googletime_samples
    ):
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

    with router_worker_traffic_guard(
        moss_td_router_server,
        label="MOSS-Transcribe-Diarize movies800times stream",
    ) as movies800times_stream_router_guard:
        movies800times_stream_outputs, movies800times_stream_wall_clock_s = (
            _run_transcribe_diarize(
                movies800times_samples,
                moss_td_router_server=moss_td_router_server,
                request_timeout_s=300,
                max_new_tokens=None,
                stream=True,
            )
        )
    movies800times_stream_results = _build_results(
        samples=movies800times_samples,
        outputs=movies800times_stream_outputs,
        wall_clock_s=movies800times_stream_wall_clock_s,
        repo_id=MOVIES800_REPO_ID,
    )
    _print_and_save_results(
        results=movies800times_stream_results,
        tmp_path=tmp_path,
        filename="moss_transcribe_diarize_stream_results.json",
        router_ready_s=moss_td_router_server.router_ready_s,
        stream=True,
    )
    _assert_movies800times_stream_results(
        checks,
        movies800times_stream_results,
        movies800times_stream_router_guard,
    )

    with router_worker_traffic_guard(
        moss_td_router_server,
        label="MOSS-Transcribe-Diarize aishell4_long",
    ):
        aishell4_outputs, aishell4_wall_clock_s = _run_transcribe_diarize(
            aishell4_long_samples,
            moss_td_router_server=moss_td_router_server,
            request_timeout_s=1800,
            max_new_tokens=MOSS_TD_LONG_MAX_NEW_TOKENS,
        )
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

    with router_worker_traffic_guard(
        moss_td_router_server,
        label="MOSS-Transcribe-Diarize aishell4_long90",
    ):
        long90_outputs, long90_wall_clock_s = _run_transcribe_diarize(
            [aishell4_long90_sample],
            moss_td_router_server=moss_td_router_server,
            request_timeout_s=1800,
            max_new_tokens=MOSS_TD_LONG90_MAX_NEW_TOKENS,
        )
    long90_results = _build_results(
        samples=[aishell4_long90_sample],
        outputs=long90_outputs,
        wall_clock_s=long90_wall_clock_s,
        repo_id=AISHELL4_REPO_ID,
        dataset="aishell4_long90",
    )
    _print_and_save_results(
        results=long90_results,
        tmp_path=tmp_path,
        filename="moss_transcribe_diarize_aishell4_long90_results.json",
        router_ready_s=moss_td_router_server.router_ready_s,
    )
    _assert_aishell4_long90_results(checks, long90_results)

    with router_worker_traffic_guard(
        moss_td_router_server,
        label="MOSS-Transcribe-Diarize googletime",
    ):
        googletime_outputs, googletime_wall_clock_s = _run_transcribe_diarize(
            googletime_samples,
            moss_td_router_server=moss_td_router_server,
            request_timeout_s=1800,
            max_new_tokens=MOSS_TD_LONG_MAX_NEW_TOKENS,
        )
    googletime_results = _build_results(
        samples=googletime_samples,
        outputs=googletime_outputs,
        wall_clock_s=googletime_wall_clock_s,
        repo_id=GOOGLETIME_REPO_ID,
    )
    _print_and_save_results(
        results=googletime_results,
        tmp_path=tmp_path,
        filename="moss_transcribe_diarize_googletime_results.json",
        router_ready_s=moss_td_router_server.router_ready_s,
    )
    _assert_googletime_results(checks, googletime_results)
    checks.assert_all()


def _run_transcribe_diarize(
    samples,
    *,
    moss_td_router_server: ManagedRouterHandle,
    request_timeout_s: int,
    max_new_tokens: int | None,
    stream: bool = False,
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
            stream=stream,
        )
    )


def _dataset_preset(repo_id: str) -> str:
    if repo_id == MOVIES800_REPO_ID:
        return "movies800times"
    if repo_id == AISHELL4_REPO_ID:
        return "aishell4_long"
    if repo_id == GOOGLETIME_REPO_ID:
        return "googletime"
    return repo_id


def _build_results(
    *,
    samples,
    outputs,
    wall_clock_s: float,
    repo_id: str,
    dataset: str | None = None,
):
    return build_evaluation_payload(
        samples=samples,
        outputs=outputs,
        wall_clock_s=wall_clock_s,
        model_path=MOSS_TD_CI_MODEL_PATH,
        concurrency=MOSS_TD_CONCURRENCY,
        repo_id=repo_id,
        split="validation",
        dataset=dataset or _dataset_preset(repo_id),
    )


def _dataset_label_from_results(results) -> str | None:
    config = results.get("config", {})
    if not isinstance(config, dict):
        return None
    return format_benchmark_dataset_label(
        dataset=config.get("dataset"),
        repo_id=config.get("repo_id"),
        split=config.get("split"),
    )


def _print_and_save_results(
    *,
    results,
    tmp_path: Path,
    filename: str,
    router_ready_s: float,
    stream: bool = False,
) -> None:
    summary = results["summary"]
    speed = results["speed"]
    diarization_metrics = results["diarization_metrics"]
    dataset_label = _dataset_label_from_results(results)
    if stream and dataset_label:
        dataset_label = f"{dataset_label} [stream]"
    print_diarization_accuracy_summary(
        summary=summary,
        diarization_metrics=diarization_metrics,
        model_name=MOSS_TD_CI_MODEL_PATH,
        concurrency=MOSS_TD_CONCURRENCY,
        dataset=dataset_label,
    )
    print_diarization_speed_summary(
        speed=speed,
        model_name=MOSS_TD_CI_MODEL_PATH,
        concurrency=MOSS_TD_CONCURRENCY,
        dataset=dataset_label,
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


def _assert_movies800times_stream_results(
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
        f"Expected all streaming samples evaluated, got {evaluated}/{total}",
    )
    checks.check(
        failed_requests == 0,
        f"Expected 0 streaming failed requests, got {failed_requests}",
    )
    checks.check(
        diarization_percent.get("count") == total,
        f"Expected streaming diarization count {total}, "
        f"got {diarization_percent.get('count')}",
    )
    checks.check(
        speed.get("text_ttft_p95_s") is not None,
        "Expected streaming text_ttft_p95_s in speed metrics",
    )
    checks.check(
        speed.get("inter_chunk_p95_s") is not None,
        "Expected streaming inter_chunk_p95_s in speed metrics",
    )
    _check_optional_max(
        checks,
        "stream cer",
        diarization_percent.get("cer"),
        MOSS_TD_STREAM_CER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "stream cer_no_spk",
        diarization_percent.get("cer_no_spk"),
        MOSS_TD_STREAM_CER_NO_SPK_PERCENT_MAX,
        unit="%",
    )
    assert_cer_partitioned(
        diarization_percent,
        max_cer_no_spk_below_50_percent=MOSS_TD_STREAM_CER_NO_SPK_BELOW_50_PERCENT_MAX,
        max_n_above_50_cer=MOSS_TD_STREAM_N_ABOVE_50_CER_MAX,
        collector=checks,
    )
    _check_optional_max(
        checks,
        "stream cp_cer",
        diarization_percent.get("cp_cer"),
        MOSS_TD_STREAM_CP_CER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "stream cer_no_spk_cp_valid",
        diarization_percent.get("cer_no_spk_cp_valid"),
        MOSS_TD_STREAM_CER_NO_SPK_CP_VALID_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "stream delta_cer",
        diarization_percent.get("delta_cer"),
        MOSS_TD_STREAM_DELTA_CER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "stream speaker_timestamp_der",
        diarization_percent.get("speaker_timestamp_der"),
        MOSS_TD_STREAM_SPEAKER_TIMESTAMP_DER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_min(
        checks,
        "stream cer_valid_samples",
        diarization_percent.get("cer_valid_samples"),
        MOSS_TD_STREAM_CER_VALID_SAMPLES_MIN,
    )
    _check_optional_min(
        checks,
        "stream cp_cer_valid_samples",
        diarization_percent.get("cp_cer_valid_samples"),
        MOSS_TD_STREAM_CP_CER_VALID_SAMPLES_MIN,
    )
    _check_optional_min(
        checks,
        "stream throughput_qps",
        speed.get("throughput_qps"),
        MOSS_TD_STREAM_THROUGHPUT_QPS_MIN,
    )
    _check_optional_max(
        checks,
        "stream latency_mean_s",
        speed.get("latency_mean_s"),
        MOSS_TD_STREAM_LATENCY_MEAN_S_MAX,
        unit="s",
    )
    _check_optional_max(
        checks,
        "stream latency_p95_s",
        speed.get("latency_p95_s"),
        MOSS_TD_STREAM_LATENCY_P95_S_MAX,
        unit="s",
    )
    _check_optional_max(
        checks,
        "stream rtf_mean",
        speed.get("rtf_mean"),
        MOSS_TD_STREAM_RTF_MEAN_MAX,
    )
    _check_optional_max(
        checks,
        "stream rtf_p95",
        speed.get("rtf_p95"),
        MOSS_TD_STREAM_RTF_P95_MAX,
    )
    _check_optional_max(
        checks,
        "stream text_ttft_p95_s",
        speed.get("text_ttft_p95_s"),
        MOSS_TD_STREAM_TEXT_TTFT_P95_S_MAX,
        unit="s",
    )
    _check_optional_max(
        checks,
        "stream inter_chunk_p95_s",
        speed.get("inter_chunk_p95_s"),
        MOSS_TD_STREAM_INTER_CHUNK_P95_S_MAX,
        unit="s",
    )
    checks.check_assertion(
        "stream router traffic",
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
    if AISHELL4_LONG_DELTA_CER_PERCENT_MAX is None:
        # Note (chenyang): Report-only: delta_cer on 20 samples is noisy,
        #  so we log the value for observability but do not assert on it.
        print(
            "[report-only] aishell4_long delta_cer="
            f"{diarization_percent.get('delta_cer')}%"
        )
    else:
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
    _check_format_validity(
        checks,
        "aishell4_long",
        diarization_percent,
        expected_valid=MOSS_TD_AISHELL4_LONG_CI_SAMPLES,
    )


def _assert_aishell4_long90_results(checks: MetricCheckCollector, results) -> None:
    summary = results["summary"]
    speed = results["speed"]
    diarization_percent = results["diarization_metrics_percent"]
    checks.check(
        summary["evaluated"] == 1,
        f"Expected the aishell4_long90 sample evaluated, got {summary['evaluated']}",
    )
    failed_requests = speed.get("failed_requests")
    checks.check(
        failed_requests == 0,
        f"Expected 0 aishell4_long90 failed requests, got {failed_requests}",
    )
    _check_format_validity(
        checks, "aishell4_long90", diarization_percent, expected_valid=1
    )
    missed = diarization_percent.get("speaker_timestamp_der_missed_detection")
    total = diarization_percent.get("speaker_timestamp_der_total_seconds")
    if total:
        ratio = missed / total
        checks.check(
            ratio <= AISHELL4_LONG90_MISSED_DETECTION_RATIO_MAX,
            "Expected aishell4_long90 timestamped segments to cover the "
            f"reference speech, got missed detection ratio {ratio:.3f}",
        )
    _check_optional_max(
        checks,
        "aishell4_long90 cer_no_spk",
        diarization_percent.get("cer_no_spk"),
        AISHELL4_LONG90_CER_NO_SPK_PERCENT_MAX,
        unit="%",
    )
    # note (db-ol): one 90 minute sample, so speaker attribution and speed
    # are logged for observability but not asserted.
    print(
        "[report-only] aishell4_long90 "
        f"cp_cer={diarization_percent.get('cp_cer')}% "
        f"speaker_timestamp_der={diarization_percent.get('speaker_timestamp_der')}% "
        f"latency_mean_s={speed.get('latency_mean_s')} "
        f"rtf_mean={speed.get('rtf_mean')}"
    )


def _check_format_validity(
    checks: MetricCheckCollector,
    label: str,
    diarization_percent,
    *,
    expected_valid: int,
) -> None:
    valid = diarization_percent.get("speaker_timestamp_der_valid_samples")
    checks.check(
        valid == expected_valid,
        f"Expected {expected_valid} {label} outputs with parseable "
        f"timestamped segments, got speaker_timestamp_der_valid_samples={valid}",
    )


def _assert_googletime_results(checks: MetricCheckCollector, results) -> None:
    summary = results["summary"]
    speed = results["speed"]
    diarization_percent = results["diarization_metrics_percent"]
    total = summary["total_samples"]
    evaluated = summary["evaluated"]
    failed_requests = speed.get("failed_requests")
    checks.check(
        total == MOSS_TD_GOOGLETIME_CI_SAMPLES,
        f"Expected {MOSS_TD_GOOGLETIME_CI_SAMPLES} googletime samples, got {total}",
    )
    checks.check(
        evaluated == total,
        f"Expected all googletime samples evaluated, got {evaluated}/{total}",
    )
    checks.check(
        failed_requests == 0,
        f"Expected 0 googletime failed requests, got {failed_requests}",
    )
    _check_optional_max(
        checks,
        "googletime cer",
        diarization_percent.get("cer"),
        GOOGLETIME_CER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_max(
        checks,
        "googletime cer_no_spk",
        diarization_percent.get("cer_no_spk"),
        GOOGLETIME_CER_NO_SPK_PERCENT_MAX,
        unit="%",
    )
    assert_cer_partitioned(
        diarization_percent,
        max_cer_no_spk_below_50_percent=GOOGLETIME_CER_NO_SPK_BELOW_50_PERCENT_MAX,
        min_n_above_50_cer=GOOGLETIME_N_ABOVE_50_CER_MIN,
        max_n_above_50_cer=GOOGLETIME_N_ABOVE_50_CER_MAX,
        collector=checks,
    )
    _check_optional_max(
        checks,
        "googletime cp_cer",
        diarization_percent.get("cp_cer"),
        GOOGLETIME_CP_CER_PERCENT_MAX,
        unit="%",
    )
    if GOOGLETIME_DELTA_CER_PERCENT_MAX is None:
        # Note (chenyang): Report-only: delta_cer on 25 samples is noisy,
        # so we log the value for observability but do not assert on it.
        print(
            "[report-only] googletime delta_cer="
            f"{diarization_percent.get('delta_cer')}%"
        )
    else:
        _check_optional_max(
            checks,
            "googletime delta_cer",
            diarization_percent.get("delta_cer"),
            GOOGLETIME_DELTA_CER_PERCENT_MAX,
            unit="%",
        )
    _check_optional_max(
        checks,
        "googletime speaker_timestamp_der",
        diarization_percent.get("speaker_timestamp_der"),
        GOOGLETIME_SPEAKER_TIMESTAMP_DER_PERCENT_MAX,
        unit="%",
    )
    _check_optional_min(
        checks,
        "googletime throughput_qps",
        speed.get("throughput_qps"),
        GOOGLETIME_THROUGHPUT_QPS_MIN,
    )
    _check_optional_max(
        checks,
        "googletime latency_mean_s",
        speed.get("latency_mean_s"),
        GOOGLETIME_LATENCY_MEAN_S_MAX,
        unit="s",
    )
    _check_optional_max(
        checks,
        "googletime latency_p95_s",
        speed.get("latency_p95_s"),
        GOOGLETIME_LATENCY_P95_S_MAX,
        unit="s",
    )
    _check_optional_max(
        checks,
        "googletime rtf_mean",
        speed.get("rtf_mean"),
        GOOGLETIME_RTF_MEAN_MAX,
    )
    _check_optional_max(
        checks,
        "googletime rtf_p95",
        speed.get("rtf_p95"),
        GOOGLETIME_RTF_P95_MAX,
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
