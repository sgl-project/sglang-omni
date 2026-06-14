# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities for model CI, metrics checks, and server lifecycle."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.benchmarker import utils as benchmark_utils
from benchmarks.tasks.tts import (
    DEFAULT_ASR_TRANSCRIBE_CONCURRENCY,
    QWEN3_ASR_MODEL_PATH,
)

if TYPE_CHECKING:
    from tests.test_model.omni_router_utils import ManagedRouterHandle

STARTUP_TIMEOUT = benchmark_utils.STARTUP_TIMEOUT
REPO_ROOT = benchmark_utils.REPO_ROOT
GPU_CLEANUP_SCRIPT = benchmark_utils.GPU_CLEANUP_SCRIPT
GPU_IDLE_THRESHOLD_MB = benchmark_utils.GPU_IDLE_THRESHOLD_MB
GPU_IDLE_WAIT_SECONDS = benchmark_utils.GPU_IDLE_WAIT_SECONDS
GPU_IDLE_POLL_SECONDS = benchmark_utils.GPU_IDLE_POLL_SECONDS
disable_proxy = benchmark_utils.disable_proxy
no_proxy_env = benchmark_utils.no_proxy_env
server_log_file = benchmark_utils.server_log_file
stop_server = benchmark_utils.stop_server
wait_for_gpu_memory_release = benchmark_utils.wait_for_gpu_memory_release
wait_healthy = benchmark_utils.wait_healthy
start_server_from_cmd = benchmark_utils.start_server_from_cmd

QWEN3_ASR_WER_MODEL_PATH = QWEN3_ASR_MODEL_PATH
QWEN3_ASR_WER_CONCURRENCY = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
QWEN3_ASR_ROUTER_STARTUP_TIMEOUT = 600


@dataclass
class ServerHandle:
    """Typed bundle of a running server's process, port, and log file."""

    proc: subprocess.Popen
    port: int
    log_file: Path | None = None


@dataclass
class MetricCheckCollector:
    """Collect metric-check failures and raise them together at the end."""

    label: str = "CI metric checks"
    failures: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.fail(message)

    def check_assertion(
        self,
        check_label: str,
        func: Callable,
        /,
        *args,
        **kwargs,
    ) -> None:
        try:
            func(*args, **kwargs)
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            self.fail(f"{check_label}: {detail}")

    def assert_all(self) -> None:
        if not self.failures:
            return
        details = "\n".join(
            f"{idx}. {failure}" for idx, failure in enumerate(self.failures, start=1)
        )
        raise AssertionError(
            f"{self.label} failed {len(self.failures)} check(s):\n{details}"
        )


@pytest.fixture
def qwen3_asr_wer_router(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator["ManagedRouterHandle"]:
    """Launch Qwen3-ASR router for WER after upstream servers release GPU."""
    from tests.test_model.omni_router_utils import launch_managed_router

    wait_for_gpu_memory_release()
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=QWEN3_ASR_WER_MODEL_PATH,
        model_name=QWEN3_ASR_WER_MODEL_PATH,
        worker_extra_args="",
        wait_timeout=QWEN3_ASR_ROUTER_STARTUP_TIMEOUT,
        log_prefix="asr_wer_router_logs",
    ) as router:
        yield router


def _metric_collector(
    collector: MetricCheckCollector | None,
    label: str,
) -> MetricCheckCollector:
    return collector if collector is not None else MetricCheckCollector(label)


def _assert_metric_collector_if_local(
    collector_arg: MetricCheckCollector | None,
    collector: MetricCheckCollector,
) -> None:
    if collector_arg is None:
        collector.assert_all()


def assert_summary_metrics(
    summary: dict,
    *,
    check_tokens: bool = True,
    collector: MetricCheckCollector | None = None,
) -> None:
    """Verify summary-level sanity invariants that must hold for every run."""
    checks = _metric_collector(collector, "summary metrics")
    failed_requests = summary.get("failed_requests")
    checks.check(
        failed_requests == 0,
        f"Expected 0 failed requests, got {failed_requests}",
    )
    audio_duration_mean_s = summary.get("audio_duration_mean_s")
    checks.check(
        audio_duration_mean_s is not None and audio_duration_mean_s > 0,
        f"Expected positive audio duration, got {audio_duration_mean_s}",
    )
    if check_tokens:
        output_tokens_mean = summary.get("output_tokens_mean", 0)
        checks.check(
            output_tokens_mean > 0,
            f"Expected positive output_tokens_mean, got {output_tokens_mean}",
        )
        prompt_tokens_mean = summary.get("prompt_tokens_mean", 0)
        checks.check(
            prompt_tokens_mean > 0,
            f"Expected positive prompt_tokens_mean, got {prompt_tokens_mean}",
        )
    _assert_metric_collector_if_local(collector, checks)


def assert_per_request_fields(
    per_request: list[dict],
    *,
    check_tokens: bool = True,
    collector: MetricCheckCollector | None = None,
) -> None:
    """Verify every request has valid audio, prompt_tokens, and completion_tokens."""
    checks = _metric_collector(collector, "per-request fields")
    for req in per_request:
        rid = req.get("id", "<missing id>")
        checks.check(
            req.get("is_success") is True, f"Request {rid} failed: {req.get('error')}"
        )
        audio_duration_s = req.get("audio_duration_s")
        checks.check(
            audio_duration_s is not None and audio_duration_s > 0,
            f"Request {rid}: audio_duration_s={audio_duration_s}, expected > 0",
        )
        if check_tokens:
            prompt_tokens = req.get("prompt_tokens")
            completion_tokens = req.get("completion_tokens")
            checks.check(
                prompt_tokens is not None and prompt_tokens > 0,
                f"Request {rid}: prompt_tokens={prompt_tokens}, expected > 0",
            )
            checks.check(
                completion_tokens is not None and completion_tokens > 0,
                f"Request {rid}: completion_tokens={completion_tokens}, expected > 0",
            )
    _assert_metric_collector_if_local(collector, checks)


def apply_slack(
    p95: dict[int, dict[str, float]],
    slack_higher: float = 0.875,
    slack_lower: float = 1.125,
) -> dict[int, dict[str, float]]:
    """Derive CI thresholds from P95 references with uniform slack.

    Higher-is-better metrics (throughput, output tok/req-s when present): threshold = P95 x slack_higher
    Lower-is-better metrics (latency, rtf):            threshold = P95 x slack_lower
    """
    result: dict[int, dict[str, float]] = {}
    for conc, m in p95.items():
        thresholds = {
            "throughput_qps_min": round(m["throughput_qps"] * slack_higher, 2),
            "latency_mean_s_max": round(m["latency_mean_s"] * slack_lower, 1),
        }
        if "output_tok_per_req_s" in m:
            thresholds["output_tok_per_req_s_min"] = round(
                m["output_tok_per_req_s"] * slack_higher,
                1,
            )
        if "rtf_mean" in m:
            thresholds["rtf_mean_max"] = round(m["rtf_mean"] * slack_lower, 2)
        result[conc] = thresholds
    return result


def persist_wer_in_benchmark_results(
    audio_dir: str,
    wer: dict,
    results_basename: str,
) -> None:
    """Merge WER into the benchmark results JSON for tune.py calibration."""
    results_path = Path(audio_dir).parent / results_basename
    data = json.loads(results_path.read_text())
    data["wer"] = wer
    results_path.write_text(json.dumps(data, indent=2))


def apply_wer_slack(reference: float, slack: float = 1.25) -> float:
    """Derive a max WER threshold from a reference value with uniform slack."""
    return round(reference * slack, 4)


def apply_mos_slack(reference: float, slack: float = 0.97) -> float:
    """Derive a min MOS threshold from a reference value with downward slack."""
    return round(reference * slack, 4)


def assert_speed_thresholds(
    summary: dict,
    thresholds: dict,
    concurrency: int,
    *,
    collector: MetricCheckCollector | None = None,
) -> None:
    """Assert speed benchmark summary meets threshold requirements.

    Whether RTF and output token throughput are checked is driven entirely by
    the thresholds dict: if ``apply_slack`` was fed a baseline that included
    ``rtf_mean`` or ``output_tok_per_req_s`` the corresponding threshold is
    present here and enforced.
    """
    checks = _metric_collector(collector, "speed thresholds")
    level_thresholds = thresholds.get(concurrency)
    if level_thresholds is None:
        checks.fail(f"No speed thresholds configured for concurrency {concurrency}")
        _assert_metric_collector_if_local(collector, checks)
        return

    throughput_qps = summary.get("throughput_qps")
    checks.check(
        throughput_qps is not None
        and throughput_qps >= level_thresholds["throughput_qps_min"],
        f"throughput_qps {throughput_qps} < "
        f"{level_thresholds['throughput_qps_min']} at concurrency {concurrency}",
    )
    if "output_tok_per_req_s_min" in level_thresholds:
        output_tok_per_req_s = summary.get("output_tok_per_req_s")
        checks.check(
            output_tok_per_req_s is not None
            and output_tok_per_req_s >= level_thresholds["output_tok_per_req_s_min"],
            f"output_tok_per_req_s {output_tok_per_req_s} < "
            f"{level_thresholds['output_tok_per_req_s_min']} "
            f"at concurrency {concurrency}",
        )
    latency_mean_s = summary.get("latency_mean_s")
    checks.check(
        latency_mean_s is not None
        and latency_mean_s <= level_thresholds["latency_mean_s_max"],
        f"latency_mean_s {latency_mean_s} > "
        f"{level_thresholds['latency_mean_s_max']} at concurrency {concurrency}",
    )
    if "rtf_mean_max" in level_thresholds:
        rtf_mean = summary.get("rtf_mean")
        checks.check(
            rtf_mean is not None and rtf_mean <= level_thresholds["rtf_mean_max"],
            f"rtf_mean {rtf_mean} > "
            f"{level_thresholds['rtf_mean_max']} at concurrency {concurrency}",
        )
    _assert_metric_collector_if_local(collector, checks)


DEFAULT_TOTAL_AUDIO_DURATION_RTOL = 0.12


def _request_by_id(requests: list[dict]) -> dict:
    return {
        request.get("id", f"<missing id {idx}>"): request
        for idx, request in enumerate(requests)
    }


def _assert_request_sets(
    non_stream_by_id: dict,
    stream_by_id: dict,
    expected_stream_count: int | None,
    collector: MetricCheckCollector,
) -> list:
    common_ids = sorted(set(non_stream_by_id) & set(stream_by_id))
    collector.check(
        bool(common_ids),
        "No overlapping request IDs between non-stream and stream runs",
    )
    collector.check(
        set(stream_by_id).issubset(set(non_stream_by_id)),
        "Streaming requests must be a subset of non-streaming requests: "
        f"non_stream={sorted(non_stream_by_id)}, stream={sorted(stream_by_id)}",
    )
    if expected_stream_count is not None:
        collector.check(
            len(stream_by_id) == expected_stream_count,
            f"Expected {expected_stream_count} streaming requests, "
            f"got {len(stream_by_id)}",
        )
    return common_ids


def _assert_relative_difference(
    metric_name: str,
    non_stream_value: float,
    stream_value: float,
    relative_tolerance: float,
    collector: MetricCheckCollector,
) -> None:
    max_value = max(non_stream_value, stream_value)
    collector.check(
        abs(non_stream_value - stream_value) <= (relative_tolerance * max_value),
        f"{metric_name} differ too much - "
        f"non_stream={non_stream_value}, stream={stream_value} "
        f"(rtol={relative_tolerance})",
    )


def assert_streaming_consistency(
    non_stream_requests: list[dict],
    stream_requests: list[dict],
    *,
    expected_stream_count: int | None = None,
    max_failed_requests: int = 0,
    total_audio_duration_rtol: float = DEFAULT_TOTAL_AUDIO_DURATION_RTOL,
    collector: MetricCheckCollector | None = None,
) -> None:
    """Assert request coverage, failure budget, and audio duration consistency
    between non-streaming and streaming runs.
    """
    checks = _metric_collector(collector, "streaming consistency")
    non_stream_by_id = _request_by_id(non_stream_requests)
    stream_by_id = _request_by_id(stream_requests)
    common_ids = _assert_request_sets(
        non_stream_by_id, stream_by_id, expected_stream_count, checks
    )
    non_stream_failed = {
        request_id
        for request_id, request in non_stream_by_id.items()
        if request.get("is_success") is not True
    }
    stream_failed = {
        request_id
        for request_id, request in stream_by_id.items()
        if request.get("is_success") is not True
    }
    checks.check(
        len(non_stream_failed) <= max_failed_requests,
        f"Non-streaming failed request count {len(non_stream_failed)} > "
        f"{max_failed_requests}",
    )
    checks.check(
        len(stream_failed) <= max_failed_requests,
        f"Streaming failed request count {len(stream_failed)} > "
        f"{max_failed_requests}",
    )
    failed_ids = non_stream_failed | stream_failed
    common_ids = [
        request_id for request_id in common_ids if request_id not in failed_ids
    ]
    checks.check(
        bool(common_ids),
        "No successful overlapping request IDs between non-stream and stream runs",
    )

    non_stream_audio_duration_total = 0.0
    stream_audio_duration_total = 0.0

    for request_id in common_ids:
        non_stream_request = non_stream_by_id[request_id]
        stream_request = stream_by_id[request_id]
        non_stream_audio = non_stream_request.get("audio_duration_s")
        stream_audio = stream_request.get("audio_duration_s")
        if non_stream_audio is None or stream_audio is None:
            checks.fail(
                f"Request {request_id}: audio_duration_s missing - "
                f"non_stream={non_stream_audio}, stream={stream_audio}"
            )
        else:
            non_stream_audio_duration_total += non_stream_audio
            stream_audio_duration_total += stream_audio

    if common_ids:
        _assert_relative_difference(
            "Total audio_duration_s",
            non_stream_audio_duration_total,
            stream_audio_duration_total,
            total_audio_duration_rtol,
            checks,
        )
    _assert_metric_collector_if_local(collector, checks)


def _wer_sample_label(sample: dict, index: int) -> str:
    sample_id = sample.get("id")
    if sample_id is None:
        return f"per_sample[{index}]"
    return f"sample {sample_id}"


def _wer_result_sections(
    results: dict,
    checks: MetricCheckCollector,
) -> tuple[dict, list[dict]]:
    summary = results.get("summary")
    if summary is None:
        checks.fail("WER results schema: missing summary")
        summary = {}
    elif not isinstance(summary, dict):
        checks.fail(
            "WER results schema: summary must be a dict, "
            f"got {type(summary).__name__}"
        )
        summary = {}

    per_sample = results.get("per_sample")
    if per_sample is None:
        checks.fail("WER results schema: missing per_sample")
        return summary, []
    if not isinstance(per_sample, list):
        checks.fail(
            "WER results schema: per_sample must be a list, "
            f"got {type(per_sample).__name__}"
        )
        return summary, []

    valid_samples: list[dict] = []
    for index, sample in enumerate(per_sample):
        if isinstance(sample, dict):
            valid_samples.append(sample)
        else:
            checks.fail(
                f"WER results schema: per_sample[{index}] must be a dict, "
                f"got {type(sample).__name__}"
            )
    return summary, valid_samples


def _check_wer_per_sample_schema(
    per_sample: list[dict],
    checks: MetricCheckCollector,
) -> None:
    for index, sample in enumerate(per_sample):
        label = _wer_sample_label(sample, index)
        if "wer" not in sample:
            checks.fail(f"WER results schema: {label} missing required 'wer' field")
            continue

        wer = sample["wer"]
        if wer is None:
            if sample.get("is_success") is True:
                checks.fail(
                    f"WER results schema: {label} has wer=None "
                    "despite is_success=True"
                )
            continue

        if isinstance(wer, bool) or not isinstance(wer, (int, float)):
            checks.fail(
                f"WER results schema: {label} wer must be numeric or None, "
                f"got {type(wer).__name__}"
            )


def assert_wer_partitioned(
    results: dict,
    *,
    max_wer_below_50_corpus: float,
    max_n_above_50: int,
    collector: MetricCheckCollector | None = None,
) -> None:
    """Verify WER results using a partitioned view of the per-sample WER
    distribution, suited to large-scale audio-QA TTS consistency tests:

    - ``max_wer_below_50_corpus``: upper bound on corpus-level WER computed
      ONLY over samples whose per-sample WER ≤ 50%. Measures transcription
      quality on the "sane" subset, insensitive to catastrophic outliers.
    - ``max_n_above_50``: upper bound on the count of samples with
      per-sample WER > 50% (catastrophic failures).

    Together these thresholds bound both the typical-case quality and the
    tail of wildly-wrong outputs, without the length-sensitivity of a
    single corpus-wide WER.
    """
    checks = _metric_collector(collector, "partitioned WER")
    summary, per_sample = _wer_result_sections(results, checks)
    _check_wer_per_sample_schema(per_sample, checks)

    failed_details = [
        f"  sample {s.get('id')}: {s.get('error')}"
        for s in per_sample
        if not s.get("is_success", True)
    ]
    evaluated = summary.get("evaluated")
    total_samples = summary.get("total_samples")
    skipped = summary.get("skipped")
    checks.check(
        evaluated == total_samples,
        f"Only {evaluated}/{total_samples} samples evaluated, "
        f"{skipped} skipped.\n"
        f"Per-sample errors:\n" + "\n".join(failed_details),
    )

    wer_below_50 = summary.get("wer_below_50_corpus")
    if wer_below_50 is None:
        checks.fail("Missing wer_below_50_corpus in WER summary")
    else:
        checks.check(
            wer_below_50 <= max_wer_below_50_corpus,
            f"Corpus WER over samples with WER<=50% is "
            f"{wer_below_50:.4f} ({wer_below_50 * 100:.2f}%) > threshold "
            f"{max_wer_below_50_corpus} ({max_wer_below_50_corpus * 100:.2f}%)",
        )

    n_above_50 = summary.get("n_above_50_pct_wer")
    if n_above_50 is None:
        checks.fail("Missing n_above_50_pct_wer in WER summary")
    else:
        checks.check(
            n_above_50 <= max_n_above_50,
            f"{n_above_50} samples have WER>50% > threshold {max_n_above_50}",
        )
    _assert_metric_collector_if_local(collector, checks)


def assert_wer_results(
    results: dict,
    max_corpus_wer: float,
    max_per_sample_wer: float | None = None,
    *,
    collector: MetricCheckCollector | None = None,
) -> None:
    """Verify WER results are within thresholds."""
    checks = _metric_collector(collector, "WER results")
    summary, per_sample = _wer_result_sections(results, checks)
    _check_wer_per_sample_schema(per_sample, checks)

    if max_per_sample_wer is not None:
        failed_details = [
            f"  sample {s.get('id')}: {s.get('error')}"
            for s in per_sample
            if not s.get("is_success", True)
        ]
        evaluated = summary.get("evaluated")
        total_samples = summary.get("total_samples")
        skipped = summary.get("skipped")
        checks.check(
            evaluated == total_samples,
            f"Only {evaluated}/{total_samples} samples evaluated, "
            f"{skipped} skipped.\n"
            f"Per-sample errors:\n" + "\n".join(failed_details),
        )

    wer_corpus = summary.get("wer_corpus")
    if wer_corpus is None:
        checks.fail("Missing wer_corpus in WER summary")
    else:
        checks.check(
            wer_corpus <= max_corpus_wer,
            f"Corpus WER {wer_corpus:.4f} ({wer_corpus * 100:.2f}%) "
            f"> threshold {max_corpus_wer} ({max_corpus_wer * 100:.0f}%)",
        )

    if max_per_sample_wer is None:
        _assert_metric_collector_if_local(collector, checks)
        return

    for sample in per_sample:
        checks.check(
            sample.get("is_success") is True,
            f"Sample {sample.get('id')} failed: {sample.get('error')}",
        )

    n_above_50 = summary.get("n_above_50_pct_wer")
    checks.check(
        n_above_50 == 0,
        f"{n_above_50} samples have >50% WER - " f"expected 0 catastrophic failures",
    )
    for sample in per_sample:
        wer = sample.get("wer")
        if (
            wer is not None
            and not isinstance(wer, bool)
            and isinstance(wer, (int, float))
        ):
            checks.check(
                wer <= max_per_sample_wer,
                f"Sample {sample.get('id')} WER {wer:.4f} > {max_per_sample_wer}",
            )
    _assert_metric_collector_if_local(collector, checks)
