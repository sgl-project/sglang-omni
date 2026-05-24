# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities — model-agnostic helpers for launching and managing servers."""

from __future__ import annotations

import os
import signal
import statistics
import subprocess
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

STARTUP_TIMEOUT = 600


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
        label: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> None:
        try:
            func(*args, **kwargs)
        except AssertionError as exc:
            detail = str(exc) or exc.__class__.__name__
            self.fail(f"{label}: {detail}")

    def assert_all(self) -> None:
        if not self.failures:
            return
        details = "\n".join(
            f"{idx}. {failure}" for idx, failure in enumerate(self.failures, start=1)
        )
        raise AssertionError(
            f"{self.label} failed {len(self.failures)} check(s):\n{details}"
        )


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


@contextmanager
def disable_proxy() -> Generator[None, None, None]:
    """Temporarily disable proxy env vars for loopback requests."""
    proxy_vars = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    )
    saved_env = {k: os.environ[k] for k in proxy_vars if k in os.environ}
    for k in proxy_vars:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k in proxy_vars:
            os.environ.pop(k, None)
        os.environ.update(saved_env)


def no_proxy_env() -> dict[str, str]:
    """Return a copy of os.environ with proxy variables removed, for subprocess use."""
    proxy_keys = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    return {k: v for k, v in os.environ.items() if k.lower() not in proxy_keys}


def server_log_file(tmp_path_factory, prefix: str = "server_logs") -> Path | None:
    """Capture server logs to a file on CI; stream to the terminal locally."""
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    if not is_ci:
        return None
    return tmp_path_factory.mktemp(prefix) / "server.log"


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop the server process group, tolerating already-dead processes."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, ChildProcessError):
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except (ProcessLookupError, ChildProcessError):
            # Process already exited — nothing left to kill.
            return


def wait_healthy(
    proc: subprocess.Popen,
    port: int,
    log_file: Path | None,
    timeout: int = STARTUP_TIMEOUT,
) -> None:
    """Wait for a server to report healthy, stopping it and raising on failure."""
    from benchmarks.benchmarker.utils import wait_for_service

    try:
        with disable_proxy():
            wait_for_service(
                f"http://localhost:{port}",
                timeout=timeout,
                server_process=proc,
                server_log_file=log_file,
                health_body_contains="healthy",
            )
    except Exception as exc:
        stop_server(proc)
        log_text = (
            log_file.read_text() if log_file is not None and log_file.exists() else ""
        )
        message = str(exc)
        if log_text and log_text not in message:
            message = f"{message}\n{log_text}"
        if isinstance(exc, TimeoutError):
            raise TimeoutError(message) from exc
        if isinstance(exc, RuntimeError):
            raise RuntimeError(message) from exc
        raise


def start_server_from_cmd(
    cmd: list[str],
    log_file: Path | None,
    port: int,
    timeout: int = STARTUP_TIMEOUT,
    env: dict[str, str] | None = None,
    tee: bool = False,
) -> subprocess.Popen:
    """Start a server from an arbitrary command and wait until healthy."""
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    if log_file is None:
        proc = subprocess.Popen(
            cmd,
            env=process_env,
            start_new_session=True,
        )
    elif tee:
        # Tee (file + stdout): TP=2 fixture wants the file for grep + live
        # output for `pytest -s`. Pattern from sglang's popen_launch_server.
        log_handle = open(log_file, "w")
        try:
            proc = subprocess.Popen(
                cmd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
                bufsize=1,
            )
        except Exception:
            log_handle.close()
            raise

        def _tee_stdout(src, sink) -> None:
            try:
                for line in iter(src.readline, ""):
                    sink.write(line)
                    sink.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
            finally:
                src.close()
                sink.close()

        # log_handle ownership is handed to the thread; its finally closes it.
        threading.Thread(
            target=_tee_stdout,
            args=(proc.stdout, log_handle),
            daemon=True,
        ).start()
    else:
        with open(log_file, "w") as log_handle:
            proc = subprocess.Popen(
                cmd,
                env=process_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    wait_healthy(proc, port, log_file, timeout=timeout)
    return proc


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
        gen_tokens_mean = summary.get("gen_tokens_mean", 0)
        checks.check(
            gen_tokens_mean > 0,
            f"Expected positive gen_tokens_mean, got {gen_tokens_mean}",
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

    Higher-is-better metrics (throughput, tok/s): threshold = P95 x slack_higher
    Lower-is-better metrics (latency, rtf):      threshold = P95 x slack_lower
    """
    result: dict[int, dict[str, float]] = {}
    for conc, m in p95.items():
        thresholds = {
            "throughput_qps_min": round(m["throughput_qps"] * slack_higher, 2),
            "tok_per_s_agg_min": round(m["tok_per_s_agg"] * slack_higher, 1),
            "latency_mean_s_max": round(m["latency_mean_s"] * slack_lower, 1),
        }
        if "rtf_mean" in m:
            thresholds["rtf_mean_max"] = round(m["rtf_mean"] * slack_lower, 2)
        result[conc] = thresholds
    return result


def apply_wer_slack(reference: float, slack: float = 1.25) -> float:
    """Derive a max WER threshold from a reference value with uniform slack."""
    return round(reference * slack, 4)


def assert_speed_thresholds(
    summary: dict,
    thresholds: dict,
    concurrency: int,
    *,
    collector: MetricCheckCollector | None = None,
) -> None:
    """Assert speed benchmark summary meets threshold requirements.

    Whether RTF is checked is driven entirely by the thresholds dict: if
    ``apply_slack`` was fed a baseline that included ``rtf_mean`` the
    corresponding ``rtf_mean_max`` is present here and enforced; otherwise
    (e.g. VLM / text-only tasks) the RTF assertion is skipped automatically.
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
    tok_per_s_agg = summary.get("tok_per_s_agg")
    checks.check(
        tok_per_s_agg is not None
        and tok_per_s_agg >= level_thresholds["tok_per_s_agg_min"],
        f"tok_per_s_agg {tok_per_s_agg} < "
        f"{level_thresholds['tok_per_s_agg_min']} at concurrency {concurrency}",
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


DEFAULT_TOTAL_COMPLETION_TOKEN_RTOL = 0.12
DEFAULT_MEDIAN_COMPLETION_TOKEN_RTOL = 0.20
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
    total_completion_token_rtol: float = DEFAULT_TOTAL_COMPLETION_TOKEN_RTOL,
    median_completion_token_rtol: float = DEFAULT_MEDIAN_COMPLETION_TOKEN_RTOL,
    total_audio_duration_rtol: float = DEFAULT_TOTAL_AUDIO_DURATION_RTOL,
    collector: MetricCheckCollector | None = None,
) -> None:
    """Assert stable invariants on the shared request subset between
    non-streaming and streaming runs (matching prompt tokens, total/median
    completion tokens within tolerance, total audio duration within tolerance).
    """
    checks = _metric_collector(collector, "streaming consistency")
    non_stream_by_id = _request_by_id(non_stream_requests)
    stream_by_id = _request_by_id(stream_requests)
    common_ids = _assert_request_sets(
        non_stream_by_id, stream_by_id, expected_stream_count, checks
    )

    non_stream_completion_tokens: list[int] = []
    stream_completion_tokens: list[int] = []
    non_stream_audio_duration_total = 0.0
    stream_audio_duration_total = 0.0

    for request_id in common_ids:
        non_stream_request = non_stream_by_id[request_id]
        stream_request = stream_by_id[request_id]
        checks.check(
            non_stream_request.get("prompt_tokens")
            == stream_request.get("prompt_tokens"),
            f"Request {request_id}: prompt_tokens mismatch - "
            f"non_stream={non_stream_request.get('prompt_tokens')}, "
            f"stream={stream_request.get('prompt_tokens')}",
        )
        non_stream_completion = non_stream_request.get("completion_tokens")
        stream_completion = stream_request.get("completion_tokens")
        non_stream_audio = non_stream_request.get("audio_duration_s")
        stream_audio = stream_request.get("audio_duration_s")
        if non_stream_completion is None or stream_completion is None:
            checks.fail(
                f"Request {request_id}: completion_tokens missing - "
                f"non_stream={non_stream_completion}, stream={stream_completion}"
            )
        else:
            non_stream_completion_tokens.append(non_stream_completion)
            stream_completion_tokens.append(stream_completion)
        if non_stream_audio is None or stream_audio is None:
            checks.fail(
                f"Request {request_id}: audio_duration_s missing - "
                f"non_stream={non_stream_audio}, stream={stream_audio}"
            )
        else:
            non_stream_audio_duration_total += non_stream_audio
            stream_audio_duration_total += stream_audio

    if non_stream_completion_tokens and stream_completion_tokens:
        _assert_relative_difference(
            "Total completion_tokens",
            sum(non_stream_completion_tokens),
            sum(stream_completion_tokens),
            total_completion_token_rtol,
            checks,
        )
        _assert_relative_difference(
            "Median completion_tokens",
            statistics.median(non_stream_completion_tokens),
            statistics.median(stream_completion_tokens),
            median_completion_token_rtol,
            checks,
        )
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
    max_per_sample_wer: float,
    *,
    collector: MetricCheckCollector | None = None,
) -> None:
    """Verify WER results are within thresholds."""
    checks = _metric_collector(collector, "WER results")
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

    wer_corpus = summary.get("wer_corpus")
    if wer_corpus is None:
        checks.fail("Missing wer_corpus in WER summary")
    else:
        checks.check(
            wer_corpus <= max_corpus_wer,
            f"Corpus WER {wer_corpus:.4f} ({wer_corpus * 100:.2f}%) "
            f"> threshold {max_corpus_wer} ({max_corpus_wer * 100:.0f}%)",
        )

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
