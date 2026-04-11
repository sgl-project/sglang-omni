# SPDX-License-Identifier: Apache-2.0
"""Shared test utilities — model-agnostic helpers for launching and managing servers."""

from __future__ import annotations

import os
import signal
import socket
import statistics
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from benchmarks.benchmarker.utils import wait_for_service

STARTUP_TIMEOUT = 600


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


def find_free_port() -> int:
    """Find and return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_handle:
        socket_handle.bind(("", 0))
        return socket_handle.getsockname()[1]


def no_proxy_env() -> dict[str, str]:
    """Return a copy of os.environ with proxy variables removed, for subprocess use."""
    proxy_keys = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    return {k: v for k, v in os.environ.items() if k.lower() not in proxy_keys}


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
    log_file: Path,
    timeout: int = STARTUP_TIMEOUT,
) -> None:
    """Wait for a server to report healthy, stopping it and raising on failure."""
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
        log_text = log_file.read_text() if log_file.exists() else ""
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
    log_file: Path,
    port: int,
    timeout: int = STARTUP_TIMEOUT,
) -> subprocess.Popen:
    """Start a server from an arbitrary command and wait until healthy."""
    with open(log_file, "w") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    wait_healthy(proc, port, log_file, timeout=timeout)
    return proc


def assert_summary_metrics(summary: dict, *, check_tokens: bool = True) -> None:
    """Verify summary-level sanity invariants that must hold for every run."""
    assert (
        summary["failed_requests"] == 0
    ), f"Expected 0 failed requests, got {summary['failed_requests']}"
    assert (
        summary["audio_duration_mean_s"] > 0
    ), f"Expected positive audio duration, got {summary['audio_duration_mean_s']}"
    if check_tokens:
        assert (
            summary.get("gen_tokens_mean", 0) > 0
        ), f"Expected positive gen_tokens_mean, got {summary.get('gen_tokens_mean', 0)}"
        assert (
            summary.get("prompt_tokens_mean", 0) > 0
        ), f"Expected positive prompt_tokens_mean, got {summary.get('prompt_tokens_mean', 0)}"


def assert_per_request_fields(
    per_request: list[dict], *, check_tokens: bool = True
) -> None:
    """Verify every request has valid audio, prompt_tokens, and completion_tokens."""
    for req in per_request:
        rid = req["id"]
        assert req["is_success"], f"Request {rid} failed: {req.get('error')}"
        assert (
            req["audio_duration_s"] is not None and req["audio_duration_s"] > 0
        ), f"Request {rid}: audio_duration_s={req['audio_duration_s']}, expected > 0"
        if check_tokens:
            assert (
                req["prompt_tokens"] is not None and req["prompt_tokens"] > 0
            ), f"Request {rid}: prompt_tokens={req['prompt_tokens']}, expected > 0"
            assert (
                req["completion_tokens"] is not None and req["completion_tokens"] > 0
            ), f"Request {rid}: completion_tokens={req['completion_tokens']}, expected > 0"


def apply_slack(
    p95: dict[int, dict[str, float]],
    slack_higher: float = 0.75,
    slack_lower: float = 1.25,
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


def assert_speed_thresholds(summary: dict, thresholds: dict, concurrency: int) -> None:
    """Assert speed benchmark summary meets threshold requirements.

    Whether RTF is checked is driven entirely by the thresholds dict: if
    ``apply_slack`` was fed a baseline that included ``rtf_mean`` the
    corresponding ``rtf_mean_max`` is present here and enforced; otherwise
    (e.g. VLM / text-only tasks) the RTF assertion is skipped automatically.
    """
    level_thresholds = thresholds[concurrency]
    assert summary["throughput_qps"] >= level_thresholds["throughput_qps_min"], (
        f"throughput_qps {summary['throughput_qps']} < "
        f"{level_thresholds['throughput_qps_min']} at concurrency {concurrency}"
    )
    assert summary["tok_per_s_agg"] >= level_thresholds["tok_per_s_agg_min"], (
        f"tok_per_s_agg {summary['tok_per_s_agg']} < "
        f"{level_thresholds['tok_per_s_agg_min']} at concurrency {concurrency}"
    )
    assert summary["latency_mean_s"] <= level_thresholds["latency_mean_s_max"], (
        f"latency_mean_s {summary['latency_mean_s']} > "
        f"{level_thresholds['latency_mean_s_max']} at concurrency {concurrency}"
    )
    if "rtf_mean_max" in level_thresholds:
        assert summary["rtf_mean"] <= level_thresholds["rtf_mean_max"], (
            f"rtf_mean {summary['rtf_mean']} > "
            f"{level_thresholds['rtf_mean_max']} at concurrency {concurrency}"
        )


def assert_streaming_consistency(
    non_stream_requests: list[dict],
    stream_requests: list[dict],
    *,
    expected_stream_count: int | None = None,
    total_completion_token_rtol: float = 0.12,
    median_completion_token_rtol: float = 0.20,
    total_audio_duration_rtol: float = 0.12,
) -> None:
    """Assert stable invariants on the shared request subset."""
    ns_by_id = {r["id"]: r for r in non_stream_requests}
    st_by_id = {r["id"]: r for r in stream_requests}
    common_ids = sorted(set(ns_by_id) & set(st_by_id))
    assert common_ids, "No overlapping request IDs between non-stream and stream runs"
    assert set(st_by_id).issubset(set(ns_by_id)), (
        "Streaming requests must be a subset of non-streaming requests: "
        f"non_stream={sorted(ns_by_id)}, stream={sorted(st_by_id)}"
    )
    if expected_stream_count is not None:
        assert len(st_by_id) == expected_stream_count, (
            f"Expected {expected_stream_count} streaming requests, "
            f"got {len(st_by_id)}"
        )

    ns_completion_tokens: list[int] = []
    st_completion_tokens: list[int] = []
    ns_audio_duration_total = 0.0
    st_audio_duration_total = 0.0

    for rid in common_ids:
        ns, st = ns_by_id[rid], st_by_id[rid]
        assert ns["prompt_tokens"] == st["prompt_tokens"], (
            f"Request {rid}: prompt_tokens mismatch — "
            f"non_stream={ns['prompt_tokens']}, stream={st['prompt_tokens']}"
        )
        ns_completion_tokens.append(ns["completion_tokens"])
        st_completion_tokens.append(st["completion_tokens"])
        ns_audio_duration_total += ns["audio_duration_s"]
        st_audio_duration_total += st["audio_duration_s"]

    ns_completion_total = sum(ns_completion_tokens)
    st_completion_total = sum(st_completion_tokens)
    max_completion_total = max(ns_completion_total, st_completion_total)
    assert abs(ns_completion_total - st_completion_total) <= (
        total_completion_token_rtol * max_completion_total
    ), (
        "Total completion_tokens differ too much — "
        f"non_stream={ns_completion_total}, stream={st_completion_total} "
        f"(rtol={total_completion_token_rtol})"
    )

    ns_completion_median = statistics.median(ns_completion_tokens)
    st_completion_median = statistics.median(st_completion_tokens)
    max_completion_median = max(ns_completion_median, st_completion_median)
    assert abs(ns_completion_median - st_completion_median) <= (
        median_completion_token_rtol * max_completion_median
    ), (
        "Median completion_tokens differ too much — "
        f"non_stream={ns_completion_median}, stream={st_completion_median} "
        f"(rtol={median_completion_token_rtol})"
    )

    max_audio_duration_total = max(ns_audio_duration_total, st_audio_duration_total)
    assert abs(ns_audio_duration_total - st_audio_duration_total) <= (
        total_audio_duration_rtol * max_audio_duration_total
    ), (
        "Total audio_duration_s differs too much — "
        f"non_stream={ns_audio_duration_total}, stream={st_audio_duration_total} "
        f"(rtol={total_audio_duration_rtol})"
    )


def assert_wer_results(
    results: dict,
    max_corpus_wer: float,
    max_per_sample_wer: float,
) -> None:
    """Verify WER results are within thresholds."""
    summary = results["summary"]
    per_sample = results["per_sample"]

    failed_details = [
        f"  sample {s['id']}: {s.get('error')}"
        for s in per_sample
        if not s.get("is_success", True)
    ]
    assert summary["evaluated"] == summary["total_samples"], (
        f"Only {summary['evaluated']}/{summary['total_samples']} samples evaluated, "
        f"{summary['skipped']} skipped.\n"
        f"Per-sample errors:\n" + "\n".join(failed_details)
    )

    assert summary["wer_corpus"] <= max_corpus_wer, (
        f"Corpus WER {summary['wer_corpus']:.4f} ({summary['wer_corpus'] * 100:.2f}%) "
        f"> threshold {max_corpus_wer} ({max_corpus_wer * 100:.0f}%)"
    )

    assert summary["n_above_50_pct_wer"] == 0, (
        f"{summary['n_above_50_pct_wer']} samples have >50% WER — "
        f"expected 0 catastrophic failures"
    )

    for sample in per_sample:
        assert sample[
            "is_success"
        ], f"Sample {sample['id']} failed: {sample.get('error')}"
        if sample["wer"] is not None:
            assert sample["wer"] <= max_per_sample_wer, (
                f"Sample {sample['id']} WER {sample['wer']:.4f} "
                f"> {max_per_sample_wer}"
            )
