# SPDX-License-Identifier: Apache-2.0
"""S2-Pro TTS benchmark CI: starts server, runs benchmarks, asserts thresholds.

Usage:
    pytest tests/test_model/test_s2pro_benchmark.py -s -x
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils import MODEL_PATH, dataset_dir, server_process  # noqa: F401

MAX_SAMPLES = 10
BENCHMARK_TIMEOUT = 600  # seconds

# Thresholds (15-25% margin from 4-run data)
# Reference: https://github.com/sgl-project/sglang-omni/issues/193
VC_NON_STREAM_MIN_TOK_PER_S = 80
VC_NON_STREAM_MAX_RTF = 2.85
VC_STREAM_MAX_LATENCY_S = 12.5
VC_STREAM_MIN_THROUGHPUT_QPS = 0.08
PLAIN_NON_STREAM_MIN_TOK_PER_S = 80
PLAIN_NON_STREAM_MAX_RTF = 0.35
PLAIN_STREAM_MAX_LATENCY_S = 4.0
PLAIN_STREAM_MIN_THROUGHPUT_QPS = 0.25


def _run_benchmark(
    port: int,
    testset: str,
    output_dir: str,
    extra_args: list[str] | None = None,
) -> dict:
    """Run benchmark_tts_speed as subprocess and return the full results dict.

    Returns the complete JSON results containing both ``summary`` and
    ``per_request`` entries.
    """
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.performance.tts.benchmark_tts_speed",
        "--model",
        MODEL_PATH,
        "--port",
        str(port),
        "--testset",
        testset,
        "--max-samples",
        str(MAX_SAMPLES),
        "--output-dir",
        output_dir,
    ]
    if extra_args:
        cmd.extend(extra_args)

    # Strip proxy env vars so the benchmark subprocess can reach localhost directly.
    # The CI environment may have HTTP_PROXY set (e.g. China-region runners), which
    # causes requests to localhost to be routed through the proxy and fail.
    proxy_keys = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    env = {k: v for k, v in os.environ.items() if k.lower() not in proxy_keys}

    proc_result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=BENCHMARK_TIMEOUT,
        env=env,
    )
    assert proc_result.returncode == 0, (
        f"Benchmark failed (rc={proc_result.returncode}).\n"
        f"stdout:\n{proc_result.stdout}\nstderr:\n{proc_result.stderr}"
    )

    results_path = Path(output_dir) / "speed_results.json"
    assert results_path.exists(), f"Results file not found: {results_path}"

    with open(results_path) as f:
        speed_results = json.load(f)
    assert (
        "summary" in speed_results
    ), f"Missing 'summary' key in results. Keys: {list(speed_results.keys())}"
    assert (
        "per_request" in speed_results
    ), f"Missing 'per_request' key in results. Keys: {list(speed_results.keys())}"
    return speed_results


def _assert_summary_metrics(summary: dict) -> None:
    """Verify summary-level sanity invariants that must hold for every run."""
    assert (
        summary["failed_requests"] == 0
    ), f"Expected 0 failed requests, got {summary['failed_requests']}"
    assert (
        summary["audio_duration_mean_s"] > 0
    ), f"Expected positive audio duration, got {summary['audio_duration_mean_s']}"
    assert (
        summary.get("gen_tokens_mean", 0) > 0
    ), f"Expected positive gen_tokens_mean, got {summary.get('gen_tokens_mean', 0)}"
    assert (
        summary.get("prompt_tokens_mean", 0) > 0
    ), f"Expected positive prompt_tokens_mean, got {summary.get('prompt_tokens_mean', 0)}"


def _assert_per_request_fields(per_request: list[dict]) -> None:
    """Verify every request has valid audio, prompt_tokens, and completion_tokens."""
    for req in per_request:
        rid = req["id"]
        assert req["is_success"], f"Request {rid} failed: {req.get('error')}"
        assert (
            req["audio_duration_s"] is not None and req["audio_duration_s"] > 0
        ), f"Request {rid}: audio_duration_s={req['audio_duration_s']}, expected > 0"
        assert (
            req["prompt_tokens"] is not None and req["prompt_tokens"] > 0
        ), f"Request {rid}: prompt_tokens={req['prompt_tokens']}, expected > 0"
        assert (
            req["completion_tokens"] is not None and req["completion_tokens"] > 0
        ), f"Request {rid}: completion_tokens={req['completion_tokens']}, expected > 0"


def _assert_streaming_consistency(
    non_stream_requests: list[dict],
    stream_requests: list[dict],
    *,
    completion_token_rtol: float = 0.10,
    audio_duration_rtol: float = 0.12,
) -> None:
    """Assert per-request metrics are close between streaming and non-streaming.

    The model is not perfectly deterministic across separate inference passes,
    so completion_tokens and audio_duration are compared with a relative
    tolerance.  prompt_tokens must still match exactly (input-dependent only).
    """
    ns_by_id = {r["id"]: r for r in non_stream_requests}
    st_by_id = {r["id"]: r for r in stream_requests}
    assert set(ns_by_id) == set(st_by_id), (
        f"Request ID mismatch: "
        f"non_stream={sorted(ns_by_id)}, stream={sorted(st_by_id)}"
    )
    for rid in sorted(ns_by_id):
        ns, st = ns_by_id[rid], st_by_id[rid]

        # completion_tokens: allow relative tolerance
        ns_ct, st_ct = ns["completion_tokens"], st["completion_tokens"]
        max_ct = max(ns_ct, st_ct)
        assert abs(ns_ct - st_ct) <= completion_token_rtol * max_ct, (
            f"Request {rid}: completion_tokens differ too much — "
            f"non_stream={ns_ct}, stream={st_ct} "
            f"(rtol={completion_token_rtol})"
        )

        assert ns["prompt_tokens"] == st["prompt_tokens"], (
            f"Request {rid}: prompt_tokens mismatch — "
            f"non_stream={ns['prompt_tokens']}, stream={st['prompt_tokens']}"
        )

        # audio_duration_s: allow relative tolerance
        ns_ad, st_ad = ns["audio_duration_s"], st["audio_duration_s"]
        max_ad = max(ns_ad, st_ad)
        assert abs(ns_ad - st_ad) <= audio_duration_rtol * max_ad, (
            f"Request {rid}: audio_duration_s differ too much — "
            f"non_stream={ns_ad}, stream={st_ad} "
            f"(rtol={audio_duration_rtol})"
        )


# Module-level storage so consistency tests can compare across streaming modes
# without re-running benchmarks.  Keys: "vc_nonstream", "vc_stream", etc.
_per_request_store: dict[str, list[dict]] = {}


@pytest.mark.benchmark
def test_voice_cloning_non_streaming(
    server_process: tuple[subprocess.Popen, int],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Voice cloning (non-streaming): tok/s >= 80, RTF <= 2.8."""
    _, port = server_process
    results = _run_benchmark(
        port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "vc_nonstream"),
    )
    summary, per_request = results["summary"], results["per_request"]
    _assert_summary_metrics(summary)
    _assert_per_request_fields(per_request)
    _per_request_store["vc_nonstream"] = per_request
    assert (
        summary["tok_per_s_agg"] >= VC_NON_STREAM_MIN_TOK_PER_S
    ), f"tok_per_s_agg {summary['tok_per_s_agg']} < {VC_NON_STREAM_MIN_TOK_PER_S}"
    assert (
        summary["rtf_mean"] <= VC_NON_STREAM_MAX_RTF
    ), f"rtf_mean {summary['rtf_mean']} > {VC_NON_STREAM_MAX_RTF}"


@pytest.mark.benchmark
def test_voice_cloning_streaming(
    server_process: tuple[subprocess.Popen, int],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Voice cloning (streaming): latency <= 12.5s, throughput >= 0.08 qps."""
    _, port = server_process
    results = _run_benchmark(
        port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "vc_stream"),
        ["--stream"],
    )
    summary, per_request = results["summary"], results["per_request"]
    _assert_summary_metrics(summary)
    _assert_per_request_fields(per_request)
    _per_request_store["vc_stream"] = per_request
    assert (
        summary["latency_mean_s"] <= VC_STREAM_MAX_LATENCY_S
    ), f"latency_mean_s {summary['latency_mean_s']} > {VC_STREAM_MAX_LATENCY_S}"
    assert (
        summary["throughput_qps"] >= VC_STREAM_MIN_THROUGHPUT_QPS
    ), f"throughput_qps {summary['throughput_qps']} < {VC_STREAM_MIN_THROUGHPUT_QPS}"


@pytest.mark.benchmark
def test_plain_tts_non_streaming(
    server_process: tuple[subprocess.Popen, int],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Plain TTS (non-streaming): tok/s >= 80, RTF <= 0.35."""
    _, port = server_process
    results = _run_benchmark(
        port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "plain_nonstream"),
        ["--no-ref-audio"],
    )
    summary, per_request = results["summary"], results["per_request"]
    _assert_summary_metrics(summary)
    _assert_per_request_fields(per_request)
    _per_request_store["plain_nonstream"] = per_request
    assert (
        summary["tok_per_s_agg"] >= PLAIN_NON_STREAM_MIN_TOK_PER_S
    ), f"tok_per_s_agg {summary['tok_per_s_agg']} < {PLAIN_NON_STREAM_MIN_TOK_PER_S}"
    assert (
        summary["rtf_mean"] <= PLAIN_NON_STREAM_MAX_RTF
    ), f"rtf_mean {summary['rtf_mean']} > {PLAIN_NON_STREAM_MAX_RTF}"


@pytest.mark.benchmark
def test_plain_tts_streaming(
    server_process: tuple[subprocess.Popen, int],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Plain TTS (streaming): latency <= 4.0s, throughput >= 0.25 qps."""
    _, port = server_process
    results = _run_benchmark(
        port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "plain_stream"),
        ["--no-ref-audio", "--stream"],
    )
    summary, per_request = results["summary"], results["per_request"]
    _assert_summary_metrics(summary)
    _assert_per_request_fields(per_request)
    _per_request_store["plain_stream"] = per_request
    assert (
        summary["latency_mean_s"] <= PLAIN_STREAM_MAX_LATENCY_S
    ), f"latency_mean_s {summary['latency_mean_s']} > {PLAIN_STREAM_MAX_LATENCY_S}"
    assert (
        summary["throughput_qps"] >= PLAIN_STREAM_MIN_THROUGHPUT_QPS
    ), f"throughput_qps {summary['throughput_qps']} < {PLAIN_STREAM_MIN_THROUGHPUT_QPS}"


# --- Cross-mode consistency tests (must run after the 4 individual tests) ---


@pytest.mark.benchmark
def test_voice_cloning_streaming_consistency(server_process) -> None:
    """Streaming vs non-streaming must produce identical per-request metrics for VC."""
    ns = _per_request_store.get("vc_nonstream")
    st = _per_request_store.get("vc_stream")
    assert ns is not None, "vc_nonstream results missing — did that test run first?"
    assert st is not None, "vc_stream results missing — did that test run first?"
    _assert_streaming_consistency(ns, st)


@pytest.mark.benchmark
def test_plain_tts_streaming_consistency(server_process) -> None:
    """Streaming vs non-streaming must produce identical per-request metrics for TTS."""
    ns = _per_request_store.get("plain_nonstream")
    st = _per_request_store.get("plain_stream")
    assert ns is not None, "plain_nonstream results missing — did that test run first?"
    assert st is not None, "plain_stream results missing — did that test run first?"
    _assert_streaming_consistency(ns, st)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
