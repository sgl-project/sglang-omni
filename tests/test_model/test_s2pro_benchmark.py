# SPDX-License-Identifier: Apache-2.0
"""S2-Pro TTS benchmark CI: starts server, runs benchmarks, asserts thresholds.

Usage:
    pytest tests/test_model/test_s2pro_benchmark.py -s -x
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from tests.test_model.helpers import disable_proxy
from tests.utils import find_free_port

logger = logging.getLogger(__name__)

MODEL_PATH = "fishaudio/s2-pro"
CONFIG_PATH = "examples/configs/s2pro_tts.yaml"
DATASET_REPO = "zhaochenyang20/seed-tts-eval-mini"
MAX_SAMPLES = 10

STARTUP_TIMEOUT = 600  # seconds
BENCHMARK_TIMEOUT = 600  # seconds

# Thresholds (15-25% margin from 4-run data)
# Reference: https://github.com/sgl-project/sglang-omni/issues/193
VC_NON_STREAM_MIN_TOK_PER_S = 80
VC_NON_STREAM_MAX_RTF = 2.8
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
    """Run benchmark_tts_speed as subprocess and return the summary dict."""
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
    return speed_results["summary"]


@pytest.fixture(scope="module")
def server_port() -> int:
    """Allocate a free TCP port for the server."""
    return find_free_port()


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download the mini seed-tts-eval dataset via huggingface_hub."""
    from huggingface_hub import snapshot_download

    cache_dir = tmp_path_factory.mktemp("seed_tts_eval")
    path = snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        local_dir=str(cache_dir / "data"),
    )
    return Path(path)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory, server_port: int):
    """Start the s2-pro server and wait until healthy."""
    log_dir = tmp_path_factory.mktemp("server_logs")
    log_file = log_dir / "server.log"
    log_handle = open(log_file, "w")

    try:
        cmd = [
            sys.executable,
            "-m",
            "sglang_omni.cli.cli",
            "serve",
            "--model-path",
            MODEL_PATH,
            "--config",
            CONFIG_PATH,
            "--port",
            str(server_port),
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        api_base = f"http://localhost:{server_port}"
        is_healthy = False
        for _ in range(STARTUP_TIMEOUT):
            if proc.poll() is not None:
                log_handle.close()
                server_log = log_file.read_text()
                pytest.fail(f"Server exited with code {proc.returncode}.\n{server_log}")
            try:
                with disable_proxy():
                    resp = requests.get(f"{api_base}/health", timeout=2)
                if resp.status_code == 200 and "healthy" in resp.text:
                    is_healthy = True
                    break
            except requests.RequestException as exc:
                logger.debug("Health check failed (transient): %s", exc)
            time.sleep(1)

        if not is_healthy:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
            log_handle.close()
            server_log = log_file.read_text()
            pytest.fail(
                f"Server did not become healthy within {STARTUP_TIMEOUT}s.\n{server_log}"
            )

        yield proc

        # Teardown
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
    finally:
        log_handle.close()


@pytest.mark.benchmark
def test_voice_cloning_non_streaming(
    server_process: subprocess.Popen,
    server_port: int,
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Voice cloning (non-streaming): tok/s >= 80, RTF <= 2.8."""
    summary = _run_benchmark(
        server_port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "vc_nonstream"),
    )
    assert (
        summary["tok_per_s_agg"] >= VC_NON_STREAM_MIN_TOK_PER_S
    ), f"tok_per_s_agg {summary['tok_per_s_agg']} < {VC_NON_STREAM_MIN_TOK_PER_S}"
    assert (
        summary["rtf_mean"] <= VC_NON_STREAM_MAX_RTF
    ), f"rtf_mean {summary['rtf_mean']} > {VC_NON_STREAM_MAX_RTF}"


@pytest.mark.benchmark
def test_voice_cloning_streaming(
    server_process: subprocess.Popen,
    server_port: int,
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Voice cloning (streaming): latency <= 12.5s, throughput >= 0.08 qps."""
    summary = _run_benchmark(
        server_port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "vc_stream"),
        ["--stream"],
    )
    assert (
        summary["latency_mean_s"] <= VC_STREAM_MAX_LATENCY_S
    ), f"latency_mean_s {summary['latency_mean_s']} > {VC_STREAM_MAX_LATENCY_S}"
    assert (
        summary["throughput_qps"] >= VC_STREAM_MIN_THROUGHPUT_QPS
    ), f"throughput_qps {summary['throughput_qps']} < {VC_STREAM_MIN_THROUGHPUT_QPS}"


@pytest.mark.benchmark
def test_plain_tts_non_streaming(
    server_process: subprocess.Popen,
    server_port: int,
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Plain TTS (non-streaming): tok/s >= 80, RTF <= 0.35."""
    summary = _run_benchmark(
        server_port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "plain_nonstream"),
        ["--no-ref-audio"],
    )
    assert (
        summary["tok_per_s_agg"] >= PLAIN_NON_STREAM_MIN_TOK_PER_S
    ), f"tok_per_s_agg {summary['tok_per_s_agg']} < {PLAIN_NON_STREAM_MIN_TOK_PER_S}"
    assert (
        summary["rtf_mean"] <= PLAIN_NON_STREAM_MAX_RTF
    ), f"rtf_mean {summary['rtf_mean']} > {PLAIN_NON_STREAM_MAX_RTF}"


@pytest.mark.benchmark
def test_plain_tts_streaming(
    server_process: subprocess.Popen,
    server_port: int,
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Plain TTS (streaming): latency <= 4.0s, throughput >= 0.25 qps."""
    summary = _run_benchmark(
        server_port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "plain_stream"),
        ["--no-ref-audio", "--stream"],
    )
    assert (
        summary["latency_mean_s"] <= PLAIN_STREAM_MAX_LATENCY_S
    ), f"latency_mean_s {summary['latency_mean_s']} > {PLAIN_STREAM_MAX_LATENCY_S}"
    assert (
        summary["throughput_qps"] >= PLAIN_STREAM_MIN_THROUGHPUT_QPS
    ), f"throughput_qps {summary['throughput_qps']} < {PLAIN_STREAM_MIN_THROUGHPUT_QPS}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
