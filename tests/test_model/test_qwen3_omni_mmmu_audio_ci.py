# SPDX-License-Identifier: Apache-2.0
"""MMMU audio consistency CI for Qwen3-Omni (Text+Image → Text+Audio, Talker ON).

Evaluates text-audio consistency by comparing the model's text output
with ASR transcription of its audio output on MMMU image-QA tasks.

Usage:
    pytest tests/test_model/test_qwen3_omni_mmmu_audio_ci.py -v -s -x
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.eval.mmmu import MMMUEvalConfig, run_mmmu_eval
from tests.utils import (
    apply_slack,
    assert_speed_thresholds,
    assert_wer_results,
    disable_proxy,
    find_free_port,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

MAX_SAMPLES = 5
MAX_TOKENS = 50
STARTUP_TIMEOUT = 900

# TODO(Yifei): Support higher concurrency and larger sample size
# once the Qwen3-Omni pipeline is optimized for concurrent requests.
CONCURRENCY = 1

# WER thresholds — text-audio consistency.
MMMU_AUDIO_WER_MAX_CORPUS = 0.08
MMMU_AUDIO_WER_MAX_PER_SAMPLE = 0.15

# TODO(Yifei): Tighten after gathering data from CI hardware.
_MMMU_AUDIO_P95 = {
    1: {
        "throughput_qps": 0.02,
        "tok_per_s_agg": 1.0,
        "latency_mean_s": 60.0,
        "rtf_mean": 5.0,
    },
}
MMMU_AUDIO_THRESHOLDS = apply_slack(_MMMU_AUDIO_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the Qwen3-Omni speech server (talker ON) and wait until healthy."""
    port = find_free_port()
    log_dir = tmp_path_factory.mktemp("server_logs")
    log_file = log_dir / "server.log"
    with open(log_file, "w") as log_handle:
        cmd = [
            sys.executable,
            "examples/run_qwen3_omni_speech_server.py",
            "--model-path",
            MODEL_PATH,
            "--gpu-thinker",
            "0",
            "--gpu-talker",
            "1",
            "--gpu-code-predictor",
            "1",
            "--gpu-code2wav",
            "1",
            "--port",
            str(port),
            "--model-name",
            "qwen3-omni",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        proc.port = port

        api_base = f"http://localhost:{port}"
        try:
            with disable_proxy():
                wait_for_service(
                    api_base,
                    timeout=STARTUP_TIMEOUT,
                    server_process=proc,
                    server_log_file=log_file,
                    health_body_contains="healthy",
                )
        except TimeoutError:
            stop_server(proc)
            server_log = log_file.read_text()
            pytest.fail(
                f"Server did not become healthy within {STARTUP_TIMEOUT}s.\n"
                f"{server_log}"
            )
        except RuntimeError as exc:
            pytest.fail(str(exc))

        yield proc

        stop_server(proc)


@pytest.mark.benchmark
def test_mmmu_audio_wer_and_speed(
    server_process: subprocess.Popen,
    tmp_path: Path,
) -> None:
    """Run MMMU eval with audio and assert WER and speed meet thresholds."""
    config = MMMUEvalConfig(
        model="qwen3-omni",
        port=server_process.port,
        max_samples=MAX_SAMPLES,
        max_tokens=MAX_TOKENS,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "mmmu_audio"),
        enable_audio=True,
    )
    results = asyncio.run(run_mmmu_eval(config))

    # Assert WER
    assert "wer" in results, "Audio WER results missing from eval output"
    assert_wer_results(
        results["wer"], MMMU_AUDIO_WER_MAX_CORPUS, MMMU_AUDIO_WER_MAX_PER_SAMPLE
    )

    # Assert speed
    speed = results["speed"]
    assert_speed_thresholds(speed, MMMU_AUDIO_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
