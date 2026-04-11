# SPDX-License-Identifier: Apache-2.0
"""MMMU TTS consistency CI for Qwen3-Omni (Text+Image → Text+Audio, Talker ON).

Evaluates text-audio consistency by comparing the model's text output
with ASR transcription of its audio output on MMMU image-QA tasks.

Usage:
    pytest tests/test_model/test_qwen3_omni_mmmu_tts_consistency_ci.py -v -s -x

Note (Chenyang):
    Currently due to the performance limitation of the Talker, we run limited
    samples for the MMMU tts CI.
    reference: https://github.com/sgl-project/sglang-omni/issues/276

Author:
    Yifei Gao https://github.com/PasserBy4
    Chenyang Zhao https://github.com/zhaochenyang20
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
from tests.utils import (
    apply_slack,
    assert_speed_thresholds,
    assert_wer_results,
    find_free_port,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

MAX_SAMPLES = 5
MAX_TOKENS = 50
STARTUP_TIMEOUT = 900

# Note (Yifei): Concurrency=1 only for now — code_predictor and code2wav
# modules serialize GPU access, so they run serially even when concurrency > 1.

CONCURRENCY = 1

# WER thresholds — text-audio consistency.
MMMU_AUDIO_WER_MAX_CORPUS = 0.10
MMMU_AUDIO_WER_MAX_PER_SAMPLE = 0.18

# Note (Yifei, Chenyang): Thresholds reference
# https://github.com/sgl-project/sglang-omni/pull/265#issuecomment-4228251028

_MMMU_AUDIO_P95 = {
    1: {
        "throughput_qps": 0.034,
        "tok_per_s_agg": 1.7,
        "latency_mean_s": 29.66,
        "rtf_mean": 2.02,
    },
}
MMMU_AUDIO_THRESHOLDS = apply_slack(_MMMU_AUDIO_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the Qwen3-Omni speech server (talker ON) and wait until healthy."""
    port = find_free_port()
    log_file = tmp_path_factory.mktemp("server_logs") / "server.log"
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
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    proc.port = port
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
        repo_id=DATASETS["mmmu-ci-50"],
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
