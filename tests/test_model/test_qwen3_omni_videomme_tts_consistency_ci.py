# SPDX-License-Identifier: Apache-2.0
"""Video-MME TTS consistency CI for Qwen3-Omni.

Runs a small Video-MME subset through Text+Video -> Text+Audio, then checks
text answer accuracy, text-audio WER, and basic speed metrics.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.videomme import VideoMMESample, load_videomme_samples
from benchmarks.eval.benchmark_omni_videomme import (
    VideoMMEEvalConfig,
    run_videomme_eval,
)
from sglang_omni.utils import find_available_port
from tests.utils import (
    ServerHandle,
    assert_speed_thresholds,
    assert_wer_results,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

CONCURRENCY = 1
MAX_SAMPLES = 3
MAX_TOKENS = 256
STARTUP_TIMEOUT = 900
SHORT_ANSWER_PROMPT = (
    "For the audio response, answer briefly in one sentence and end with "
    "'Answer: $LETTER'. Do not include step-by-step reasoning."
)

VIDEOMME_TTS_MIN_ACCURACY = 0.20
VIDEOMME_TTS_MAX_FAILED = 0
VIDEOMME_TTS_WER_MAX_CORPUS = 0.25
VIDEOMME_TTS_WER_MAX_PER_SAMPLE = 0.50

VIDEOMME_TTS_THRESHOLDS = {
    1: {
        "throughput_qps_min": 0.001,
        "tok_per_s_agg_min": 0.1,
        "latency_mean_s_max": 240.0,
        "rtf_mean_max": 12.0,
    },
}


def _load_short_answer_samples() -> list[VideoMMESample]:
    samples = load_videomme_samples(
        max_samples=MAX_SAMPLES,
        repo_id=DATASETS["videomme-ci-50"],
    )
    for sample in samples:
        sample.prompt = f"{sample.prompt}\n{SHORT_ANSWER_PROMPT}"
    return samples


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the Qwen3-Omni speech server and wait until healthy."""
    port = find_available_port()
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
        "--thinker-max-seq-len",
        "32768",
        "--thinker-mem-fraction-static",
        "0.78",
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    yield ServerHandle(proc=proc, port=port)
    stop_server(proc)


@pytest.mark.benchmark
def test_videomme_tts_accuracy_wer_and_speed(
    server_process: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run Video-MME with Talker enabled and assert text/audio metrics."""
    config = VideoMMEEvalConfig(
        model="qwen3-omni",
        port=server_process.port,
        max_samples=MAX_SAMPLES,
        max_tokens=MAX_TOKENS,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videomme_audio"),
        repo_id=DATASETS["videomme-ci-50"],
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        enable_audio=True,
        asr_device="cuda:0",
        disable_tqdm=True,
    )
    results = asyncio.run(
        run_videomme_eval(config, samples=_load_short_answer_samples())
    )

    summary = results["summary"]
    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    assert failed <= VIDEOMME_TTS_MAX_FAILED, (
        f"Video-MME TTS had {failed}/{total} failed requests, "
        f"which exceeds the threshold {VIDEOMME_TTS_MAX_FAILED}"
    )
    assert summary["accuracy"] >= VIDEOMME_TTS_MIN_ACCURACY, (
        f"Video-MME TTS accuracy {summary['accuracy']:.4f} "
        f"({summary['accuracy'] * 100:.1f}%) < "
        f"threshold {VIDEOMME_TTS_MIN_ACCURACY} "
        f"({VIDEOMME_TTS_MIN_ACCURACY * 100:.0f}%)"
    )

    assert "wer" in results, "Audio WER results missing from Video-MME TTS output"
    assert_wer_results(
        results["wer"], VIDEOMME_TTS_WER_MAX_CORPUS, VIDEOMME_TTS_WER_MAX_PER_SAMPLE
    )
    assert_speed_thresholds(results["speed"], VIDEOMME_TTS_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
