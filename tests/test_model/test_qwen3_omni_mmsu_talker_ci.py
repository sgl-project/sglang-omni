# SPDX-License-Identifier: Apache-2.0
"""MMSU Talker CI for Qwen3-Omni (Text + Audio → Text+Audio, Talker ON).

Evaluates text-audio consistency by comparing the model's text output with
ASR transcription of its audio output on MMSU audio-QA tasks. Uses a
chain-of-thought prompt (mirroring MMMU style) so the model reasons step
by step before giving the final answer letter, producing longer responses
more suitable for WER evaluation.

Usage:
    pytest tests/test_model/test_qwen3_omni_mmsu_talker_ci.py -v -s -x

Author:
    Yifei Gao https://github.com/PasserBy4
    Huapeng Zhou https://github.com/PopSoda2002
    Chenyang Zhao https://github.com/zhaochenyang20
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu
from benchmarks.metrics.mmsu import print_mmsu_summary
from benchmarks.metrics.wer import print_wer_summary
from sglang_omni.utils import find_available_port
from tests.utils import (
    ServerHandle,
    apply_slack,
    assert_speed_thresholds,
    assert_wer_partitioned,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

MAX_SAMPLES = 20
MAX_TOKENS = 256
STARTUP_TIMEOUT = 300

CONCURRENCY = 8

# Note (Yifei): "2-3 sentences" floor prevents terse "Answer: X" replies that
# would starve the WER signal; the 120-word cap keeps p95 output well under
# MAX_TOKENS so the final 'Answer: $LETTER' line is never truncated.
MMSU_TTS_PROMPT = (
    "Listen to the audio and answer the multiple-choice question.\n"
    "Briefly explain your reasoning in 2-3 sentences, then on a new final "
    "line output exactly:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of the options.\n"
    "Do not exceed 120 words in total."
)

# Threshold reference: https://github.com/sgl-project/sglang-omni/pull/382#issuecomment-4366925373

# Accuracy floor — audio-mode MMSU.
MMSU_AUDIO_MIN_ACCURACY = 0.60

# WER thresholds use a partitioned view of the per-sample distribution:
#  - corpus WER over the "sane" subset (per-sample WER <= 50%)
#  - count of catastrophic failures (per-sample WER > 50%)
MMSU_AUDIO_WER_BELOW_50_CORPUS_MAX = 0.04
# Relaxed in V1 refactor: v0=0 → v1=1.
MMSU_AUDIO_N_ABOVE_50_MAX = 1

_MMSU_AUDIO_P95 = {
    8: {
        "throughput_qps": 0.266,
        "tok_per_s_agg": 2.5,
        "latency_mean_s": 24.663,
        "rtf_mean": 1.3743,
    },
}
MMSU_AUDIO_THRESHOLDS = apply_slack(_MMSU_AUDIO_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    port = find_available_port()
    log_file = server_log_file(tmp_path_factory)
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
    yield ServerHandle(proc=proc, port=port)
    stop_server(proc)


def _build_args(port: int, output_dir: str) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=None,
        host="localhost",
        port=port,
        model="qwen3-omni",
        modalities="text+audio",
        output_dir=output_dir,
        max_samples=MAX_SAMPLES,
        task_names=None,
        categories=None,
        prompt=MMSU_TTS_PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        warmup=0,
        max_concurrency=CONCURRENCY,
        request_rate=float("inf"),
        save_audio=True,
        disable_tqdm=False,
        seed=None,
        lang="en",
        asr_device="cuda:0",
        timeout_s=500,
    )


@pytest.mark.benchmark
def test_mmsu_audio_wer_and_speed(
    server_process: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run MMSU eval with audio and assert WER and speed meet thresholds."""
    args = _build_args(server_process.port, str(tmp_path / "mmsu_audio"))

    samples = load_mmsu_samples(
        max_samples=MAX_SAMPLES, repo_id=DATASETS["mmsu-ci-2000"]
    )

    results = asyncio.run(run_mmsu(args, samples=samples))

    print_mmsu_summary(results["accuracy"], args.model, speed_metrics=results["speed"])
    if "wer" in results:
        print_wer_summary(results["wer"]["summary"], args.model)

    failed = results["accuracy"].get("failed_samples", 0)
    total = results["accuracy"].get("total_samples", 0)
    assert failed == 0, (
        f"MMSU Talker had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test"
    )

    accuracy = results["accuracy"]["overall_accuracy"]
    assert accuracy >= MMSU_AUDIO_MIN_ACCURACY, (
        f"MMSU audio accuracy {accuracy:.4f} ({accuracy * 100:.1f}%) < "
        f"threshold {MMSU_AUDIO_MIN_ACCURACY} "
        f"({MMSU_AUDIO_MIN_ACCURACY * 100:.0f}%)"
    )

    assert "wer" in results, "Audio WER results missing from eval output"
    assert_wer_partitioned(
        results["wer"],
        max_wer_below_50_corpus=MMSU_AUDIO_WER_BELOW_50_CORPUS_MAX,
        max_n_above_50=MMSU_AUDIO_N_ABOVE_50_MAX,
    )

    assert_speed_thresholds(results["speed"], MMSU_AUDIO_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
