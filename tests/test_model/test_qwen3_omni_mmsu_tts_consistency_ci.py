# SPDX-License-Identifier: Apache-2.0
"""MMSU TTS consistency CI for Qwen3-Omni (Text + Audio → Text+Audio, Talker ON).

Evaluates text-audio consistency by comparing the model's text output with
ASR transcription of its audio output on MMSU audio-QA tasks. Uses a
chain-of-thought prompt (mirroring MMMU style) so the model reasons step
by step before giving the final answer letter, producing longer responses
more suitable for WER evaluation.

Usage:
    pytest tests/test_model/test_qwen3_omni_mmsu_tts_consistency_ci.py -v -s -x

Author:
    Yifei Gao https://github.com/PasserBy4
    Huapeng Zhou https://github.com/PopSoda2002
    Chenyang Zhao https://github.com/zhaochenyang20
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu
from sglang_omni.utils import find_available_port
from tests.utils import (
    ServerHandle,
    apply_slack,
    assert_speed_thresholds,
    assert_wer_results,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# TODO (Yifei): enable larger subset/max_tokens when concurrency > 1 is supported.
MAX_SAMPLES = 5
MAX_TOKENS = 50
STARTUP_TIMEOUT = 900

# Note (Yifei): Concurrency=1 only for now — code_predictor and code2wav
# modules serialize GPU access, so they run serially even when concurrency > 1.
CONCURRENCY = 1

# Note (Yifei): Use chain-of-thought prompt (mirroring MMMU style) instead of the default
# "Reply with only A, B, C, or D." to elicit longer responses for WER.
MMSU_TTS_PROMPT = (
    "Listen to the audio and answer the multiple-choice question. "
    "Think step by step before answering. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) "
    "where LETTER is one of the options."
)

# TODO (Yifei): update thresholds when concurrency > 1 is supported.

MMSU_AUDIO_WER_MAX_CORPUS = 0.12
MMSU_AUDIO_WER_MAX_PER_SAMPLE = 0.36

_MMSU_AUDIO_P95 = {
    1: {
        "throughput_qps": 0.04,
        "tok_per_s_agg": 1.80,
        "latency_mean_s": 27.376,
        "rtf_mean": 1.5734,
    },
}
MMSU_AUDIO_THRESHOLDS = apply_slack(_MMSU_AUDIO_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
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
        warmup=1,
        max_concurrency=CONCURRENCY,
        request_rate=float("inf"),
        save_audio=True,
        disable_tqdm=True,
        seed=None,
        lang="en",
        asr_device="cuda:0",
    )


@pytest.mark.benchmark
def test_mmsu_audio_wer_and_speed(
    server_process: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run MMSU eval with audio and assert WER and speed meet thresholds."""
    args = _build_args(server_process.port, str(tmp_path / "mmsu_audio"))

    # NOTE (Yifei):
    # Regression guard for issue #299: append a dup of sample[0] so the
    # audio encoder sees a cached+non-cached mixed batch. Inert at
    # concurrency=1; starts to take effect once concurrency is raised.
    base = load_mmsu_samples(max_samples=MAX_SAMPLES, repo_id=DATASETS["mmsu-ci-2000"])
    dup = deepcopy(base[0])
    dup.sample_id = f"{base[0].sample_id}__dup"
    samples = [*base, dup]

    results = asyncio.run(run_mmsu(args, samples=samples))

    failed = results["accuracy"].get("failed_samples", 0)
    total = results["accuracy"].get("total_samples", 0)
    assert failed == 0, (
        f"MMSU TTS consistency had {failed}/{total} failed requests "
        f"(timeouts or empty responses); any failure fails the test"
    )

    assert "wer" in results, "Audio WER results missing from eval output"
    assert_wer_results(
        results["wer"], MMSU_AUDIO_WER_MAX_CORPUS, MMSU_AUDIO_WER_MAX_PER_SAMPLE
    )

    assert_speed_thresholds(results["speed"], MMSU_AUDIO_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
