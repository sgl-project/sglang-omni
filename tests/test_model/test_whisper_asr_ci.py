# SPDX-License-Identifier: Apache-2.0
"""Whisper ASR correctness CI for SGLang Omni.

The test uses the first 20 English SeedTTS samples as a lightweight speech
corpus. It compares normalized transcriptions from the SGLang Omni Whisper
server against the dataset reference text.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import requests
from jiwer import process_words

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.tasks.tts import normalize_text
from tests.utils import (
    disable_proxy,
    no_proxy_env,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

WHISPER_MODEL_PATH = "openai/whisper-large-v3"
SEEDTTS_ASR_CORRECTNESS_SAMPLES = 20
SEEDTTS_ASR_CORPUS_WER_MAX = 0.01
SEEDTTS_ASR_SAMPLE_WER_MAX = 0.20
STARTUP_TIMEOUT = 600
REQUEST_TIMEOUT = 300
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Whisper ASR correctness CI")


@pytest.fixture(scope="module")
def seedtts_en_samples() -> list[SampleInput]:
    return load_seedtts_samples(
        DATASETS["seedtts"],
        max_samples=SEEDTTS_ASR_CORRECTNESS_SAMPLES,
        split="en",
    )


def _start_whisper_server(tmp_path_factory: pytest.TempPathFactory):
    from sglang_omni.utils import find_available_port

    port = find_available_port()
    log_file = server_log_file(tmp_path_factory, prefix="whisper_asr_logs")
    env = no_proxy_env()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        WHISPER_MODEL_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    proc = start_server_from_cmd(
        cmd,
        log_file,
        port,
        timeout=STARTUP_TIMEOUT,
        env=env,
    )
    return proc, port


def _transcribe_with_omni(port: int, sample: SampleInput) -> tuple[str, float]:
    start = time.perf_counter()
    with open(sample.ref_audio, "rb") as audio_file:
        with disable_proxy():
            response = requests.post(
                f"http://127.0.0.1:{port}/v1/audio/transcriptions",
                data={
                    "model": WHISPER_MODEL_PATH,
                    "language": "en",
                    "temperature": "0",
                },
                files={
                    "file": (
                        os.path.basename(sample.ref_audio),
                        audio_file,
                        "audio/wav",
                    )
                },
                timeout=REQUEST_TIMEOUT,
            )
    response.raise_for_status()
    return str(response.json()["text"]), time.perf_counter() - start


@pytest.mark.benchmark
def test_whisper_asr_matches_seedtts_reference_text(
    seedtts_en_samples: list[SampleInput],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _require_cuda()
    assert len(seedtts_en_samples) == SEEDTTS_ASR_CORRECTNESS_SAMPLES

    proc, port = _start_whisper_server(tmp_path_factory)
    omni_outputs: dict[str, str] = {}
    omni_latency_s = 0.0
    try:
        for sample in seedtts_en_samples:
            text, latency_s = _transcribe_with_omni(port, sample)
            omni_outputs[sample.sample_id] = text
            omni_latency_s += latency_s
    finally:
        stop_server(proc)

    sample_diffs: list[str] = []
    high_wer_samples: list[str] = []
    ref_norms: list[str] = []
    hyp_norms: list[str] = []
    for sample in seedtts_en_samples:
        omni_text = omni_outputs[sample.sample_id]
        ref_norm = normalize_text(sample.ref_text, "en")
        omni_norm = normalize_text(omni_text, "en")
        ref_norms.append(ref_norm)
        hyp_norms.append(omni_norm)
        sample_wer = process_words(ref_norm, omni_norm).wer
        if sample_wer > 0:
            diff = "\n".join(
                [
                    f"sample_id={sample.sample_id}",
                    f"ref_text={sample.ref_text!r}",
                    f"omni={omni_text!r}",
                    f"sample_wer={sample_wer:.4f}",
                    f"ref_norm={ref_norm!r}",
                    f"omni_norm={omni_norm!r}",
                ]
            )
            sample_diffs.append(diff)
            if sample_wer > SEEDTTS_ASR_SAMPLE_WER_MAX:
                high_wer_samples.append(diff)

    if sample_diffs:
        print("\n[Whisper ASR correctness diffs]\n" + "\n\n".join(sample_diffs))

    corpus_wer = process_words(ref_norms, hyp_norms).wer
    print(
        "\n[Whisper ASR correctness] "
        f"samples={len(seedtts_en_samples)} "
        f"omni_total_s={omni_latency_s:.3f} "
        f"diff_samples={len(sample_diffs)} "
        f"corpus_wer={corpus_wer:.4f}"
    )
    assert corpus_wer <= SEEDTTS_ASR_CORPUS_WER_MAX, (
        f"Whisper ASR corpus WER {corpus_wer:.4f} exceeds "
        f"{SEEDTTS_ASR_CORPUS_WER_MAX:.4f}"
    )
    assert (
        not high_wer_samples
    ), "Whisper ASR high-WER SeedTTS samples:\n" + "\n\n".join(high_wer_samples)
