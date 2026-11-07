# SPDX-License-Identifier: Apache-2.0
"""Text-audio WER in the MMSU, MMMU and Video-MME evaluation CLIs."""

import argparse
import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner
from benchmarks.dataset.mmmu import MMMUSample
from benchmarks.dataset.mmsu import MmsuSample
from benchmarks.dataset.videomme import VideoMMESample
from benchmarks.eval import (
    benchmark_omni_mmmu,
    benchmark_omni_mmsu,
    benchmark_omni_videomme,
)

ASR_ROUTER_PORT = 8001
SERVER_URL = "http://omni.test"
WER_RESULT = {"summary": {"evaluated": 1}, "per_sample": []}


async def answer_every_sample(
    runner: BenchmarkRunner, samples: list[object], send_fn: object
) -> list[RequestResult]:
    return [
        RequestResult(
            request_id=sample.sample_id,
            text="A",
            is_success=True,
            latency_s=1.0,
            audio_duration_s=1.0,
            wav_path=f"{sample.sample_id}.wav",
        )
        for sample in samples
    ]


def run_mmsu(asr_router_port: int | None, tmp_path: Path) -> dict:
    sample = MmsuSample(
        sample_id="mmsu-0",
        audio_path="question.wav",
        question="Is it raining?",
        choices=["yes", "no"],
        answer_text="yes",
        answer_index=0,
        task_name="task",
        category="category",
        sub_category="",
        sub_sub_category="",
        linguistics_sub_discipline="",
    )
    args = argparse.Namespace(
        base_url=SERVER_URL,
        host="localhost",
        port=8000,
        model="qwen3-omni",
        modalities="text+audio",
        output_dir=None,
        max_samples=None,
        task_names=None,
        categories=None,
        prompt=None,
        max_tokens=32,
        temperature=0.0,
        warmup=0,
        max_concurrency=1,
        request_rate=float("inf"),
        save_audio=False,
        disable_tqdm=True,
        seed=None,
        fingerprint=False,
        lang="en",
        asr_device="cuda:0",
        asr_concurrency=1,
        asr_router_port=asr_router_port,
        timeout_s=10,
    )
    return asyncio.run(benchmark_omni_mmsu.run(args, samples=[sample]))


def run_mmmu(asr_router_port: int | None, tmp_path: Path) -> dict:
    config = benchmark_omni_mmmu.MMMUEvalConfig(
        model="qwen3-omni",
        base_url=SERVER_URL,
        warmup=0,
        disable_tqdm=True,
        enable_audio=True,
        asr_router_port=asr_router_port,
        output_dir=str(tmp_path),
    )
    return asyncio.run(benchmark_omni_mmmu.run_mmmu_eval(config))


def run_videomme(asr_router_port: int | None, tmp_path: Path) -> dict:
    sample = VideoMMESample(
        sample_id="video-0",
        video_path="clip.mp4",
        question="Is it raining?",
        options=["yes", "no"],
        answer="A",
        all_choices=["A", "B"],
    )
    config = benchmark_omni_videomme.VideoEvalConfig(
        model="qwen3-omni",
        base_url=SERVER_URL,
        warmup=0,
        disable_tqdm=True,
        enable_audio=True,
        asr_router_port=asr_router_port,
        output_dir=str(tmp_path),
    )
    return asyncio.run(
        benchmark_omni_videomme.run_video_eval(
            config,
            samples=[sample],
            task_label="Video-MME",
            output_filename="videomme_results.json",
            audio_output_dir_default=str(tmp_path),
        )
    )


EVAL_RUNS = pytest.mark.parametrize(
    "run_eval", [run_mmsu, run_mmmu, run_videomme], ids=["mmsu", "mmmu", "videomme"]
)


@pytest.fixture(autouse=True)
def fake_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BenchmarkRunner, "run", answer_every_sample)
    mmmu_sample = MMMUSample(
        sample_id="mmmu-0",
        question="Is it raining?",
        options=["yes", "no"],
        answer="A",
        image_data_uris=(),
        subject="Art",
        prompt="Is it raining?",
        all_choices=["A", "B"],
        index2ans={"A": "yes", "B": "no"},
    )
    monkeypatch.setattr(
        benchmark_omni_mmmu,
        "load_mmmu_samples",
        lambda max_samples, **options: [mmmu_sample],
    )


@EVAL_RUNS
def test_text_audio_wer_uses_the_given_asr_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_eval: Callable[[int | None, Path], dict],
) -> None:
    """Audio-output runs had no way to name an ASR server, so WER always failed."""
    asr_router_ports = []

    def record_wer(
        request_results: list[RequestResult],
        lang: str,
        asr_device: str,
        *,
        asr_router_port: int | None = None,
        asr_concurrency: int = 1,
    ) -> dict:
        asr_router_ports.append(asr_router_port)
        return WER_RESULT

    for module in (benchmark_omni_mmsu, benchmark_omni_mmmu, benchmark_omni_videomme):
        monkeypatch.setattr(module, "compute_text_audio_consistency", record_wer)

    results = run_eval(ASR_ROUTER_PORT, tmp_path)

    assert results["wer"] == WER_RESULT
    assert asr_router_ports == [ASR_ROUTER_PORT]


@EVAL_RUNS
def test_text_audio_run_without_asr_server_keeps_its_results(
    tmp_path: Path, run_eval: Callable[[int | None, Path], dict]
) -> None:
    """Without an ASR server, audio-output runs crashed after generation and lost every result."""
    results = run_eval(None, tmp_path)

    assert "wer" not in results
