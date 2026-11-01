# SPDX-License-Identifier: Apache-2.0
"""Full SeedTTS generation and transcription coverage for NPU CustomVoice."""

from __future__ import annotations

import array
import asyncio
import json
import os
import wave
from dataclasses import replace
from pathlib import Path

import pytest
import requests

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.eval.benchmark_tts_seedtts import (
    DEFAULT_TTS_BENCHMARK_CONCURRENCY,
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_benchmark,
    run_tts_seedtts_transcribe,
)
from benchmarks.tasks.tts import run_seedtts_utmos
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from tests.test_model.tts_ci_config import select_tts_ci_preset
from tests.utils import (
    QWEN3_ASR_WER_CONCURRENCY,
    MetricCheckCollector,
    assert_open_loop_latency_results,
    assert_streaming_consistency,
    assert_tts_audio_result_integrity,
    print_latency_point,
)
from tests.utils.npu_ci import load_full_seedtts_samples, managed_npu_server

pytestmark = pytest.mark.skipif(
    os.environ.get("OMNI_RUN_NPU_TESTS") != "1",
    reason="Set OMNI_RUN_NPU_TESTS=1 with a reserved NPU and local models",
)

PRESET = select_tts_ci_preset("qwen3-tts-custom-voice")[1]
TTS_CONCURRENCY = DEFAULT_TTS_BENCHMARK_CONCURRENCY


def validate_wav(audio_path: Path) -> None:
    with wave.open(str(audio_path), "rb") as waveform:
        assert (
            waveform.getnchannels(),
            waveform.getsampwidth(),
            waveform.getframerate(),
        ) == (1, 2, 24000)
        frames = waveform.getnframes()
        pcm = waveform.readframes(frames)
    assert frames > 0 and len(pcm) == frames * 2, f"Truncated WAV: {audio_path}"
    samples = array.array("h")
    samples.frombytes(pcm)
    assert any(samples), f"All-silent output: {audio_path}"


def test_full_seedtts(tmp_path: Path) -> None:
    samples = load_full_seedtts_samples("en")
    expected_ids = {sample.sample_id for sample in samples}
    output_dir = Path(os.environ.get("OMNI_NPU_TTS_OUTPUT", str(tmp_path))) / "en"
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations: list[TtsSeedttsBenchmarkConfig] = []
    checks = MetricCheckCollector("NPU TTS SeedTTS EN")
    admission_limits = Qwen3TTSPipelineConfig.generation_admission_defaults()

    with managed_npu_server("TTS", output_dir) as tts_server:
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                f"{tts_server.base_url}/v1/audio/speech", json={}, timeout=10
            )
            checks.check(response.status_code in {400, 422}, response.text)
            with session.post(
                f"{tts_server.base_url}/v1/audio/speech",
                json={
                    "model": tts_server.model_path,
                    "input": samples[0].target_text,
                    "task_type": "CustomVoice",
                    "voice": PRESET.model.voice,
                    "stream": True,
                    "response_format": "pcm",
                },
                stream=True,
                timeout=(10, 600),
            ) as response:
                assert response.status_code == 200, response.text
                for header, expected in (
                    ("x-sample-rate", "24000"),
                    ("x-channels", "1"),
                    ("x-bit-depth", "16"),
                ):
                    checks.check(response.headers.get(header) == expected, header)
                pcm = response.content
                checks.check(bool(pcm) and len(pcm) % 2 == 0, "Incomplete PCM16 stream")
        for stream in (False, True):
            mode = "stream" if stream else "nonstream"
            configuration = TtsSeedttsBenchmarkConfig(
                model=tts_server.model_path,
                meta=SEEDTTS_DATASET_ID,
                base_url=tts_server.base_url,
                voice=PRESET.model.voice,
                task_type="CustomVoice",
                voice_clone=PRESET.model.voice_clone,
                ref_format=PRESET.model.ref_format,
                token_count=PRESET.model.token_count,
                lang="en",
                device="npu:0",
                output_dir=str(output_dir / mode),
                server_config=os.environ["OMNI_NPU_TTS_CONFIG"],
                stream=stream,
                response_format="pcm" if stream else "wav",
                concurrency=TTS_CONCURRENCY,
                asr_concurrency=QWEN3_ASR_WER_CONCURRENCY,
                warmup=1,
                **admission_limits,
            )
            configurations.append(configuration)
            asyncio.run(run_tts_seedtts_benchmark(configuration, samples=samples))

    # UTMOS is a TorchScript evaluator; score on CPU without an NPU JIT port.
    utmos = run_seedtts_utmos(replace(configurations[0], device="cpu"))
    checks.check(
        utmos["summary"]["total_samples"]
        == utmos["summary"]["evaluated"]
        == len(samples),
        f"Incomplete UTMOS scoring: {utmos['summary']}",
    )

    with managed_npu_server("ASR", output_dir) as asr_server:
        for configuration in configurations:
            configuration.asr_model_path = asr_server.model_path
            results = run_tts_seedtts_transcribe(
                configuration, asr_router_port=asr_server.port
            )
            summary = results["wer_summary"]
            checks.check(
                summary["total_samples"] == summary["evaluated"] == len(samples),
                f"Incomplete generation/transcription in {configuration.output_dir}: {summary}",
            )
            checks.check(
                all(sample.hyp_norm.strip() for sample in results["per_sample"]),
                f"Empty transcript in {configuration.output_dir}",
            )
            checks.check(
                {sample.sample_id for sample in results["per_sample"]} == expected_ids,
                f"Transcription sample IDs differ in {configuration.output_dir}",
            )

    nonstream = json.loads((output_dir / "nonstream/speed_results.json").read_text())
    streaming = json.loads((output_dir / "stream/speed_results.json").read_text())
    for results in (nonstream, streaming):
        requests_results = results["per_request"]
        assert_tts_audio_result_integrity(
            results["summary"],
            requests_results,
            label="NPU TTS SeedTTS EN",
            collector=checks,
        )
        checks.check(
            len(requests_results) == len(samples)
            and {request["id"] for request in requests_results} == expected_ids,
            "Generation did not cover the full SeedTTS split",
        )
        for request in requests_results:
            checks.check(
                request["finish_reason"] != "length",
                f"Truncated generation: {request['id']}",
            )
            if request["is_success"]:
                validate_wav(Path(request["wav_path"]))
            else:
                pass
    assert_streaming_consistency(
        nonstream["per_request"],
        streaming["per_request"],
        expected_stream_count=len(samples),
        max_failed_requests=0,
        collector=checks,
    )
    checks.assert_all()


def test_streaming_first_audio_latency(tmp_path: Path) -> None:
    samples = load_full_seedtts_samples("en")
    output_dir = Path(os.environ.get("OMNI_NPU_TTS_OUTPUT", str(tmp_path))) / "latency"
    output_dir.mkdir(parents=True, exist_ok=True)
    latency = PRESET.latency
    assert latency is not None
    checks = MetricCheckCollector("NPU TTS streaming first-audio latency")
    admission_limits = Qwen3TTSPipelineConfig.generation_admission_defaults()
    with managed_npu_server("TTS", output_dir) as tts_server:
        for point in latency.points:
            configuration = TtsSeedttsBenchmarkConfig(
                model=tts_server.model_path,
                meta=SEEDTTS_DATASET_ID,
                base_url=tts_server.base_url,
                output_dir=str(output_dir / f"r{point.request_rate:g}"),
                server_config=os.environ["OMNI_NPU_TTS_CONFIG"],
                voice=PRESET.model.voice,
                task_type="CustomVoice",
                voice_clone=PRESET.model.voice_clone,
                ref_format=PRESET.model.ref_format,
                token_count=PRESET.model.token_count,
                device="npu:0",
                stream=True,
                response_format="pcm",
                concurrency=admission_limits["max_running_requests"],
                max_samples=point.samples,
                request_rate=point.request_rate,
                arrival_seed=0,
                warmup=1,
                **admission_limits,
            )
            results = asyncio.run(
                run_tts_seedtts_benchmark(
                    configuration, samples=samples[: point.samples]
                )
            )
            label = f"NPU TTS latency {point.request_rate:g} rps"
            print_latency_point(results["summary"], label=label)
            assert_open_loop_latency_results(
                results, samples=point.samples, label=label, collector=checks
            )
    checks.assert_all()
