# SPDX-License-Identifier: Apache-2.0
"""Opt-in real worker generation, repeatability, and cancellation validation."""

import io
import json
import os
import platform
import statistics
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np
import pytest
from numpy.typing import NDArray

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.voxtream2.config import VoXtream2PipelineConfig
from sglang_omni.models.voxtream2.remote import RemoteVoXtreamBackend
from sglang_omni.models.voxtream2.stages import create_executor
from sglang_omni.proto import OmniRequest, StagePayload

pytestmark = pytest.mark.accelerator

TEXTS = (
    "The train will arrive at the next station in five minutes.",
    "Please check the weather before leaving home, and remember to bring your umbrella.",
)
SEEDS = (101, 202)
REPEATABILITY_ATOL = {
    "float16": 2**-11,
    "bfloat16": 2**-8,
    "float32": 1e-5,
}


@dataclass(kw_only=True)
class GenerationMeasurement:
    request_id: str
    text: str
    seed: int
    sample_rate: int
    samples: int
    chunks: int
    first_audio_seconds: float
    elapsed_seconds: float
    audio_seconds: float
    real_time_factor: float


def generate_and_record(
    backend: RemoteVoXtreamBackend,
    output_dir: Path,
    request_id: str,
    text: str,
    seed: int,
) -> NDArray[np.float32]:
    started = time.perf_counter()
    first_audio_seconds = None
    chunks: list[NDArray[np.float32]] = []
    iterator = backend.iter_audio(
        StagePayload(request_id, OmniRequest(text, {"seed": seed}), None)
    )
    try:
        for chunk in iterator:
            if first_audio_seconds is None:
                first_audio_seconds = time.perf_counter() - started
            else:
                pass
            chunks.append(chunk.copy())
    finally:
        iterator.close()
    elapsed_seconds = time.perf_counter() - started
    assert chunks and first_audio_seconds is not None, "Model produced no audio"
    audio = np.concatenate(chunks)
    assert audio.ndim == 1 and np.isfinite(audio).all()
    assert 0 < audio.size <= backend.max_samples
    assert np.any(audio != 0), "Model returned silence"
    audio_seconds = audio.size / backend.sample_rate
    measurement = GenerationMeasurement(
        request_id=request_id,
        text=text,
        seed=seed,
        sample_rate=backend.sample_rate,
        samples=audio.size,
        chunks=len(chunks),
        first_audio_seconds=first_audio_seconds,
        elapsed_seconds=elapsed_seconds,
        audio_seconds=audio_seconds,
        real_time_factor=elapsed_seconds / audio_seconds,
    )
    np.save(output_dir / f"{request_id}.npy", audio)
    with wave.open(str(output_dir / f"{request_id}.wav"), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(backend.sample_rate)
        recording.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    (output_dir / f"{request_id}.json").write_text(
        json.dumps(asdict(measurement), indent=2), encoding="utf-8"
    )
    return audio


@pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["eager", "graph"])
def test_real_worker_lifecycle(use_cuda_graph: bool) -> None:
    config_path = os.environ.get("VOXTREAM2_TEST_CONFIG")
    if config_path is None:
        pytest.skip("Set VOXTREAM2_TEST_CONFIG to a deployment YAML with real assets")
    else:
        pass
    output_root = os.environ.get("VOXTREAM2_TEST_OUTPUT")
    assert output_root, "Set VOXTREAM2_TEST_OUTPUT to retain validation artifacts"
    config = ConfigManager.from_file(config_path).config
    assert isinstance(config, VoXtream2PipelineConfig)
    assert len(config.stages) == 1
    assert config.stages[0].gpu is not None, "This acceptance test requires CUDA"
    config.use_cuda_graph = use_cuda_graph
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    output_dir = Path(
        tempfile.mkdtemp(prefix=f"graph-{use_cuda_graph}-", dir=output_path)
    )
    (output_dir / "config.json").write_text(config.model_dump_json(indent=2))
    scheduler = create_executor(
        model_path=config.model_path,
        gpu_id=config.stages[0].gpu,
        **config.stage_factory_kwargs("tts_engine"),
    )
    backend = scheduler.backend
    assert isinstance(backend, RemoteVoXtreamBackend)
    passed = False
    try:
        (output_dir / "runtime.json").write_text(
            json.dumps(
                {"parent_platform": platform.platform(), "worker": backend.runtime},
                indent=2,
            ),
            encoding="utf-8",
        )
        generate_and_record(backend, output_dir, "warmup", TEXTS[0], SEEDS[0])
        for index, text in enumerate(TEXTS):
            for seed in SEEDS:
                first = generate_and_record(
                    backend, output_dir, f"text-{index}-seed-{seed}-first", text, seed
                )
                repeated = generate_and_record(
                    backend, output_dir, f"text-{index}-seed-{seed}-repeat", text, seed
                )
                assert first.shape == repeated.shape
                difference = first.astype(np.float64) - repeated
                (output_dir / f"repeatability-{index}-{seed}.json").write_text(
                    json.dumps(
                        {
                            "dtype": config.dtype,
                            "absolute_tolerance": REPEATABILITY_ATOL[config.dtype],
                            "max_absolute_error": float(np.max(np.abs(difference))),
                            "root_mean_square_error": float(
                                np.sqrt(np.mean(difference**2))
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                np.testing.assert_allclose(
                    first, repeated, rtol=0, atol=REPEATABILITY_ATOL[config.dtype]
                )
        cancelled = backend.iter_audio(
            StagePayload("cancelled", OmniRequest(TEXTS[1], {"seed": SEEDS[0]}), None)
        )
        try:
            assert next(cancelled).size > 0
        finally:
            cancelled.close()
        assert backend.healthy and backend.active_request is None
        generate_and_record(backend, output_dir, "after-cancel", TEXTS[0], SEEDS[0])
        rejected = backend.iter_audio(StagePayload("invalid", OmniRequest(""), None))
        try:
            with pytest.raises(RuntimeError, match="VoXtream2 invalid request"):
                next(rejected)
        finally:
            rejected.close()
        generate_and_record(backend, output_dir, "after-invalid", TEXTS[0], SEEDS[0])
        measurements = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(output_dir.glob("text-*.json"))
        ]
        first_audio_seconds = statistics.median(
            measurement["first_audio_seconds"] for measurement in measurements
        )
        real_time_factor = statistics.median(
            measurement["real_time_factor"] for measurement in measurements
        )
        (output_dir / "measurements.md").write_text(
            "| Execution | Measured requests | Median first audio (s) | Median RTF |\n"
            "|---|---:|---:|---:|\n"
            f"| {'CUDA Graph' if use_cuda_graph else 'Eager'} | {len(measurements)} "
            f"| {first_audio_seconds:.6f} | {real_time_factor:.6f} |\n\n"
            "Includes worker transport. Excludes warmup and recovery requests. "
            "Two texts and two seeds; not a broad quality or speedup claim.\n",
            encoding="utf-8",
        )
        passed = True
    finally:
        backend.close()
        passed = (
            passed
            and backend.process.poll() is not None
            and not backend.reader.is_alive()
        )
        (output_dir / "result.json").write_text(
            json.dumps({"passed": passed, "use_cuda_graph": use_cuda_graph}),
            encoding="utf-8",
        )
    assert backend.process.poll() is not None
    assert not backend.reader.is_alive()


def test_real_http_lifecycle() -> None:
    endpoint = os.environ.get("VOXTREAM2_TEST_URL")
    if endpoint is None:
        pytest.skip("Set VOXTREAM2_TEST_URL to the running VoXtream2 server URL")
    else:
        pass
    output_root = os.environ.get("VOXTREAM2_TEST_OUTPUT")
    model_name = os.environ.get("VOXTREAM2_TEST_MODEL")
    assert (
        output_root and model_name
    ), "Set VOXTREAM2_TEST_OUTPUT and VOXTREAM2_TEST_MODEL"
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="http-", dir=output_path))
    request_body = {
        "model": model_name,
        "input": TEXTS[0],
        "voice": "reference",
        "language": "English",
        "seed": SEEDS[0],
        "response_format": "wav",
    }
    passed = False
    try:
        with httpx.Client(base_url=endpoint.rstrip("/"), timeout=180) as client:
            response = client.post("/v1/audio/speech", json=request_body)
            response.raise_for_status()
            (output_dir / "non-streaming.wav").write_bytes(response.content)
            with wave.open(io.BytesIO(response.content), "rb") as recording:
                assert recording.getnchannels() == 1 and recording.getnframes() > 0
                assert recording.getframerate() > 0
            with client.stream(
                "POST",
                "/v1/audio/speech",
                json={**request_body, "stream": True, "response_format": "pcm"},
            ) as streamed:
                streamed.raise_for_status()
                with (output_dir / "streaming.pcm").open("wb") as recording:
                    for chunk in streamed.iter_bytes():
                        recording.write(chunk)
            assert (output_dir / "streaming.pcm").stat().st_size > 0
            for streaming in (False, True):
                for invalid in (
                    {"language": "Japanese"},
                    {"instructions": "Whisper"},
                    {"stage_params": {"tts_engine": {"speaking_rate_sps": -1}}},
                ):
                    response = client.post(
                        "/v1/audio/speech",
                        json={
                            **request_body,
                            "stream": streaming,
                            "response_format": "pcm" if streaming else "wav",
                            **invalid,
                        },
                    )
                    assert response.status_code == 400, response.text
            with client.stream(
                "POST",
                "/v1/audio/speech",
                json={**request_body, "stream": True, "response_format": "pcm"},
            ) as cancelled:
                cancelled.raise_for_status()
                assert next(cancelled.iter_bytes())
            response = client.post("/v1/audio/speech", json=request_body)
            response.raise_for_status()
            (output_dir / "after-disconnect.wav").write_bytes(response.content)
            with wave.open(io.BytesIO(response.content), "rb") as recording:
                assert recording.getnframes() > 0
        passed = True
    finally:
        (output_dir / "result.json").write_text(
            json.dumps({"passed": passed, "endpoint": endpoint, "model": model_name}),
            encoding="utf-8",
        )
