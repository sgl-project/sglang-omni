# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS deterministic inference checks."""

from __future__ import annotations

import asyncio
import io
import os
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import aiohttp
import pytest
import torch
import yaml

from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.dataset.seedtts import load_seedtts_samples
from tests.test_model.omni_router_utils import _find_available_port_range

MODEL_PATH = os.environ.get(
    "QWEN3_TTS_TEST_MODEL",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
)
DATASET = os.environ.get("QWEN3_TTS_TEST_DATASET", DATASETS["seedtts-50"])
SEED = 123456
STARTUP_TIMEOUT = 600


class Qwen3TTSServer(NamedTuple):
    base_url: str
    log_file: Path


def _pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        pcm = wav.readframes(wav.getnframes())
    assert pcm
    return pcm


async def _model_info(base_url: str, stop: asyncio.Event) -> int:
    max_batch_size = 0
    timeout = aiohttp.ClientTimeout(total=2)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while not stop.is_set():
            try:
                async with session.get(f"{base_url}/model_info") as response:
                    response.raise_for_status()
                    payload = await response.json()
            except TimeoutError:
                pass
            else:
                for result in payload.get("results", []):
                    if result.get("stage") == "tts_engine":
                        max_batch_size = max(
                            max_batch_size,
                            int(result.get("data", {}).get("running_batch_size", 0)),
                        )
            await asyncio.sleep(0.05)
    return max_batch_size


async def _generate(
    session: aiohttp.ClientSession,
    base_url: str,
    payload: dict,
) -> bytes:
    async with session.post(f"{base_url}/v1/audio/speech", json=payload) as response:
        response.raise_for_status()
        return _pcm(await response.read())


async def _generate_serial(
    base_url: str,
    payloads: list[dict],
) -> list[bytes]:
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return [await _generate(session, base_url, payload) for payload in payloads]


async def _generate_batch(
    base_url: str,
    payloads: list[dict],
) -> list[bytes]:
    stop = asyncio.Event()
    poller = asyncio.create_task(_model_info(base_url, stop))
    timeout = aiohttp.ClientTimeout(total=300)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            outputs = await asyncio.gather(
                *(_generate(session, base_url, payload) for payload in payloads)
            )
    finally:
        stop.set()
        max_batch_size = await poller

    assert max_batch_size == 8
    return outputs


def _payload(sample, *, subtalker_dosample: bool | None = None) -> dict:
    return {
        "model": MODEL_PATH,
        "input": sample.target_text,
        "ref_audio": sample.ref_audio,
        "ref_text": sample.ref_text,
        "response_format": "wav",
        "seed": SEED,
        "max_new_tokens": 2048,
        "stage_params": {"tts_engine": {"subtalker_dosample": subtalker_dosample}},
    }


@contextmanager
def _qwen3_tts_server(
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
) -> Iterator[Qwen3TTSServer]:
    config_path = tmp_path_factory.mktemp(f"qwen3_tts_{name}") / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "config_cls": "Qwen3TTSPipelineConfig",
                "model_path": MODEL_PATH,
                "enable_deterministic_inference": True,
                "stages": {
                    "tts_engine": {
                        "engine": {
                            "attention_backend": "triton",
                            "max_running_requests": 8,
                            "cuda_graph_max_bs": 8,
                            "cuda_graph_bs": [1, 2, 4, 8],
                            "torch_compile_max_bs": 8,
                        }
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    port = _find_available_port_range(1)
    log_file = tmp_path_factory.mktemp(f"qwen3_tts_{name}_logs") / "server.log"
    with managed_omni_server(
        model_path=MODEL_PATH,
        port=port,
        host="127.0.0.1",
        log_file=log_file,
        server_config=str(config_path),
        timeout=STARTUP_TIMEOUT,
    ):
        yield Qwen3TTSServer(f"http://127.0.0.1:{port}", log_file)


def _assert_uncached_reference_encode(log_file: Path) -> None:
    assert (
        "Qwen3-TTS ad-hoc reference reference encode stats: "
        "{'hits': 0, 'misses': 1" in log_file.read_text(encoding="utf-8")
    )


@pytest.mark.benchmark
def test_qwen3_tts_deterministic_batch_invariance(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Match fresh batch-one and batch-eight executions."""
    if not torch.cuda.is_available():
        pytest.skip("Qwen3-TTS batch invariance requires CUDA")
    if not Path(DATASET).exists():
        download_dataset(DATASET, quiet=True)

    payloads = []
    for sample in load_seedtts_samples(DATASET, 8, split="en"):
        payload = _payload(sample)
        payload["input"] = " ".join([sample.target_text] * 3)
        payloads.append(payload)
    assert len({payload["ref_text"] for payload in payloads}) == 8

    payload = payloads[0]
    with _qwen3_tts_server(tmp_path_factory, "b1") as server:
        b1 = asyncio.run(_generate_serial(server.base_url, payloads))
        repeated_b1 = asyncio.run(_generate_serial(server.base_url, [payload] * 2))
        _assert_uncached_reference_encode(server.log_file)

    with _qwen3_tts_server(tmp_path_factory, "b8") as server:
        mixed_b8 = asyncio.run(_generate_batch(server.base_url, payloads))
        _assert_uncached_reference_encode(server.log_file)
        repeated_b8 = asyncio.run(_generate_batch(server.base_url, [payload] * 8))

    assert mixed_b8 == b1
    assert all(pcm == b1[0] for pcm in repeated_b1)
    assert all(pcm == b1[0] for pcm in repeated_b8)


@pytest.mark.benchmark
def test_qwen3_tts_mixed_sampled_argmax_batch_invariance(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Qwen3-TTS batch invariance requires CUDA")
    if not Path(DATASET).exists():
        download_dataset(DATASET, quiet=True)

    payloads = []
    for index, sample in enumerate(load_seedtts_samples(DATASET, 8, split="en")):
        payload = _payload(sample, subtalker_dosample=index % 2 == 0)
        payload["input"] = " ".join([sample.target_text] * 3)
        payloads.append(payload)
    assert len({payload["ref_text"] for payload in payloads}) == 8
    assert {
        payload["stage_params"]["tts_engine"]["subtalker_dosample"]
        for payload in payloads
    } == {False, True}

    payload = payloads[0]
    with _qwen3_tts_server(tmp_path_factory, "b1") as server:
        b1 = asyncio.run(_generate_serial(server.base_url, payloads))
        repeated_b1 = asyncio.run(_generate_serial(server.base_url, [payload] * 2))
        _assert_uncached_reference_encode(server.log_file)

    with _qwen3_tts_server(tmp_path_factory, "b8") as server:
        mixed_b8 = asyncio.run(_generate_batch(server.base_url, payloads))
        _assert_uncached_reference_encode(server.log_file)
        repeated_b8 = asyncio.run(_generate_batch(server.base_url, [payload] * 8))

    assert mixed_b8 == b1
    assert all(pcm == b1[0] for pcm in repeated_b1)
    assert all(pcm == b1[0] for pcm in repeated_b8)
