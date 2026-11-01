# SPDX-License-Identifier: Apache-2.0
"""Full SeedTTS inputs and managed servers for NPU model CI."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests
import torch

from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.utils.connection import find_available_port


def load_full_seedtts_samples(language: Literal["en", "zh"]) -> list[SampleInput]:
    samples = load_seedtts_samples(SEEDTTS_DATASET_ID, split=language)
    expected_count = {"en": 1088, "zh": 2020}[language]
    assert len(samples) == expected_count, (
        f"Expected the full SeedTTS {language} split ({expected_count} samples), "
        f"got {len(samples)}"
    )
    assert (
        len({sample.sample_id for sample in samples}) == expected_count
    ), f"Duplicate sample IDs in SeedTTS {language}"
    return samples


@dataclass(frozen=True, kw_only=True)
class NpuServer:
    port: int
    model_path: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@contextmanager
def managed_npu_server(
    model_type: Literal["ASR", "TTS"],
    output_dir: Path,
) -> Iterator[NpuServer]:
    device_module = torch.get_device_module("npu")
    assert device_module.is_available() and device_module.device_count() >= 1
    model_path = Path(os.environ[f"OMNI_NPU_{model_type}_MODEL"]).resolve()
    server_config = Path(os.environ[f"OMNI_NPU_{model_type}_CONFIG"]).resolve()
    assert (model_path / "config.json").is_file()
    assert server_config.is_file()
    if model_type == "TTS":
        assert (model_path / "speech_tokenizer").is_dir()
        # note (cocoa): Full-load CI needs both admission limits, not the small example queue.
        admission_limits = Qwen3TTSPipelineConfig.generation_admission_defaults()
    else:
        admission_limits = {}
    port = find_available_port()
    with managed_omni_server(
        model_path=str(model_path),
        server_config=str(server_config),
        port=port,
        host="127.0.0.1",
        log_file=output_dir / f"{model_type.lower()}-server.log",
        wait_for_gpu_release=False,
        **admission_limits,
    ):
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(f"http://127.0.0.1:{port}/v1/models", timeout=10)
            response.raise_for_status()
            model_id = response.json()["data"][0]["id"]
        yield NpuServer(port=port, model_path=model_id)
