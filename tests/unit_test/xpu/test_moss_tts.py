# SPDX-License-Identifier: Apache-2.0
"""Intel XPU policy tests for MOSS-TTS."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.models.moss_tts.vocoder as vocoder_module
from sglang_omni import platforms
from sglang_omni.models.moss_tts import engine_builder
from sglang_omni.models.moss_tts import request_builders as rb
from sglang_omni.models.moss_tts import stages
from sglang_omni.models.moss_tts.audio_tokenizer import codec_autocast_enabled
from sglang_omni.models.moss_tts.vocoder import autocast_if_supported


def _mock_xpu_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(platforms.current_platform, "device_type", "xpu", raising=False)


def test_moss_tts_codec_device_follows_xpu_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)
    processor = SimpleNamespace(
        audio_tokenizer=None,
        model_config=SimpleNamespace(n_vq=32, audio_tokenizer_name_or_path="codec"),
    )
    encoder = SimpleNamespace(
        sample_rate=24000,
        number_channels=1,
        device="cpu",
        model=SimpleNamespace(encoder_dtype=torch.float32),
    )
    loaded: list[str] = []

    def load_encoder(model_path, *, device, compute_dtype, attention_backend):
        del model_path, compute_dtype, attention_backend
        loaded.append(device)
        return encoder

    monkeypatch.setattr(stages, "load_moss_processor", lambda model_path: processor)
    monkeypatch.setattr(stages, "load_moss_audio_encoder", load_encoder)

    try:
        stages.create_preprocessing_executor("model", gpu_id=2, ref_audio_cache=False)
    finally:
        rb.clear_moss_tts_preprocessing_context()

    assert loaded == ["xpu:2"]


def test_moss_tts_xpu_defaults_to_eager_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)

    defaults = engine_builder.MossTtsEngineBuilder().generation_defaults(
        dtype="bfloat16"
    )

    assert defaults["disable_cuda_graph"] is True


@pytest.mark.parametrize(
    ("device_type", "dtype", "expected"),
    [
        ("xpu", torch.bfloat16, True),
        ("xpu", torch.float16, True),
        ("xpu", torch.float32, False),
        ("cuda", torch.bfloat16, True),
        ("cpu", torch.bfloat16, True),
        ("cpu", torch.float16, False),
    ],
)
def test_moss_tts_codec_autocast_covers_xpu(
    device_type: str, dtype: torch.dtype, expected: bool
) -> None:
    # The streaming decoder and the segment vocoder share this predicate, so
    # XPU streaming and non-streaming PCM use the same autocast boundary.
    assert codec_autocast_enabled(torch.device(device_type), dtype) is expected


def test_moss_tts_vocoder_enables_bfloat16_autocast_on_xpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, torch.dtype]] = []

    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        return nullcontext()

    monkeypatch.setattr(vocoder_module.torch, "autocast", fake_autocast)

    with autocast_if_supported(torch.device("xpu:0"), torch.bfloat16):
        pass

    assert calls == [("xpu", torch.bfloat16)]
