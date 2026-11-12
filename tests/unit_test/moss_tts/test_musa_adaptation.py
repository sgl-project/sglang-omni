# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

import sglang_omni.models.moss_tts.vocoder as vocoder_module
from sglang_omni.models.moss_tts.audio_tokenizer import MossAudioEncoder
from sglang_omni.models.moss_tts.vocoder import autocast_if_supported


def test_musa_audio_adaptations(monkeypatch) -> None:
    waveform = torch.zeros(1, 8)
    loader_calls = []

    def fake_load(path: str, *, source_name: str):
        loader_calls.append((path, source_name))
        return waveform, 16_000

    monkeypatch.setattr(
        "sglang_omni.utils.audio.load_with_torchaudio",
        fake_load,
    )
    encoder = object.__new__(MossAudioEncoder)
    encoder.sample_rate = 16_000

    loaded = encoder.load_paths(["reference.wav"])

    assert loader_calls == [("reference.wav", "reference")]
    assert loaded == [(waveform, 16_000)]

    autocast_calls = []

    @contextmanager
    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        autocast_calls.append((device_type, dtype))
        yield

    monkeypatch.setattr(vocoder_module.torch, "autocast", fake_autocast)

    with autocast_if_supported(SimpleNamespace(type="musa"), torch.bfloat16):
        pass
    with autocast_if_supported(SimpleNamespace(type="musa"), torch.float32):
        pass

    assert autocast_calls == [("musa", torch.bfloat16)]
