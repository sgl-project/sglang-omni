# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

import sglang_omni.models.moss_tts.vocoder as vocoder_module
from sglang_omni.models.moss_tts.audio_tokenizer import MossAudioEncoder
from sglang_omni.models.moss_tts.vocoder import autocast_if_supported


def test_reference_audio_uses_shared_audio_loader(monkeypatch) -> None:
    waveform = torch.zeros(1, 8)
    calls = []

    def fake_load(path: str, *, source_name: str):
        calls.append((path, source_name))
        return waveform, 16_000

    monkeypatch.setattr(
        "sglang_omni.utils.audio.load_with_torchaudio",
        fake_load,
    )
    encoder = object.__new__(MossAudioEncoder)
    encoder.sample_rate = 16_000

    loaded = encoder.load_paths(["reference.wav"])

    assert calls == [("reference.wav", "reference")]
    assert len(loaded) == 1
    assert loaded[0][0] is waveform
    assert loaded[0][1] == 16_000


def test_autocast_if_supported_enables_musa_bf16(monkeypatch) -> None:
    calls = []

    @contextmanager
    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        calls.append((device_type, dtype))
        yield

    monkeypatch.setattr(vocoder_module.torch, "autocast", fake_autocast)

    with autocast_if_supported(SimpleNamespace(type="musa"), torch.bfloat16):
        pass
    with autocast_if_supported(SimpleNamespace(type="musa"), torch.float32):
        pass

    assert calls == [("musa", torch.bfloat16)]
