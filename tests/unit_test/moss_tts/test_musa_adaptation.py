# SPDX-License-Identifier: Apache-2.0

"""Focused tests for MOSS-TTS MUSA adaptation boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

import sglang_omni.models.moss_tts.vocoder as vocoder_module
from sglang_omni.models.moss_tts.audio_tokenizer import MossAudioEncoder
from sglang_omni.models.moss_tts.vocoder import autocast_if_supported


@dataclass(frozen=True)
class FakeDevice:
    type: str


def test_audio_tokenizer_uses_shared_audio_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = torch.zeros(1, 8)
    loader_calls: list[tuple[str, str]] = []

    def fake_load(path: str, *, source_name: str) -> tuple[Tensor, int]:
        loader_calls.append((path, source_name))
        return waveform, 16_000

    monkeypatch.setattr(
        "sglang_omni.utils.audio.load_with_torchaudio",
        fake_load,
    )
    encoder = object.__new__(MossAudioEncoder)
    encoder.sample_rate = 16_000

    loaded = encoder.load_paths([Path("reference.wav")])

    assert loader_calls == [("reference.wav", "reference")]
    assert loaded == [(waveform, 16_000)]


def test_vocoder_enables_musa_autocast_for_reduced_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autocast_calls: list[tuple[str, torch.dtype]] = []

    @contextmanager
    def fake_autocast(*, device_type: str, dtype: torch.dtype) -> Iterator[None]:
        autocast_calls.append((device_type, dtype))
        yield

    monkeypatch.setattr(vocoder_module.torch, "autocast", fake_autocast)

    musa_device = cast(torch.device, FakeDevice("musa"))
    with autocast_if_supported(musa_device, torch.bfloat16):
        pass
    with autocast_if_supported(musa_device, torch.float32):
        pass

    assert autocast_calls == [("musa", torch.bfloat16)]
