# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from sglang_omni.models.moss_transcribe_diarize.sglang_model import (
    MossTranscribeDiarizeForConditionalGeneration,
    _normalize_compile_targets,
)


class FakeWhisperEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


class FakeModel(nn.Module):
    pass


def _model_for_compile() -> MossTranscribeDiarizeForConditionalGeneration:
    model = object.__new__(MossTranscribeDiarizeForConditionalGeneration)
    nn.Module.__init__(model)
    model.whisper_encoder = FakeWhisperEncoder()
    model.vq_adaptor = nn.Linear(2, 2)
    model._audio_encoder_compiled = False
    model._audio_compile_targets = set()
    return model


def _install_fake_compile(monkeypatch):
    calls = []

    def fake_compile(module, *, mode, dynamic):
        wrapper = FakeModel()
        wrapper.original = module
        wrapper.mode = mode
        wrapper.dynamic = dynamic
        calls.append((module, mode, dynamic, wrapper))
        return wrapper

    fake_runner = ModuleType("sglang.srt.model_executor.cuda_graph_runner")
    fake_runner.set_torch_compile_config = lambda: None
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.cuda_graph_runner",
        fake_runner,
    )
    monkeypatch.setattr(torch, "compile", fake_compile)
    return calls


def test_normalize_compile_targets_defaults_to_whisper_layers() -> None:
    assert _normalize_compile_targets(None) == ("whisper_layers",)
    assert _normalize_compile_targets("encoder, adaptor") == ("whisper", "adaptor")
    assert _normalize_compile_targets("off") == ()


def test_normalize_compile_targets_rejects_conflicting_whisper_boundaries() -> None:
    with pytest.raises(ValueError, match="cannot combine"):
        _normalize_compile_targets("whisper,whisper_layers")


def test_compile_audio_encoder_compiles_whisper_layers(monkeypatch) -> None:
    calls = _install_fake_compile(monkeypatch)
    model = _model_for_compile()
    original_encoder = model.whisper_encoder
    original_layers = list(original_encoder.layers)

    model.compile_audio_encoder(mode="reduce-overhead", target="whisper_layers")

    assert model.whisper_encoder is original_encoder
    assert len(calls) == len(original_layers)
    assert [call[0] for call in calls] == original_layers
    assert all(call[1] == "reduce-overhead" for call in calls)
    assert all(call[2] is True for call in calls)
    assert all(isinstance(layer, FakeModel) for layer in model.whisper_encoder.layers)
    assert model._audio_compile_targets == {"whisper_layers"}


def test_compile_audio_encoder_can_compile_adaptor_without_encoder(monkeypatch) -> None:
    calls = _install_fake_compile(monkeypatch)
    model = _model_for_compile()
    original_encoder = model.whisper_encoder
    original_adaptor = model.vq_adaptor

    model.compile_audio_encoder(target="adaptor")

    assert model.whisper_encoder is original_encoder
    assert calls[0][0] is original_adaptor
    assert model.vq_adaptor is calls[0][3]
    assert model._audio_compile_targets == {"adaptor"}


def test_compile_audio_encoder_can_still_compile_whole_encoder(monkeypatch) -> None:
    calls = _install_fake_compile(monkeypatch)
    model = _model_for_compile()
    original_encoder = model.whisper_encoder

    model.compile_audio_encoder(target="whisper")

    assert calls[0][0] is original_encoder
    assert model.whisper_encoder is calls[0][3]
    assert model._audio_compile_targets == {"whisper"}
