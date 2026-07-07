# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang")

from sglang_omni.models.moss_transcribe_diarize import sglang_model  # noqa: E402

_ENCODER_CACHE_MAX_ENTRIES = sglang_model._ENCODER_CACHE_MAX_ENTRIES
MossModel = sglang_model.MossTranscribeDiarizeForConditionalGeneration


def _make_model(max_bytes: int) -> MossModel:
    model = MossModel.__new__(MossModel)
    torch.nn.Module.__init__(model)
    model.vq_adaptor = torch.nn.Linear(4, 4)
    model.init_encoder_cache(max_bytes)
    return model


def _stub_encode(model: MossModel):
    calls = {"count": 0}

    def _fake(items, forward_batch):  # noqa: ANN001
        calls["count"] += 1
        return torch.ones(4)

    model._get_audio_feature_uncached = _fake  # type: ignore[assignment]
    return calls


def _item(audio_hash: int) -> SimpleNamespace:
    return SimpleNamespace(hash=audio_hash)


def test_identical_hash_encodes_once() -> None:
    model = _make_model(max_bytes=1 << 20)
    calls = _stub_encode(model)

    first = model.get_audio_feature([_item(123)], forward_batch=None)
    second = model.get_audio_feature([_item(123)], forward_batch=None)

    assert calls["count"] == 1
    assert torch.equal(first, second)


def test_different_hash_encodes_each() -> None:
    model = _make_model(max_bytes=1 << 20)
    calls = _stub_encode(model)

    model.get_audio_feature([_item(1)], forward_batch=None)
    model.get_audio_feature([_item(2)], forward_batch=None)

    assert calls["count"] == 2


def test_disabled_cache_always_encodes() -> None:
    model = _make_model(max_bytes=0)
    calls = _stub_encode(model)

    assert model._encoder_cache is None
    model.get_audio_feature([_item(7)], forward_batch=None)
    model.get_audio_feature([_item(7)], forward_batch=None)

    assert calls["count"] == 2


def test_multi_item_batch_bypasses_cache() -> None:
    model = _make_model(max_bytes=1 << 20)
    calls = _stub_encode(model)

    model.get_audio_feature([_item(1), _item(2)], forward_batch=None)
    model.get_audio_feature([_item(1), _item(2)], forward_batch=None)

    assert calls["count"] == 2


def test_lru_evicts_when_over_budget() -> None:
    model = _make_model(max_bytes=16)
    calls = _stub_encode(model)

    model.get_audio_feature([_item(1)], forward_batch=None)
    model.get_audio_feature([_item(2)], forward_batch=None)
    model.get_audio_feature([_item(1)], forward_batch=None)

    assert calls["count"] == 3
    assert model._encoder_cache is not None
    assert model._encoder_cache.eviction_count >= 1


def test_entry_count_cap_matches_constant() -> None:
    model = _make_model(max_bytes=1 << 30)
    assert model._encoder_cache is not None
    assert model._encoder_cache.max_size == _ENCODER_CACHE_MAX_ENTRIES


def test_hit_returns_model_device_tensors() -> None:
    model = _make_model(max_bytes=1 << 20)
    _stub_encode(model)
    expected_device = next(model.vq_adaptor.parameters()).device

    model.get_audio_feature([_item(42)], forward_batch=None)
    cached = model.get_audio_feature([_item(42)], forward_batch=None)

    assert cached.device == expected_device
