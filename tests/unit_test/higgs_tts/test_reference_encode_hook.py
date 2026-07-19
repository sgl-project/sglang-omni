# SPDX-License-Identifier: Apache-2.0
"""Contract for the Higgs M4a reference-encode hook (waveform-content keys)."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.stages import (
    _HiggsReferenceEncodeHook,
    _HiggsReferenceInput,
)
from sglang_omni.models.higgs_tts.utils import apply_delay_pattern
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService


class _FakeCodec:
    def __init__(self, num_codebooks: int = 8) -> None:
        self.num_codebooks = num_codebooks
        self.calls = 0

    def encode_reference(self, waveform: torch.Tensor, *, sample_rate: int):
        assert sample_rate == 24000
        self.calls += 1
        frames = 3
        return torch.arange(frames * self.num_codebooks).reshape(
            frames, self.num_codebooks
        )


def _service(codec: _FakeCodec) -> ReferenceEncodeService:
    hook = _HiggsReferenceEncodeHook(
        codec, num_codebooks=codec.num_codebooks, model_identity="ckpt"
    )
    return ReferenceEncodeService(hook, max_items=8, max_bytes=1 << 20)


def test_same_content_key_hits_cache_and_round_trips_long() -> None:
    codec = _FakeCodec()
    service = _service(codec)
    wav = torch.zeros(1, 1, 240)

    first = service.get_or_encode(_HiggsReferenceInput(wav, "waveform:abc"))
    second = service.get_or_encode(_HiggsReferenceInput(wav, "waveform:abc"))

    assert codec.calls == 1
    assert service.stats()["hits"] == 1
    expected = apply_delay_pattern(
        torch.arange(3 * codec.num_codebooks).reshape(3, codec.num_codebooks)
    )
    assert first.dtype == torch.long
    assert second.dtype == torch.long
    assert torch.equal(first, expected.to(torch.long))
    assert torch.equal(second, expected.to(torch.long))


def test_missing_content_key_bypasses_cache() -> None:
    codec = _FakeCodec()
    service = _service(codec)
    wav = torch.zeros(1, 1, 240)

    service.get_or_encode(_HiggsReferenceInput(wav, None))
    service.get_or_encode(_HiggsReferenceInput(wav, None))

    assert codec.calls == 2
    assert service.stats()["uncacheable"] == 2
    assert service.stats()["entries"] == 0


def test_codec_shape_mismatch_fails_loud() -> None:
    class _WrongShapeCodec(_FakeCodec):
        def encode_reference(self, waveform, *, sample_rate):
            return torch.zeros(3, self.num_codebooks + 1)

    service = _service(_WrongShapeCodec())
    with pytest.raises(ValueError, match="codec output must be"):
        service.get_or_encode(_HiggsReferenceInput(torch.zeros(1, 1, 240), "k"))
