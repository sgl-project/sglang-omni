# SPDX-License-Identifier: Apache-2.0
"""Regression tests for S2-Pro streaming helpers."""

from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_s2_pro.pipeline.stages import (
    _STREAM_CODES_KEY,
    _STREAM_LAST_VOCODE_TOKENS_KEY,
    _build_incremental_audio_chunk,
    _resolve_stream_overlap_tokens,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeCodec:
    sample_rate = 24000
    delay = 3
    frame_length = 2

    def from_indices(self, codes):
        num_tokens = int(codes.shape[-1])
        audio = torch.arange(num_tokens * 2, dtype=torch.float32)
        return audio.reshape(1, 1, -1)


def test_resolve_stream_overlap_tokens_uses_codec_delay_math() -> None:
    assert _resolve_stream_overlap_tokens(_FakeCodec(), None) == 2


def test_build_incremental_audio_chunk_emits_delta_audio() -> None:
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs="hello"),
        data={
            _STREAM_CODES_KEY: [
                torch.tensor([[1, 2], [3, 4]]),
                torch.tensor([[5], [6]]),
            ]
        },
    )

    first_chunk = _build_incremental_audio_chunk(
        payload,
        codec=_FakeCodec(),
        device="cpu",
        stream_overlap_tokens=2,
    )
    assert first_chunk is not None
    assert first_chunk["modality"] == "audio"
    assert len(first_chunk["audio_data"]) == 6
    assert payload.data[_STREAM_LAST_VOCODE_TOKENS_KEY] == 3

    payload.data[_STREAM_CODES_KEY].append(torch.tensor([[7], [8]]))
    second_chunk = _build_incremental_audio_chunk(
        payload,
        codec=_FakeCodec(),
        device="cpu",
        stream_overlap_tokens=2,
    )
    assert second_chunk is not None
    assert len(second_chunk["audio_data"]) == 2
    assert second_chunk["audio_data"] == [4.0, 5.0]
    assert payload.data[_STREAM_LAST_VOCODE_TOKENS_KEY] == 4


def test_build_incremental_audio_chunk_crossfades_chunk_boundaries() -> None:
    payload = StagePayload(
        request_id="req-xfade",
        request=OmniRequest(inputs="hello", params={"stream": True}),
        data={
            _STREAM_CODES_KEY: [torch.tensor([[1, 2, 3], [4, 5, 6]])],
        },
    )

    first_chunk = _build_incremental_audio_chunk(
        payload,
        codec=_FakeCodec(),
        device="cpu",
        stream_overlap_tokens=2,
        stream_crossfade_samples=2,
    )
    assert first_chunk is not None
    assert first_chunk["audio_data"] == [0.0, 1.0, 2.0, 3.0]

    payload.data[_STREAM_CODES_KEY].append(torch.tensor([[7], [8]]))
    second_chunk = _build_incremental_audio_chunk(
        payload,
        codec=_FakeCodec(),
        device="cpu",
        stream_overlap_tokens=2,
        stream_crossfade_samples=2,
    )
    assert second_chunk is None

    final_chunk = _build_incremental_audio_chunk(
        payload,
        codec=_FakeCodec(),
        device="cpu",
        stream_overlap_tokens=2,
        stream_crossfade_samples=2,
        is_final=True,
    )
    assert final_chunk is not None
    assert final_chunk["audio_data"] == [4.0, 5.0]
