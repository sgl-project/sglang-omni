# SPDX-License-Identifier: Apache-2.0
"""Regression tests for S2-Pro streaming helpers."""

from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_s2_pro.pipeline.stages import (
    _STREAM_CODES_KEY,
    _STREAM_EMITTED_SAMPLES_KEY,
    _STREAM_LAST_VOCODE_TOKENS_KEY,
    _build_incremental_audio_chunk,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeCodec:
    sample_rate = 24000

    def from_indices(self, codes):
        num_tokens = int(codes.shape[-1])
        audio = torch.arange(num_tokens * 2, dtype=torch.float32)
        return audio.reshape(1, 1, -1)


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
        payload, codec=_FakeCodec(), device="cpu"
    )
    assert first_chunk is not None
    assert first_chunk["modality"] == "audio"
    assert len(first_chunk["audio_data"]) == 6
    assert payload.data[_STREAM_EMITTED_SAMPLES_KEY] == 6
    assert payload.data[_STREAM_LAST_VOCODE_TOKENS_KEY] == 3

    payload.data[_STREAM_CODES_KEY].append(torch.tensor([[7], [8]]))
    second_chunk = _build_incremental_audio_chunk(
        payload, codec=_FakeCodec(), device="cpu"
    )
    assert second_chunk is not None
    assert len(second_chunk["audio_data"]) == 2
