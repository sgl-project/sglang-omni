# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni encoder batching behavior."""
from __future__ import annotations

from typing import Any

import pytest
import torch

from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.stages import (
    _batch_audio_encoder_payloads,
    _encoder_batch_wait_ms,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeAudioEncoder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        lengths = kwargs["audio_feature_lengths"].to(dtype=torch.long).view(-1)
        self.calls.append(int(lengths.shape[0]))
        total = int(lengths.sum().item())
        return {
            "audio_embeds": torch.arange(total, dtype=torch.float32).unsqueeze(1),
            "audio_feature_lengths": lengths,
            "audio_output_lengths": lengths,
        }


@pytest.mark.parametrize("raw", [None, "invalid", "-1"])
def test_encoder_batch_wait_falls_back_to_zero(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    if raw is None:
        monkeypatch.delenv("SGLANG_OMNI_ENCODER_BATCH_WAIT_MS", raising=False)
    else:
        monkeypatch.setenv("SGLANG_OMNI_ENCODER_BATCH_WAIT_MS", raw)

    assert _encoder_batch_wait_ms() == 0


def _payload(request_id: str, cache_key: str, time_steps: int) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hi"),
        data={
            "encoder_inputs": {
                "audio_encoder": {
                    "cache_key": cache_key,
                    "input_features": torch.ones(4, time_steps),
                    "audio_feature_lengths": torch.tensor([time_steps]),
                }
            }
        },
    )


def test_audio_encoder_batch_dedups_same_cache_key() -> None:
    model = _FakeAudioEncoder()
    payloads = [
        _payload("req-a1", "spk-a", 3),
        _payload("req-b", "spk-b", 5),
        _payload("req-a2", "spk-a", 3),
    ]

    out = _batch_audio_encoder_payloads(payloads, model=model, cache=None)

    assert model.calls == [2]
    assert len(out) == 3
    states = [Qwen3OmniPipelineState.from_dict(p.data) for p in out]
    embeds = [s.encoder_outs["audio_encoder"]["audio_embeds"] for s in states]
    assert all(e.shape[0] == n for e, n in zip(embeds, [3, 5, 3]))
    assert torch.equal(embeds[0], embeds[2])


def test_audio_encoder_batch_without_cache_keys_runs_every_request() -> None:
    model = _FakeAudioEncoder()
    payloads = [_payload("req-1", "k1", 3), _payload("req-2", "k2", 3)]
    for payload in payloads:
        payload.data["encoder_inputs"]["audio_encoder"].pop("cache_key")

    out = _batch_audio_encoder_payloads(payloads, model=model, cache=None)

    assert model.calls == [2]
    assert len(out) == 2
