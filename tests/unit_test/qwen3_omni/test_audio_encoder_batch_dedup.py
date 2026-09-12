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


def test_audio_encoder_cache_preserves_hits_in_mixed_batch():
    from sglang_omni.scheduling.stage_cache import StageOutputCache

    model = _FakeAudioEncoder()
    cache = StageOutputCache(max_size=64, max_bytes=4 * 1024**3, cache_device="cpu")
    first = _batch_audio_encoder_payloads(
        [_payload("a1", "a", 3), _payload("b1", "b", 5), _payload("a2", "a", 3)],
        model=model,
        cache=cache,
    )
    second = _batch_audio_encoder_payloads(
        [_payload("b2", "b", 5), _payload("c", "c", 2), _payload("a3", "a", 3)],
        model=model,
        cache=cache,
    )
    assert model.calls == [2, 1]
    for old, new in [(first[1], second[0]), (first[0], second[2])]:
        old_state = Qwen3OmniPipelineState.from_dict(old.data).encoder_outs[
            "audio_encoder"
        ]
        new_state = Qwen3OmniPipelineState.from_dict(new.data).encoder_outs[
            "audio_encoder"
        ]
        assert old_state.keys() == new_state.keys()
        for key in old_state:
            assert torch.equal(old_state[key], new_state[key])


@pytest.mark.accelerator
def test_audio_encoder_cache_owns_cuda_output_before_replay():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cuda_device = torch.device("cuda", torch.cuda.current_device())
    from sglang_omni.scheduling.stage_cache import StageOutputCache

    class ReusedOutputEncoder:
        def __init__(self):
            self.calls = 0
            self.buffer = torch.zeros(8, 1, device=cuda_device)
            self.length = torch.zeros(1, dtype=torch.long, device=cuda_device)

        def __call__(self, **kwargs):
            self.calls += 1
            size = int(kwargs["audio_feature_lengths"].sum())
            self.buffer.fill_(self.calls)
            self.length.fill_(size)
            return {
                "audio_embeds": self.buffer[:size],
                "audio_feature_lengths": self.length,
                "audio_output_lengths": self.length,
            }

    model = ReusedOutputEncoder()
    cache = StageOutputCache(max_size=64, max_bytes=4 * 1024**3, cache_device="cpu")
    first = _batch_audio_encoder_payloads(
        [_payload("a1", "a", 3)], model=model, cache=cache
    )
    expected = {
        key: value.cpu().clone()
        for key, value in Qwen3OmniPipelineState.from_dict(first[0].data)
        .encoder_outs["audio_encoder"]
        .items()
    }
    _batch_audio_encoder_payloads([_payload("b", "b", 5)], model=model, cache=cache)
    again = _batch_audio_encoder_payloads(
        [_payload("a2", "a", 3)], model=model, cache=cache
    )
    assert model.calls == 2
    actual = Qwen3OmniPipelineState.from_dict(again[0].data).encoder_outs[
        "audio_encoder"
    ]
    for key, value in expected.items():
        assert actual[key].device.type == "cpu"
        assert torch.equal(actual[key], value)
