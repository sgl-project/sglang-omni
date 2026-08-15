# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.vocoder import (
    DotsTTSBatchVocoder,
    DotsTTSStreamingVocoder,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


class _FakeInference:
    def __init__(self, hop_size: int) -> None:
        self.hop_size = hop_size
        self.inputs: list[torch.Tensor] = []
        self.input_data_ptrs: list[int] = []

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        self.input_data_ptrs.append(latents.data_ptr())
        self.inputs.append(latents.clone())
        rows = []
        for row in latents:
            samples = row[:, 0].repeat_interleave(self.hop_size)
            rows.append(samples.unsqueeze(0))
        return torch.stack(rows)


def _codec(*, latent_dim: int = 3, hop_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        latent_dim=latent_dim,
        hop_size=hop_size,
        sample_rate=48000,
        patch_size=4,
        lock=threading.RLock(),
        inference=_FakeInference(hop_size),
    )


def _latents(frames: int, value: float, *, latent_dim: int = 3) -> torch.Tensor:
    return torch.full((1, frames, latent_dim), value)


def _decode(
    vocoder: DotsTTSBatchVocoder, latents: list[torch.Tensor]
) -> list[tuple[torch.Tensor, int]]:
    items = [(DotsTTSState(), item) for item in latents]
    return asyncio.run(vocoder.decode_batch(items))


def _payload(
    request_id: str, frames: int, value: float, *, stream: bool = False
) -> StagePayload:
    state = DotsTTSState(generated_latents=_latents(frames, value))
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": stream}),
        data=state.to_dict(),
    )


def test_equal_length_inputs_use_one_audiovae_forward() -> None:
    codec = _codec()
    vocoder = DotsTTSBatchVocoder(codec)
    outputs = _decode(
        vocoder,
        [_latents(16, 1), _latents(16, 2), _latents(16, 3)],
    )

    assert vocoder._logged_batch
    assert len(codec.inference.inputs) == 1
    assert codec.inference.inputs[0].shape == (3, 16, 3)
    assert [waveform.shape for waveform, _ in outputs] == [(1, 1, 32)] * 3


def test_mixed_length_bucket_pads_and_crops_each_output() -> None:
    codec = _codec()
    outputs = _decode(
        DotsTTSBatchVocoder(codec),
        [_latents(17, 1), _latents(31, 2)],
    )

    [padded] = codec.inference.inputs
    assert padded.shape == (2, 31, 3)
    assert torch.count_nonzero(padded[0, 17:]) == 0
    assert [waveform.shape[-1] for waveform, _ in outputs] == [34, 62]
    assert torch.all(outputs[0][0] == 1)
    assert torch.all(outputs[1][0] == 2)


def test_multiple_buckets_restore_original_request_order() -> None:
    codec = _codec()
    outputs = _decode(
        DotsTTSBatchVocoder(codec),
        [_latents(33, 1), _latents(16, 2), _latents(40, 3), _latents(8, 4)],
    )

    assert [batch.shape[0] for batch in codec.inference.inputs] == [2, 2]
    assert [waveform[0, 0, 0].item() for waveform, _ in outputs] == [1, 2, 3, 4]


def test_single_input_preserves_batch_and_waveform_shapes() -> None:
    codec = _codec()
    vocoder = DotsTTSBatchVocoder(codec)
    latents = _latents(12, 5)
    [output] = _decode(vocoder, [latents])

    assert not vocoder._logged_batch
    assert codec.inference.input_data_ptrs == [latents.data_ptr()]
    assert codec.inference.inputs[0].shape == (1, 12, 3)
    assert output[0].shape == (1, 1, 24)
    assert output[1] == 48000


@pytest.mark.parametrize(
    ("latents", "message"),
    [
        (torch.zeros(4, 3), "shape"),
        (torch.zeros(2, 4, 3), "shape"),
        (torch.zeros(1, 0, 3), "at least one frame"),
        (torch.zeros(1, 4, 2), "latent_dim"),
    ],
)
def test_invalid_latents_are_rejected(latents: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _decode(DotsTTSBatchVocoder(_codec()), [latents])


def test_streaming_vocoder_enables_payload_and_chunk_batching() -> None:
    codec = _codec()
    scheduler = DotsTTSStreamingVocoder(
        codec,
        optimize=False,
    )

    assert scheduler._batch_fn is not None
    assert scheduler._max_batch_size == 4
    assert scheduler._stream_chunk_batch_max == 4
    assert scheduler._max_batch_wait_s == 0.002
    assert scheduler._can_batch_stream_chunks
    results = asyncio.run(
        scheduler._batch_fn([_payload("a", 16, 1), _payload("b", 16, 2)])
    )
    assert len(codec.inference.inputs) == 1
    assert [result.request_id for result in results] == ["a", "b"]
    assert scheduler.is_streaming_payload(_payload("stream", 16, 1, stream=True))


def test_non_streaming_batch_isolates_invalid_payload() -> None:
    codec = _codec()
    scheduler = DotsTTSStreamingVocoder(codec, optimize=False)
    invalid = _payload("bad", 16, 2)
    invalid.data["generated_latents"] = torch.zeros(2, 4, 3)

    scheduler._handle_new_request_batch(
        [
            IncomingMessage("good-a", "new_request", _payload("good-a", 16, 1)),
            IncomingMessage("bad", "new_request", invalid),
            IncomingMessage("good-b", "new_request", _payload("good-b", 16, 3)),
        ]
    )

    outputs = [scheduler.outbox.get_nowait() for _ in range(3)]
    by_request = {output.request_id: output for output in outputs}
    assert by_request["bad"].type == "error"
    assert isinstance(by_request["bad"].data, ValueError)
    assert by_request["good-a"].type == "result"
    assert by_request["good-b"].type == "result"
    assert [item.shape for item in codec.inference.inputs] == [(2, 16, 3)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"max_batch_wait_ms": -1}, "max_batch_wait_ms"),
        ({"stream_slots": 0}, "stream_slots"),
    ],
)
def test_invalid_batch_config_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DotsTTSStreamingVocoder(_codec(), optimize=False, **kwargs)
