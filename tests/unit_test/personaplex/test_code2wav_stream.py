# SPDX-License-Identifier: Apache-2.0
"""The streaming codec stage renders per-request chunks into the whole-reply waveform."""

from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.personaplex.architecture import SAMPLE_RATE
from sglang_omni.models.personaplex.code2wav_stream import PersonaPlexCode2WavScheduler
from sglang_omni.models.personaplex.components.mimi import MimiCodec
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest


def _random_codec() -> MimiCodec:
    torch.manual_seed(0)
    codec = MimiCodec().eval()
    with torch.no_grad():
        for parameter in codec.parameters():
            parameter.normal_(std=0.05)
        for module in codec.modules():
            if hasattr(module, "embedding_sum"):
                module.embedding_sum.normal_()
                module.cluster_usage.fill_(1.0)
    return codec


def _waveform(payload: dict) -> torch.Tensor:
    assert payload["sample_rate"] == SAMPLE_RATE
    return torch.from_numpy(
        np.frombuffer(payload["audio_waveform"], dtype=np.float32).copy()
    )


def _start(scheduler, request_id: str) -> StagePayload:
    payload = StagePayload(request_id, request=OmniRequest(inputs={}), data={})
    scheduler._stream_payloads[request_id] = payload
    scheduler.on_streaming_new_request(request_id, payload)
    return payload


def test_interleaved_requests_stream_their_own_waveforms():
    codec = _random_codec()
    scheduler = PersonaPlexCode2WavScheduler(codec, compute_fn=lambda payload: payload)
    codes = {
        "a": torch.randint(0, 2048, (4, 8), generator=torch.Generator().manual_seed(1)),
        "b": torch.randint(0, 2048, (4, 8), generator=torch.Generator().manual_seed(2)),
    }
    whole = {rid: codec.decode(c.T[None])[0, 0] for rid, c in codes.items()}
    payloads = {rid: _start(scheduler, rid) for rid in codes}

    streamed = {rid: [] for rid in codes}

    def push(rid: str, chunk: torch.Tensor) -> None:
        (message,) = scheduler.on_stream_chunk(rid, SimpleNamespace(data=chunk))
        assert message.type == "stream"
        streamed[rid].append(_waveform(message.data))

    for frame in range(4):
        push("a", codes["a"][frame : frame + 1])
        if frame % 2 == 1:
            push("b", codes["b"][frame - 1 : frame + 1])

    for rid in codes:
        torch.testing.assert_close(
            torch.cat(streamed[rid]), whole[rid], atol=1e-5, rtol=1e-5
        )
        (result,) = scheduler.on_stream_done(rid)
        assert result.type == "result"
        assert result.data.request is payloads[rid].request
        torch.testing.assert_close(
            _waveform(result.data.data), whole[rid], atol=1e-5, rtol=1e-5
        )


def test_abort_clears_stream_state():
    scheduler = PersonaPlexCode2WavScheduler(
        _random_codec(), compute_fn=lambda payload: payload
    )
    payload = _start(scheduler, "a")
    assert scheduler.is_streaming_payload(payload)
    scheduler.clear_stream_state("a")
    assert not scheduler.is_streaming_payload(payload)
    assert scheduler.on_stream_done("never-started") == []
