# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
    FunCosyVoice3StreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


class _TokenVocoder:
    """Encode token IDs as samples so reordering and loss remain observable."""

    def __init__(self):
        self.batch_sizes = []

    def first_hop_batch(self, items):
        self.batch_sizes.append(len(items))
        return [
            item.token[:, None, :-3].float().repeat_interleave(2, dim=-1)
            for item in items
        ]

    def _hift_delta(self, mel, *, hift_mel, speech_offset, finalize):
        return mel.reshape(-1), None, 0

    def token2wav_chunk(self, *, token, token_offset, finalize, **kwargs):
        end = token.shape[-1] if finalize else token.shape[-1] - 3
        wav = token[:, token_offset:end].float().repeat_interleave(2, dim=-1)
        return wav.reshape(-1), None, 0


def _payload(request_id, *, stream=True):
    state = FunCosyVoice3State(
        stream=stream,
        flow_prompt_speech_token=torch.zeros(1, 25, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 50, 80),
        flow_embedding=torch.ones(1, 192),
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": stream}),
        data=state.to_dict(),
    )


def _chunk(request_id, tokens, chunk_id=0, *, conditioning=False):
    metadata = {"modality": "audio_codes", "stream": True}
    if conditioning:
        metadata.update(_payload(request_id).data)
    return IncomingMessage(
        request_id=request_id,
        type="stream_chunk",
        data=StreamItem(
            chunk_id=chunk_id,
            data=torch.tensor(tokens, dtype=torch.long),
            from_stage="tts_engine",
            metadata=metadata,
        ),
    )


@pytest.mark.parametrize("concurrency", [1, 16])
@pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
@pytest.mark.parametrize("late_payload", [False, True])
def test_backlogged_stream_preserves_every_token(concurrency, batch_size, late_payload):
    vocoder = _TokenVocoder()
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        vocoder, max_batch_size=batch_size
    )
    scheduler._first_hop_peer_wait_ms = 0
    request_ids = [f"req-{i}" for i in range(concurrency)]
    expected = {
        rid: list(range(i * 100, i * 100 + 83)) for i, rid in enumerate(request_ids)
    }
    for rid in request_ids:
        if not late_payload:
            scheduler._on_streaming_new_request(rid, _payload(rid))
        scheduler.inbox.put(_chunk(rid, expected[rid][:28], conditioning=True))
        scheduler.inbox.put(_chunk(rid, expected[rid][28:53], 1))
    for rid in request_ids:
        scheduler.inbox.put(_chunk(rid, expected[rid][53:78], 2))
        scheduler.inbox.put(_chunk(rid, expected[rid][78:], 3))
        scheduler.inbox.put(IncomingMessage(rid, "stream_done"))
        if late_payload:
            scheduler.inbox.put(IncomingMessage(rid, "new_request", _payload(rid)))

    while scheduler._pending_messages or not scheduler.inbox.empty():
        scheduler._handle_message(scheduler._next_message(), None)

    received = {rid: [] for rid in request_ids}
    completed = []
    while not scheduler.outbox.empty():
        msg = scheduler.outbox.get_nowait()
        assert msg.type != "error", str(msg.data)
        if msg.type == "stream":
            assert msg.request_id not in completed
            received[msg.request_id].append(
                np.frombuffer(msg.data["audio_waveform"], dtype=np.float32)
            )
        elif msg.type == "result":
            completed.append(msg.request_id)
    assert sorted(completed) == sorted(request_ids)
    for rid in request_ids:
        np.testing.assert_array_equal(
            np.concatenate(received[rid]), np.repeat(expected[rid], 2)
        )
    assert not scheduler._stream_states
    assert not scheduler._pending_done
    if concurrency > 1 and batch_size > 1:
        assert any(size > 1 for size in vocoder.batch_sizes)


@pytest.mark.parametrize(
    "reader",
    ["_ingest_ready_inbox", "_wait_for_first_hop_peers", "_wait_for_follow_up_peers"],
)
def test_peer_readers_consume_pending_tokens_before_newer_inbox_tokens(reader):
    scheduler = FunCosyVoice3StreamingVocoderScheduler(_TokenVocoder())
    scheduler._first_hop_peer_wait_ms = 0
    for rid in ("a", "b"):
        scheduler._on_streaming_new_request(rid, _payload(rid))
    scheduler._ingest_stream_item("a", _chunk("a", list(range(78))).data)
    scheduler._ingest_stream_item("b", _chunk("b", [0]).data)
    if reader == "_wait_for_follow_up_peers":
        for state in scheduler._stream_states.values():
            state.token_offset = 25
            state.hop_len = 50
    scheduler._pending_messages.append(_chunk("b", [1], 1))
    scheduler.inbox.put(_chunk("b", [2], 2))

    getattr(scheduler, reader)()

    assert scheduler._stream_states["b"].tokens == [0, 1, 2]
    assert not scheduler._pending_messages


@pytest.mark.parametrize("barrier_type", ["stream_done", "new_request"])
def test_peer_ingest_does_not_cross_pending_control_message(barrier_type):
    scheduler = FunCosyVoice3StreamingVocoderScheduler(_TokenVocoder())
    barrier = IncomingMessage("a", barrier_type, _payload("a", stream=False))
    scheduler._pending_messages.append(barrier)
    newer = _chunk("b", [99])
    scheduler.inbox.put(newer)

    scheduler._ingest_ready_inbox()

    assert list(scheduler._pending_messages) == [barrier]
    assert scheduler.inbox.get_nowait() is newer
    assert not scheduler._stream_states


def test_chunk_collector_preserves_existing_pending_order():
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        _TokenVocoder(), max_batch_size=2
    )
    second = _chunk("a", [1], 1)
    third = _chunk("a", [2], 2)
    peer = _chunk("b", [10])
    done = IncomingMessage("a", "stream_done")
    scheduler._pending_messages.extend([second, third, peer])
    scheduler.inbox.put(done)

    batch = scheduler._collect_stream_chunk_batch(_chunk("a", [0]))

    assert [msg.request_id for msg in batch] == ["a", "b"]
    assert scheduler._next_message() is second
    assert scheduler._next_message() is third
    assert scheduler._next_message() is done
