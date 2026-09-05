# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MiniCPM-o cross-request encoder batching (GPU-free)."""

from __future__ import annotations

import asyncio
import json

import pytest
import torch

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.stages import (
    _batch_encoder_payloads,
    _create_encoder_executor,
    _encoder_request_cost,
)
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.stage_cache import StageOutputCache

MEL = 8


def _payload(state: MiniCPMOPipelineState, request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hi", params={}, metadata={}),
        data=state.to_dict(),
    )


def _audio_payload(
    request_id: str, lens: list[int], *, time: int = 300, cache_key: str | None = None
) -> StagePayload:
    inputs = {
        "audio_features": torch.zeros(len(lens), MEL, time),
        "audio_feature_lens": torch.tensor(lens, dtype=torch.long),
    }
    if cache_key is not None:
        inputs["cache_key"] = cache_key
    return _payload(
        MiniCPMOPipelineState(encoder_inputs={"audio_encoder": inputs}), request_id
    )


def _image_payload(
    request_id: str, num_slices: int, *, cache_key: str | None = None
) -> StagePayload:
    inputs = {
        "pixel_values": [torch.zeros(3, 2, 2) for _ in range(num_slices)],
        "tgt_sizes": torch.ones(num_slices, 2, dtype=torch.int32),
    }
    if cache_key is not None:
        inputs["cache_key"] = cache_key
    return _payload(
        MiniCPMOPipelineState(encoder_inputs={"image_encoder": inputs}), request_id
    )


def _embeds(payload: StagePayload, stage: str, key: str) -> torch.Tensor:
    return MiniCPMOPipelineState.from_dict(payload.data).encoder_outs[stage][key]


class _FakeAudioEncoder:
    """One output row per 100 mel frames; each row carries its chunk's length
    so a split can be checked against the request that owned the chunk."""

    def __init__(self):
        self.calls: list[tuple[tuple[int, ...], list[int]]] = []

    def pooled_feature_lens(self, audio_feature_lens: torch.Tensor) -> torch.Tensor:
        return (audio_feature_lens.reshape(-1) // 100).to(torch.int32)

    def __call__(self, *, audio_features, audio_feature_lens):
        lens = audio_feature_lens.reshape(-1).tolist()
        self.calls.append((tuple(audio_features.shape), lens))
        rows = [
            torch.full((int(p), 4), float(length))
            for p, length in zip(self.pooled_feature_lens(audio_feature_lens), lens)
        ]
        return {"audio_embeds": torch.cat(rows, dim=0)}


class _FakeImageEncoder:
    """query_num rows per slice; each row carries the slice's batch index."""

    query_num = 3

    def __init__(self):
        self.calls: list[int] = []

    def __call__(self, *, pixel_values, tgt_sizes):
        self.calls.append(len(pixel_values))
        rows = [
            torch.full((self.query_num, 4), float(i)) for i in range(len(pixel_values))
        ]
        return {"image_embeds": torch.cat(rows, dim=0)}


def test_audio_batch_is_one_forward_split_by_pooled_rows():
    model = _FakeAudioEncoder()
    payloads = [
        _audio_payload("a", [300, 300, 100]),
        _audio_payload("b", [300, 100], time=200),
        _audio_payload("c", [200]),
    ]
    out = _batch_encoder_payloads(
        payloads, stage_name="audio_encoder", model=model, cache=None
    )

    # one forward over all six chunks, padded to the longest mel axis
    assert model.calls == [((6, MEL, 300), [300, 300, 100, 300, 100, 200])]
    assert [p.request_id for p in out] == ["a", "b", "c"]
    a, b, c = (_embeds(p, "audio_encoder", "audio_embeds") for p in out)
    assert a[:, 0].tolist() == [300.0] * 6 + [100.0]
    assert b[:, 0].tolist() == [300.0] * 3 + [100.0]
    assert c[:, 0].tolist() == [200.0] * 2


def test_image_batch_is_one_forward_split_by_slices():
    model = _FakeImageEncoder()
    out = _batch_encoder_payloads(
        [_image_payload("a", 2), _image_payload("b", 1)],
        stage_name="image_encoder",
        model=model,
        cache=None,
    )
    assert model.calls == [3]
    a, b = (_embeds(p, "image_encoder", "image_embeds") for p in out)
    assert a[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert b[:, 0].tolist() == [2.0, 2.0, 2.0]


def test_single_active_payload_uses_plain_forward():
    model = _FakeAudioEncoder()
    out = _batch_encoder_payloads(
        [_audio_payload("a", [300, 100])],
        stage_name="audio_encoder",
        model=model,
        cache=None,
    )
    assert model.calls == [((2, MEL, 300), [300, 100])]
    assert _embeds(out[0], "audio_encoder", "audio_embeds").shape == (4, 4)


def test_same_media_in_one_batch_is_encoded_once():
    model = _FakeAudioEncoder()
    cache = StageOutputCache(max_size=8, max_bytes=1 << 20, cache_device="cpu")
    out = _batch_encoder_payloads(
        [
            _audio_payload("lead", [300], cache_key="same"),
            _audio_payload("dup", [300], cache_key="same"),
            _audio_payload("other", [200], cache_key="other"),
        ],
        stage_name="audio_encoder",
        model=model,
        cache=cache,
    )
    # the duplicate's chunk is not in the forward; it copies the leader's rows
    assert model.calls == [((2, MEL, 300), [300, 200])]
    assert [p.request_id for p in out] == ["lead", "dup", "other"]
    lead, dup, other = (_embeds(p, "audio_encoder", "audio_embeds") for p in out)
    assert torch.equal(lead, dup)
    assert other[:, 0].tolist() == [200.0, 200.0]
    assert cache.get("same") is not None and cache.get("other") is not None


def test_cache_hits_skip_the_forward():
    model = _FakeAudioEncoder()
    cache = StageOutputCache(max_size=8, max_bytes=1 << 20, cache_device="cpu")
    cache.put("hit", {"audio_embeds": torch.full((3, 4), 7.0)})
    out = _batch_encoder_payloads(
        [_audio_payload("a", [300], cache_key="hit")],
        stage_name="audio_encoder",
        model=model,
        cache=cache,
    )
    assert model.calls == []
    assert _embeds(out[0], "audio_encoder", "audio_embeds")[:, 0].tolist() == [7.0] * 3


def test_payload_without_encoder_input_passes_through():
    model = _FakeAudioEncoder()
    state = MiniCPMOPipelineState(prompt={"prompt_text": "x"})
    out = _batch_encoder_payloads(
        [_payload(state, "skip"), _audio_payload("a", [300])],
        stage_name="audio_encoder",
        model=model,
        cache=None,
    )
    assert [p.request_id for p in out] == ["skip", "a"]
    skipped = MiniCPMOPipelineState.from_dict(out[0].data)
    assert skipped.encoder_outs["audio_encoder"] == {}
    assert model.calls == [((1, MEL, 300), [300])]


def test_request_cost_counts_items():
    cost = _encoder_request_cost("audio_encoder")
    assert cost(_audio_payload("a", [300, 300, 100])) == 3
    assert cost(_payload(MiniCPMOPipelineState(), "skip")) == 0
    image_cost = _encoder_request_cost("image_encoder")
    assert image_cost(_image_payload("b", 5)) == 5


def test_executor_batches_whatever_is_queued():
    model = _FakeAudioEncoder()
    sched = _create_encoder_executor(
        model, stage_name="audio_encoder", max_batch_size=4, max_batch_cost=48
    )
    for rid, lens in (("a", [300]), ("b", [300, 100]), ("c", [200])):
        sched.inbox.put(
            IncomingMessage(
                request_id=rid, type="new_request", data=_audio_payload(rid, lens)
            )
        )
    loop = asyncio.new_event_loop()
    try:
        first = sched._next_message()
        batch = sched._collect_batch(first)
        sched._run_batch(batch, loop)
    finally:
        loop.close()

    assert len(model.calls) == 1
    assert model.calls[0][1] == [300, 300, 100, 200]
    results = [sched.outbox.get_nowait() for _ in range(3)]
    assert [m.request_id for m in results] == ["a", "b", "c"]
    assert all(m.type == "result" for m in results)


def test_batch_cost_budget_defers_the_overflowing_payload():
    model = _FakeAudioEncoder()
    sched = _create_encoder_executor(
        model, stage_name="audio_encoder", max_batch_size=8, max_batch_cost=3
    )
    for rid, lens in (("a", [300, 300]), ("b", [300, 100]), ("c", [200])):
        sched.inbox.put(
            IncomingMessage(
                request_id=rid, type="new_request", data=_audio_payload(rid, lens)
            )
        )
    batch = sched._collect_batch(sched._next_message())
    # 2 + 2 chunks exceed the 3-chunk budget, so "b" waits for the next batch
    assert [m.request_id for m in batch] == ["a"]


@pytest.fixture()
def event_dir(tmp_path):
    recorder = get_recorder()
    assert not recorder.is_active()
    recorder.start("test-run", str(tmp_path), "encoders")
    yield tmp_path
    recorder.stop()


def _events(event_dir) -> list[dict]:
    events = []
    for path in sorted(event_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fp:
            events.extend(json.loads(line) for line in fp)
    return events


def test_batch_events_carry_batch_size_and_dedup(event_dir):
    model = _FakeAudioEncoder()
    cache = StageOutputCache(max_size=8, max_bytes=1 << 20, cache_device="cpu")
    _batch_encoder_payloads(
        [
            _audio_payload("lead", [300], cache_key="same"),
            _audio_payload("dup", [300], cache_key="same"),
            _audio_payload("other", [200], cache_key="other"),
        ],
        stage_name="audio_encoder",
        model=model,
        cache=cache,
    )
    ends = {
        e["request_id"]: e["metadata"]
        for e in _events(event_dir)
        if e["event_name"] == "encoder_end"
    }
    assert ends["lead"] == {
        "modality": "audio",
        "batch_size": 2,
        "cacheable": True,
        "cache_hit": False,
        "status": "ok",
    }
    assert ends["other"]["batch_size"] == 2
    assert ends["dup"]["cache_hit"] is True
    assert ends["dup"]["dedup_same_batch"] is True
    starts = [e for e in _events(event_dir) if e["event_name"] == "encoder_start"]
    assert {e["request_id"] for e in starts} == {"lead", "dup", "other"}
