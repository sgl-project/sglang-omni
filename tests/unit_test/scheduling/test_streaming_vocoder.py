# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for StreamingVocoderBase.

A fake subclass records every hook invocation so the tests can assert the base
skeleton drives each lifecycle event (one-backbone guarantee): state-registry
lifecycle, abort/late-chunk handling, stream_done-before-payload buffering, the
nothing-emitted fallback, contract-latch immutability, threshold/emit ordering,
coalesced stepping, and subclass-named error propagation.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling import streaming_vocoder
from sglang_omni.scheduling.streaming_vocoder import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    StreamingVocoderBase,
    resolve_initial_codec_chunk_frames,
)

SAMPLE_RATE = 16000


@dataclass
class _FakeStreamState:
    frames: list[torch.Tensor] = field(default_factory=list)
    decoded_upto: int = 0
    n_vq: int | None = None
    released: bool = False


class _FakeStreamingVocoder(StreamingVocoderBase[_FakeStreamState, Any]):
    """Minimal subclass: 1-D long rows accumulate; a decode emits every frame
    not yet decoded, concatenated in arrival order."""

    def __init__(
        self,
        *,
        threshold: int = 1,
        fallback: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        self.calls: list[str] = []
        self._threshold = threshold
        self._fallback = fallback
        super().__init__(None, sample_rate=SAMPLE_RATE, **kwargs)

    def create_stream_state(self, request_id: str) -> _FakeStreamState:
        self.calls.append(f"create:{request_id}")
        return _FakeStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: _FakeStreamState,
        source: Any,
        *,
        origin: str,
    ) -> None:
        self.calls.append(f"latch:{request_id}:{origin}")
        if origin != "stream metadata":
            return
        n_vq = source.get("n_vq")
        if n_vq is None:
            return
        n_vq = int(n_vq)
        if state.n_vq is not None and state.n_vq != n_vq:
            raise ValueError(
                f"fake stream n_vq changed for {request_id!r}: "
                f"{state.n_vq} -> {n_vq}"
            )
        state.n_vq = n_vq

    def validate_chunk(
        self, request_id: str, state: _FakeStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        self.calls.append(f"validate:{request_id}")
        if codes.ndim != 1:
            raise ValueError(f"fake stream chunk must be 1-D, got {tuple(codes.shape)}")
        return codes.to(dtype=torch.long)

    def ingest(
        self, request_id: str, state: _FakeStreamState, codes: torch.Tensor
    ) -> None:
        self.calls.append(f"ingest:{request_id}")
        state.frames.append(codes)

    def should_decode(self, state: _FakeStreamState, *, is_final: bool) -> bool:
        self.calls.append("should_decode")
        return len(state.frames) - state.decoded_upto >= self._threshold

    def decode_delta(
        self, request_id: str, state: _FakeStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        self.calls.append(f"decode:{request_id}:{'final' if is_final else 'delta'}")
        fresh = state.frames[state.decoded_upto :]
        if not fresh:
            return None
        state.decoded_upto = len(state.frames)
        return torch.cat(fresh).to(torch.float32)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: _FakeStreamState
    ) -> dict[str, Any]:
        self.calls.append(f"final:{request_id}")
        return {
            "modality": "audio",
            "sample_rate": self._sample_rate,
            "frames": len(state.frames),
        }

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: _FakeStreamState
    ) -> torch.Tensor | None:
        self.calls.append(f"fallback:{request_id}")
        return self._fallback

    def release_stream_resources(
        self, request_id: str, state: _FakeStreamState
    ) -> None:
        self.calls.append(f"release:{request_id}")
        state.released = True

    def on_serving_stop(self) -> None:
        self.calls.append("serving_stop")


class _CoalescingFakeVocoder(_FakeStreamingVocoder):
    """Coalescing variant: due streams (>= threshold undecoded frames) step
    together; run_step consumes each participant's fresh frames."""

    _can_batch_stream_chunks = True

    def __init__(self, *, fail_steps: bool = False, **kwargs: Any) -> None:
        self._fail_steps = fail_steps
        super().__init__(**kwargs)

    def select_step_participants(self) -> list[tuple[str, _FakeStreamState]]:
        self.calls.append("select")
        return [
            (request_id, state)
            for request_id, state in self._stream_state_items()
            if len(state.frames) - state.decoded_upto >= self._threshold
        ]

    def build_step_plan(
        self, participants: list[tuple[str, _FakeStreamState]]
    ) -> list[tuple[str, _FakeStreamState]]:
        self.calls.append("plan")
        return participants

    def run_step(
        self,
        participants: list[tuple[str, _FakeStreamState]],
        plan: list[tuple[str, _FakeStreamState]],
    ) -> dict[str, torch.Tensor]:
        self.calls.append("step")
        if self._fail_steps:
            raise RuntimeError("fake step exploded")
        out: dict[str, torch.Tensor] = {}
        for request_id, state in participants:
            fresh = state.frames[state.decoded_upto :]
            state.decoded_upto = len(state.frames)
            out[request_id] = torch.cat(fresh).to(torch.float32)
        return out


def _payload(
    request_id: str = "r", params: dict[str, Any] | None = None
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="", params={"stream": True, **(params or {})}),
        data={},
    )


def _item(values: list[int], metadata: Any = ...) -> StreamItem:
    if metadata is ...:
        metadata = {"stream": True}
    return StreamItem(
        chunk_id=0,
        data=torch.tensor(values, dtype=torch.long),
        from_stage="tts_engine",
        metadata=metadata,
    )


def _drain(scheduler: StreamingVocoderBase) -> list:
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def _waveform(data: dict[str, Any]) -> np.ndarray:
    assert data["audio_waveform_dtype"] == "float32"
    return np.frombuffer(data["audio_waveform"], dtype=np.float32).reshape(
        data["audio_waveform_shape"]
    )


def test_resolve_initial_codec_chunk_frames() -> None:
    assert resolve_initial_codec_chunk_frames(None, steady_chunk_frames=10) == 0
    assert resolve_initial_codec_chunk_frames({}, steady_chunk_frames=10) == 0
    assert (
        resolve_initial_codec_chunk_frames(
            {INITIAL_CODEC_CHUNK_FRAMES_PARAM: 3}, steady_chunk_frames=10
        )
        == 3
    )
    assert (
        resolve_initial_codec_chunk_frames(
            {INITIAL_CODEC_CHUNK_FRAMES_PARAM: 99}, steady_chunk_frames=10
        )
        == 10
    )
    with pytest.raises(TypeError, match="must be an integer"):
        resolve_initial_codec_chunk_frames(
            {INITIAL_CODEC_CHUNK_FRAMES_PARAM: "x"}, steady_chunk_frames=10
        )
    with pytest.raises(ValueError, match="must be >= 0"):
        resolve_initial_codec_chunk_frames(
            {INITIAL_CODEC_CHUNK_FRAMES_PARAM: -1}, steady_chunk_frames=10
        )
    with pytest.raises(ValueError, match="steady_chunk_frames"):
        resolve_initial_codec_chunk_frames({}, steady_chunk_frames=0)


def test_is_streaming_payload_gate_names_subclass() -> None:
    scheduler = _FakeStreamingVocoder()
    assert scheduler.is_streaming_payload(_payload())
    assert not scheduler.is_streaming_payload(
        StagePayload(request_id="r", request=OmniRequest(inputs="", params={}), data={})
    )
    bad = SimpleNamespace(request=SimpleNamespace(params="nope"))
    with pytest.raises(TypeError, match="_FakeStreamingVocoder"):
        scheduler.is_streaming_payload(bad)


def test_registry_lifecycle_and_hook_call_order() -> None:
    scheduler = _FakeStreamingVocoder(threshold=2)
    scheduler._on_chunk("r", _item([1]))
    assert "r" in scheduler._stream_states
    assert _drain(scheduler) == []
    scheduler._on_chunk("r", _item([2]))
    messages = _drain(scheduler)
    assert scheduler.calls == [
        "create:r",
        "latch:r:stream metadata",
        "validate:r",
        "ingest:r",
        "should_decode",
        "latch:r:stream metadata",
        "validate:r",
        "ingest:r",
        "should_decode",
        "decode:r:delta",
    ]
    # note (Gaokai): the base emit skeleton wraps every chunk (one-backbone);
    # payload shape and metadata must come from the base, not the fake.
    assert len(messages) == 1
    assert messages[0].type == "stream"
    assert messages[0].metadata == {"modality": "audio"}
    assert messages[0].data["sample_rate"] == SAMPLE_RATE
    assert messages[0].data["modality"] == "audio"
    np.testing.assert_array_equal(_waveform(messages[0].data), [1.0, 2.0])

    scheduler.calls.clear()
    scheduler._on_done("r")
    scheduler._on_streaming_new_request("r", _payload())
    messages = _drain(scheduler)
    assert scheduler.calls == [
        "latch:r:payload",
        "decode:r:final",
        "final:r",
        "release:r",
    ]
    assert [m.type for m in messages] == ["result"]
    result = messages[0].data
    assert isinstance(result, StagePayload)
    assert result.data == {
        "modality": "audio",
        "sample_rate": SAMPLE_RATE,
        "frames": 2,
    }
    assert scheduler._stream_states == {}
    assert scheduler._emitted_stream_ids == set()


def test_threshold_accumulate_then_flush_on_done() -> None:
    scheduler = _FakeStreamingVocoder(threshold=10)
    for value in (1, 2, 3):
        scheduler._on_chunk("r", _item([value]))
    assert _drain(scheduler) == []
    assert not any(call.startswith("decode:") for call in scheduler.calls)
    scheduler._on_done("r")
    scheduler._on_streaming_new_request("r", _payload())
    messages = _drain(scheduler)
    # note (Gaokai): done sequencing is flush remainder -> final stream chunk ->
    # terminal result; the flush must precede the result in the outbox.
    assert [m.type for m in messages] == ["stream", "result"]
    np.testing.assert_array_equal(_waveform(messages[0].data), [1.0, 2.0, 3.0])
    assert "fallback:r" not in scheduler.calls


def test_stream_done_before_payload_is_buffered() -> None:
    scheduler = _FakeStreamingVocoder(threshold=10)
    scheduler._on_chunk("r", _item([7]))
    scheduler._on_done("r")
    assert "r" in scheduler._pending_done
    assert _drain(scheduler) == []
    assert not any(call.startswith("decode:") for call in scheduler.calls)
    scheduler._on_streaming_new_request("r", _payload())
    messages = _drain(scheduler)
    assert [m.type for m in messages] == ["stream", "result"]
    np.testing.assert_array_equal(_waveform(messages[0].data), [7.0])
    assert scheduler._stream_states == {}


def test_nothing_emitted_fallback_decodes_whole_utterance() -> None:
    scheduler = _FakeStreamingVocoder(fallback=torch.tensor([9.0, 9.0]))
    scheduler._on_done("r")
    scheduler._on_streaming_new_request("r", _payload())
    messages = _drain(scheduler)
    assert "fallback:r" in scheduler.calls
    assert [m.type for m in messages] == ["stream", "result"]
    np.testing.assert_array_equal(_waveform(messages[0].data), [9.0, 9.0])


def test_nothing_emitted_fallback_none_keeps_terminal_result_only() -> None:
    scheduler = _FakeStreamingVocoder(fallback=None)
    scheduler._on_done("r")
    scheduler._on_streaming_new_request("r", _payload())
    messages = _drain(scheduler)
    assert "fallback:r" in scheduler.calls
    assert [m.type for m in messages] == ["result"]


def test_fallback_skipped_when_stream_emitted() -> None:
    scheduler = _FakeStreamingVocoder(threshold=1, fallback=torch.tensor([9.0]))
    scheduler._on_chunk("r", _item([1]))
    scheduler._on_done("r")
    scheduler._on_streaming_new_request("r", _payload())
    messages = _drain(scheduler)
    assert "fallback:r" not in scheduler.calls
    assert [m.type for m in messages] == ["stream", "result"]


def test_abort_releases_resources_and_late_chunks_never_recreate_state() -> None:
    scheduler = _FakeStreamingVocoder(threshold=10)
    scheduler._on_chunk("r", _item([1]))
    state = scheduler._stream_states["r"]
    scheduler.abort("r")
    assert "release:r" in scheduler.calls
    assert state.released
    assert scheduler._stream_states == {}
    assert scheduler._is_aborted("r")

    scheduler.calls.clear()
    scheduler._on_chunk("r", _item([2]))
    assert scheduler.calls == []
    assert scheduler._stream_states == {}
    assert _drain(scheduler) == []


def test_late_chunk_after_completed_stream_never_recreates_state() -> None:
    scheduler = _FakeStreamingVocoder(threshold=1)
    scheduler._on_streaming_new_request("r", _payload())
    scheduler._on_chunk("r", _item([1]))
    scheduler._on_done("r")
    assert [m.type for m in _drain(scheduler)] == ["stream", "result"]
    assert scheduler._stream_states == {}

    scheduler.calls.clear()
    scheduler._on_chunk("r", _item([2]))
    assert scheduler.calls == []
    assert scheduler._stream_states == {}
    assert _drain(scheduler) == []

    # note (Gaokai): a fresh new_request may reuse the id; only late chunks
    # without a new_request stay dropped.
    scheduler._on_streaming_new_request("r", _payload())
    scheduler._on_chunk("r", _item([3]))
    assert "r" in scheduler._stream_states


def test_completed_stream_ids_evict_oldest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streaming_vocoder, "_COMPLETED_STREAM_REQUEST_ID_LIMIT", 3)
    monkeypatch.setattr(streaming_vocoder, "_COMPLETED_STREAM_REQUEST_ID_RETAINED", 2)
    scheduler = _FakeStreamingVocoder(threshold=1)
    for rid in ("r0", "r1", "r2", "r3"):
        scheduler._record_completed_stream_request_id(rid)
    assert list(scheduler._completed_stream_request_ids) == ["r2", "r3"]


def test_contract_latch_is_immutable_per_request() -> None:
    scheduler = _FakeStreamingVocoder(threshold=10)
    scheduler._on_chunk("r", _item([1], {"stream": True, "n_vq": 4}))
    assert scheduler._stream_states["r"].n_vq == 4
    with pytest.raises(ValueError, match="n_vq changed"):
        scheduler.on_stream_chunk("r", _item([2], {"stream": True, "n_vq": 8}))
    # note (Gaokai): the latch runs before ingest, so the rejected chunk must
    # not have been buffered.
    assert len(scheduler._stream_states["r"].frames) == 1
    # note (Gaokai): re-latching the identical contract must stay legal.
    scheduler._on_chunk("r", _item([3], {"stream": True, "n_vq": 4}))
    assert scheduler._stream_states["r"].n_vq == 4
    assert len(scheduler._stream_states["r"].frames) == 2


def test_chunk_scaffold_errors_name_subclass() -> None:
    scheduler = _FakeStreamingVocoder()
    with pytest.raises(RuntimeError, match="_FakeStreamingVocoder.*missing metadata"):
        scheduler.on_stream_chunk("r1", _item([1], None))
    with pytest.raises(ValueError, match="modality must be audio_codes"):
        scheduler.on_stream_chunk(
            "r2", _item([1], {"stream": True, "modality": "text"})
        )
    with pytest.raises(RuntimeError, match=r"metadata\['stream'\] == True"):
        scheduler.on_stream_chunk("r3", _item([1], {"modality": "audio_codes"}))
    with pytest.raises(TypeError, match="_FakeStreamingVocoder.*torch.Tensor"):
        scheduler.on_stream_chunk(
            "r4",
            StreamItem(
                chunk_id=0,
                data=[1, 2],
                from_stage="tts_engine",
                metadata={"stream": True},
            ),
        )


def test_scaffold_errors_use_stream_source_hint() -> None:
    # note (Gaokai): client-visible scaffold errors carry the display hint
    # (e.g. moss passes "MOSS-TTS Local"), keeping migrated error text
    # byte-identical to the pre-refactor schedulers.
    scheduler = _FakeStreamingVocoder(stream_source_hint="Fake TTS")
    bad = SimpleNamespace(request=SimpleNamespace(params="nope"))
    with pytest.raises(TypeError, match="Fake TTS request params must be a dict"):
        scheduler.is_streaming_payload(bad)
    with pytest.raises(RuntimeError, match="Fake TTS.*missing metadata"):
        scheduler.on_stream_chunk("r", _item([1], None))


def test_create_stream_state_none_guard_names_subclass() -> None:
    class _NoneStateVocoder(_FakeStreamingVocoder):
        def create_stream_state(self, request_id: str) -> _FakeStreamState:
            return None  # type: ignore[return-value]

    scheduler = _NoneStateVocoder()
    with pytest.raises(RuntimeError, match="_NoneStateVocoder.create_stream_state"):
        scheduler.on_stream_chunk("r", _item([1]))


def test_coalesced_step_emits_for_all_participants_in_order() -> None:
    scheduler = _CoalescingFakeVocoder(threshold=2)
    scheduler.on_stream_chunk_batch([("a", _item([1])), ("b", _item([10]))])
    assert _drain(scheduler) == []
    scheduler.calls.clear()
    scheduler.on_stream_chunk_batch([("a", _item([2])), ("b", _item([20]))])
    messages = _drain(scheduler)
    hook_calls = [c for c in scheduler.calls if c in ("select", "plan", "step")]
    assert hook_calls == ["select", "plan", "step", "select"]
    assert [(m.request_id, m.type) for m in messages] == [
        ("a", "stream"),
        ("b", "stream"),
    ]
    np.testing.assert_array_equal(_waveform(messages[0].data), [1.0, 2.0])
    np.testing.assert_array_equal(_waveform(messages[1].data), [10.0, 20.0])
    assert scheduler._stream_has_emitted("a")
    assert scheduler._stream_has_emitted("b")


def test_step_omitting_a_participant_keeps_nothing_emitted_fallback() -> None:
    class _HoldbackVocoder(_CoalescingFakeVocoder):
        """run_step consumes both buffers but only returns audio for "a",
        modelling holdback/crossfade vocoders that emit nothing early on."""

        def run_step(
            self,
            participants: list[tuple[str, _FakeStreamState]],
            plan: list[tuple[str, _FakeStreamState]],
        ) -> dict[str, torch.Tensor]:
            self.calls.append("step")
            out: dict[str, torch.Tensor] = {}
            for request_id, state in participants:
                fresh = state.frames[state.decoded_upto :]
                state.decoded_upto = len(state.frames)
                if request_id != "b":
                    out[request_id] = torch.cat(fresh).to(torch.float32)
            return out

    scheduler = _HoldbackVocoder(threshold=1, fallback=torch.tensor([9.0]))
    scheduler.on_stream_chunk_batch([("a", _item([1])), ("b", _item([2]))])
    messages = _drain(scheduler)
    assert [(m.request_id, m.type) for m in messages] == [("a", "stream")]
    assert scheduler._stream_has_emitted("a")
    assert not scheduler._stream_has_emitted("b")

    scheduler._on_done("b")
    scheduler._on_streaming_new_request("b", _payload("b"))
    messages = _drain(scheduler)
    assert "fallback:b" in scheduler.calls
    assert [m.type for m in messages] == ["stream", "result"]
    np.testing.assert_array_equal(_waveform(messages[0].data), [9.0])


def test_coalescing_single_chunk_path_pumps_same_backbone() -> None:
    scheduler = _CoalescingFakeVocoder(threshold=1)
    scheduler._on_chunk("a", _item([5]))
    messages = _drain(scheduler)
    assert [c for c in scheduler.calls if c in ("select", "plan", "step")] == [
        "select",
        "plan",
        "step",
        "select",
    ]
    assert [(m.request_id, m.type) for m in messages] == [("a", "stream")]
    assert messages[0].metadata == {"modality": "audio"}


def test_step_failure_aborts_every_participant() -> None:
    scheduler = _CoalescingFakeVocoder(threshold=1, fail_steps=True)
    scheduler.on_stream_chunk_batch([("a", _item([1])), ("b", _item([2]))])
    messages = _drain(scheduler)
    errors = [m for m in messages if m.type == "error"]
    assert {m.request_id for m in errors} == {"a", "b"}
    assert all(m.type == "error" for m in messages)
    assert scheduler._stream_states == {}
    assert scheduler._is_aborted("a") and scheduler._is_aborted("b")
    assert "release:a" in scheduler.calls and "release:b" in scheduler.calls
    # note (Gaokai): late chunks after the failed step must never recreate
    # state: upstream AR stages abort lazily and can still deliver in-flight
    # chunks for an already-aborted request.
    scheduler.calls.clear()
    scheduler.on_stream_chunk_batch([("a", _item([3]))])
    assert scheduler._stream_states == {}
    assert not any(call.startswith("create:") for call in scheduler.calls)


def test_batched_ingest_failure_aborts_only_offender() -> None:
    scheduler = _CoalescingFakeVocoder(threshold=2)
    scheduler.on_stream_chunk_batch(
        [
            ("ok", _item([1])),
            ("bad", _item([1], None)),
            ("ok", _item([2])),
        ]
    )
    messages = _drain(scheduler)
    assert [m.request_id for m in messages if m.type == "error"] == ["bad"]
    assert scheduler._is_aborted("bad")
    assert "bad" not in scheduler._stream_states
    ok_streams = [m for m in messages if m.type == "stream"]
    assert [m.request_id for m in ok_streams] == ["ok"]
    np.testing.assert_array_equal(_waveform(ok_streams[0].data), [1.0, 2.0])


def test_missing_coalescing_hooks_name_subclass() -> None:
    class _FlaggedVocoder(_FakeStreamingVocoder):
        _can_batch_stream_chunks = True

    scheduler = _FlaggedVocoder(threshold=1)
    with pytest.raises(RuntimeError, match="_FlaggedVocoder.*select_step_participants"):
        scheduler.on_stream_chunk_batch([("r", _item([1]))])


def test_direct_chunk_entry_rejected_for_coalescing_schedulers() -> None:
    scheduler = _CoalescingFakeVocoder(threshold=1)
    with pytest.raises(
        RuntimeError, match="_CoalescingFakeVocoder.*on_stream_chunk_batch"
    ):
        scheduler.on_stream_chunk("r", _item([1]))
    # note (Gaokai): the rejection must precede ingestion so a wrong entry
    # point cannot leave half-registered state behind.
    assert scheduler._stream_states == {}


def test_stop_releases_live_streams_then_calls_serving_stop_hook() -> None:
    scheduler = _FakeStreamingVocoder(threshold=10)
    scheduler._on_chunk("a", _item([1]))
    scheduler._on_chunk("b", _item([2]))
    states = dict(scheduler._stream_states)
    scheduler.stop()
    assert scheduler._stream_states == {}
    assert scheduler._emitted_stream_ids == set()
    assert all(state.released for state in states.values())
    assert scheduler.calls[-3:] == ["release:a", "release:b", "serving_stop"]


def test_stop_release_failure_still_tears_down_session() -> None:
    class _ExplodingVocoder(_FakeStreamingVocoder):
        def release_stream_resources(
            self, request_id: str, state: _FakeStreamState
        ) -> None:
            super().release_stream_resources(request_id, state)
            if request_id == "a":
                raise RuntimeError("release exploded")

    scheduler = _ExplodingVocoder(threshold=10)
    scheduler._on_chunk("a", _item([1]))
    scheduler._on_chunk("b", _item([2]))
    scheduler.stop()
    assert scheduler._stream_states == {}
    assert scheduler.calls[-3:] == ["release:a", "release:b", "serving_stop"]
