# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.dots_tts.vocoder import DotsTTSStreamingVocoder
from sglang_omni.models.dots_tts.vocoder_slot_pool import (
    DotsVocoderSlotPool,
    append_decoder_input_per_row,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload


class RecordingSlotPool:
    def __init__(self, *, num_slots: int = 4) -> None:
        self.num_slots = num_slots
        self.free = list(reversed(range(num_slots)))
        self.in_use: set[int] = set()
        self.steps: list[dict[int, torch.Tensor]] = []
        self.flushes: list[int] = []

    def acquire(self) -> int:
        if not self.free:
            raise RuntimeError(
                f"dots.tts streaming vocoder admission failed: ran out of slots "
                f"(num_slots={self.num_slots})"
            )
        slot = self.free.pop()
        self.in_use.add(slot)
        return slot

    def release(self, slot: int) -> None:
        slot = int(slot)
        if slot not in self.in_use:
            return
        self.in_use.remove(slot)
        self.free.append(slot)

    def step(self, slot_latents: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        assert all(latents.shape[1] < 32 for latents in slot_latents.values())
        self.steps.append(
            {slot: latents.clone() for slot, latents in slot_latents.items()}
        )
        return {slot: torch.full((1, 1, 8), float(slot + 1)) for slot in slot_latents}

    def flush(self, slot: int) -> torch.Tensor:
        self.flushes.append(int(slot))
        return torch.full((1, 1, 4), float(slot + 1))


class FakeInference:
    """Minimal VocoderInference surface for DotsVocoderSlotPool unit tests."""

    def __init__(self, *, latent_dim: int = 5, hop_size: int = 2) -> None:
        self.vocoder = SimpleNamespace(
            hop_size=hop_size,
            h=SimpleNamespace(latent_dim=latent_dim, causal=True),
        )
        self.batch_steps: list[tuple[int, int]] = []
        self.latent_dim = latent_dim
        self.hop_size = hop_size

    def init_stream_state(self, *, batch_size: int, chunk_size: int):
        window = torch.zeros(batch_size, self.latent_dim, chunk_size + 4)
        hidden = torch.zeros(1, batch_size, 8)
        return SimpleNamespace(
            lstm_hidden=(hidden, hidden.clone()),
            decoder=SimpleNamespace(window=window, chunk_size=chunk_size),
        )

    def _decoder_stream_lookahead(
        self,
    ) -> int:  # noqa: leading-underscore  # upstream name
        return 1

    def _validate_stream_latents(
        self, latents: torch.Tensor
    ) -> None:  # noqa: leading-underscore  # upstream name
        if latents.ndim != 3 or int(latents.shape[1]) != self.latent_dim:
            raise ValueError(f"bad latents {tuple(latents.shape)}")

    def _decode_stream_latents(
        self, latents, hidden
    ):  # noqa: leading-underscore  # upstream name
        batch, channels, frames = latents.shape
        self.batch_steps.append((batch, frames))
        decoder_input = latents.clone()
        return decoder_input, (hidden[0].clone(), hidden[1].clone())

    def _decode_stream_window(
        self, window: torch.Tensor
    ) -> torch.Tensor:  # noqa: leading-underscore  # upstream name
        return torch.zeros(
            window.size(0), 1, window.size(-1) * self.hop_size, dtype=window.dtype
        )


def make_codec(*, latent_dim: int = 5, patch_size: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        inference=FakeInference(latent_dim=latent_dim),
        lock=threading.RLock(),
        sample_rate=48000,
        patch_size=patch_size,
        latent_dim=latent_dim,
        device=torch.device("cpu"),
        hop_size=2,
    )


def patch(value: float = 0.0, *, frames: int = 3, dim: int = 5) -> torch.Tensor:
    return torch.full((1, frames, dim), value)


def test_append_decoder_input_per_row_matches_scalar_when_ages_equal() -> None:
    window = torch.zeros(2, 2, 6)
    window[0, :, :2] = 1
    window[1, :, :2] = 1
    decoder_input = torch.full((2, 2, 2), 3.0)
    valid = torch.tensor([2, 2], dtype=torch.int64)
    out = append_decoder_input_per_row(decoder_input, window, valid)
    assert out.shape == window.shape
    assert torch.equal(out[0], out[1])


def test_append_decoder_input_keeps_independent_row_ages() -> None:
    window = torch.zeros(2, 1, 6)
    window[0, :, :1] = 1
    window[1, :, :4] = 2
    decoder_input = torch.tensor([[[7.0, 8.0]], [[9.0, 10.0]]])
    valid = torch.tensor([1, 4], dtype=torch.int64)
    out = append_decoder_input_per_row(decoder_input, window, valid)
    assert out[0, 0, :3].tolist() == [1.0, 7.0, 8.0]
    assert out[1, 0, :6].tolist() == [2.0, 2.0, 2.0, 2.0, 9.0, 10.0]


def test_slot_pool_batches_equal_t_and_preserves_independent_counters() -> None:
    inference = FakeInference()
    pool = DotsVocoderSlotPool(inference, num_slots=4, chunk_size=6)
    s0 = pool.acquire()
    s1 = pool.acquire()
    older = torch.ones(1, 3, 5)
    newer = torch.full((1, 3, 5), 2.0)

    # note (guozhihao-224): age s0 alone so total_frames diverge before the
    # shared step.
    pool.step({s0: older})
    assert inference.batch_steps == [(1, 3)]

    pool.step({s0: older, s1: newer})
    assert inference.batch_steps[-1] == (2, 3)
    assert pool.total_frames[s0] == 6
    assert pool.total_frames[s1] == 3

    pool.release(s0)
    reused = pool.acquire()
    assert reused == s0
    assert pool.total_frames[reused] == 0


def test_slot_pool_rejects_mixed_step_lengths() -> None:
    pool = DotsVocoderSlotPool(FakeInference(), num_slots=2, chunk_size=6)
    a = pool.acquire()
    b = pool.acquire()
    with pytest.raises(ValueError, match="uniform latent length"):
        pool.step({a: torch.zeros(1, 2, 5), b: torch.zeros(1, 3, 5)})


def test_streaming_coalesces_equal_t_requests_into_one_pool_step() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        max_batch_size=4,
        stream_slots=4,
        slot_pool=pool,
    )
    assert vocoder.can_batch_stream_chunks is True
    assert vocoder.stream_chunk_batch_max == 4

    for request_id in ("a", "b"):
        state = vocoder.create_stream_state(request_id)
        vocoder.stream_states[request_id] = state
        vocoder.ingest(request_id, state, patch(1.0 if request_id == "a" else 2.0))

    participants = vocoder.select_step_participants()
    assert {request_id for request_id, _ in participants} == {"a", "b"}
    plan = vocoder.build_step_plan(participants)
    assert plan.take_patches == 1
    assert len(plan.slot_latents) == 2
    decoded = vocoder.run_step(participants, plan)

    assert len(pool.steps) == 1
    assert len(pool.steps[0]) == 2
    assert set(decoded) == {"a", "b"}


def test_select_step_participants_respects_max_batch_size() -> None:
    pool = RecordingSlotPool(num_slots=8)
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        max_batch_size=2,
        stream_slots=8,
        slot_pool=pool,
    )
    assert vocoder.stream_chunk_batch_max == 2

    for request_id in ("a", "b", "c", "d"):
        state = vocoder.create_stream_state(request_id)
        vocoder.stream_states[request_id] = state
        vocoder.ingest(request_id, state, patch(float(ord(request_id))))

    participants = vocoder.select_step_participants()
    assert len(participants) == 2
    plan = vocoder.build_step_plan(participants)
    vocoder.run_step(participants, plan)
    assert len(pool.steps[0]) == 2

    # note (guozhihao-224): pump drains the backlog across capped steps.
    remaining = vocoder.select_step_participants()
    assert len(remaining) == 2


def test_stream_chunk_batch_cap_follows_max_batch_size_not_slots() -> None:
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        max_batch_size=8,
        stream_slots=1,
        slot_pool=RecordingSlotPool(num_slots=1),
    )
    assert vocoder.stream_chunk_batch_max == 8
    assert vocoder.stream_slots == 1


def test_streaming_groups_by_exact_frame_count() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        stream_slots=4,
        slot_pool=pool,
    )
    early = vocoder.create_stream_state("early")
    steady = vocoder.create_stream_state("steady")
    vocoder.stream_states["early"] = early
    vocoder.stream_states["steady"] = steady

    vocoder.ingest("early", early, patch(1.0))
    # note (guozhihao-224): past the first-two-patch fast path so take_patches
    # diverges from early.
    steady.received_patches = 3
    steady.pending = [patch(2.0), patch(3.0)]
    steady.slot = pool.acquire()

    participants = vocoder.select_step_participants()
    frames = {vocoder.step_frames(state) for _, state in participants}
    assert len(frames) == 1
    plan = vocoder.build_step_plan(participants)
    assert len({int(t.shape[1]) for t in plan.slot_latents.values()}) == 1


def test_stream_done_flushes_and_releases_slot() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        slot_pool=pool,
    )
    state = vocoder.create_stream_state("req")
    vocoder.ingest("req", state, patch(1.0))
    slot = state.slot
    assert slot is not None
    waveform = vocoder.decode_delta("req", state, is_final=True)
    assert waveform is not None
    assert state.slot is None
    assert pool.flushes == [slot]
    assert slot not in pool.in_use


def test_release_stream_resources_returns_slot() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=False,
        slot_pool=pool,
    )
    state = vocoder.create_stream_state("req")
    vocoder.ingest("req", state, patch())
    slot = state.slot
    vocoder.release_stream_resources("req", state)
    assert state.slot is None
    assert slot not in pool.in_use


def test_slot_exhaustion_raises_clear_admission_error() -> None:
    pool = RecordingSlotPool(num_slots=1)
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        stream_slots=1,
        slot_pool=pool,
    )
    first = vocoder.create_stream_state("a")
    vocoder.ingest("a", first, patch())
    second = vocoder.create_stream_state("b")
    with pytest.raises(RuntimeError, match="ran out of slots"):
        vocoder.ingest("b", second, patch())


def test_on_stream_chunk_batch_uses_pool_not_compiled_stream_step() -> None:
    pool = RecordingSlotPool()
    codec = make_codec()
    vocoder = DotsTTSStreamingVocoder(
        codec,
        optimize=True,
        merge_steps=2,
        slot_pool=pool,
    )
    payload_state = vocoder.create_stream_state("req")
    vocoder.stream_states["req"] = payload_state
    vocoder.stream_payloads["req"] = SimpleNamespace(
        request_id="req",
        request=SimpleNamespace(params={"stream": True}),
        data={},
    )

    item = StreamItem(
        chunk_id=0,
        data=patch(1.0),
        from_stage="latent_engine",
        metadata={"modality": "audio_latents", "stream": True},
    )
    vocoder.on_stream_chunk_batch([("req", item)])

    assert pool.steps, "expected coalesced pool step"
    assert len(pool.steps[0]) == 1


def test_decode_delta_non_final_is_a_no_op() -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        merge_steps=2,
        slot_pool=pool,
    )
    state = vocoder.create_stream_state("req")
    vocoder.ingest("req", state, patch(1.0))
    assert vocoder.decode_delta("req", state, is_final=False) is None
    assert state.pending
    assert state.slot is not None
    assert not pool.steps


@pytest.mark.parametrize("enabled", [False, True])
def test_buffer_scheduling_preserves_patch_order_and_completes(enabled: bool) -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        slot_pool=pool,
        enable_buffer_scheduling=enabled,
    )
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs=[], params={"stream": True}),
        data={},
    )
    vocoder.handle_streaming_new_request("req", payload)
    for index in range(7):
        vocoder.on_stream_chunk_batch(
            [
                (
                    "req",
                    StreamItem(
                        chunk_id=index,
                        data=patch(float(index)),
                        from_stage="latent_engine",
                        metadata={"modality": "audio_latents", "stream": True},
                    ),
                )
            ]
        )
    if enabled:
        assert not pool.steps
    else:
        assert pool.steps
    vocoder.handle_stream_done("req")
    if enabled:
        assert not pool.steps
        for _ in range(8):
            if not vocoder.has_ready_work():
                break
            vocoder.run_ready_step()
    assert len(pool.flushes) == 1
    slot = pool.flushes[0]
    consumed = [step[slot] for step in pool.steps]
    assert [tensor.shape[1] for tensor in consumed] == [3, 3, 12, 3]
    torch.testing.assert_close(
        torch.cat(consumed, dim=1),
        torch.cat([patch(float(i)) for i in range(7)], dim=1),
    )
    messages = []
    while not vocoder.outbox.empty():
        messages.append(vocoder.outbox.get_nowait())
    assert [message.type for message in messages] == ["stream"] * 4 + ["result"]
    assert not pool.in_use
    assert not vocoder.stream_states
    assert not vocoder.stream_payloads


@pytest.mark.parametrize("termination", ["abort", "stop", "failure"])
def test_buffer_scheduling_releases_pending_slots(
    termination: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(), optimize=True, slot_pool=pool, enable_buffer_scheduling=True
    )
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs=[], params={"stream": True}),
        data={},
    )
    vocoder.handle_streaming_new_request("req", payload)
    state = vocoder.stream_states["req"]
    vocoder.ingest("req", state, patch())
    vocoder.handle_stream_done("req")
    assert state.done
    if termination == "abort":
        vocoder.abort("req")
    elif termination == "stop":
        vocoder.stop()
    else:

        def fail_step(slot_latents: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
            raise RuntimeError("decode failed")

        monkeypatch.setattr(pool, "step", fail_step)
        vocoder.run_ready_step()
    assert not vocoder.has_ready_work()
    assert not pool.in_use
    assert not vocoder.stream_states
    assert not pool.steps


def test_buffer_scheduling_prioritizes_first_audio_without_starving_started_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        make_codec(), optimize=True, slot_pool=pool, enable_buffer_scheduling=True
    )
    monkeypatch.setattr(
        "sglang_omni.models.dots_tts.vocoder.time.perf_counter", lambda: 1.0
    )
    for request_id in ["old-a", "old-b", "new"]:
        state = vocoder.create_stream_state(request_id)
        vocoder.stream_states[request_id] = state
        for _ in range(4):
            vocoder.ingest(request_id, state, patch())
        if request_id != "new":
            state.received_patches += 2
            state.playback_end_seconds = 2.0
            vocoder.mark_stream_emitted(request_id)
    vocoder.stream_states["new"].first_ingest_seconds = 0.97
    assert [request_id for request_id, _ in vocoder.select_step_participants()] == [
        "new"
    ]
    vocoder.stream_states["old-a"].ready_since_seconds = 0.8
    assert [request_id for request_id, _ in vocoder.select_step_participants()] == [
        "old-a",
        "old-b",
    ]
    pool.num_slots = 3
    vocoder.stream_states["old-a"].done = True
    vocoder.record_aborted_request_id("old-a")
    assert all(
        request_id != "old-a" for request_id, _ in vocoder.select_step_participants()
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("num_slots", [1, 2])
def test_eos_releases_full_pool_before_ingesting_successor(
    enabled: bool, num_slots: int
) -> None:
    pool = RecordingSlotPool(num_slots=num_slots)
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        slot_pool=pool,
        stream_slots=num_slots,
        enable_buffer_scheduling=enabled,
    )
    for generation in ["old", "new"]:
        for index in range(num_slots):
            request_id = f"{generation}-{index}"
            payload = StagePayload(
                request_id=request_id,
                request=OmniRequest(inputs=[], params={"stream": True}),
                data={},
            )
            vocoder.handle_streaming_new_request(request_id, payload)
            for chunk_id in range(16):
                vocoder.on_stream_chunk_batch(
                    [
                        (
                            request_id,
                            StreamItem(
                                chunk_id=chunk_id,
                                data=patch(float(chunk_id)),
                                from_stage="latent_engine",
                                metadata={"modality": "audio_latents", "stream": True},
                            ),
                        )
                    ]
                )
        for index in range(num_slots):
            vocoder.handle_stream_done(f"{generation}-{index}")
    while vocoder.has_ready_work():
        vocoder.run_ready_step()
    assert len(pool.flushes) == 2 * num_slots
    assert all(tensor.shape[1] <= 12 for step in pool.steps for tensor in step.values())
    assert not pool.in_use
    assert not vocoder.stream_states


def test_pressure_drain_failure_aborts_old_request_but_admits_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = RecordingSlotPool(num_slots=1)
    vocoder = DotsTTSStreamingVocoder(
        make_codec(),
        optimize=True,
        slot_pool=pool,
        stream_slots=1,
        enable_buffer_scheduling=True,
    )
    callbacks = []
    vocoder.abort_callback = callbacks.append
    for request_id in ["old", "new"]:
        vocoder.handle_streaming_new_request(
            request_id,
            StagePayload(
                request_id=request_id,
                request=OmniRequest(inputs=[], params={"stream": True}),
                data={},
            ),
        )
    vocoder.ingest("old", vocoder.stream_states["old"], patch())
    vocoder.handle_stream_done("old")

    def fail_step(slot_latents: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        raise RuntimeError("old decode failed")

    monkeypatch.setattr(pool, "step", fail_step)
    vocoder.on_stream_chunk_batch(
        [
            (
                "new",
                StreamItem(
                    chunk_id=0,
                    data=patch(),
                    from_stage="latent_engine",
                    metadata={"modality": "audio_latents", "stream": True},
                ),
            )
        ]
    )
    assert callbacks == ["old"]
    assert vocoder.is_aborted("old")
    assert not vocoder.is_aborted("new")
    assert vocoder.stream_states["new"].slot is not None
    messages = []
    while not vocoder.outbox.empty():
        messages.append(vocoder.outbox.get_nowait())
    assert [(message.request_id, message.type) for message in messages] == [
        ("old", "error")
    ]
    vocoder.stop()
    assert not pool.in_use
