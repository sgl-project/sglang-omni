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


class _RecordingSlotPool:
    def __init__(self, *, num_slots: int = 4) -> None:
        self.num_slots = num_slots
        self._free = list(reversed(range(num_slots)))
        self._in_use: set[int] = set()
        self.steps: list[dict[int, torch.Tensor]] = []
        self.flushes: list[int] = []

    def acquire(self) -> int:
        if not self._free:
            raise RuntimeError(
                f"dots.tts streaming vocoder admission failed: ran out of slots "
                f"(num_slots={self.num_slots})"
            )
        slot = self._free.pop()
        self._in_use.add(slot)
        return slot

    def release(self, slot: int) -> None:
        slot = int(slot)
        if slot not in self._in_use:
            return
        self._in_use.remove(slot)
        self._free.append(slot)

    def step(self, slot_latents: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        self.steps.append(
            {slot: latents.clone() for slot, latents in slot_latents.items()}
        )
        return {slot: torch.full((1, 1, 8), float(slot + 1)) for slot in slot_latents}

    def flush(self, slot: int) -> torch.Tensor:
        self.flushes.append(int(slot))
        return torch.full((1, 1, 4), float(slot + 1))


class _FakeInference:
    """Minimal VocoderInference surface for DotsVocoderSlotPool unit tests."""

    def __init__(self, *, latent_dim: int = 5, hop_size: int = 2) -> None:
        self.vocoder = SimpleNamespace(
            hop_size=hop_size,
            h=SimpleNamespace(latent_dim=latent_dim, causal=True),
        )
        self.batch_steps: list[tuple[int, int]] = []
        self._latent_dim = latent_dim
        self._hop_size = hop_size

    def init_stream_state(self, *, batch_size: int, chunk_size: int):
        window = torch.zeros(batch_size, self._latent_dim, chunk_size + 4)
        hidden = torch.zeros(1, batch_size, 8)
        return SimpleNamespace(
            lstm_hidden=(hidden, hidden.clone()),
            decoder=SimpleNamespace(window=window, chunk_size=chunk_size),
        )

    def _decoder_stream_lookahead(self) -> int:
        return 1

    def _validate_stream_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 3 or int(latents.shape[1]) != self._latent_dim:
            raise ValueError(f"bad latents {tuple(latents.shape)}")

    def _decode_stream_latents(self, latents, hidden):
        batch, _channels, frames = latents.shape
        self.batch_steps.append((batch, frames))
        decoder_input = latents.clone()
        return decoder_input, (hidden[0].clone(), hidden[1].clone())

    def _decode_stream_window(self, window: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            window.size(0), 1, window.size(-1) * self._hop_size, dtype=window.dtype
        )


def _codec(*, latent_dim: int = 5, patch_size: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        inference=_FakeInference(latent_dim=latent_dim),
        lock=threading.RLock(),
        sample_rate=48000,
        patch_size=patch_size,
        latent_dim=latent_dim,
        device=torch.device("cpu"),
        hop_size=2,
    )


def _patch(value: float = 0.0, *, frames: int = 3, dim: int = 5) -> torch.Tensor:
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
    inference = _FakeInference()
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
    assert pool._total_frames[s0] == 6
    assert pool._total_frames[s1] == 3

    pool.release(s0)
    reused = pool.acquire()
    assert reused == s0
    assert pool._total_frames[reused] == 0


def test_slot_pool_rejects_mixed_step_lengths() -> None:
    pool = DotsVocoderSlotPool(_FakeInference(), num_slots=2, chunk_size=6)
    a = pool.acquire()
    b = pool.acquire()
    with pytest.raises(ValueError, match="uniform latent length"):
        pool.step({a: torch.zeros(1, 2, 5), b: torch.zeros(1, 3, 5)})


def test_streaming_coalesces_equal_t_requests_into_one_pool_step() -> None:
    pool = _RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        _codec(),
        optimize=True,
        merge_steps=2,
        max_batch_size=4,
        stream_slots=4,
        slot_pool=pool,
    )
    assert vocoder._can_batch_stream_chunks is True
    assert vocoder._stream_chunk_batch_max == 4

    for request_id in ("a", "b"):
        state = vocoder.create_stream_state(request_id)
        vocoder._stream_states[request_id] = state
        vocoder.ingest(request_id, state, _patch(1.0 if request_id == "a" else 2.0))

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
    pool = _RecordingSlotPool(num_slots=8)
    vocoder = DotsTTSStreamingVocoder(
        _codec(),
        optimize=True,
        merge_steps=2,
        max_batch_size=2,
        stream_slots=8,
        slot_pool=pool,
    )
    assert vocoder._stream_chunk_batch_max == 2

    for request_id in ("a", "b", "c", "d"):
        state = vocoder.create_stream_state(request_id)
        vocoder._stream_states[request_id] = state
        vocoder.ingest(request_id, state, _patch(float(ord(request_id))))

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
        _codec(),
        optimize=True,
        max_batch_size=8,
        stream_slots=1,
        slot_pool=_RecordingSlotPool(num_slots=1),
    )
    assert vocoder._stream_chunk_batch_max == 8
    assert vocoder.stream_slots == 1


def test_streaming_groups_by_exact_frame_count() -> None:
    pool = _RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        _codec(),
        optimize=True,
        merge_steps=2,
        stream_slots=4,
        slot_pool=pool,
    )
    early = vocoder.create_stream_state("early")
    steady = vocoder.create_stream_state("steady")
    vocoder._stream_states["early"] = early
    vocoder._stream_states["steady"] = steady

    vocoder.ingest("early", early, _patch(1.0))
    # note (guozhihao-224): past the first-two-patch fast path so take_patches
    # diverges from early.
    steady.received_patches = 3
    steady.pending = [_patch(2.0), _patch(3.0)]
    steady.slot = pool.acquire()

    participants = vocoder.select_step_participants()
    frames = {vocoder._step_frames(state) for _, state in participants}
    assert len(frames) == 1
    plan = vocoder.build_step_plan(participants)
    assert len({int(t.shape[1]) for t in plan.slot_latents.values()}) == 1


def test_stream_done_flushes_and_releases_slot() -> None:
    pool = _RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        _codec(),
        optimize=True,
        merge_steps=2,
        slot_pool=pool,
    )
    state = vocoder.create_stream_state("req")
    vocoder.ingest("req", state, _patch(1.0))
    slot = state.slot
    assert slot is not None
    waveform = vocoder.decode_delta("req", state, is_final=True)
    assert waveform is not None
    assert state.slot is None
    assert pool.flushes == [slot]
    assert slot not in pool._in_use


def test_release_stream_resources_returns_slot() -> None:
    pool = _RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        _codec(),
        optimize=False,
        slot_pool=pool,
    )
    state = vocoder.create_stream_state("req")
    vocoder.ingest("req", state, _patch())
    slot = state.slot
    vocoder.release_stream_resources("req", state)
    assert state.slot is None
    assert slot not in pool._in_use


def test_slot_exhaustion_raises_clear_admission_error() -> None:
    pool = _RecordingSlotPool(num_slots=1)
    vocoder = DotsTTSStreamingVocoder(
        _codec(),
        optimize=True,
        stream_slots=1,
        slot_pool=pool,
    )
    first = vocoder.create_stream_state("a")
    vocoder.ingest("a", first, _patch())
    second = vocoder.create_stream_state("b")
    with pytest.raises(RuntimeError, match="ran out of slots"):
        vocoder.ingest("b", second, _patch())


def test_on_stream_chunk_batch_uses_pool_not_compiled_stream_step() -> None:
    pool = _RecordingSlotPool()
    codec = _codec()
    vocoder = DotsTTSStreamingVocoder(
        codec,
        optimize=True,
        merge_steps=2,
        slot_pool=pool,
    )
    payload_state = vocoder.create_stream_state("req")
    vocoder._stream_states["req"] = payload_state
    vocoder._stream_payloads["req"] = SimpleNamespace(
        request_id="req",
        request=SimpleNamespace(params={"stream": True}),
        data={},
    )

    item = StreamItem(
        chunk_id=0,
        data=_patch(1.0),
        from_stage="latent_engine",
        metadata={"modality": "audio_latents", "stream": True},
    )
    vocoder.on_stream_chunk_batch([("req", item)])

    assert pool.steps, "expected coalesced pool step"
    assert len(pool.steps[0]) == 1


def test_decode_delta_non_final_is_a_no_op() -> None:
    pool = _RecordingSlotPool()
    vocoder = DotsTTSStreamingVocoder(
        _codec(),
        optimize=True,
        merge_steps=2,
        slot_pool=pool,
    )
    state = vocoder.create_stream_state("req")
    vocoder.ingest("req", state, _patch(1.0))
    assert vocoder.decode_delta("req", state, is_final=False) is None
    assert state.pending
    assert state.slot is not None
    assert not pool.steps
