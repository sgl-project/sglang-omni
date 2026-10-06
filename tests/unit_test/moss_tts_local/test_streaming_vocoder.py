# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local streaming vocoder tests.

All tests are CPU-only and drive the scheduler hooks synchronously in the real
pipeline order (chunks -> stream_done -> terminal payload replay). The fake
codec implements the native indexed streaming interface with a decode whose output
depends on each slot's cumulative frame offset, so any state-advance error,
cross-slot leak, or missed reset changes the waveform. The headline assertion
is that streamed PCM concatenates to exactly the offline decode of the same
codes — the property the v2 codec provides by construction.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from sglang_omni.models.moss_tts_local import stages
from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
from sglang_omni.models.moss_tts_local.request_builders import (
    build_moss_tts_local_stream_metadata,
)
from sglang_omni.models.moss_tts_local.streaming_vocoder import (
    CodecStreamSession,
    MossTTSLocalStreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.streaming_vocoder import INITIAL_CODEC_CHUNK_FRAMES_PARAM

N_VQ = 4
SAMPLES_PER_FRAME = 4
SAMPLE_RATE = 48000


class FakeQuantizer:
    """Batched-path quantizer: [N, B, T] codes -> [B, 1, T] hidden values
    ``sum(codes[:, b, t]) + 1000 * t`` (the other half of ``reference_waveform``)."""

    def __init__(self) -> None:
        self.calls = 0

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        values = codes.to(torch.float64).sum(dim=0)
        offsets = torch.arange(int(codes.shape[2]), dtype=torch.float64) * 1000.0
        return (values + offsets).to(torch.float32).unsqueeze(1)


class FakeDecoderStage(nn.Module):
    module_type = "PatchedPretransform"
    patch_size = SAMPLES_PER_FRAME
    is_downsample = False

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # note (Zhang Yiyang): [B, 1, T] -> [B, 2, T * SAMPLES_PER_FRAME]:
        # replicate per-frame values over the frame's samples (negated on the
        # second channel).
        audio = x.repeat_interleave(SAMPLES_PER_FRAME, dim=-1)
        return (
            torch.cat([audio, -audio], dim=1),
            input_lengths * SAMPLES_PER_FRAME,
        )


class FakeCodec(nn.Module):
    """Native codec fake whose PCM depends on each persistent slot's offset."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(sampling_rate=SAMPLE_RATE)
        self.decoder = nn.ModuleList([FakeDecoderStage()])
        self.quantizer = FakeQuantizer()
        self.offsets: torch.Tensor | None = None
        self.batch_shapes: list[tuple[int, ...]] = []
        self.slot_ids: list[torch.Tensor] = []
        self.frame_calls = 0
        self.decode_calls = 0

    def initialize_decoder_state_pool(
        self, state_capacity: int, scratch_capacity: int = 0
    ) -> None:
        if self.offsets is not None:
            raise RuntimeError("already streaming")
        self.offsets = torch.zeros(state_capacity + scratch_capacity, dtype=torch.long)

    def reset_decoder_state_slots(self, slot_ids: torch.Tensor) -> None:
        assert self.offsets is not None
        self.offsets[slot_ids.to("cpu")] = 0

    def close_decoder_state_pool(self) -> None:
        self.offsets = None

    def decode_streaming_tensors(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.offsets is not None
        self.frame_calls += 1
        self.batch_shapes.append(tuple(codes.shape))
        self.slot_ids.append(slot_ids.detach().to("cpu").clone())
        _, batch_size, step_t = codes.shape
        audio = torch.zeros(batch_size, 2, step_t * SAMPLES_PER_FRAME)
        audio_lengths = torch.zeros(batch_size, dtype=torch.long)
        for row, slot in enumerate(slot_ids.tolist()):
            if not bool(valid_rows[row]):
                continue
            length = int(codes_lengths[row])
            base = int(self.offsets[slot])
            for frame in range(length):
                value = float(codes[:, row, frame].sum()) + 1000.0 * (base + frame)
                start = frame * SAMPLES_PER_FRAME
                audio[row, 0, start : start + SAMPLES_PER_FRAME] = value
                audio[row, 1, start : start + SAMPLES_PER_FRAME] = -value
            audio_lengths[row] = length * SAMPLES_PER_FRAME
            self.offsets[slot] += length
        return audio, audio_lengths

    def decode(self, *args, **kwargs):
        self.decode_calls += 1
        raise AssertionError("offline requests must use the batched decoder")


class FakeProcessor:
    def __init__(self) -> None:
        self.audio_tokenizer = FakeCodec()
        self.model_config = SimpleNamespace(n_vq=N_VQ, sampling_rate=SAMPLE_RATE)
        self.decode_calls = 0

    def decode_audio_codes(self, codes_list, *, return_stereo: bool = True):
        self.decode_calls += 1
        raise AssertionError("vocoder must not re-enter processor decode")


def reference_waveform(rows: torch.Tensor) -> torch.Tensor:
    """Stateless offline decode of [T, n_vq] codes rows."""
    codes = rows[:, :N_VQ]
    frames = int(codes.shape[0])
    audio = torch.zeros(2, frames * SAMPLES_PER_FRAME)
    for t in range(frames):
        value = float(codes[t].sum()) + 1000.0 * t
        start = t * SAMPLES_PER_FRAME
        audio[0, start : start + SAMPLES_PER_FRAME] = value
        audio[1, start : start + SAMPLES_PER_FRAME] = -value
    return audio


def make_scheduler(
    monkeypatch: pytest.MonkeyPatch, processor: FakeProcessor, **kwargs: int
) -> MossTTSLocalStreamingVocoderScheduler:
    patch_vocoder_factory_loaders(monkeypatch, processor, processor.audio_tokenizer)
    scheduler = stages.create_vocoder_executor("fake-model", device="cpu", **kwargs)
    assert isinstance(scheduler, MossTTSLocalStreamingVocoderScheduler)
    return scheduler


def patch_vocoder_factory_loaders(
    monkeypatch: pytest.MonkeyPatch,
    processor: FakeProcessor,
    codec: FakeCodec,
) -> None:
    monkeypatch.setattr(
        stages,
        "load_moss_tts_local_processor",
        lambda model_path: processor,
    )
    monkeypatch.setattr(
        stages,
        "load_moss_audio_vocoder",
        lambda model_path, **kwargs: SimpleNamespace(
            model=codec,
            sample_rate=SAMPLE_RATE,
        ),
    )


def make_rows(frames: int, *, seed: int) -> torch.Tensor:
    """Full AR rows [frames, 1 + n_vq]: text token + codes."""
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 100, (frames, N_VQ), generator=generator)
    text = torch.full((frames, 1), 7, dtype=torch.long)
    return torch.cat([text, codes], dim=1)


def make_metadata(**extra: Any) -> dict[str, Any]:
    return {"stream": True, "modality": "audio_codes", "n_vq": N_VQ, **extra}


def stream_item(row: torch.Tensor, metadata: dict[str, Any], chunk_id: int = 0):
    return StreamItem(
        chunk_id=chunk_id,
        data=row.clone(),
        from_stage="tts_engine",
        metadata=metadata,
    )


def terminal_payload(
    rows: torch.Tensor | None,
    *,
    request_id: str = "req",
    params: dict[str, Any] | None = None,
) -> StagePayload:
    state = MossTTSLocalState(
        text="hello",
        audio_codes=rows[:, 1:].clone() if rows is not None else None,
        prompt_tokens=3,
        completion_tokens=int(rows.shape[0]) if rows is not None else 0,
        engine_time_s=0.5,
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="", params={"stream": True, **(params or {})}),
        data=state.to_dict(),
    )


def offline_payload(rows: torch.Tensor, request_id: str) -> StagePayload:
    state = MossTTSLocalState(
        text="x",
        audio_codes=rows[:, 1:].clone(),
        prompt_tokens=2,
        completion_tokens=int(rows.shape[0]),
        engine_time_s=0.25,
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="", params={}),
        data=state.to_dict(),
    )


def drain(scheduler) -> list:
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def run_stream(
    scheduler,
    rows: torch.Tensor,
    *,
    request_id: str = "req",
    metadata: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> list:
    metadata = metadata if metadata is not None else make_metadata()
    for index, row in enumerate(rows):
        scheduler.handle_stream_chunk(request_id, stream_item(row, metadata, index))
    # Real pipeline order: chunks -> stream_done -> terminal payload replay.
    scheduler.handle_stream_done(request_id)
    scheduler.handle_streaming_new_request(
        request_id, terminal_payload(rows, request_id=request_id, params=params)
    )
    return drain(scheduler)


def decode_audio(data: dict[str, Any]) -> np.ndarray:
    assert data["audio_waveform_dtype"] == "float32"
    array = np.frombuffer(data["audio_waveform"], dtype=np.float32)
    return array.reshape(data["audio_waveform_shape"])


def concat_stream_audio(messages: list, request_id: str) -> np.ndarray:
    chunks = [
        decode_audio(msg.data)
        for msg in messages
        if msg.type == "stream" and msg.request_id == request_id
    ]
    assert chunks, "no stream chunks emitted"
    for chunk in chunks:
        assert chunk.ndim == 2 and chunk.shape[0] == 2  # stereo kept end to end
    return np.concatenate(chunks, axis=1)


def test_stream_metadata_builder() -> None:
    def payload(params: dict[str, Any]) -> StagePayload:
        return StagePayload(
            request_id="req",
            request=OmniRequest(inputs="", params=params),
            data={},
        )

    assert build_moss_tts_local_stream_metadata(payload({}), n_vq=12) is None
    metadata = build_moss_tts_local_stream_metadata(
        payload({"stream": True, INITIAL_CODEC_CHUNK_FRAMES_PARAM: 3}), n_vq=12
    )
    assert metadata == {
        "stream": True,
        "modality": "audio_codes",
        "n_vq": 12,
        INITIAL_CODEC_CHUNK_FRAMES_PARAM: 3,
    }


def test_stream_concatenates_to_offline_decode(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=10,
        initial_chunk_frames=5,
    )
    rows = make_rows(23, seed=1)
    messages = run_stream(scheduler, rows)

    stream_msgs = [m for m in messages if m.type == "stream"]
    # 23 frames at initial=5/steady=10: chunks of 5, 10, and the 8-frame tail.
    assert [
        decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME for m in stream_msgs
    ] == [5, 10, 8]
    for msg in stream_msgs:
        assert msg.data["sample_rate"] == SAMPLE_RATE
        assert msg.data["modality"] == "audio"
        assert msg.metadata == {"modality": "audio"}

    audio = concat_stream_audio(messages, "req")
    np.testing.assert_array_equal(audio, reference_waveform(rows[:, 1:]).numpy())

    results = [m for m in messages if m.type == "result"]
    assert len(results) == 1
    final = results[0].data
    assert isinstance(final, StagePayload)
    assert final.data["modality"] == "audio"
    assert final.data["sample_rate"] == SAMPLE_RATE
    assert final.data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 23,
        "total_tokens": 26,
        "engine_time_s": 0.5,
    }


def test_default_session_streaming_lane_capacity(monkeypatch) -> None:
    scheduler = make_scheduler(monkeypatch, FakeProcessor())
    session = scheduler.ensure_session()

    assert scheduler.max_batch_size == 8
    # note (Zhang Yiyang): 15 streaming lanes + the 1 lane freed by removing the
    # offline reserve = 16; the freed lane goes to streaming, not the trash.
    assert scheduler.stream_slots == 16
    assert not hasattr(session, "_offline_slots")
    assert session.stream_slots == 16
    assert session.graph_batch_sizes() == [1, 2, 4, 8, 12, 16]


@pytest.mark.parametrize(
    "stream_slots, active_batch_size, expected_batch_size",
    [
        (1, 1, 1),
        (3, 3, 3),
        (16, 15, 16),
        (32, 17, 20),
        (64, 17, 20),
        (64, 33, 36),
        (25, 25, 25),
    ],
)
def test_streaming_session_pads_to_nearest_batch_bucket(
    stream_slots: int, active_batch_size: int, expected_batch_size: int
) -> None:
    codec = FakeCodec()
    session = CodecStreamSession(codec, stream_slots=stream_slots, n_vq=N_VQ)
    slots = [session.acquire() for _ in range(active_batch_size)]
    codes = make_rows(2, seed=101)[:, 1:].transpose(0, 1).contiguous()

    audio = session.step({slot: codes for slot in slots})

    assert codec.batch_shapes == [(N_VQ, expected_batch_size, 2)]
    assert set(audio) == set(slots)
    for waveform in audio.values():
        assert waveform.shape[-1] == 2 * SAMPLES_PER_FRAME
    session.close()


def test_streaming_session_initializes_and_closes_codec_state() -> None:
    codec = FakeCodec()
    session = CodecStreamSession(codec, stream_slots=2, n_vq=N_VQ)
    assert codec.offsets.tolist() == [0, 0, 0, 0]
    session.close()
    assert codec.offsets is None


def test_compact_native_session_uses_active_batch_and_reuses_sparse_slots() -> None:
    codec = FakeCodec()
    session = CodecStreamSession(
        codec,
        stream_slots=2,
        n_vq=N_VQ,
    )
    slot_a = session.acquire()
    slot_b = session.acquire()
    assert (slot_a, slot_b) == (1, 0)

    codes_a = make_rows(2, seed=101)[:, 1:].transpose(0, 1).contiguous()
    codes_b = make_rows(1, seed=102)[:, 1:].transpose(0, 1).contiguous()
    first = session.step({slot_a: codes_a})
    assert first[slot_a].shape[-1] == 2 * SAMPLES_PER_FRAME
    both = session.step({slot_a: codes_b, slot_b: codes_b})
    assert set(both) == {slot_a, slot_b}
    assert codec.batch_shapes == [
        (N_VQ, 1, 2),
        (N_VQ, 2, 1),
    ]
    assert [slots.tolist() for slots in codec.slot_ids] == [[1], [1, 0]]

    session.release(slot_a)
    reused = session.acquire()
    assert reused == slot_a
    reset_audio = session.step({reused: codes_b})[reused]
    expected = reference_waveform(make_rows(1, seed=102)[:, 1:]).numpy()
    np.testing.assert_array_equal(reset_audio.numpy(), expected)
    session.close()
    assert codec.offsets is None


def test_compact_native_session_replays_graph_and_slices_bucket() -> None:
    codec = FakeCodec()
    session = CodecStreamSession(
        codec,
        stream_slots=3,
        n_vq=N_VQ,
    )
    runner = FakeVocoderCudaGraphRunner(
        bucket_size=2,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    session.cg_runner = runner
    slot = session.acquire()
    assert slot == 2
    codes = make_rows(5, seed=103)[:, 1:].transpose(0, 1).contiguous()

    output = session.step({slot: codes})[slot]

    assert output.shape == (2, 5 * SAMPLES_PER_FRAME)
    assert output[:, 0].tolist() == [3.0, -3.0]
    assert runner.calls == [((N_VQ, 1, 5), [2])]
    session.close()


def test_compact_runner_requires_scratch_rows() -> None:
    from sglang_omni.models.moss_tts_local.vocoder_cuda_graph import (
        MossVocoderCudaGraphRunner,
    )

    with pytest.raises(ValueError, match="scratch_capacity"):
        MossVocoderCudaGraphRunner(
            FakeCodec(),
            real_state_capacity=4,
            scratch_capacity=1,
            batch_sizes=[1, 2],
            frame_sizes=[5],
            num_quantizers=N_VQ,
        )


def test_runner_skips_capture_on_cpu() -> None:
    from sglang_omni.models.moss_tts_local.vocoder_cuda_graph import (
        MossVocoderCudaGraphRunner,
    )

    runner = MossVocoderCudaGraphRunner(
        FakeCodec(),
        real_state_capacity=4,
        scratch_capacity=4,
        batch_sizes=[1, 2, 4],
        frame_sizes=[5],
        num_quantizers=N_VQ,
    )
    assert runner.warmup() == []
    assert runner.capture_sizes == []


@pytest.mark.parametrize("graph_miss", [False, True])
def test_streaming_session_padding_preserves_inactive_slots(graph_miss) -> None:
    codec = FakeCodec()
    session = CodecStreamSession(
        codec,
        stream_slots=8,
        n_vq=N_VQ,
    )
    # An idle request retains its state while a sparse batch uses scratch padding.
    codes = make_rows(2, seed=104)[:, 1:].transpose(0, 1).contiguous()
    session.step({2: codes})
    expected_offsets = codec.offsets.clone()
    if graph_miss:
        session.cg_runner = FakeCudaGraphRunner([5])

    active_slots = [7, 0, 4]
    output = session.step({slot: codes for slot in active_slots})

    assert codec.batch_shapes[-1] == (N_VQ, 4, 2)
    assert codec.slot_ids[-1].tolist() == [7, 0, 4, 8]
    for slot in active_slots:
        torch.testing.assert_close(output[slot], reference_waveform(codes.T))
    expected_offsets[active_slots] = 2
    torch.testing.assert_close(codec.offsets, expected_offsets)
    session.close()


def test_factory_default_decouples_first_chunk_from_join_floor(monkeypatch) -> None:
    """The model first-chunk default and coalescing join floor are independent."""
    processor = FakeProcessor()
    scheduler = make_scheduler(monkeypatch, processor, stream_chunk_frames=10)
    assert scheduler.default_initial_chunk_frames == 5
    assert scheduler.coalesce_floor_frames == 5
    rows = make_rows(12, seed=99)
    messages = run_stream(scheduler, rows)
    sizes = [
        decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages
        if m.type == "stream"
    ]
    assert sizes[0] == 5
    np.testing.assert_array_equal(
        concat_stream_audio(messages, "req"),
        reference_waveform(rows[:, 1:]).numpy(),
    )


def run_stream_batched(
    scheduler,
    rows: torch.Tensor,
    *,
    request_id: str = "req",
    metadata: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> list:
    """Drive chunks through the real batched seam: enqueue stream_chunk messages and
    pump the serving loop so collect_stream_chunk_batch coalesces already-queued chunks.
    """
    metadata = metadata if metadata is not None else make_metadata()
    for index, row in enumerate(rows):
        scheduler.inbox.put(
            IncomingMessage(
                request_id, "stream_chunk", stream_item(row, metadata, index)
            )
        )
    pump_queued_stream_chunks(scheduler)
    scheduler.handle_stream_done(request_id)
    scheduler.handle_streaming_new_request(
        request_id, terminal_payload(rows, request_id=request_id, params=params)
    )
    return drain(scheduler)


def pump_queued_stream_chunks(scheduler) -> None:
    while True:
        msg = scheduler.next_message()
        if msg is None:
            break
        scheduler.handle_message(msg, None)


def test_batched_coalescing_matches_offline_decode(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_slots=8,
        stream_chunk_frames=10,
        initial_chunk_frames=5,
    )
    assert scheduler.can_batch_stream_chunks is True
    assert (
        scheduler.stream_chunk_batch_max == 8
    )  # follows stream_slots, not max_batch_size
    rows = make_rows(23, seed=1)
    messages = run_stream_batched(scheduler, rows)

    sizes = [
        decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages
        if m.type == "stream"
    ]
    # One row per request is consumed in each pump, preserving the configured
    # first-chunk boundary even when this request has a backlog.
    assert sizes == [5, 10, 8]
    assert sum(sizes) == 23

    audio = concat_stream_audio(messages, "req")
    np.testing.assert_array_equal(audio, reference_waveform(rows[:, 1:]).numpy())


def test_batched_step_capped_at_chunk_frames(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_slots=16,
        max_batch_size=4,  # offline knob < stream_slots; drain cap must track stream_slots
        stream_chunk_frames=10,
        initial_chunk_frames=5,
    )
    assert scheduler.stream_chunk_batch_max == 16
    scheduler.stream_chunk_batch_distinct_requests = False
    rows = make_rows(16, seed=3)
    messages = run_stream_batched(scheduler, rows)

    sizes = [
        decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages
        if m.type == "stream"
    ]
    # With cap=stream_slots=16, the first pump sees all 16 queued chunks. It
    # preserves the 5-frame initial boundary, then caps the steady step at
    # stream_chunk_frames=10 before draining the final frame.
    assert max(sizes) <= 10
    assert sizes == [5, 10, 1]
    audio = concat_stream_audio(messages, "req")
    np.testing.assert_array_equal(audio, reference_waveform(rows[:, 1:]).numpy())


def test_batched_coalescing_handles_two_streaming_lanes(monkeypatch) -> None:
    processor = FakeProcessor()
    codec = processor.audio_tokenizer
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_slots=2,
        stream_chunk_frames=3,
        initial_chunk_frames=3,
    )
    rows_a = make_rows(6, seed=4)
    rows_b = make_rows(6, seed=5)
    metadata = make_metadata()
    chunk_id = 0
    for index in range(6):
        scheduler.inbox.put(
            IncomingMessage(
                "a", "stream_chunk", stream_item(rows_a[index], metadata, chunk_id)
            )
        )
        chunk_id += 1
        scheduler.inbox.put(
            IncomingMessage(
                "b", "stream_chunk", stream_item(rows_b[index], metadata, chunk_id)
            )
        )
        chunk_id += 1

    pump_queued_stream_chunks(scheduler)
    scheduler.handle_stream_done("a")
    scheduler.handle_streaming_new_request(
        "a", terminal_payload(rows_a, request_id="a")
    )
    scheduler.handle_stream_done("b")
    scheduler.handle_streaming_new_request(
        "b", terminal_payload(rows_b, request_id="b")
    )
    messages = drain(scheduler)

    stream_boundaries = [
        (m.request_id, decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME)
        for m in messages
        if m.type == "stream"
    ]
    assert stream_boundaries == [("a", 3), ("b", 3), ("a", 3), ("b", 3)]
    assert codec.frame_calls == 2
    np.testing.assert_array_equal(
        concat_stream_audio(messages, "a"),
        reference_waveform(rows_a[:, 1:]).numpy(),
    )
    np.testing.assert_array_equal(
        concat_stream_audio(messages, "b"),
        reference_waveform(rows_b[:, 1:]).numpy(),
    )


def test_batched_ingest_failure_aborts_and_cleans_up_off_lock(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=2,
        initial_chunk_frames=2,
    )
    cleanup_calls: list[str] = []
    cleanup_saw_lock_owned: list[bool] = []

    def cleanup(request_id: str) -> None:
        is_owned = getattr(
            scheduler.state_lock, "_is_owned", lambda: False
        )  # noqa: leading-underscore  # upstream name
        cleanup_saw_lock_owned.append(bool(is_owned()))
        cleanup_calls.append(request_id)

    monkeypatch.setattr(scheduler, "cleanup_aborted_request", cleanup)
    rows = make_rows(2, seed=6)
    metadata = make_metadata()
    scheduler.inbox.put(
        IncomingMessage("ok", "stream_chunk", stream_item(rows[0], metadata, 0))
    )
    scheduler.inbox.put(
        IncomingMessage(
            "bad",
            "stream_chunk",
            stream_item(torch.tensor([7], dtype=torch.long), metadata, 1),
        )
    )
    scheduler.inbox.put(
        IncomingMessage("ok", "stream_chunk", stream_item(rows[1], metadata, 2))
    )

    pump_queued_stream_chunks(scheduler)
    scheduler.handle_stream_done("ok")
    scheduler.handle_streaming_new_request(
        "ok", terminal_payload(rows, request_id="ok")
    )
    messages = drain(scheduler)

    assert cleanup_calls == ["bad"]
    assert cleanup_saw_lock_owned == [False]
    assert scheduler.is_aborted("bad")
    assert "bad" not in scheduler.stream_states
    assert any(m.request_id == "bad" and m.type == "error" for m in messages)
    np.testing.assert_array_equal(
        concat_stream_audio(messages, "ok"),
        reference_waveform(rows[:, 1:]).numpy(),
    )


def test_initial_chunk_frames_request_override(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=10,
        initial_chunk_frames=5,
    )
    rows = make_rows(14, seed=2)
    metadata = make_metadata(**{INITIAL_CODEC_CHUNK_FRAMES_PARAM: 2})
    messages = run_stream(scheduler, rows, metadata=metadata)
    sizes = [
        decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages
        if m.type == "stream"
    ]
    assert sizes == [2, 10, 2]
    audio = concat_stream_audio(messages, "req")
    np.testing.assert_array_equal(audio, reference_waveform(rows[:, 1:]).numpy())


def test_explicit_zero_initial_chunk_means_steady_only(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=10,
        initial_chunk_frames=5,
    )
    rows = make_rows(12, seed=3)
    metadata = make_metadata(**{INITIAL_CODEC_CHUNK_FRAMES_PARAM: 0})
    messages = run_stream(scheduler, rows, metadata=metadata)
    sizes = [
        decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages
        if m.type == "stream"
    ]
    assert sizes == [10, 2]


def test_interleaved_streams_are_isolated(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=6,
        initial_chunk_frames=3,
    )
    rows_a = make_rows(17, seed=10)
    rows_b = make_rows(9, seed=11)
    metadata = make_metadata()
    chunk_id = 0
    for index in range(max(len(rows_a), len(rows_b))):
        if index < len(rows_a):
            scheduler.handle_stream_chunk(
                "a", stream_item(rows_a[index], metadata, chunk_id)
            )
            chunk_id += 1
        if index < len(rows_b):
            scheduler.handle_stream_chunk(
                "b", stream_item(rows_b[index], metadata, chunk_id)
            )
            chunk_id += 1
    scheduler.handle_stream_done("b")
    scheduler.handle_streaming_new_request(
        "b", terminal_payload(rows_b, request_id="b")
    )
    scheduler.handle_stream_done("a")
    scheduler.handle_streaming_new_request(
        "a", terminal_payload(rows_a, request_id="a")
    )
    messages = drain(scheduler)

    audio_a = concat_stream_audio(messages, "a")
    audio_b = concat_stream_audio(messages, "b")
    np.testing.assert_array_equal(audio_a, reference_waveform(rows_a[:, 1:]).numpy())
    np.testing.assert_array_equal(audio_b, reference_waveform(rows_b[:, 1:]).numpy())


def test_near_due_streams_coalesce_into_one_step(monkeypatch) -> None:
    """A due stream must not step alone past near-due peers.

    A decode step costs one forward over the full slot width regardless of
    how many slots are active, so when A (6 buffered, due) steps while B sits
    at 5, B's own step a moment later doubles the GPU work. The pump must
    instead step both at T=5 in a single _decode_frame call.
    """
    processor = FakeProcessor()
    codec = processor.audio_tokenizer
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=6,
        initial_chunk_frames=3,
    )
    rows_a = make_rows(9, seed=40)
    rows_b = make_rows(8, seed=41)
    metadata = make_metadata()
    messages: list = []
    chunk_id = 0
    # Warm both streams past their initial chunk so both sit at the steady
    # threshold (6) with empty buffers.
    for index in range(3):
        scheduler.handle_stream_chunk(
            "a", stream_item(rows_a[index], metadata, chunk_id)
        )
        chunk_id += 1
    for index in range(3):
        scheduler.handle_stream_chunk(
            "b", stream_item(rows_b[index], metadata, chunk_id)
        )
        chunk_id += 1
    messages += drain(scheduler)
    # B buffers 5 frames (one short of due); then A crosses its threshold.
    for index in range(3, 8):
        scheduler.handle_stream_chunk(
            "b", stream_item(rows_b[index], metadata, chunk_id)
        )
        chunk_id += 1
    calls_before = codec.frame_calls
    for index in range(3, 9):
        scheduler.handle_stream_chunk(
            "a", stream_item(rows_a[index], metadata, chunk_id)
        )
        chunk_id += 1
    assert codec.frame_calls - calls_before == 1
    coalesced = drain(scheduler)
    sizes = {
        msg.request_id: decode_audio(msg.data).shape[1]
        for msg in coalesced
        if msg.type == "stream"
    }
    assert sizes == {
        "a": 5 * SAMPLES_PER_FRAME,
        "b": 5 * SAMPLES_PER_FRAME,
    }
    messages += coalesced
    # Finishing both streams must still produce exactly the offline waveform,
    # proving the rider step advanced B's slot state correctly.
    scheduler.handle_stream_done("a")
    scheduler.handle_streaming_new_request(
        "a", terminal_payload(rows_a, request_id="a")
    )
    scheduler.handle_stream_done("b")
    scheduler.handle_streaming_new_request(
        "b", terminal_payload(rows_b, request_id="b")
    )
    messages += drain(scheduler)
    np.testing.assert_array_equal(
        concat_stream_audio(messages, "a"),
        reference_waveform(rows_a[:, 1:]).numpy(),
    )
    np.testing.assert_array_equal(
        concat_stream_audio(messages, "b"),
        reference_waveform(rows_b[:, 1:]).numpy(),
    )


def test_explicit_zero_initial_chunk_is_not_pulled_below_steady(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=6,
        initial_chunk_frames=2,
    )
    rows_a = make_rows(2, seed=42)
    rows_b = make_rows(6, seed=43)
    metadata_a = make_metadata()
    metadata_b = make_metadata(**{INITIAL_CODEC_CHUNK_FRAMES_PARAM: 0})
    chunk_id = 0

    # B explicitly opts out of a smaller first chunk, so five buffered frames
    # must not ride along when A crosses its own first-chunk threshold.
    for index in range(5):
        scheduler.handle_stream_chunk(
            "b", stream_item(rows_b[index], metadata_b, chunk_id)
        )
        chunk_id += 1
    for index in range(2):
        scheduler.handle_stream_chunk(
            "a", stream_item(rows_a[index], metadata_a, chunk_id)
        )
        chunk_id += 1

    messages = drain(scheduler)
    assert [m.request_id for m in messages if m.type == "stream"] == ["a"]

    scheduler.handle_stream_chunk("b", stream_item(rows_b[5], metadata_b, chunk_id))
    messages += drain(scheduler)
    b_chunks = [
        decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages
        if m.type == "stream" and m.request_id == "b"
    ]
    assert b_chunks == [6]


def test_positive_initial_chunk_is_not_pulled_below_threshold(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=6,
        initial_chunk_frames=2,
    )
    rows_a = make_rows(1, seed=44)
    rows_b = make_rows(5, seed=45)
    metadata_a = make_metadata(**{INITIAL_CODEC_CHUNK_FRAMES_PARAM: 1})
    metadata_b = make_metadata(**{INITIAL_CODEC_CHUNK_FRAMES_PARAM: 5})
    chunk_id = 0

    # B asked for a 5-frame first chunk; four buffered frames must not ride
    # along when A becomes due with a 1-frame floor.
    for index in range(4):
        scheduler.handle_stream_chunk(
            "b", stream_item(rows_b[index], metadata_b, chunk_id)
        )
        chunk_id += 1
    scheduler.handle_stream_chunk("a", stream_item(rows_a[0], metadata_a, chunk_id))
    chunk_id += 1

    messages = drain(scheduler)
    assert [m.request_id for m in messages if m.type == "stream"] == ["a"]

    scheduler.handle_stream_chunk("b", stream_item(rows_b[4], metadata_b, chunk_id))
    messages += drain(scheduler)
    b_chunks = [
        decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages
        if m.type == "stream" and m.request_id == "b"
    ]
    assert b_chunks == [5]


def test_slot_reuse_after_release(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_slots=1,
        stream_chunk_frames=4,
        initial_chunk_frames=2,
    )
    rows_a = make_rows(7, seed=20)
    messages_a = run_stream(scheduler, rows_a, request_id="a")
    np.testing.assert_array_equal(
        concat_stream_audio(messages_a, "a"),
        reference_waveform(rows_a[:, 1:]).numpy(),
    )
    # The single slot was released and reset; a second stream must start from
    # a fresh offset, not continue where "a" left off.
    rows_c = make_rows(6, seed=21)
    messages_c = run_stream(scheduler, rows_c, request_id="c")
    np.testing.assert_array_equal(
        concat_stream_audio(messages_c, "c"),
        reference_waveform(rows_c[:, 1:]).numpy(),
    )


def test_slot_reacquisition_preserves_initial_chunk_boundary(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_slots=1,
        stream_chunk_frames=6,
        initial_chunk_frames=1,
    )
    metadata = make_metadata()
    hold_rows = make_rows(1, seed=22)
    starved_rows = make_rows(4, seed=23)

    scheduler.handle_stream_chunk("hold", stream_item(hold_rows[0], metadata))
    drain(scheduler)
    for index, row in enumerate(starved_rows[:3]):
        scheduler.handle_stream_chunk("starved", stream_item(row, metadata, index))
    assert not [
        msg
        for msg in drain(scheduler)
        if msg.type == "stream" and msg.request_id == "starved"
    ]

    scheduler.handle_stream_done("hold")
    scheduler.handle_streaming_new_request(
        "hold", terminal_payload(hold_rows, request_id="hold")
    )
    drain(scheduler)

    scheduler.handle_stream_chunk(
        "starved", stream_item(starved_rows[3], metadata, len(starved_rows) - 1)
    )
    messages = drain(scheduler)
    first_chunk_sizes = [
        decode_audio(msg.data).shape[1] // SAMPLES_PER_FRAME
        for msg in messages
        if msg.type == "stream" and msg.request_id == "starved"
    ]
    assert first_chunk_sizes == [1]

    scheduler.handle_stream_done("starved")
    scheduler.handle_streaming_new_request(
        "starved", terminal_payload(starved_rows, request_id="starved")
    )
    messages += drain(scheduler)
    np.testing.assert_array_equal(
        concat_stream_audio(messages, "starved"),
        reference_waveform(starved_rows[:, 1:]).numpy(),
    )


def test_slot_exhaustion_falls_back_to_batched_decode(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_slots=1,
        stream_chunk_frames=4,
        initial_chunk_frames=2,
    )
    metadata = make_metadata()
    rows_a = make_rows(9, seed=30)
    rows_b = make_rows(8, seed=31)
    messages: list = []
    for index, row in enumerate(rows_a[:5]):
        scheduler.handle_stream_chunk("a", stream_item(row, metadata, index))
    # "b" cannot get a slot while "a" holds the only one: nothing may stream.
    for index, row in enumerate(rows_b):
        scheduler.handle_stream_chunk("b", stream_item(row, metadata, index))
    messages += drain(scheduler)
    assert all(m.request_id != "b" for m in messages if m.type == "stream")
    scheduler.handle_stream_done("b")
    scheduler.handle_streaming_new_request(
        "b", terminal_payload(rows_b, request_id="b")
    )
    messages_b = drain(scheduler)
    sizes_b = [
        decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages_b
        if m.type == "stream"
    ]
    # note (Zhang Yiyang): the whole buffer decodes in one batched call.
    assert sizes_b == [8]
    np.testing.assert_array_equal(
        concat_stream_audio(messages_b, "b"),
        reference_waveform(rows_b[:, 1:]).numpy(),
    )
    # note (Zhang Yiyang): "a" is unaffected by b's batched decode.
    for index, row in enumerate(rows_a[5:], start=5):
        scheduler.handle_stream_chunk("a", stream_item(row, metadata, index))
    scheduler.handle_stream_done("a")
    scheduler.handle_streaming_new_request(
        "a", terminal_payload(rows_a, request_id="a")
    )
    messages += drain(scheduler)
    np.testing.assert_array_equal(
        concat_stream_audio(messages, "a"),
        reference_waveform(rows_a[:, 1:]).numpy(),
    )


def test_done_without_chunks_decodes_payload_codes(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(monkeypatch, processor)
    rows = make_rows(5, seed=40)
    scheduler.handle_stream_done("req")
    scheduler.handle_streaming_new_request("req", terminal_payload(rows))
    messages = drain(scheduler)
    np.testing.assert_array_equal(
        concat_stream_audio(messages, "req"),
        reference_waveform(rows[:, 1:]).numpy(),
    )
    assert [m.type for m in messages] == ["stream", "result"]


def test_abort_releases_slot(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_slots=1,
        stream_chunk_frames=4,
        initial_chunk_frames=2,
    )
    metadata = make_metadata()
    rows_a = make_rows(3, seed=50)
    for index, row in enumerate(rows_a):
        scheduler.handle_stream_chunk("a", stream_item(row, metadata, index))
    scheduler.abort("a")
    drain(scheduler)
    rows_b = make_rows(6, seed=51)
    messages_b = run_stream(scheduler, rows_b, request_id="b")
    np.testing.assert_array_equal(
        concat_stream_audio(messages_b, "b"),
        reference_waveform(rows_b[:, 1:]).numpy(),
    )


def test_non_streaming_path_leaves_startup_session_untouched(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(monkeypatch, processor)
    startup_session = scheduler.ensure_session()
    assert scheduler.codec.offsets is not None

    rows = make_rows(11, seed=59)
    original_decoder = scheduler.codec.decoder
    (result,) = scheduler.vocode_batch([offline_payload(rows, "r1")])

    assert processor.decode_calls == 0
    # note (Zhang Yiyang): the batched path bypasses codec.decode entirely (no
    # decoder swap, no streaming-loop entry): quantizer decode happens once,
    # under no session, and never touches the startup session.
    assert scheduler.codec.decode_calls == 0
    assert scheduler.codec.quantizer.calls == 1
    assert scheduler.codec.decoder is original_decoder
    assert scheduler.session is startup_session
    assert scheduler.codec.offsets is not None
    np.testing.assert_array_equal(
        decode_audio(result.data), reference_waveform(rows[:, 1:]).numpy()
    )


def test_non_streaming_empty_audio_codes_skip_decode(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(monkeypatch, processor)
    rows = torch.empty(0, N_VQ + 1, dtype=torch.long)

    (result,) = scheduler.vocode_batch([offline_payload(rows, "empty")])

    assert processor.decode_calls == 0
    assert scheduler.codec.decode_calls == 0
    assert "audio_codes" not in result.data
    assert "audio_waveform" not in result.data


def test_non_streaming_path_with_and_without_live_session(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(monkeypatch, processor)

    rows_1 = make_rows(101, seed=60)
    rows_2 = make_rows(4, seed=61)
    original_decoder = scheduler.codec.decoder

    # note (Zhang Yiyang): before any stream, use the batched full-sequence
    # path (quantizer + batched decoder), never codec.decode.
    results = scheduler.vocode_batch(
        [offline_payload(rows_1, "r1"), offline_payload(rows_2, "r2")]
    )
    assert processor.decode_calls == 0
    assert scheduler.codec.decode_calls == 0
    assert scheduler.codec.quantizer.calls == 1
    assert scheduler.codec.decoder is original_decoder
    waves_before = [decode_audio(result.data) for result in results]
    for result in results:
        assert result.data["sample_rate"] == SAMPLE_RATE
        assert result.data["modality"] == "audio"
        assert result.data["usage"]["prompt_tokens"] == 2

    # A streaming request opens the persistent session...
    run_stream(scheduler, make_rows(6, seed=62))
    assert scheduler.session is not None

    # note (Zhang Yiyang): ...after which offline decodes still use the batched
    # full-sequence path (non-streaming work never enters the streaming
    # session), producing identical audio.
    results = scheduler.vocode_batch(
        [offline_payload(rows_1, "r3"), offline_payload(rows_2, "r4")]
    )
    assert processor.decode_calls == 0
    assert scheduler.codec.decode_calls == 0
    assert scheduler.codec.quantizer.calls == 2
    waves_after = [decode_audio(result.data) for result in results]
    for before, after in zip(waves_before, waves_after):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(
        waves_after[0], reference_waveform(rows_1[:, 1:]).numpy()
    )


def test_offline_batch_uses_batched_path_with_live_session(monkeypatch) -> None:
    del monkeypatch
    processor = FakeProcessor()
    # Constructed directly: max_step_frames is not a factory knob.
    scheduler = MossTTSLocalStreamingVocoderScheduler(
        processor.audio_tokenizer,
        n_vq=N_VQ,
        sample_rate=SAMPLE_RATE,
        max_batch_size=2,
        max_step_frames=3,
        stream_chunk_frames=3,
    )
    run_stream(scheduler, make_rows(5, seed=70))  # open the session
    rows_list = [make_rows(7, seed=71), make_rows(2, seed=72), make_rows(5, seed=73)]
    payloads = []
    for index, rows in enumerate(rows_list):
        state = MossTTSLocalState(text="x", audio_codes=rows[:, 1:].clone())
        payloads.append(
            StagePayload(
                request_id=f"r{index}",
                request=OmniRequest(inputs="", params={}),
                data=state.to_dict(),
            )
        )
    results = scheduler.vocode_batch(payloads)
    codec = processor.audio_tokenizer
    # note (Zhang Yiyang): two batched waves (3 items, max_batch_size=2); no
    # session stepping.
    assert codec.decode_calls == 0
    assert codec.quantizer.calls == 2
    for rows, result in zip(rows_list, results):
        np.testing.assert_array_equal(
            decode_audio(result.data), reference_waveform(rows[:, 1:]).numpy()
        )


def test_offline_batch_leaves_stream_slots_untouched(monkeypatch) -> None:
    """With streams holding slots, an offline batch decodes via the batched path
    without borrowing or resetting any stream slot."""
    processor = FakeProcessor()
    codec = processor.audio_tokenizer
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_slots=3,
        max_batch_size=2,
        stream_chunk_frames=3,
        initial_chunk_frames=3,
    )

    for request_id, seed in (("hold-a", 80), ("hold-b", 81)):
        scheduler.handle_stream_chunk(
            request_id, stream_item(make_rows(1, seed=seed)[0], make_metadata())
        )
    assert codec.frame_calls == 0
    session = scheduler.session
    assert session is not None
    assert len(session.free_stream_slots) == 1

    offline_rows = [make_rows(2, seed=82), make_rows(2, seed=83)]
    results = scheduler.vocode_batch(
        [
            offline_payload(offline_rows[0], "offline-a"),
            offline_payload(offline_rows[1], "offline-b"),
        ]
    )
    assert codec.decode_calls == 0
    assert codec.quantizer.calls == 1
    # note (Zhang Yiyang): the batched path never runs streaming steps.
    assert codec.frame_calls == 0
    assert scheduler.session is session
    assert len(session.free_stream_slots) == 1
    for rows, result in zip(offline_rows, results):
        np.testing.assert_array_equal(
            decode_audio(result.data), reference_waveform(rows[:, 1:]).numpy()
        )

    rows = make_rows(2, seed=84)
    messages = run_stream(scheduler, rows, request_id="stream-after-offline")
    np.testing.assert_array_equal(
        concat_stream_audio(messages, "stream-after-offline"),
        reference_waveform(rows[:, 1:]).numpy(),
    )


def test_stop_closes_persistent_streaming_session(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = make_scheduler(monkeypatch, processor)
    scheduler.handle_stream_chunk(
        "req", stream_item(make_rows(1, seed=74)[0], make_metadata())
    )
    assert scheduler.session is not None
    assert scheduler.codec.offsets is not None

    scheduler.stop()

    assert scheduler.session is None
    assert scheduler.stream_states == {}
    assert scheduler.codec.offsets is None

    # Reusing the same codec instance after stop must be able to open a fresh
    # streaming context instead of tripping the codec's nested-session guard.
    restarted = make_scheduler(monkeypatch, processor)
    restarted.handle_stream_chunk(
        "req2", stream_item(make_rows(1, seed=75)[0], make_metadata())
    )
    assert restarted.session is not None
    assert restarted.codec.offsets is not None
    restarted.stop()


class FailingCodec(FakeCodec):
    """FakeCodec whose Nth streaming decode call raises."""

    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call

    def decode_streaming_tensors(self, codes, codes_lengths, slot_ids, valid_rows):
        if self.frame_calls + 1 == self.fail_on_call:
            self.frame_calls += 1
            raise RuntimeError("codec decode exploded")
        return super().decode_streaming_tensors(
            codes, codes_lengths, slot_ids, valid_rows
        )


def test_decode_step_failure_fails_participants_only(monkeypatch) -> None:
    """A failed decode step errors every participant, releases their slots,
    leaves non-participants and already-emitted audio untouched, and keeps
    the scheduler usable for new streams.
    """
    processor = FakeProcessor()
    processor.audio_tokenizer = FailingCodec(fail_on_call=3)
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=10,
        initial_chunk_frames=2,
    )
    metadata = make_metadata()

    # Decode #1 succeeds: "c" emits its initial 2-frame chunk.
    rows_c = make_rows(2, seed=100)
    for index, row in enumerate(rows_c):
        scheduler.handle_stream_chunk("c", stream_item(row, metadata, index))
    early = drain(scheduler)
    assert [m.type for m in early] == ["stream"]
    assert early[0].request_id == "c"

    # "b" first emits its configured initial chunk, then buffers below its
    # steady threshold. When "a" crosses its 2-frame initial threshold, "b" can
    # ride along because it did not opt out of smaller initial chunks.
    rows_b = make_rows(5, seed=101)
    for index, row in enumerate(rows_b):
        scheduler.handle_stream_chunk("b", stream_item(row, metadata, index))
    early_b = drain(scheduler)
    assert [m.type for m in early_b] == ["stream"]
    assert early_b[0].request_id == "b"
    rows_a = make_rows(2, seed=102)
    for index, row in enumerate(rows_a):
        scheduler.handle_stream_chunk("a", stream_item(row, metadata, index))

    messages = drain(scheduler)
    errors = [m for m in messages if m.type == "error"]
    assert {m.request_id for m in errors} == {"a", "b"}
    assert all(m.request_id != "c" for m in messages if m.type == "stream")

    # Both participants' state is gone and their slots are back in the pool.
    assert "a" not in scheduler.stream_states
    assert "b" not in scheduler.stream_states
    assert len(scheduler.session.free_stream_slots) == scheduler.stream_slots - 1

    # The scheduler keeps serving: a fresh stream decodes normally.
    rows_d = make_rows(6, seed=103)
    messages_d = run_stream(scheduler, rows_d, request_id="d")
    np.testing.assert_array_equal(
        concat_stream_audio(messages_d, "d"),
        reference_waveform(rows_d[:, 1:]).numpy(),
    )


def test_stream_chunk_requires_metadata_contract(monkeypatch) -> None:
    # note (Gaokai): base-owned scaffold errors are contract-tested in
    # tests/unit_test/scheduling/test_streaming_vocoder.py; only the
    # model-owned row-shape and n_vq checks belong here.
    processor = FakeProcessor()
    scheduler = make_scheduler(monkeypatch, processor)
    row = make_rows(1, seed=80)[0]
    scheduler.on_stream_chunk_batch(
        [("req", stream_item(torch.zeros(2, dtype=torch.long), make_metadata()))]
    )
    scheduler.on_stream_chunk_batch([("req2", stream_item(row, make_metadata()))])
    scheduler.on_stream_chunk_batch(
        [("req2", stream_item(row, make_metadata(n_vq=N_VQ + 1)))]
    )
    errors = {m.request_id: m.data for m in drain(scheduler) if m.type == "error"}
    assert "channels" in str(errors["req"])
    assert "n_vq changed" in str(errors["req2"])
    # note (Gaokai): the serving path aborts a request whose chunk breaks the
    # model contract, so neither request may keep stream state.
    assert scheduler.stream_states == {}


def test_stream_chunk_accepts_batched_ar_rows(monkeypatch) -> None:
    scheduler = make_scheduler(monkeypatch, FakeProcessor())
    state = scheduler.create_stream_state("req")
    rows = make_rows(3, seed=81)

    codes = scheduler.validate_chunk("req", state, rows)
    scheduler.ingest("req", state, codes)

    assert len(state.pending) == 3
    assert torch.equal(torch.stack(state.pending), rows[:, 1:])


# --- CUDA-graph config + recapture / factory-capture / anti-storm lifecycle (CPU fakes) ---


class FakeCudaGraphRunner:
    """Stand-in runner so CPU tests can exercise the capture/recapture lifecycle without a GPU.
    decode_step returns None, so the session falls through to the correct eager FakeCodec decode.
    """

    def __init__(self, frames: Any) -> None:
        self.frames = list(frames)

    def captured_frames(self) -> list:
        return list(self.frames)

    def decode_step(self, codes_step, state_slot_ids, valid_rows=None):
        return None


class FakeVocoderCudaGraphRunner:
    """Small runner double for the native compact replay contract."""

    def __init__(self, *, bucket_size: int, samples_per_frame: int) -> None:
        self.bucket_size = bucket_size
        self.samples_per_frame = samples_per_frame
        self.calls: list[tuple[tuple[int, ...], list[int]]] = []

    def captured_frames(self) -> list[int]:
        return [5]

    def decode_step(
        self,
        codes_step: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(
            (tuple(codes_step.shape), state_slot_ids.detach().to("cpu").tolist())
        )
        _, batch_size, step_t = codes_step.shape
        audio = torch.zeros(
            self.bucket_size,
            2,
            step_t * self.samples_per_frame,
        )
        audio[:batch_size, 0].fill_(3.0)
        audio[:batch_size, 1].fill_(-3.0)
        lengths = torch.zeros(self.bucket_size, dtype=torch.long)
        lengths[:batch_size] = step_t * self.samples_per_frame
        return audio, lengths


def install_fake_capture(monkeypatch, calls: list, *, seal: bool = True) -> None:
    """Make the scheduler treat the codec as CUDA-resident and swap real graph capture for a fake
    that records each call (so re-probe is observable) and, when seal=True, attaches a runner.
    """
    monkeypatch.setattr(
        MossTTSLocalStreamingVocoderScheduler, "codec_on_cuda", lambda self: True
    )

    def fake_warmup(self, frames, *, min_free_gb: float = 3.0) -> list:
        self.warmup_attempted = True
        calls.append(id(self))
        self.cg_runner = FakeCudaGraphRunner(frames) if seal else None
        return self.cg_runner.captured_frames() if seal else []

    monkeypatch.setattr(CodecStreamSession, "warmup_cuda_graph", fake_warmup)


def test_create_vocoder_executor_threads_cuda_graph_config(monkeypatch) -> None:
    from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig

    config = MossTTSLocalPipelineConfig(
        model_path="fake-model", vocoder_cuda_graph=False
    )
    scheduler = make_scheduler(
        monkeypatch, FakeProcessor(), **config.stage_factory_kwargs("vocoder")
    )
    assert scheduler.vocoder_cuda_graph is False
    config2 = MossTTSLocalPipelineConfig(
        model_path="fake-model",
        vocoder_cuda_graph_frames=[5, 25],
        vocoder_cuda_graph_min_free_gb=7.0,
    )
    scheduler2 = make_scheduler(
        monkeypatch, FakeProcessor(), **config2.stage_factory_kwargs("vocoder")
    )
    assert scheduler2.vocoder_cuda_graph_frames == [5, 25]
    assert scheduler2.vocoder_cuda_graph_min_free_gb == 7.0


def test_vocoder_factory_resolves_graph_policy_before_loading(monkeypatch) -> None:
    calls: list[bool | None] = []

    def resolve(vocoder_cuda_graph: bool | None) -> bool:
        calls.append(vocoder_cuda_graph)
        raise ValueError("unsafe graph")

    monkeypatch.setattr(stages, "resolve_vocoder_cuda_graph", resolve)
    monkeypatch.setattr(
        stages,
        "load_moss_tts_local_processor",
        lambda model_path: (_ for _ in ()).throw(AssertionError("loaded codec")),
    )

    with pytest.raises(ValueError, match="unsafe graph"):
        stages.create_vocoder_executor(
            "fake-model", device="cpu", vocoder_cuda_graph=True
        )
    assert calls == [True]


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, list(range(1, 26))),
        ({"stream_chunk_frames": 4, "initial_chunk_frames": 2}, [1, 2, 3, 4]),
    ],
)
def test_default_cuda_graph_frames_cover_stream_chunk_exactly(
    monkeypatch, kwargs, expected
) -> None:
    captured: list[list[int]] = []
    monkeypatch.setattr(
        MossTTSLocalStreamingVocoderScheduler, "codec_on_cuda", lambda self: True
    )

    def fake_warmup(self, frames, *, min_free_gb: float = 3.0) -> list[int]:
        self.warmup_attempted = True
        captured.append(list(frames))
        self.cg_runner = FakeCudaGraphRunner(frames)
        return self.cg_runner.captured_frames()

    monkeypatch.setattr(CodecStreamSession, "warmup_cuda_graph", fake_warmup)

    scheduler = make_scheduler(monkeypatch, FakeProcessor(), **kwargs)
    try:
        assert captured == [expected]
        assert scheduler.session is not None
        assert scheduler.session.captured_frames() == expected
    finally:
        scheduler.stop()


def test_create_vocoder_executor_uses_separate_codec(monkeypatch) -> None:
    processor = FakeProcessor()
    codec = FakeCodec()
    patch_vocoder_factory_loaders(monkeypatch, processor, codec)

    scheduler = stages.create_vocoder_executor("fake-model", device="cpu")

    assert scheduler.codec is codec
    assert scheduler.codec is not processor.audio_tokenizer

    rows = make_rows(7, seed=98)
    (result,) = scheduler.vocode_batch([offline_payload(rows, "separate-codec")])

    assert processor.decode_calls == 0
    assert processor.audio_tokenizer.decode_calls == 0
    assert codec.decode_calls == 0
    assert codec.quantizer.calls == 1
    np.testing.assert_array_equal(
        decode_audio(result.data), reference_waveform(rows[:, 1:]).numpy()
    )


def test_create_vocoder_executor_validates_process_memory_after_warmup(
    monkeypatch,
) -> None:
    processor = FakeProcessor()
    codec = FakeCodec()
    patch_vocoder_factory_loaders(monkeypatch, processor, codec)
    validations: list[dict] = []
    monkeypatch.setattr(
        stages,
        "validate_loaded_process_memory_budget",
        lambda **kwargs: validations.append(kwargs),
    )

    stages.create_vocoder_executor(
        "fake-model",
        device="cpu",
        gpu_id=3,
        total_gpu_memory_fraction=0.18,
        process_total_gpu_memory_fraction=0.95,
    )

    assert validations == [
        {
            "stage_name": "MOSS-TTS Local vocoder",
            "gpu_id": 3,
            "total_gpu_memory_fraction": 0.95,
        }
    ]


def test_create_vocoder_executor_uses_model_config_codec_path(monkeypatch) -> None:
    processor = FakeProcessor()
    processor.model_config.audio_tokenizer_name_or_path = "codec-from-model-config"
    codec = FakeCodec()
    loaded_codec_paths = []

    def fake_load_audio_vocoder(model_path, **kwargs):
        loaded_codec_paths.append(model_path)
        return SimpleNamespace(model=codec, sample_rate=SAMPLE_RATE)

    monkeypatch.setattr(
        stages,
        "load_moss_tts_local_processor",
        lambda model_path: processor,
    )
    monkeypatch.setattr(
        stages,
        "load_moss_audio_vocoder",
        fake_load_audio_vocoder,
    )

    stages.create_vocoder_executor("fake-model", device="cpu")

    assert loaded_codec_paths == ["codec-from-model-config"]


def test_pipeline_config_injects_vocoder_graph_settings() -> None:
    from sglang_omni.models.moss_tts_local.config import (
        MossTTSLocalPipelineConfig,
        MossTTSLocalSplitPipelineConfig,
    )

    cfg = MossTTSLocalPipelineConfig(model_path="x")
    voc = cfg.stage_named("vocoder")
    assert voc.factory.dtype == "float32"
    assert voc.factory.compute_dtype == "bfloat16"
    assert voc.factory.attention_backend == "auto"
    kwargs = cfg.stage_factory_kwargs("vocoder")
    assert kwargs["vocoder_cuda_graph"] is True
    assert kwargs["vocoder_cuda_graph_frames"] is None
    assert kwargs["vocoder_cuda_graph_min_free_gb"] == 3.0

    cfg2 = MossTTSLocalPipelineConfig(
        model_path="x",
        vocoder_cuda_graph=False,
        vocoder_cuda_graph_frames=[5, 25],
        vocoder_cuda_graph_min_free_gb=4.5,
    )
    kwargs2 = cfg2.stage_factory_kwargs("vocoder")
    assert kwargs2["vocoder_cuda_graph"] is False
    assert kwargs2["vocoder_cuda_graph_frames"] == [5, 25]
    assert kwargs2["vocoder_cuda_graph_min_free_gb"] == 4.5

    # The split variant overrides `stages`; the injection must still reach its vocoder.
    split = MossTTSLocalSplitPipelineConfig(model_path="x", vocoder_cuda_graph=False)
    assert split.stage_factory_kwargs("vocoder")["vocoder_cuda_graph"] is False


def test_pipeline_config_rejects_invalid_vocoder_graph_settings() -> None:
    from sglang_omni.models.moss_tts_local.config import MossTTSLocalPipelineConfig

    # note (Zhang Yiyang): Use vocoder_cuda_graph: false to disable graphs.
    with pytest.raises(ValueError, match="vocoder_cuda_graph_frames must be non-empty"):
        MossTTSLocalPipelineConfig(model_path="x", vocoder_cuda_graph_frames=[])
    # Non-positive frame counts must error, not be silently filtered.
    with pytest.raises(ValueError, match="positive ints"):
        MossTTSLocalPipelineConfig(model_path="x", vocoder_cuda_graph_frames=[5, 0])
    with pytest.raises(ValueError, match="positive ints"):
        MossTTSLocalPipelineConfig(model_path="x", vocoder_cuda_graph_frames=[-1])
    # Negative VRAM headroom is nonsensical (would disable the guard); error.
    with pytest.raises(ValueError, match="vocoder_cuda_graph_min_free_gb"):
        MossTTSLocalPipelineConfig(model_path="x", vocoder_cuda_graph_min_free_gb=-1.0)


def test_scheduler_rejects_frame_above_max_step(monkeypatch) -> None:
    # A configured frame count above max_step_frames is invalid (no such step occurs); fail fast,
    # do not silently drop it.
    with pytest.raises(ValueError, match="exceed max_step_frames"):
        MossTTSLocalStreamingVocoderScheduler(
            FakeCodec(),
            n_vq=N_VQ,
            sample_rate=SAMPLE_RATE,
            max_step_frames=25,
            vocoder_cuda_graph_frames=[5, 100],
        )


def test_factory_captures_graphs_before_returning(monkeypatch) -> None:
    """create_vocoder_executor runs warmup_now synchronously, so the scheduler it returns already
    has graphs captured (the stage process is only marked ready after the factory returns).
    """
    calls: list = []
    install_fake_capture(monkeypatch, calls, seal=True)
    scheduler = make_scheduler(monkeypatch, FakeProcessor())
    assert calls, "factory must capture before returning"
    assert scheduler.session is not None
    assert scheduler.session.has_cuda_graph_runner()


@pytest.mark.parametrize("trigger", ["chunk", "slot_starved", "no_chunk_done"])
def test_streaming_reuses_graphed_session_after_nonstreaming(
    monkeypatch, trigger
) -> None:
    """Non-streaming traffic leaves the graphed startup session intact; later streaming work
    reuses it with no re-capture. Streaming sessions are created by the first-chunk path
    (_ensure_slot), or by the factory warmup; a slot-starved finish and a no-chunk finish use
    the batched non-streaming route and never create a session by themselves.
    """
    calls: list = []
    install_fake_capture(monkeypatch, calls, seal=True)
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch,
        processor,
        stream_slots=1,
        stream_chunk_frames=4,
        initial_chunk_frames=2,
    )
    assert scheduler.session is not None
    assert scheduler.session.has_cuda_graph_runner()
    startup_session = scheduler.session

    # note (Zhang Yiyang): non-streaming decode uses the batched path and
    # leaves the startup session untouched.
    nonstream_rows = make_rows(5, seed=1)
    (result,) = scheduler.vocode_batch([offline_payload(nonstream_rows, "n1")])
    assert scheduler.session is startup_session
    assert not startup_session.closed
    assert len(calls) == 1
    np.testing.assert_array_equal(
        decode_audio(result.data), reference_waveform(nonstream_rows[:, 1:]).numpy()
    )

    if trigger == "chunk":
        scheduler.handle_stream_chunk(
            "s", stream_item(make_rows(6, seed=2)[0], make_metadata())
        )
    elif trigger == "slot_starved":
        # note (Zhang Yiyang): "hold" takes the persistent session's only
        # slot; "starve" buffers without a slot, then finishes through the
        # batched path.
        for i, row in enumerate(make_rows(5, seed=3)):
            scheduler.handle_stream_chunk("hold", stream_item(row, make_metadata(), i))
        for i, row in enumerate(make_rows(6, seed=4)):
            scheduler.handle_stream_chunk(
                "starve", stream_item(row, make_metadata(), 100 + i)
            )
        scheduler.handle_stream_done("starve")
        scheduler.handle_streaming_new_request(
            "starve", terminal_payload(make_rows(6, seed=4), request_id="starve")
        )
    else:  # no_chunk_done: terminal payload replay with no chunks -> _decode_payload_codes
        scheduler.handle_stream_done("nc")
        scheduler.handle_streaming_new_request(
            "nc", terminal_payload(make_rows(5, seed=5), request_id="nc")
        )

    if trigger == "no_chunk_done":
        # note (Zhang Yiyang): batched non-streaming decode creates no session;
        # the factory-captured startup session stays the only one.
        assert scheduler.session is startup_session
        assert len(calls) == 1
        messages = drain(scheduler)
        np.testing.assert_array_equal(
            concat_stream_audio(messages, "nc"),
            reference_waveform(make_rows(5, seed=5)[:, 1:]).numpy(),
        )
        return

    # note (Zhang Yiyang): streaming work reuses the factory-captured session —
    # no re-capture.
    assert scheduler.session is startup_session
    assert (
        len(calls) == 1
    ), f"{trigger}: expected 1 warmup (factory only, no recapture), got {len(calls)}"
    assert scheduler.session.has_cuda_graph_runner()


def test_low_vram_capture_attempted_once_no_reprobe(monkeypatch) -> None:
    """A capture that seals nothing (low VRAM) sets warmup_attempted, so streaming on that session
    never re-probes capture per step."""
    calls: list = []
    install_fake_capture(monkeypatch, calls, seal=False)
    processor = FakeProcessor()
    scheduler = make_scheduler(
        monkeypatch, processor, stream_chunk_frames=4, initial_chunk_frames=2
    )
    assert scheduler.session is not None
    assert not scheduler.session.has_cuda_graph_runner()
    assert len(calls) == 1

    rows = make_rows(20, seed=9)
    messages = run_stream(scheduler, rows, request_id="s")
    assert (
        len(calls) == 1
    ), f"re-probe storm: capture attempted {len(calls)}x, expected 1"
    np.testing.assert_array_equal(
        concat_stream_audio(messages, "s"), reference_waveform(rows[:, 1:]).numpy()
    )
