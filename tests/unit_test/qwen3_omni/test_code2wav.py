# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.streaming_vocoder import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
)
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel, make_qwen_payload


def test_qwen_code2wav_streams_incrementally_and_abort_clears_state() -> None:
    """Preserves incremental waveform emission and request-state cleanup on abort."""
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )
    scheduler._payloads["req-1"] = make_qwen_payload(request_id="req-1")
    scheduler._ensure_request_state("req-1")

    chunk_meta = {"stream": False}  # non-streaming: final result carries full PCM
    scheduler._on_chunk(
        "req-1",
        StreamItem(0, torch.tensor([1, 10]), "talker", metadata=chunk_meta),
    )
    scheduler._on_chunk(
        "req-1",
        StreamItem(1, torch.tensor([2, 20]), "talker", metadata=chunk_meta),
    )
    scheduler._on_chunk(
        "req-1",
        StreamItem(2, torch.tensor([3, 30]), "talker", metadata=chunk_meta),
    )
    scheduler._on_done("req-1")

    message = scheduler.outbox.get_nowait()
    audio = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 2, 2), (1, 2, 2)]
    assert audio.shape == (6,)

    scheduler._payloads["req-2"] = make_qwen_payload(request_id="req-2")
    scheduler._ensure_request_state("req-2")
    scheduler._pending_done.add("req-2")
    scheduler.abort("req-2")
    assert "req-2" not in scheduler._code_chunks
    assert "req-2" not in scheduler._payloads
    assert "req-2" not in scheduler._pending_done


# ---------------------------------------------------------------------------
# Configurable initial / steady chunk sizes
# ---------------------------------------------------------------------------


def test_code2wav_config_stage_passes_chunk_size_args() -> None:
    """_code2wav_stage must expose stream_chunk_size, left_context_size and
    initial_codec_chunk_frames via factory_args so config controls both the
    steady and the first-chunk sizes."""
    from sglang_omni.models.qwen3_omni.config import _code2wav_stage

    stage = _code2wav_stage(gpu=0, process="code2wav")
    assert stage.factory_args["stream_chunk_size"] == 25
    assert stage.factory_args["left_context_size"] == 25
    # 0 = off by default (backwards compatible); per-request opt-in still works.
    assert stage.factory_args["initial_codec_chunk_frames"] == 0


def _code_chunk(idx: int, *, stream: bool = True) -> StreamItem:
    """One codec frame, two codebooks, non-EOS."""
    return StreamItem(
        chunk_id=idx,
        data=torch.tensor([idx + 1, (idx + 1) * 10], dtype=torch.long),
        from_stage="talker",
        metadata={"stream": stream},
    )


def _drain_stream_audio(scheduler: Code2WavScheduler) -> list[np.ndarray]:
    """Collect every emitted 'stream' message's PCM as a float32 array."""
    parts: list[np.ndarray] = []
    while not scheduler.outbox.empty():
        msg = scheduler.outbox.get_nowait()
        if msg.type == "stream":
            parts.append(
                np.frombuffer(msg.data["audio_waveform"], dtype=np.float32).copy()
            )
    return parts


def test_code2wav_smaller_first_chunk_then_steady_with_seamless_overlap() -> None:
    """First emit uses the small initial chunk; the second (steady) window
    reuses the *entire* initial chunk as left context and trims the overlap,
    so the size change across the seam is sample-accurate.

    initial=2, steady=3, left=5, upsample=2:
      - emit 1: window [0:2], context=min(5,0)=0, trim=0  -> 4 samples
      - emit 2: window [0:5], context=min(5,2)=2, trim=4  -> 6 samples
    The second window's 2-frame left context == the whole first chunk.
    """
    model = FakeCode2WavModel(total_upsample=2)
    sched = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=3,
        left_context_size=5,
        initial_codec_chunk_frames=2,
        sample_rate=24000,
    )
    sched._payloads["req-1"] = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": True}),
        data={},
    )

    # Frame 1: ready=1 < initial=2 -> no emit.
    sched._on_chunk("req-1", _code_chunk(0))
    assert model.calls == []
    # Frame 2: ready=2 >= initial=2 -> first emit (2-frame window).
    sched._on_chunk("req-1", _code_chunk(1))
    assert model.calls == [(1, 2, 2)]
    # Frames 3-4: steady threshold=3, not enough yet.
    sched._on_chunk("req-1", _code_chunk(2))
    sched._on_chunk("req-1", _code_chunk(3))
    assert model.calls == [(1, 2, 2)]
    # Frame 5: ready=3 >= steady=3 -> second emit (5-frame window = 2 left + 3 new).
    sched._on_chunk("req-1", _code_chunk(4))
    assert model.calls == [(1, 2, 2), (1, 2, 5)]

    parts = _drain_stream_audio(sched)
    assert len(parts) == 2
    assert parts[0].shape == (4,), "first chunk: 2 frames * 2 upsample"
    assert parts[1].shape == (6,), "second chunk: 5 frames * 2 - 4 trim"


def test_code2wav_per_request_initial_chunk_overrides_config_default() -> None:
    """params['initial_codec_chunk_frames'] overrides the scheduler default."""
    model = FakeCode2WavModel(total_upsample=2)
    sched = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=0,
        initial_codec_chunk_frames=4,  # config default
        sample_rate=24000,
    )
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs=[],
            params={"stream": True, INITIAL_CODEC_CHUNK_FRAMES_PARAM: 2},
        ),
        data={},
    )
    # Realistic entry: stores payload + calls on_streaming_new_request (latch).
    sched._handle_streaming_new_request("req-1", payload)
    assert sched._req_initial_chunk_frames["req-1"] == 2

    for i in range(2):
        sched._on_chunk("req-1", _code_chunk(i))
    # First emit at 2 (per-request override), not 4 (config) or 10 (steady).
    assert model.calls == [(1, 2, 2)]


def test_code2wav_per_request_initial_chunk_zero_opts_out() -> None:
    """An explicit params['initial_codec_chunk_frames']=0 must opt out of the
    smaller first chunk, falling back to the steady size even when the
    scheduler config default is non-zero."""
    model = FakeCode2WavModel(total_upsample=2)
    sched = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=3,
        left_context_size=0,
        initial_codec_chunk_frames=2,  # config default
        sample_rate=24000,
    )
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs=[],
            params={"stream": True, INITIAL_CODEC_CHUNK_FRAMES_PARAM: 0},
        ),
        data={},
    )
    sched._handle_streaming_new_request("req-1", payload)
    assert sched._req_initial_chunk_frames["req-1"] == 0

    # Feed 2 frames: config default would emit at 2, but opt-out means steady=3.
    for i in range(2):
        sched._on_chunk("req-1", _code_chunk(i))
    assert model.calls == []
    # Frame 3: steady threshold reached.
    sched._on_chunk("req-1", _code_chunk(2))
    assert model.calls == [(1, 2, 3)]


def test_code2wav_initial_chunk_clamped_to_steady() -> None:
    """A per-request initial chunk larger than steady is clamped to steady
    (resolve_initial_codec_chunk_frames semantics), so it degrades gracefully
    to the steady size rather than over-collecting."""
    model = FakeCode2WavModel(total_upsample=2)
    sched = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=3,
        left_context_size=0,
        sample_rate=24000,
    )
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs=[],
            params={"stream": True, INITIAL_CODEC_CHUNK_FRAMES_PARAM: 100},
        ),
        data={},
    )
    sched._handle_streaming_new_request("req-1", payload)
    # Clamped to steady=3.
    assert sched._req_initial_chunk_frames["req-1"] == 3

    for i in range(3):
        sched._on_chunk("req-1", _code_chunk(i))
    assert model.calls == [(1, 2, 3)]
