"""Realization model work leaves the websocket loop live and closes after GPU work."""

import asyncio
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from io import BytesIO
from typing import Literal
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from sglang.multimodal_gen.configs.pipeline_configs.duo import (
    DUO_DECODE_SPECS,
    DuoDecodeMode,
)
from sglang.multimodal_gen.runtime.models.vaes.duo.codec import DuoCodecs

from sglang_omni.models.duo.live import LiveDuoSessionHooks, LiveQwenASR
from sglang_omni.models.duo.live_intent import IntentVisual, LiveIntentPlanner, LivePlan
from sglang_omni.models.duo.realtime import (
    DuoLiveResources,
    DuoRealtimeAdapter,
    encode_jpeg,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    OutputEvent,
    ResponseFinished,
    TurnFailure,
    VideoDelta,
)
from sglang_omni.serve.realtime.types import Unit


@pytest.mark.asyncio
async def test_jpeg_delivery_drains_before_encoding_all_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = np.full((4, 32, 32, 3), 96, dtype=np.uint8)
    outgoing: asyncio.Queue[VideoDelta] = asyncio.Queue()
    encoded_frames = 0
    first_drained_count = 0

    def observed_encode(frame: np.ndarray) -> bytes:
        nonlocal encoded_frames
        encoded_frames += 1
        return encode_jpeg(frame)

    async def emit(event: OutputEvent, unit: Unit | None = None) -> None:
        if isinstance(event, VideoDelta):
            outgoing.put_nowait(event)
        else:
            pass

    async def drain() -> list[VideoDelta]:
        nonlocal first_drained_count
        video = [await outgoing.get()]
        first_drained_count = encoded_frames
        for _ in range(3):
            video.append(await outgoing.get())
        return video

    monkeypatch.setattr("sglang_omni.models.duo.realtime.encode_jpeg", observed_encode)
    adapter = DuoRealtimeAdapter(DuoLiveResources(Mock(spec=LiveDuoSessionHooks), 1))
    await adapter.open("jpeg-worker", {}, emit)
    consumer = asyncio.create_task(drain())
    try:
        await adapter.emit_block(
            Unit(0, 0, b"", 0),
            [
                TimedChunk(
                    modality="video",
                    t_start_ms=0,
                    duration_ms=1000 / 6,
                    seq=0,
                    payload={"data": frames.tobytes(), "shape": frames.shape},
                )
            ],
        )
        video = await asyncio.wait_for(consumer, 2)
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await adapter.close()
    assert first_drained_count < len(frames)
    assert [event.frame for event in video] == [encode_jpeg(frame) for frame in frames]
    assert [event.t_start_ms for event in video] == pytest.approx(
        [index * 1000 / 24 for index in range(4)]
    )


@pytest.mark.asyncio
async def test_realization_worker_keeps_loop_live_and_defers_cleanup() -> None:
    hooks = Mock(spec=LiveDuoSessionHooks)
    adapter = DuoRealtimeAdapter(DuoLiveResources(hooks, max_blocks=3))
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()

    def block(
        chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "model worker was not released"
        finished.set()
        return payload

    def close_session(session_id: SessionIdentity) -> None:
        assert finished.is_set(), "state was freed during a forward"

    async def emit(event: OutputEvent, unit: Unit | None = None) -> None:
        pytest.fail("a cancelled unit must not emit output")

    hooks.append.side_effect = block
    hooks.close.side_effect = close_session
    with BytesIO() as buffer:
        Image.new("RGB", (64, 96)).save(buffer, format="PNG")
        reference_image = base64.b64encode(buffer.getvalue()).decode()
    await adapter.open(
        "worker-test",
        {"sglang": {"realization": {"reference_image": reference_image}}},
        emit,
    )
    with BytesIO() as buffer:
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
        frame = buffer.getvalue()
    unit = Unit(0, 0, b"\0\0" * 2667, 2667, images=(frame,) * 4)
    processing = asyncio.create_task(adapter.process(unit))
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        generation_request = hooks.open.call_args.args[1]
        assert generation_request.inputs["height"] == 96
        assert generation_request.inputs["width"] == 64
        observation = hooks.append.call_args.args[0].payload
        assert observation["height"] == 32
        assert observation["width"] == 32
        assert not processing.done()
        closing = asyncio.create_task(adapter.close())
        await asyncio.sleep(0)
        hooks.close.assert_not_called()
        assert adapter.cancelled.is_set()
    finally:
        release.set()
        await processing
        if closing is not None:
            await closing
        else:
            await adapter.close()
    hooks.close.assert_called_once()
    assert adapter.session_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "interval_seconds,expected_resets", [(30.0, [43, 86]), (0.0, [])]
)
async def test_server_resets_dit_on_generated_media_boundaries(
    interval_seconds: float, expected_resets: list[int]
) -> None:
    """Reset generated history without reopening Intent or requiring a client update."""
    hooks = Mock(spec=LiveDuoSessionHooks)
    adapter = DuoRealtimeAdapter(
        DuoLiveResources(
            hooks,
            max_blocks=88,
            supports_realization_kv_reset=True,
            realization_kv_reset_interval_seconds=interval_seconds,
        )
    )

    async def emit(event: OutputEvent, unit: Unit | None = None) -> None:
        pass

    configuration = {"sglang": {"realization": {"seed": 123, "decode": "duo_5_17"}}}
    await adapter.open("reset-test", configuration, emit)
    with BytesIO() as buffer:
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
        frame = buffer.getvalue()
    spec = DUO_DECODE_SPECS["duo_5_17"]
    try:
        for block in range(88):
            await adapter.process(
                Unit(
                    block,
                    block * spec.input_audio_samples,
                    b"\0\0" * spec.input_audio_samples,
                    spec.input_audio_samples,
                    images=(frame,) * 17,
                )
            )
        resets = [
            index
            for index, call in enumerate(hooks.append.call_args_list)
            if call.args[0].payload["reset_realization_kv"]
        ]
        assert resets == expected_resets
        hooks.open.assert_called_once()
        hooks.close.assert_not_called()
        assert hooks.open.call_args.args[1].inputs["seed"] == 123
    finally:
        await adapter.close()
    await adapter.open("second-session", configuration, emit)
    try:
        await adapter.process(
            Unit(
                0,
                0,
                b"\0\0" * spec.input_audio_samples,
                spec.input_audio_samples,
                images=(frame,) * 17,
            )
        )
        assert not hooks.append.call_args.args[0].payload["reset_realization_kv"]
    finally:
        await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("decode,frame_count", [("duo_1_4", 4), ("duo_5_17", 17)])
async def test_realization_empty_eos_finishes_without_an_extra_model_block(
    decode: str, frame_count: int
) -> None:
    """Ending drained input previously tried to generate a block with no images."""
    with BytesIO() as buffer:
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
        frame = buffer.getvalue()
    event_sequences: list[list[OutputEvent]] = []
    for separate_eos in (False, True):
        hooks = Mock(spec=LiveDuoSessionHooks)
        adapter = DuoRealtimeAdapter(DuoLiveResources(hooks, max_blocks=1))
        events: list[OutputEvent] = []

        async def emit(event: OutputEvent, unit: Unit | None = None) -> None:
            events.append(event)

        await adapter.open(
            "eos-test", {"sglang": {"realization": {"decode": decode}}}, emit
        )
        unit = Unit(
            0,
            0,
            b"\0\0" * 2667,
            2667,
            eos=not separate_eos,
            images=(frame,) * frame_count,
        )
        assert await adapter.process(unit) == unit.real_samples
        if separate_eos:
            assert await adapter.process(Unit(1, 2667, b"", 0, eos=True)) == 0
        else:
            pass
        hooks.open.assert_called_once()
        hooks.append.assert_called_once()
        event_sequences.append(events)
        await adapter.close()
    assert event_sequences[0] == event_sequences[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["duo_1_4", "duo_5_17"])
@pytest.mark.parametrize("action", ["next", "pause", "cancel", "fail"])
async def test_realization_decode_overlaps_next_plan_and_preserves_delivery_lifetime(
    monkeypatch: pytest.MonkeyPatch,
    mode: DuoDecodeMode,
    action: Literal["next", "pause", "cancel", "fail"],
) -> None:
    monkeypatch.setattr(torch.cuda, "device", Mock(return_value=nullcontext()))
    loop = asyncio.get_running_loop()
    decoding = asyncio.Event()
    next_planned = asyncio.Event()
    returned = asyncio.Event()
    release_decode = threading.Event()
    decode_finished = threading.Event()
    order: list[str] = []
    events: list[tuple[OutputEvent, Unit | None]] = []
    spec = DUO_DECODE_SPECS[mode]
    planner = Mock(spec=LiveIntentPlanner)
    planner.device = torch.device("cpu")
    realization = Mock()
    realization.device = torch.device("cpu")
    realization.graph_counts = (0, 0)
    realization.usage.return_value = ResourceUsage(kv_tokens=4, bytes=16)
    codecs = Mock(spec=DuoCodecs)
    codecs.device = torch.device("cpu")
    codecs.video_lookahead_blocks = 0
    codecs.encode_reference.return_value = torch.zeros(24, 1, 4, 4)
    codecs.finish_5_17.return_value = None
    asr = Mock(spec=LiveQwenASR)
    asr_started = threading.Event()
    visual_ready = threading.Event()
    visual_calls = 0
    asr_calls = 0

    def transcribe(audio_pcm16: bytes) -> str:
        nonlocal asr_calls
        asr_calls += 1
        asr_started.set()
        assert visual_ready.wait(5), "ASR prevented current video preprocessing"
        return "hello" if asr_calls == 1 else " world"

    def prepare_visual(frames: np.ndarray) -> IntentVisual:
        nonlocal visual_calls
        visual_calls += 1
        if visual_calls == 2:
            assert asr_started.wait(5), "video preprocessing did not overlap ASR"
            visual_ready.set()
        else:
            pass
        return {"patches": torch.zeros(1, 3), "grid": [1, 1, 1]}

    asr.transcribe.side_effect = transcribe
    planner.visual_patch.side_effect = prepare_visual

    def plan(
        session_id: str,
        frames: np.ndarray,
        user_text: str,
        block_index: int,
        cancelled: threading.Event,
        *,
        frames_per_block: int,
        block_duration_ms: float,
        prepared_visual: IntentVisual | None = None,
    ) -> LivePlan:
        order.append(f"plan{block_index}")
        if block_index == 1:
            expected_text = (
                "hello world" if 2 * spec.input_audio_samples > 16000 else "hello"
            )
            assert user_text == expected_text
            assert not decode_finished.is_set(), "Intent waited for the preceding Codec"
            loop.call_soon_threadsafe(next_planned.set)
        else:
            pass
        return LivePlan(
            shared_h50=torch.zeros(1, 8),
            plan_h50=torch.zeros(1, 8),
            body_ids=[1],
            text="<wait>",
            prefix_tokens=1,
            reused_tokens=block_index,
            kv_tokens=block_index + 1,
        )

    def generate(
        chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if chunk.seq == 1:
            assert decode_finished.is_set(), "TP8 overlapped the preceding Codec"
            assert payload.data["reset_realization_kv"]
        else:
            pass
        order.append(f"realization{chunk.seq}")
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "video_latent": payload.data["video"].tokens,
                "audio_latent": payload.data["audio"].tokens,
            },
        )

    def decode(
        reference: torch.Tensor,
        previous: list[torch.Tensor],
        video: torch.Tensor,
        audio: torch.Tensor,
        *,
        session_id: str,
        audio_samples: int,
        mode: DuoDecodeMode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not previous:
            order.append("decode0-start")
            loop.call_soon_threadsafe(decoding.set)
            assert release_decode.wait(5), "decode worker was not released"
            decode_finished.set()
            order.append("decode0-end")
            if action == "fail":
                raise RuntimeError("decode failed")
            else:
                pass
        else:
            order.append("decode1")
        return torch.zeros(
            spec.input_frames, 32, 32, 3, dtype=torch.uint8
        ), torch.zeros(2, audio_samples)

    async def emit(event: OutputEvent, unit: Unit | None = None) -> None:
        events.append((event, unit))
        if isinstance(event, (AudioDelta, TurnFailure)):
            returned.set()
        else:
            pass

    def close_stream(session_id: str) -> None:
        assert decode_finished.is_set(), "codec state was freed during decoding"

    planner.append.side_effect = plan
    realization.append.side_effect = generate
    codecs.decode.side_effect = decode
    codecs.close_stream.side_effect = close_stream
    hooks = LiveDuoSessionHooks(planner, realization, codecs, max_blocks=2, asr=asr)
    with BytesIO() as image_buffer:
        Image.new("RGB", (32, 32)).save(image_buffer, format="PNG")
        frame = image_buffer.getvalue()
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        ThreadPoolExecutor(max_workers=1) as input_executor,
    ):
        hooks.input_executor = input_executor
        adapter = DuoRealtimeAdapter(
            DuoLiveResources(
                hooks,
                max_blocks=2,
                decode_executor=executor,
                supports_realization_kv_reset=True,
                realization_kv_reset_interval_seconds=spec.duration_ms / 1000,
            )
        )
        await adapter.open(
            "overlap", {"sglang": {"realization": {"decode": mode}}}, emit
        )
        unit0 = Unit(
            0,
            0,
            b"\0\0" * spec.input_audio_samples,
            spec.input_audio_samples,
            images=(frame,) * spec.input_frames,
        )
        second: asyncio.Task[int] | None = None
        closing: asyncio.Task[None] | None = None
        try:
            assert await adapter.process(unit0) == unit0.real_samples
            await asyncio.wait_for(decoding.wait(), 2)
            assert not events
            if action == "next":
                unit1 = Unit(
                    1,
                    spec.input_audio_samples,
                    unit0.pcm,
                    unit0.real_samples,
                    images=unit0.images,
                    eos=True,
                )
                second = asyncio.create_task(adapter.process(unit1))
                await asyncio.wait_for(next_planned.wait(), 2)
                assert realization.append.call_count == 1
            elif action == "cancel":
                closing = asyncio.create_task(adapter.close())
                await asyncio.sleep(0)
                assert not closing.done()
                codecs.close_stream.assert_not_called()
            else:
                pass
        finally:
            release_decode.set()
            if second is not None:
                await second
            else:
                pass
            if closing is not None:
                await closing
            elif action == "fail":
                await asyncio.wait_for(returned.wait(), 2)
                assert isinstance(events[-1][0], TurnFailure)
                assert "decode failed" in events[-1][0].message
                await adapter.close()
            else:
                await asyncio.wait_for(returned.wait(), 2)
                if action == "pause":
                    await adapter.process(
                        Unit(1, spec.input_audio_samples, b"", 0, eos=True)
                    )
                else:
                    pass
                await adapter.close()
    if action == "next":
        assert (
            order.index("plan1")
            < order.index("decode0-end")
            < order.index("realization1")
        )
        audio_units = [
            unit.index for event, unit in events if isinstance(event, AudioDelta)
        ]
        assert audio_units == [0, 1]
        assert (
            sum(isinstance(event, VideoDelta) for event, unit in events)
            == 2 * spec.input_frames
        )
        assert isinstance(events[-1][0], ResponseFinished)
    elif action == "pause":
        assert any(isinstance(event, AudioDelta) for event, unit in events)
        assert isinstance(events[-1][0], ResponseFinished)
    elif action == "cancel":
        assert not events
    else:
        assert len(events) == 1
    assert not hooks.states
    planner.open.assert_called_once()
    realization.open.assert_called_once()
    planner.close.assert_called_once()
    realization.close.assert_called_once()


@pytest.mark.asyncio
async def test_wrong_reference_shape_does_not_reach_parallel_workers() -> None:
    """A wrong-sized reference previously stopped all persistent DiT ranks."""
    hooks = Mock(spec=LiveDuoSessionHooks)
    adapter = DuoRealtimeAdapter(DuoLiveResources(hooks, 4, reference_size=(64, 64)))

    async def emit(event: OutputEvent, unit: Unit | None = None) -> None:
        pass

    await adapter.open("wrong-reference", {}, emit)
    with BytesIO() as buffer:
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
        image = buffer.getvalue()
    try:
        with pytest.raises(ValueError, match="Reference height and width"):
            await adapter.process(Unit(0, 0, bytes(5334), 2667, images=(image,) * 4))
        hooks.open.assert_not_called()
        hooks.append.assert_not_called()
    finally:
        await adapter.close()
