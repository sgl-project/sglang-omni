# SPDX-License-Identifier: Apache-2.0
"""D2 frame order, interrupt boundaries, and session cleanup through the terminal stage."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmarks.duplex.v15_audio import reconstruct_output
from sglang_omni.admission import QueueFullError
from sglang_omni.client.client import Client
from sglang_omni.models.qwen3_omni_d2.config import (
    MAX_SESSIONS,
    Qwen3OmniD2PipelineConfig,
)
from sglang_omni.models.qwen3_omni_d2.realtime import (
    Qwen3OmniD2OutputConverter,
    create_realtime_deployment,
)
from sglang_omni.models.qwen3_omni_d2.runtime import Qwen3OmniD2Runtime
from sglang_omni.models.qwen3_omni_d2.session import Qwen3OmniD2Frame
from sglang_omni.models.qwen3_omni_d2.stages import Qwen3OmniD2SessionHooks
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    OutputChunk,
    SessionIdentity,
    SessionOperation,
    SessionOperationKind,
    TimedChunk,
)
from sglang_omni.scheduling.session import SessionScheduler
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    ResponseFinished,
    ResponseStarted,
)
from sglang_omni.serve.realtime.output_buffer import OutputBuffer
from sglang_omni.serve.realtime.projection import project_output
from sglang_omni.serve.realtime.types import RuntimeLimits


def duplex_stage(runtime: Qwen3OmniD2Runtime, interval_ms: int) -> SessionScheduler:
    return SessionScheduler(
        Qwen3OmniD2SessionHooks(runtime, interval_ms),
        max_open_sessions=MAX_SESSIONS,
        max_concurrency=1,
    )


def submit(
    stage: SessionScheduler,
    identity: SessionIdentity,
    operation: SessionOperationKind,
    chunk: TimedChunk | None = None,
) -> list[OutputChunk]:
    session_operation = SessionOperation(operation, identity, ("duplex",), chunk)
    request = OmniRequest(
        None, metadata={SESSION_METADATA_KEY: session_operation.to_dict()}
    )
    stage.compute(StagePayload(operation, request, {}))
    chunks = []
    while not stage.outbox.empty():
        message = stage.outbox.get_nowait()
        assert message.type == "stream" and chunk is not None
        frame = TimedChunk.from_dict(message.data)
        chunks.append(
            OutputChunk(
                session_identity=identity, input_seq=chunk.seq, **frame.to_dict()
            )
        )
    return chunks


def playback(
    sequence: int, text: str, *, interrupted: bool = False
) -> Qwen3OmniD2Frame:
    return Qwen3OmniD2Frame(
        pcm16=bytes([sequence, 255]) * 1920,
        text_delta=text,
        sequence=sequence,
        is_interruption=interrupted,
    )


def test_interrupt_inside_one_macro_and_end_on_its_last_frame(tmp_path: Path) -> None:
    runtime = Mock(spec=Qwen3OmniD2Runtime)
    frames = [
        playback(0, "old"),
        playback(1, "", interrupted=True),
        playback(2, "new"),
        playback(3, "!"),
    ]
    runtime.step.return_value = frames
    stage = duplex_stage(runtime, interval_ms=320)
    identity = SessionIdentity("conversation")
    submit(stage, identity, "open")
    pcm16 = b"\xfe\x80" * 4800 + bytes(640)
    chunk = TimedChunk("audio", 0, 300, 0, pcm16, "pcm16", eos=True)
    emitted = submit(stage, identity, "append", chunk)
    runtime.step.assert_called_once_with(pcm16)
    assert [(frame.t_start_ms, frame.duration_ms, frame.eos) for frame in emitted] == [
        (0, 80, False),
        (80, 80, False),
        (160, 80, False),
        (240, 80, True),
    ]
    convert = Qwen3OmniD2OutputConverter()
    outbox = OutputBuffer(RuntimeLimits())
    for frame in emitted:
        for event in convert(frame):
            outbox.emit(event, unit=None, output_modalities=("audio",))

    # Note (cuzimi): A slow socket can receive the entire macro in one burst.
    events = []
    trace = [
        {
            "direction": "send",
            "time_s": 0.0,
            "event": {"type": "input_audio_buffer.append", "sglang": {"t_start_ms": 0}},
        }
    ]
    while (envelope := outbox.dequeue()) is not None:
        outbox.before_send(envelope)
        events.append(envelope.event)
        projected = project_output(envelope.event, output_modalities=("audio",))
        trace.append({"direction": "receive", "time_s": 0.0, "event": projected})
        outbox.sent(envelope)
    assert [event.pcm for event in events if isinstance(event, AudioDelta)] == [
        frame.pcm16 for frame in frames
    ]
    finished = [event for event in events if isinstance(event, ResponseFinished)]
    assert [(event.text, event.status) for event in finished] == [
        ("old", "cancelled"),
        ("new!", "completed"),
    ]
    started_ids = [
        event.response_id for event in events if isinstance(event, ResponseStarted)
    ]
    assert started_ids == [event.response_id for event in finished]
    assert len(set(started_ids)) == 2
    assert events.index(finished[0]) < next(
        i for i, event in enumerate(events) if event.response_id == started_ids[1]
    )
    (tmp_path / "continuous.jsonl").write_text("\n".join(map(json.dumps, trace)))
    replay = reconstruct_output(tmp_path, profile="qwen3-omni-d2-320ms")
    assert replay["errors"] == []
    assert replay["pcm_sha256"] == {
        "output-media.wav": hashlib.sha256(
            b"".join(f.pcm16 for f in frames)
        ).hexdigest(),
        "output-playout.wav": hashlib.sha256(
            b"".join(f.pcm16 for f in frames[1:])
        ).hexdigest(),
    }
    submit(stage, identity, "close")


def test_failed_unit_releases_capacity_for_a_fresh_conversation(tmp_path: Path) -> None:
    runtime = Mock(spec=Qwen3OmniD2Runtime)
    stage = duplex_stage(runtime, interval_ms=80)
    (tmp_path / "d2.json").write_text(
        '{"format":"d2.release.v1","family":"qwen","latency_ms":80}'
    )
    deployment = create_realtime_deployment(
        Mock(spec=Client), Qwen3OmniD2PipelineConfig(model_path=str(tmp_path))
    )
    convert = deployment.adapter_factory().output_converter
    identity = SessionIdentity("same", 1)
    reopened = SessionIdentity("same", 2)
    submit(stage, identity, "open")
    with pytest.raises(QueueFullError):
        submit(stage, reopened, "open")
    runtime.step.return_value = [playback(0, "old")]
    interval = TimedChunk("audio", 0, 80, 0, bytes(2560), "pcm16")
    first = [
        event
        for frame in submit(stage, identity, "append", interval)
        for event in convert(frame)
    ]
    with pytest.raises(ValueError, match="input bytes per interval"):
        submit(
            stage, identity, "append", TimedChunk("audio", 80, 1, 1, bytes(32), "pcm16")
        )
    submit(stage, reopened, "open")
    convert = deployment.adapter_factory().output_converter
    runtime.step.return_value = [playback(0, "fresh")]
    second = [
        event
        for frame in submit(stage, reopened, "append", interval)
        for event in convert(frame)
    ]
    end = TimedChunk("audio", 80, 0, 1, b"", "pcm16", eos=True)
    second.extend(
        event
        for frame in submit(stage, reopened, "append", end)
        for event in convert(frame)
    )
    assert runtime.step.call_count == 2
    assert isinstance(first[0], ResponseStarted) and isinstance(
        second[0], ResponseStarted
    )
    assert first[0].response_id != second[0].response_id
    assert isinstance(second[-1], ResponseFinished) and second[-1].text == "fresh"
    submit(stage, reopened, "close")
    assert runtime.open_session.call_count == 2
