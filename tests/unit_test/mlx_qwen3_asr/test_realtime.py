# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from sglang_omni_mlx.qwen3_asr.realtime import (  # noqa: E402
    RealtimeSession,
    join_transcript_parts,
    realtime_settings,
)
from tests.unit_test.mlx_qwen3_asr.fakes import FakeWorker  # noqa: E402

SAMPLES_PER_MS = 16


def pcm_event(milliseconds: int, amplitude: float = 0.2) -> dict[str, object]:
    samples = np.full(milliseconds * SAMPLES_PER_MS, amplitude, dtype=np.float32)
    pcm = (samples * 32767).astype("<i2").tobytes()
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode(),
    }


class Recorder:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def __call__(self, event: dict[str, object]) -> None:
        self.events.append(event)

    def of_type(self, event_type: str) -> list[dict[str, object]]:
        return [event for event in self.events if event["type"] == event_type]


def new_session(
    worker: FakeWorker, max_segment_seconds: float = 30.0
) -> tuple[RealtimeSession, Recorder]:
    recorder = Recorder()
    session = RealtimeSession(
        worker=worker,
        settings=realtime_settings(
            decode_interval_ms=1000,
            first_decode_ms=100,
            max_segment_seconds=max_segment_seconds,
        ),
        send=recorder,
    )
    return session, recorder


async def append(
    session: RealtimeSession, milliseconds: int, amplitude: float = 0.2
) -> None:
    await session.handle(pcm_event(milliseconds, amplitude))
    if session.refresh_task is not None:
        await session.refresh_task
    else:
        pass


@pytest.mark.asyncio
async def test_only_manual_turns_are_accepted() -> None:
    session, recorder = new_session(FakeWorker())
    assert await session.handle(
        {
            "type": "session.update",
            "session": {"turn_detection": {"type": "server_vad"}},
        }
    )
    assert recorder.events[-1]["error"]["code"] == "unsupported_session"
    assert await session.handle(
        {
            "type": "session.update",
            "session": {"turn_detection": None, "language": "en"},
        }
    )
    assert recorder.events[-1]["type"] == "transcription_session.updated"
    assert session.language == "English"
    assert [event["event_index"] for event in recorder.events] == [1, 2]


@pytest.mark.asyncio
async def test_preview_decodes_after_100_ms_then_once_a_second() -> None:
    worker = FakeWorker()
    session, recorder = new_session(worker)
    await append(session, 60)
    assert worker.calls == []
    await append(session, 40)
    assert [call.sample_count for call in worker.calls] == [1600]
    await append(session, 900)
    assert len(worker.calls) == 1
    await append(session, 100)
    assert [call.sample_count for call in worker.calls] == [1600, 17600]
    previews = recorder.of_type("transcription.segment")
    assert [(event["text"], event["is_final"]) for event in previews] == [
        ("heard 1600", False),
        ("heard 17600", False),
    ]


@pytest.mark.asyncio
async def test_leading_silence_keeps_the_early_first_decode() -> None:
    worker = FakeWorker()
    session, _ = new_session(worker)
    await append(session, 500, amplitude=0.0)
    assert worker.calls == []
    await append(session, 20)
    assert [call.sample_count for call in worker.calls] == [520 * SAMPLES_PER_MS]


@pytest.mark.asyncio
async def test_refreshes_continue_from_the_shown_text_once_the_language_is_known() -> (
    None
):
    worker = FakeWorker(fixed_text="a b c d e f g")
    session, _ = new_session(worker)
    for milliseconds in (100, 1000, 1000):
        await append(session, milliseconds)
    # The third decode continues from the shown text minus its last five tokens.
    assert [call.options.prefix_text for call in worker.calls] == ["", "", "a b"]
    assert worker.calls[2].options.prefix_token_ids == (0, 1)
    assert [call.options.language for call in worker.calls] == [
        None,
        "English",
        "English",
    ]


@pytest.mark.asyncio
async def test_refreshes_start_over_while_the_language_is_unknown() -> None:
    worker = FakeWorker(fixed_text="a b c d e f g", language=None)
    session, _ = new_session(worker)
    for milliseconds in (100, 1000, 1000):
        await append(session, milliseconds)
    assert [call.options.prefix_text for call in worker.calls] == ["", "", ""]


@pytest.mark.asyncio
async def test_commit_finalizes_the_segment_and_starts_a_new_one() -> None:
    worker = FakeWorker()
    session, recorder = new_session(worker)
    await append(session, 300)
    assert await session.handle({"type": "input_audio_buffer.commit"})
    finals = [
        event
        for event in recorder.of_type("transcription.segment")
        if event["is_final"]
    ]
    assert [(event["segment_id"], event["text"]) for event in finals] == [
        (0, "heard 4800")
    ]
    await append(session, 200)
    assert recorder.of_type("transcription.segment")[-1]["segment_id"] == 1


@pytest.mark.asyncio
async def test_a_silent_segment_finalizes_empty_without_decoding() -> None:
    worker = FakeWorker()
    session, recorder = new_session(worker)
    await append(session, 400, amplitude=0.0)
    await session.handle({"type": "input_audio_buffer.commit"})
    assert worker.calls == []
    assert recorder.of_type("transcription.segment")[-1] == {
        **recorder.of_type("transcription.segment")[-1],
        "text": "",
        "is_final": True,
    }


@pytest.mark.asyncio
async def test_long_audio_is_cut_at_the_segment_limit() -> None:
    worker = FakeWorker()
    session, recorder = new_session(worker, max_segment_seconds=2.0)
    for _ in range(5):
        await append(session, 1000)
    finals = [
        event
        for event in recorder.of_type("transcription.segment")
        if event["is_final"]
    ]
    assert [(event["segment_id"], event["text"]) for event in finals] == [
        (0, "heard 32000"),
        (1, "heard 32000"),
    ]
    assert session.segment is not None and session.segment.segment_id == 2


@pytest.mark.asyncio
async def test_done_finalizes_and_completes_with_the_joined_text() -> None:
    worker = FakeWorker()
    session, recorder = new_session(worker)
    await append(session, 300)
    await session.handle({"type": "input_audio_buffer.commit"})
    await append(session, 200)
    assert not await session.handle({"type": "transcription.done"})
    assert recorder.events[-1]["type"] == "transcription.completed"
    assert recorder.events[-1]["text"] == "heard 4800 heard 3200"


@pytest.mark.asyncio
@pytest.mark.parametrize("audio", ["***", base64.b64encode(b"\x00").decode(), 7])
async def test_audio_that_is_not_base64_pcm16_is_an_error(audio: object) -> None:
    session, recorder = new_session(FakeWorker())
    assert await session.handle({"type": "input_audio_buffer.append", "audio": audio})
    assert recorder.events[-1]["error"]["code"] == "invalid_audio"


@pytest.mark.asyncio
async def test_unknown_events_are_errors() -> None:
    session, recorder = new_session(FakeWorker())
    assert await session.handle({"type": "response.create"})
    assert recorder.events[-1]["error"]["code"] == "invalid_event"


@pytest.mark.asyncio
async def test_closing_cancels_the_decode_in_flight() -> None:
    worker = FakeWorker(gate=asyncio.Event())
    session, recorder = new_session(worker)
    await session.handle(pcm_event(200))
    await asyncio.sleep(0)
    assert len(worker.calls) == 1
    session.close()
    worker.gate.set()
    await session.refresh_task
    assert recorder.of_type("transcription.segment") == []


@pytest.mark.asyncio
async def test_a_failed_preview_decode_reports_an_error() -> None:
    worker = FakeWorker(failure=RuntimeError("model output"))
    session, recorder = new_session(worker)
    await append(session, 200)
    errors = recorder.of_type("error")
    assert [event["error"]["code"] for event in errors] == ["transcription_failed"]
    assert "model output" not in str(recorder.events)


@pytest.mark.asyncio
async def test_a_failed_final_decode_reports_an_error_and_the_session_goes_on() -> None:
    worker = FakeWorker(failure=RuntimeError("model output"))
    session, recorder = new_session(worker, max_segment_seconds=30.0)
    await session.handle(pcm_event(50))
    assert await session.handle({"type": "input_audio_buffer.commit"})
    assert recorder.events[-1]["error"]["code"] == "transcription_failed"
    worker.failure = None
    assert not await session.handle({"type": "transcription.done"})
    assert recorder.events[-1]["type"] == "transcription.completed"


@pytest.mark.asyncio
async def test_an_unknown_session_language_is_used_as_given() -> None:
    worker = FakeWorker()
    session, recorder = new_session(worker)
    await session.handle(
        {
            "type": "session.update",
            "session": {"turn_detection": None, "language": "Klingon"},
        }
    )
    await append(session, 200)
    assert worker.calls[0].options.language == "Klingon"
    await session.handle(
        {"type": "session.update", "session": {"turn_detection": None, "language": ""}}
    )
    assert session.language is None


@pytest.mark.asyncio
async def test_a_very_short_final_tail_is_decoded() -> None:
    worker = FakeWorker()
    session, recorder = new_session(worker, max_segment_seconds=1.0)
    # 1 s cut, then a 3 ms tail: the tail is still finalized.
    await session.handle(pcm_event(1003))
    await session.handle({"type": "input_audio_buffer.commit"})
    finals = [
        event
        for event in recorder.of_type("transcription.segment")
        if event["is_final"]
    ]
    assert [event["text"] for event in finals] == ["heard 16000", "heard 48"]


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (["Hello", "world"], "Hello world"),
        (["你好", "世界"], "你好世界"),
        (["Hello ", " 世界"], "Hello世界"),
        (["", " ok ", ""], "ok"),
    ],
)
def test_join_transcript_parts(parts: list[str], expected: str) -> None:
    assert join_transcript_parts(parts) == expected
