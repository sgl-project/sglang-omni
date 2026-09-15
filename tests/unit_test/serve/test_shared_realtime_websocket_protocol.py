# SPDX-License-Identifier: Apache-2.0
import base64

import httpx
import pytest
import websockets

from sglang_omni.serve.realtime.output import (
    TranscriptionDelta,
    TranscriptionFinished,
    TranscriptionRevision,
    TranscriptionSegment,
)
from sglang_omni.serve.realtime.runtime import Capabilities, RuntimeLimits
from tests.unit_test.fixtures.realtime_websocket import (
    Producer,
    append,
    endpoint,
    recv,
    send,
    until,
)


@pytest.mark.asyncio
async def test_mounted_created_strict_recovery_eos_and_ack():
    async with endpoint() as (http, url, producer, app):
        async with httpx.AsyncClient() as client:
            response = await client.get(http + "/v1/realtime/capabilities")
            assert response.status_code == 200
            assert response.json()["supports_resume"] is False
        assert producer.opened == 0
        async with websockets.connect(url) as ws:
            created = await recv(ws)
            assert created["type"] == "session.created"
            assert created["sglang"] == dict(epoch=0)
            assert producer.opened == 0
            await append(ws, 0)
            assert (await recv(ws))["error"]["code"] == "invalid_state"
            for frame in (
                "{",
                "[]",
                b"binary",
                "{}",
                '{"event_id":"nan","type":"session.update","session":{"x":NaN}}',
            ):
                await ws.send(frame)
                error = await recv(ws)
                assert error["type"] == "error" and error["sglang"]["fatal"] is False
            await send(
                ws, "session.update", "open", session={"output_modalities": ["audio"]}
            )
            updated = await recv(ws)
            assert (
                updated["session"]["id"]
                == created["session"]["id"]
                == producer.session_id
            )
            assert updated["client_event_id"] == "open"
            for audio in ("@@@", base64.b64encode(b"x").decode()):
                await send(
                    ws,
                    "input_audio_buffer.append",
                    "bad",
                    audio=audio,
                    sglang=dict(seq=0),
                )
                assert (await recv(ws))["error"]["event_id"] == "bad"
            await append(ws, True)
            assert (await recv(ws))["error"]["code"] == "invalid_state"
            await append(ws, 0, start=4)
            assert (await recv(ws))["error"]["code"] == "invalid_state"
            await send(ws, "session.update", "patch", session=dict(model="other"))
            assert (await recv(ws))["error"]["code"] == "invalid_state"
            await append(ws, 0)
            assert (await recv(ws))["accepted_end_ms"] == 5
            await send(ws, "sglang.input_audio.end", "eos")
            ended = await recv(ws)
            assert ended["type"] == "sglang.input_audio.ended"
            drained, events = await until(ws, "sglang.input_audio.drained")
            assert drained["consumed_ms"] == 5 and drained["discarded_ms"] == 0
            assert [e["type"] for e in events[:3]] == [
                "response.created",
                "response.output_audio_transcript.delta",
                "response.output_audio.delta",
            ]
            assert sum(e["type"] == "response.done" for e in events) == 1
            await send(
                ws,
                "sglang.playback.ack",
                "too_far",
                response_id="r0",
                item_id="i0",
                content_index=0,
                audio_end_ms=11,
                sglang=dict(epoch=0),
            )
            assert (await recv(ws))["error"]["code"] == "invalid_request"
            await send(
                ws,
                "sglang.playback.ack",
                "ack",
                response_id="r0",
                item_id="i0",
                content_index=0,
                audio_end_ms=10,
                sglang=dict(epoch=0),
            )
            await send(ws, "response.cancel", "cancel")
            cancelled = await recv(ws)
            assert cancelled["type"] == "sglang.response.cancelled"
            assert cancelled["epoch"] == 1 and cancelled["input_policy"] == "preserve"
            await send(ws, "session.close", "close")
            closed = await recv(ws)
            assert closed["type"] == "session.closed" and closed["held"]["bytes"] == 0
        assert producer.closed == 1


@pytest.mark.asyncio
async def test_input_overflow_preserves_retry_clock_and_clear_accounting():
    class CancellableProducer(Producer):
        async def cancel(self):
            self.release.set()

    producer = CancellableProducer()
    producer.release.clear()
    async with endpoint(producer, limits=RuntimeLimits(max_input_bytes=800)) as (
        _,
        url,
        _,
        app,
    ):
        async with websockets.connect(url) as ws:
            await recv(ws)
            await send(ws, "session.update", session={})
            await recv(ws)
            await append(ws, 0, 320)
            await recv(ws)
            await producer.started.wait()
            await append(ws, 1, 80)
            assert (await recv(ws))["seq"] == 1
            await append(ws, 2, 80)
            assert (await recv(ws))["error"]["code"] == "buffer_overflow"
            await send(ws, "input_audio_buffer.clear", "clear")
            assert (await recv(ws))["sglang"]["discarded_ms"] == 5
            await append(ws, 2, 80, start=25)
            assert (await recv(ws))["accepted_end_ms"] == 30
            await send(ws, "response.cancel", "cancel")
            cancelled, _ = await until(ws, "sglang.response.cancelled")
            assert cancelled["epoch"] == 1
            await send(ws, "sglang.input_audio.end", "end")
            drained, events = await until(ws, "sglang.input_audio.drained")
            assert drained["consumed_ms"] == 25 and drained["discarded_ms"] == 5
            assert all(e["sglang"]["epoch"] == 1 for e in events)


@pytest.mark.asyncio
async def test_session_timeout_closes_with_explicit_reason():
    async with endpoint(limits=RuntimeLimits(session_timeout_s=0.05)) as (
        _,
        url,
        producer,
        app,
    ):
        async with websockets.connect(url) as ws:
            await recv(ws)
            error = await recv(ws)
            assert error["error"]["code"] == "session_timeout"
            assert (await recv(ws))["type"] == "session.closed"


@pytest.mark.asyncio
async def test_same_turn_two_segments_revision_final_freeze():
    class ASR(Producer):
        async def process(self, unit, epoch):
            for seg in ("s1", "s2"):
                fields = dict(item_id="turn", segment_id=seg, start_ms=0, end_ms=5)
                await self.emit(TranscriptionDelta(text="helo", **fields), epoch)
                await self.emit(
                    TranscriptionRevision(
                        text="hello", base_revision_id=1, revision_id=2, **fields
                    ),
                    epoch,
                )
                await self.emit(
                    TranscriptionSegment(
                        text="hello" if seg == "s1" else "hello!", **fields
                    ),
                    epoch,
                )
                await self.emit(
                    TranscriptionRevision(
                        text="late", base_revision_id=2, revision_id=3, **fields
                    ),
                    epoch,
                )
            await self.emit(TranscriptionFinished("turn", "hello hello!"), epoch)
            return unit.real_samples

    async with endpoint(
        ASR(), caps=Capabilities(interaction="transcription", partial_style="revising")
    ) as (_, url, _, _):
        async with websockets.connect(url) as ws:
            await recv(ws)
            await send(ws, "session.update", session={})
            await recv(ws)
            await append(ws, 0)
            await recv(ws)
            await send(ws, "sglang.input_audio.end")
            _, events = await until(ws, "sglang.input_audio.drained")
            finals = [e for e in events if e["type"].endswith("transcription.segment")]
            assert len(finals) == 2
            assert {e["id"] for e in finals} == {"s1", "s2"}
            assert all(e["item_id"] == "turn" and "turn_id" not in e for e in finals)
            assert [e["text"] for e in finals] == ["hello", "hello!"]
            assert (
                sum(e["type"].endswith("transcription.completed") for e in events) == 1
            )
            assert not any(e.get("text") == "late" for e in events)
            deltas = [e for e in events if e["type"].endswith("transcription.delta")]
            assert len(deltas) == 2
            assert all(
                "segment_id" not in e and e["sglang"]["segment_id"] in {"s1", "s2"}
                for e in deltas
            )


@pytest.mark.asyncio
async def test_requested_text_subset_filters_audio_and_terminal_content():
    async with endpoint() as (_, url, _, _):
        async with websockets.connect(url) as ws:
            await recv(ws)
            await send(ws, "session.update", session={"output_modalities": ["text"]})
            assert (await recv(ws))["session"]["sglang"]["granted"][
                "output_modalities"
            ] == ["text"]
            await append(ws, 0)
            await recv(ws)
            await send(ws, "sglang.input_audio.end")
            _, events = await until(ws, "sglang.input_audio.drained")
            assert not any(
                e["type"].startswith("response.output_audio") for e in events
            )
            response = next(
                e["response"] for e in events if e["type"] == "response.done"
            )
            assert [c["type"] for c in response["output"][0]["content"]] == [
                "output_text"
            ]


@pytest.mark.asyncio
async def test_ga_format_namespace_and_atomic_modality_update():
    async with endpoint() as (_, url, producer, app):
        async with websockets.connect(url) as ws:
            await recv(ws)
            await send(
                ws,
                "session.update",
                session={
                    "audio": {"input": {"format": {"type": "pcm16", "rate": 16000}}}
                },
            )
            error = await recv(ws)
            assert error["error"]["param"] == "session.audio.input.format"
            assert error["sglang"]["fatal"] is False and producer.opened == 0
            await send(
                ws,
                "session.update",
                session={
                    "output_modalities": ["text"],
                    "audio": {
                        "input": {"format": {"type": "audio/pcm", "rate": 16000}}
                    },
                },
            )
            session = (await recv(ws))["session"]
            assert "granted" not in session and "granted" in session["sglang"]
            assert session["audio"]["input"]["format"]["type"] == "audio/pcm"
            await send(ws, "session.update", session={"output_modalities": ["audio"]})
            assert (await recv(ws))["session"]["output_modalities"] == ["audio"]
            await send(
                ws,
                "session.update",
                session={"output_modalities": ["text"], "model": "different"},
            )
            assert (await recv(ws))["error"]["code"] == "invalid_state"
            await send(ws, "session.update", session={})
            assert (await recv(ws))["session"]["output_modalities"] == ["audio"]
            await append(ws, 0)
            await recv(ws)
            await send(ws, "sglang.input_audio.end")
            _, events = await until(ws, "sglang.input_audio.drained")
            transcript = next(
                e
                for e in events
                if e["type"] == "response.output_audio_transcript.delta"
            )
            audio = next(
                e for e in events if e["type"] == "response.output_audio.delta"
            )
            assert transcript["content_index"] == audio["content_index"] == 0
            assert (
                "incarnation" not in audio["sglang"]
                and "segment_seq" not in audio["sglang"]
            )
            assert "media_time" in audio["sglang"] and "chunk_seq" in audio["sglang"]
