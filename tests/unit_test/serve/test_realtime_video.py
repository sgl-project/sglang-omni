# SPDX-License-Identifier: Apache-2.0
"""Video frames retain their media clock and lifecycle over realtime sessions."""

import base64
from dataclasses import replace
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sglang_omni.client.client import Client
from sglang_omni.proto.session import OutputChunk, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    PCM16,
    TURN_DONE,
    VIDEO,
    VideoFrameFormat,
    VideoPort,
)
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.control import Closed
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    ResponseFinished,
    ResponseStarted,
    TextDelta,
    TextFinished,
    VideoDelta,
    VideoFinished,
)
from sglang_omni.serve.realtime.output_buffer import OutputBuffer
from sglang_omni.serve.realtime.session_bridge import SessionBridge, capabilities_of
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits
from tests.unit_test.fixtures.realtime_client import (
    UNIT_BYTES,
    append_audio,
    build_test_client,
    receive_until,
    send_event,
)
from tests.unit_test.fixtures.realtime_fakes import HealthCoordinator, ScriptedAdapter
from tests.unit_test.fixtures.session_client import PORTS, FakeSessionClient


@pytest.fixture
def png_frame() -> bytes:
    with BytesIO() as encoded:
        Image.new("RGB", (8, 6), color="red").save(encoded, format="PNG")
        return encoded.getvalue()


@pytest.mark.parametrize(
    "modalities",
    [("video",), ("audio", "video"), ("text", "audio", "video"), ("audio",)],
)
@pytest.mark.parametrize("frame_format", ["image/jpeg", "image/png"])
def test_video_wire_preserves_output_clock_and_negotiated_content(
    png_frame: bytes, modalities: tuple[str, ...], frame_format: VideoFrameFormat
) -> None:
    with BytesIO() as encoded:
        Image.open(BytesIO(png_frame)).save(
            encoded, format="JPEG" if frame_format == "image/jpeg" else "PNG"
        )
        frame_bytes = encoded.getvalue()
    adapter = ScriptedAdapter(
        reply=[
            ResponseStarted("response"),
            TextDelta("response", "item", "hello"),
            VideoDelta("response", "item", frame_bytes, frame_format, 1000.0, 40.0),
            AudioDelta("response", "item", b"\0\0" * 160, 1000.0, 10.0),
            VideoDelta("response", "item", frame_bytes, frame_format, 1040.0, 40.0),
            VideoFinished("response", "item"),
            AudioFinished("response", "item"),
            TextFinished("response", "item", "hello"),
            ResponseFinished(
                "response", "item", "hello", True, "completed", "stop", has_video=True
            ),
        ]
    )
    capabilities = Capabilities(
        output_modalities=("text", "audio", "video"),
        max_granted_modalities=3,
        output_video_format=frame_format,
    )
    with build_test_client(adapter, capabilities=capabilities) as client:
        with client.websocket_connect("/v1/realtime") as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            send_event(
                websocket,
                "session.update",
                session={"output_modalities": list(modalities)},
            )
            updated = receive_until(websocket, "session.updated")[-1]
            assert updated["session"]["output_modalities"] == list(modalities)
            assert updated["session"]["sglang"]["granted"]["output_video_format"] == {
                "type": frame_format
            }
            append_audio(websocket, b"\0" * UNIT_BYTES, 0)
            events = receive_until(websocket, "response.done")
            frames = [e for e in events if e["type"] == "sglang.response.video.delta"]
            assert len(frames) == (2 if "video" in modalities else 0)
            for index, frame in enumerate(frames):
                assert base64.b64decode(frame["delta"]) == frame_bytes
                assert frame["format"] == frame_format
                assert frame["sglang"]["media_time"] == {
                    "t_start_ms": 1000.0 + index * 40.0,
                    "duration_ms": 40.0,
                }
                assert frame["sglang"]["unit_id"] == "unit_0"
            if "audio" in modalities:
                audio = next(
                    e for e in events if e["type"] == "response.output_audio.delta"
                )
                assert audio["sglang"]["media_time"] == {
                    "t_start_ms": 1000.0,
                    "duration_ms": 10.0,
                }
            else:
                assert not any("output_audio" in e["type"] for e in events)
            content = events[-1]["response"]["output"][0]["content"]
            assert [part["type"] for part in content] == [
                f"output_{modality}"
                for modality in ("text", "audio", "video")
                if modality in modalities
            ]
            assert sum(e["type"] == "sglang.response.video.done" for e in events) == (
                1 if "video" in modalities else 0
            )
            if modalities == ("video",):
                assert not any("output_text" in e["type"] for e in events)
            else:
                pass
            send_event(websocket, "session.close")
            receive_until(websocket, "session.closed")
    assert adapter.is_closed


def test_audio_deployment_does_not_grant_video() -> None:
    capabilities = Capabilities(
        output_modalities=("text", "audio"), max_granted_modalities=2
    )
    with build_test_client(ScriptedAdapter(), capabilities=capabilities) as client:
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            send_event(
                websocket, "session.update", session={"output_modalities": ["video"]}
            )
            assert (
                receive_until(websocket, "error")[-1]["error"]["code"]
                == "invalid_request"
            )
            send_event(
                websocket,
                "session.update",
                session={"output_modalities": ["audio", "video"]},
            )
            updated = receive_until(websocket, "session.updated")[-1]
            assert updated["session"]["output_modalities"] == ["audio"]
            assert (
                updated["session"]["sglang"]["granted"]["rejections"][0]["field"]
                == "output_modalities"
            )
            send_event(websocket, "session.close")
            receive_until(websocket, "session.closed")


def test_session_bridge_streams_video_and_audio_from_timed_chunks(
    png_frame: bytes,
) -> None:
    class MediaClient(FakeSessionClient):
        def answer(self, chunk: TimedChunk) -> None:
            for sequence, modality, output_format, payload, duration_ms in (
                (0, VIDEO, "image/png", png_frame, 40.0),
                (1, AUDIO, PCM16, b"\0\0" * 160, 10.0),
            ):
                self.outputs.put_nowait(
                    OutputChunk(
                        self.identity,
                        sequence,
                        chunk.seq,
                        modality,
                        700.0,
                        duration_ms,
                        payload,
                        output_format,
                        turn=0,
                    )
                )
            self.control(TURN_DONE, turn=0)

    ports = replace(
        PORTS,
        output=replace(PORTS.output, format=PCM16),
        video_output=VideoPort("image/png"),
    )
    session_client = MediaClient()
    deployment = RealtimeDeployment(
        capabilities=capabilities_of(ports),
        adapter_factory=lambda: SessionBridge(session_client, ports),
    )
    app = create_app(
        Client(HealthCoordinator(True)),
        model_name="video-test",
        realtime_deployment=deployment,
    )
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            send_event(
                websocket,
                "session.update",
                session={"output_modalities": ["audio", "video"]},
            )
            receive_until(websocket, "session.updated")
            append_audio(websocket, b"\0" * 6400, 0)
            events = receive_until(websocket, "response.done")
            media = [
                e
                for e in events
                if e["type"]
                in ("sglang.response.video.delta", "response.output_audio.delta")
            ]
            assert [e["sglang"]["media_time"]["t_start_ms"] for e in media] == [
                700.0,
                700.0,
            ]
            assert base64.b64decode(media[0]["delta"]) == png_frame
            assert events[-1]["response"]["output"][0]["content"] == [
                {"type": "output_audio", "transcript": ""},
                {"type": "output_video"},
            ]
            send_event(websocket, "session.close")
            receive_until(websocket, "session.closed")
    assert session_client.closed.is_set()


def test_video_queued_bytes_count_base64_and_enforce_budget(png_frame: bytes) -> None:
    buffer = OutputBuffer(RuntimeLimits(max_output_bytes=1024))
    buffer.emit(ResponseStarted("response"), None, ("video",))
    buffer.dequeue()
    with pytest.raises(RuntimeError, match="budget exhausted"):
        buffer.emit(
            VideoDelta("response", "item", png_frame * 20, "image/png", 0.0, 40.0),
            None,
            ("video",),
        )
    assert buffer.queued_bytes == 0


def test_cancellation_drops_only_cancelled_response_video(png_frame: bytes) -> None:
    buffer = OutputBuffer(RuntimeLimits())
    for response_id in ("first", "second"):
        buffer.emit(ResponseStarted(response_id), None, ("video",))
        buffer.emit(
            VideoDelta(response_id, response_id, png_frame, "image/png", 0.0, 40.0),
            None,
            ("video",),
        )
    buffer.emit(
        ResponseFinished(
            "first", "first", "", False, "cancelled", "interrupted", has_video=True
        ),
        None,
        ("video",),
    )
    remaining_video = []
    while (envelope := buffer.dequeue()) is not None:
        if isinstance(envelope.event, VideoDelta):
            remaining_video.append(envelope.event.response_id)
    assert remaining_video == ["second"]
    assert buffer.queued_bytes == 0


def test_close_finishes_visible_video_and_purges_unsent_frames(
    png_frame: bytes,
) -> None:
    buffer = OutputBuffer(RuntimeLimits())
    buffer.emit(ResponseStarted("response"), None, ("video",))
    buffer.before_send(buffer.dequeue())
    frame = VideoDelta("response", "item", png_frame, "image/png", 0.0, 40.0)
    buffer.emit(frame, None, ("video",))
    buffer.before_send(buffer.dequeue())
    buffer.emit(replace(frame, t_start_ms=40.0), None, ("video",))
    buffer.finish_responses("cancelled", "client_closed")
    buffer.finish_responses("cancelled", "client_closed")
    buffer.enqueue_terminal(Closed("client_closed"))
    events = []
    while (envelope := buffer.dequeue()) is not None:
        events.append(envelope.event)
    assert [type(event) for event in events] == [
        VideoFinished,
        ResponseFinished,
        Closed,
    ]
    assert events[1].has_video


@pytest.mark.parametrize(
    "start_ms,duration_ms",
    [(-1.0, 40.0), (0.0, 0.0), (float("nan"), 40.0), (0.0, float("inf"))],
)
def test_video_requires_finite_presentation_time(
    png_frame: bytes, start_ms: float, duration_ms: float
) -> None:
    buffer = OutputBuffer(RuntimeLimits())
    buffer.emit(ResponseStarted("response"), None, ("video",))
    with pytest.raises(RuntimeError, match="untimed video"):
        buffer.emit(
            VideoDelta(
                "response", "item", png_frame, "image/png", start_ms, duration_ms
            ),
            None,
            ("video",),
        )


def test_video_port_survives_graph_composition() -> None:
    video = replace(PORTS, video_output=VideoPort("image/png"))
    assert PORTS.with_node(video).video_output == video.video_output
    assert video.with_node(PORTS).video_output == video.video_output
    with pytest.raises(ValueError, match="share one frame format"):
        video.with_node(replace(PORTS, video_output=VideoPort("image/jpeg")))


def test_active_response_keeps_video_across_modality_update(png_frame: bytes) -> None:
    buffer = OutputBuffer(RuntimeLimits())
    buffer.emit(ResponseStarted("video_response"), None, ("video",))
    buffer.emit(
        VideoDelta("video_response", "item", png_frame, "image/png", 0.0, 40.0),
        None,
        ("audio",),
    )
    buffer.emit(ResponseStarted("audio_response"), None, ("audio",))
    buffer.emit(
        VideoDelta("audio_response", "item", png_frame, "image/png", 40.0, 40.0),
        None,
        ("audio",),
    )
    frames = []
    while (envelope := buffer.dequeue()) is not None:
        if isinstance(envelope.event, VideoDelta):
            frames.append(envelope.event.response_id)
    assert frames == ["video_response"]


def test_video_format_mismatch_closes_session(png_frame: bytes) -> None:
    adapter = ScriptedAdapter(
        reply=[
            ResponseStarted("response"),
            VideoDelta("response", "item", png_frame, "image/png", 0.0, 40.0),
        ]
    )
    capabilities = Capabilities(
        output_modalities=("video",), output_video_format="image/jpeg"
    )
    with build_test_client(adapter, capabilities=capabilities) as client:
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_json()
            send_event(websocket, "session.update", session={})
            receive_until(websocket, "session.updated")
            append_audio(websocket, b"\0" * UNIT_BYTES, 0)
            events = receive_until(websocket, "session.closed")
            assert any(e["type"] == "error" for e in events)
            assert not any(e["type"] == "sglang.response.video.delta" for e in events)
    assert adapter.is_closed


@pytest.mark.parametrize("slice_count,sample_count", [(1, 2667), (2, 11333)])
def test_fractional_video_cadence_cuts_exact_audio_samples(
    slice_count: int, sample_count: int
) -> None:
    """Rounding 17/24 seconds to integer milliseconds used to cut input early."""
    adapter = ScriptedAdapter(
        reply=[
            ResponseStarted("response"),
            ResponseFinished("response", "item", "", False, "completed", "stop"),
        ]
    )
    capabilities = Capabilities(
        input_modalities=("audio", "image"),
        image_frames_per_unit=(4, 17),
        input_samples_per_slice=(2667, 11333),
    )
    pcm = b"\x01\x02" * (sample_count * 2)
    with build_test_client(adapter, capabilities=capabilities) as client:
        with client.websocket_connect("/v1/realtime") as websocket:
            assert websocket.receive_json()["type"] == "session.created"
            send_event(
                websocket,
                "session.update",
                session={"sglang": {"max_slice_nums": slice_count}},
            )
            updated = receive_until(websocket, "session.updated")[-1]
            assert updated["session"]["sglang"]["granted"]["native_unit_ms"] == round(
                sample_count / 16
            )
            for index in range(2):
                response_id = f"response_{index}"
                adapter.reply = [
                    ResponseStarted(response_id),
                    ResponseFinished(
                        response_id, "item", "", False, "completed", "stop"
                    ),
                ]
                append_audio(
                    websocket,
                    pcm[index * sample_count * 2 : (index + 1) * sample_count * 2],
                    index,
                )
                receive_until(websocket, "response.done")
    assert [unit.real_samples for unit in adapter.units] == [sample_count, sample_count]
    assert [unit.start_sample for unit in adapter.units] == [0, sample_count]
    assert b"".join(unit.pcm for unit in adapter.units) == pcm
