# SPDX-License-Identifier: Apache-2.0
"""Per-unit image protocol contracts and audio wire compatibility."""

import base64
import json
from dataclasses import replace

import pytest
from pydantic import ValidationError
from starlette.testclient import WebSocketTestSession

from sglang_omni.serve.realtime.schema import CLIENT_EVENT, JsonObject
from sglang_omni.serve.realtime.types import Capabilities
from tests.unit_test.serve.test_realtime_coordinator_adapter import (
    SESSION_CONFIG,
    RecordingSink,
    SessionCoordinator,
    build_adapter,
    build_unit,
)
from tests.unit_test.serve.test_realtime_duplex_negotiation import FRAMES_BY_SLICES
from tests.unit_test.serve.test_realtime_duplex_session import (
    UNIT_BYTES,
    ScriptedAdapter,
    append_audio,
    build_test_client,
    open_session,
    receive_until,
    send_event,
)

JPEG = b"\xff\xd8frame"
PNG = b"\x89PNGtest"


def image_websocket(
    adapter: ScriptedAdapter,
    input_modalities: tuple[str, ...] = ("audio", "image"),
    **capabilities: int | tuple[int, ...],
) -> WebSocketTestSession:
    return build_test_client(
        adapter,
        capabilities=Capabilities(input_modalities=input_modalities, **capabilities),
    ).websocket_connect("/v1/realtime")


def image_event(t_ms: float = 0.0, image: bytes = JPEG) -> JsonObject:
    return dict(
        type="sglang.input_image.append",
        event_id="frame",
        image=base64.b64encode(image).decode(),
        sglang=dict(t_ms=t_ms),
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"sglang": {}},
        {"sglang": {"t_ms": float("nan")}},
        {"image": b"abc"},
        {"extra": 1},
    ],
)
def test_image_schema_rejects_invalid_event(fields: JsonObject) -> None:
    with pytest.raises(ValidationError):
        CLIENT_EVENT.validate_python({**image_event(), **fields})


def test_frame_binding_missing_frame_and_accounting() -> None:
    adapter = ScriptedAdapter()
    with image_websocket(adapter) as websocket:
        open_session(websocket)
        websocket.send_json(image_event(39.999))
        accepted = websocket.receive_json()
        assert accepted["type"] == "sglang.input_image.accepted"
        assert accepted["unit_id"] == "unit_1"
        assert accepted["client_event_id"] == "frame"
        append_audio(websocket, b"\0" * UNIT_BYTES * 2, 0)
        receive_until(websocket, "sglang.unit.done")
        receive_until(websocket, "sglang.unit.done")
        send_event(websocket, "sglang.input_audio.end")
        drained = receive_until(websocket, "sglang.input_audio.drained")[-1]
    assert [unit.images for unit in adapter.units] == [(), (JPEG,), ()]
    assert [
        drained[key]
        for key in ("accepted_end_ms", "consumed_ms", "discarded_ms", "padding_ms")
    ] == [40.0, 40.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("scenario", "fields", "code"),
    [
        ("late", {}, "invalid_state"),
        ("ended", {}, "invalid_state"),
        ("unsupported", {}, "not_supported"),
        ("lookahead", {"sglang": {"t_ms": 60.0}}, "buffer_overflow"),
        ("too_large", {"image": "!" * 13}, "buffer_overflow"),
        ("not_base64", {"image": "!!!!"}, "invalid_request"),
        ("gif", {"image": base64.b64encode(b"GIF89a").decode()}, "invalid_request"),
    ],
)
def test_frame_rejections_are_nonfatal(
    scenario: str, fields: JsonObject, code: str
) -> None:
    modalities = ("audio",) if scenario == "unsupported" else ("audio", "image")
    with image_websocket(ScriptedAdapter(), modalities, max_image_bytes=8) as websocket:
        open_session(websocket)
        if scenario == "late":
            append_audio(websocket, b"\0" * UNIT_BYTES, 0)
            receive_until(websocket, "sglang.unit.done")
        elif scenario == "ended":
            send_event(websocket, "sglang.input_audio.end")
            receive_until(websocket, "sglang.input_audio.drained")
        else:
            pass
        websocket.send_json({**image_event(), **fields})
        error = websocket.receive_json()
        assert (error["error"]["code"], error["error"]["event_id"]) == (code, "frame")
        assert error["sglang"]["fatal"] is False
        send_event(websocket, "session.update", session={})
        assert websocket.receive_json()["type"] == "session.updated"


@pytest.mark.parametrize("modalities", [("audio",), ("audio", "image")])
def test_image_grant_follows_capabilities(modalities: tuple[str, ...]) -> None:
    client = build_test_client(
        ScriptedAdapter(),
        capabilities=Capabilities(input_modalities=modalities, max_image_bytes=100),
    )
    advertised = client.get("/v1/realtime/capabilities").json()
    with client.websocket_connect("/v1/realtime") as websocket:
        events = [websocket.receive_json()]
        send_event(websocket, "session.update", session={})
        events.append(websocket.receive_json())
        append_audio(websocket, b"\0" * UNIT_BYTES, 0)
        events.extend(receive_until(websocket, "sglang.unit.done"))
    granted = events[1]["session"]["sglang"]["granted"]
    assert granted["input_modalities"] == list(modalities)
    if "image" in modalities:
        image_format = {
            "types": ["image/jpeg", "image/png"],
            "max_bytes": 100,
            "max_frames_per_unit": 1,
            "max_slice_nums": 1,
        }
        assert advertised["input_image_format"] == image_format
        assert granted["input_image_format"] == image_format
    else:
        assert "image" not in json.dumps(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("image", [None, JPEG])
async def test_adapter_payload_preserves_audio_and_bundles_image(
    image: bytes | None,
) -> None:
    coordinator = SessionCoordinator([])
    adapter = build_adapter(coordinator)
    await adapter.open("session", SESSION_CONFIG, RecordingSink())
    unit = replace(build_unit(2, is_eos=True), images=() if image is None else (image,))
    assert await adapter.process(unit) == unit.real_samples
    await adapter.close()
    chunk = coordinator.appended[0]
    assert chunk.payload == (
        unit.pcm if image is None else {"pcm": unit.pcm, "images": [image]}
    )
    assert (chunk.modality, chunk.format, chunk.seq, chunk.eos) == (
        "audio",
        "pcm16",
        2,
        True,
    )


def test_clear_drops_frames_and_moves_frame_origin() -> None:
    adapter = ScriptedAdapter()
    with image_websocket(adapter) as websocket:
        open_session(websocket)
        websocket.send_json(image_event(0.0, JPEG))
        assert websocket.receive_json()["unit_id"] == "unit_0"
        for sequence in range(10):
            append_audio(websocket, b"\0" * (UNIT_BYTES // 2), sequence)
            send_event(websocket, "input_audio_buffer.clear")
            cleared = receive_until(websocket, "input_audio_buffer.cleared")[-1]
            assert cleared["sglang"]["discarded_ms"] == 10.0
        websocket.send_json(image_event(80.0, JPEG))
        assert websocket.receive_json()["error"]["code"] == "invalid_state"
        websocket.send_json(image_event(160.0, JPEG))
        assert websocket.receive_json()["error"]["code"] == "buffer_overflow"
        websocket.send_json(image_event(140.0, JPEG))
        assert websocket.receive_json()["unit_id"] == "unit_2"
        websocket.send_json(image_event(100.0, PNG))
        assert websocket.receive_json()["unit_id"] == "unit_0"
        append_audio(websocket, b"\0" * UNIT_BYTES, 10)
        receive_until(websocket, "sglang.unit.done")
    assert adapter.units[0].images == (PNG,)


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        ([(15.0, PNG), (5.0, JPEG)], (JPEG, PNG)),
        ([(0.0, PNG), (0.0, JPEG)], (PNG, JPEG)),
    ],
)
def test_unit_frames_follow_slice_grant_and_media_time(
    frames: list[tuple[float, bytes]], expected: tuple[bytes, ...]
) -> None:
    adapter = ScriptedAdapter()
    with image_websocket(adapter, image_frames_per_unit=FRAMES_BY_SLICES) as websocket:
        websocket.receive_json()
        send_event(
            websocket, "session.update", session={"sglang": {"max_slice_nums": 4}}
        )
        websocket.receive_json()
        for t_ms, image in frames:
            websocket.send_json(image_event(t_ms, image))
            assert websocket.receive_json()["type"] == "sglang.input_image.accepted"
        websocket.send_json(image_event(10.0, JPEG))
        assert websocket.receive_json()["error"]["code"] == "buffer_overflow"
        append_audio(websocket, b"\0" * UNIT_BYTES, 0)
        receive_until(websocket, "sglang.unit.done")
    assert adapter.units[0].images == expected
