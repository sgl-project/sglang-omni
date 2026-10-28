# SPDX-License-Identifier: Apache-2.0
"""Plugin ports over /v1/realtime: advertised only when a Session declares them, forwarded both ways as they are."""

import asyncio
import base64
from dataclasses import replace

import pytest

from sglang_omni.admission import QueueFullError
from sglang_omni.client.client import Client
from sglang_omni.models.minicpm_o.session_adapters import (
    OutputConverter as MiniCPMOOutputConverter,
)
from sglang_omni.proto.session import OutputChunk, SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    IMAGE,
    IMAGE_FRAME,
    PCM16,
    TURN_DONE,
    TURN_INTERRUPTED,
    TURN_STARTED,
    ImagePort,
    PluginPort,
)
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.output import (
    PluginChunk,
    ResponseFinished,
    ResponseStarted,
    TextFinished,
)
from sglang_omni.serve.realtime.schema import JsonObject
from sglang_omni.serve.realtime.session_bridge import capabilities_of
from sglang_omni.serve.realtime.types import (
    Capabilities,
    InteractionAdapter,
    ProtocolError,
    RuntimeLimits,
)
from tests.unit_test.fixtures.realtime_fakes import ScriptedAdapter
from tests.unit_test.fixtures.session_client import PORTS
from tests.unit_test.serve.test_realtime_coordinator_adapter import (
    SESSION_CONFIG,
    RecordingSink,
    SessionCoordinator,
    build_request,
    build_unit,
)
from tests.unit_test.serve.test_realtime_duplex_session import (
    UNIT_BYTES,
    append_audio,
    build_test_client,
    events_of_type,
    open_session,
    receive_until,
    send_event,
)
from tests.unit_test.serve.test_session_bridge import Harness, settle, unit

# A plugin media format and an instant one, as a model package would declare them.
MOTION = PluginPort("motion", "smplx_motion", {"fps": 25})
IMAGE_ASSET = PluginPort(IMAGE, "image_asset")
MOTION_DICT = {"modality": "motion", "format": "smplx_motion", "params": {"fps": 25}}
IMAGE_ASSET_DICT = {"modality": "image", "format": "image_asset", "params": {}}
PLUGIN_PORTS = Capabilities(
    plugin_input_ports=(MOTION,), plugin_output_ports=(MOTION, IMAGE_ASSET)
)


def motion_output(
    identity: SessionIdentity, t_start_ms: float, turn: int | None
) -> OutputChunk:
    return OutputChunk(
        identity,
        0,
        0,
        "motion",
        t_start_ms,
        80.0,
        b"pose",
        "smplx_motion",
        turn=turn,
    )


def test_only_declared_plugin_ports_are_advertised_beside_the_granted_formats():
    assert capabilities_of(PORTS).plugin_ports() == {}
    pasting = replace(PORTS, plugin_input=(IMAGE_ASSET,))
    assert capabilities_of(pasting).input_modalities == ("audio",)
    assert capabilities_of(pasting).plugin_ports() == {
        "plugin_input_ports": [IMAGE_ASSET_DICT],
        "plugin_output_ports": [],
    }
    assert (
        capabilities_of(pasting).to_granted_capabilities()
        == capabilities_of(PORTS).to_granted_capabilities()
    )
    joined = PORTS.with_node(replace(PORTS, plugin_output=(MOTION,))).with_node(
        replace(PORTS, plugin_output=(MOTION, IMAGE_ASSET))
    )
    assert joined.plugin_output == (MOTION, IMAGE_ASSET)


@pytest.mark.asyncio
async def test_plugin_outputs_are_published_as_they_are_within_their_turn():
    h = Harness(replace(PORTS, plugin_output=(MOTION, IMAGE_ASSET)))
    await h.open()
    h.client.outputs.put_nowait(motion_output(h.client.identity, 600.0, 1))
    h.client.emit(IMAGE, "image_asset", {"url": "a.png"}, turn=1)
    h.client.control(TURN_DONE, turn=1)
    h.client.emit(IMAGE, "image_asset", {"url": "b.png"})
    await settle()
    assert h.published == [
        ResponseStarted("resp_0"),
        PluginChunk("resp_0", "motion", "smplx_motion", 600.0, 80.0, 1, b"pose"),
        PluginChunk("resp_0", IMAGE, "image_asset", 0, 0, 1, {"url": "a.png"}),
        TextFinished("resp_0", "item_0", ""),
        ResponseFinished("resp_0", "item_0", "", False, "completed", "stop"),
        PluginChunk(None, IMAGE, "image_asset", 0, 0, None, {"url": "b.png"}),
    ]
    await h.bridge.close()


@pytest.mark.asyncio
async def test_a_plugin_chunk_keeps_its_own_turn_s_response_and_never_reopens_a_finished_one():
    """Avatar frames trail the reply audio by the delivery lead, so they arrive after their turn's turn.done and while the next turn is open."""
    h = Harness(replace(PORTS, plugin_output=(MOTION,)))
    await h.open()
    h.client.control(TURN_STARTED, turn=1)
    h.client.control(TURN_DONE, turn=1)
    h.client.outputs.put_nowait(motion_output(h.client.identity, 0, 1))
    h.client.control(TURN_STARTED, turn=2)
    h.client.outputs.put_nowait(motion_output(h.client.identity, 0, 1))
    h.client.outputs.put_nowait(motion_output(h.client.identity, 0, 3))
    h.client.outputs.put_nowait(motion_output(h.client.identity, 0, 2))
    h.client.control(TURN_INTERRUPTED, turn=2)
    h.client.outputs.put_nowait(motion_output(h.client.identity, 0, 1))
    h.client.outputs.put_nowait(motion_output(h.client.identity, 0, 3))
    await settle()
    chunks = [e for e in h.published if isinstance(e, PluginChunk)]
    assert [(c.turn, c.response_id) for c in chunks] == [
        (1, "resp_0"),
        (1, "resp_0"),
        (3, None),
        (2, "resp_1"),
        (1, None),
        (3, "resp_2"),
    ]
    assert [e.response_id for e in h.published if isinstance(e, ResponseStarted)] == [
        "resp_0",
        "resp_1",
        "resp_2",
    ]
    await h.bridge.close()


@pytest.mark.asyncio
async def test_a_client_chunk_reaches_a_declared_input_port_only():
    h = Harness(
        replace(
            PORTS, input=(*PORTS.input, ImagePort(IMAGE_FRAME)), plugin_input=(MOTION,)
        )
    )
    await h.open()
    await h.bridge.process(unit(0))
    await h.bridge.append_plugin_chunk("motion", "smplx_motion", 120.0, 80.0, b"pose")
    await h.bridge.append_plugin_chunk(
        "motion", "smplx_motion", 200.0, 0, {"pose": [1]}
    )
    assert [
        (c.modality, c.format, c.t_start_ms, c.duration_ms, c.seq, c.payload)
        for c in h.client.appended[1:]
    ] == [
        ("motion", "smplx_motion", 120.0, 80.0, 1, b"pose"),
        ("motion", "smplx_motion", 200.0, 0, 2, {"pose": [1]}),
    ]
    for modality, output_format in ((AUDIO, PCM16), (IMAGE, IMAGE_FRAME), ("x", "y")):
        with pytest.raises(ProtocolError, match="no plugin input port"):
            await h.bridge.append_plugin_chunk(modality, output_format, 0, 0, b"")
    accepting = h.client.append_session
    for error, code in (
        (QueueFullError(), "buffer_overflow"),
        (ValueError("input timing overlaps or moves backwards"), "invalid_request"),
    ):

        async def reject(session_identity, chunk, error=error):
            h.client.append_session = accepting
            raise error

        h.client.append_session = reject
        with pytest.raises(ProtocolError) as rejected:
            await h.bridge.append_plugin_chunk(
                "motion", "smplx_motion", 280.0, 80.0, b"pose"
            )
        assert rejected.value.code == code
    assert len(h.client.appended) == 3
    await h.bridge.close()


@pytest.mark.asyncio
async def test_a_plugin_port_s_chunks_bypass_the_model_converter_as_published_chunks():
    """A gesture node beside a native duplex model publishes smplx_motion; the MiniCPM-o converter keeps seeing only its own chunks."""
    coordinator = SessionCoordinator([])
    reply = {"text": "hi", "pcm": b"\1\2", "end_of_turn": True, "is_speech_cut": False}

    def answer(chunk: TimedChunk) -> None:
        coordinator.emit("motion", "smplx_motion", b"pose", input_seq=chunk.seq)
        coordinator.emit("text", None, reply, input_seq=chunk.seq)

    coordinator.answer = answer
    sink = RecordingSink()
    adapter = CoordinatorAdapter(
        Client(coordinator),
        request_builder=build_request,
        output_converter=MiniCPMOOutputConverter(),
        atomic_consumption=True,
        plugin_output_ports=(MOTION,),
    )
    await adapter.open("sess_1", SESSION_CONFIG, sink)
    unit = build_unit(0)
    await adapter.process(unit)
    # Note (Junnan Li): A downstream node publishes after the unit's input_done, and under the next input's seq once it is appended.
    coordinator.emit("motion", "smplx_motion", b"late", input_seq=0)
    coordinator.emit("motion", "smplx_motion", b"next", input_seq=1)
    await asyncio.sleep(0.01)
    await adapter.close()
    assert sink.published == [
        (PluginChunk(None, "motion", "smplx_motion", 0, 0, None, b"pose"), None),
        *(
            (event, unit)
            for event in MiniCPMOOutputConverter()(
                OutputChunk(coordinator.identity, 0, 0, "text", 0, 0, reply, None)
            )
        ),
        (PluginChunk(None, "motion", "smplx_motion", 0, 0, None, b"late"), None),
        (PluginChunk(None, "motion", "smplx_motion", 0, 0, None, b"next"), None),
    ]


def test_declared_plugin_ports_are_advertised_in_events_and_capabilities() -> None:
    client = build_test_client(ScriptedAdapter(), capabilities=PLUGIN_PORTS)
    with client.websocket_connect("/v1/realtime") as websocket:
        created = websocket.receive_json()
        send_event(websocket, "session.update", session={})
        updated = websocket.receive_json()
    capabilities = client.get("/v1/realtime/capabilities").json()
    ports = {
        "plugin_input_ports": [MOTION_DICT],
        "plugin_output_ports": [MOTION_DICT, IMAGE_ASSET_DICT],
    }
    assert created["session"]["sglang"] == {"granted": None, **ports}
    for advertised in (updated["session"]["sglang"], capabilities):
        assert {key: advertised[key] for key in ports} == ports


def test_a_session_without_plugin_ports_advertises_none() -> None:
    client = build_test_client(ScriptedAdapter())
    with client.websocket_connect("/v1/realtime") as websocket:
        created = websocket.receive_json()
        send_event(websocket, "session.update", session={})
        updated = websocket.receive_json()
    capabilities = client.get("/v1/realtime/capabilities").json()
    assert created["session"]["sglang"] == {"granted": None}
    for advertised in (updated["session"]["sglang"], capabilities):
        assert "plugin_input_ports" not in advertised
        assert "plugin_output_ports" not in advertised


def test_plugin_input_chunks_reach_the_adapter_as_bytes_or_json() -> None:
    adapter = ScriptedAdapter()
    with build_test_client(adapter, capabilities=PLUGIN_PORTS).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        for payload in (
            {"payload_b64": base64.b64encode(b"pose").decode()},
            {"payload": {"pose": [1, 2]}},
        ):
            send_event(
                websocket,
                "sglang.input_chunk.append",
                modality="motion",
                format="smplx_motion",
                t_ms=120.0,
                duration_ms=80.0,
                **payload,
            )
        append_audio(websocket, b"\0" * UNIT_BYTES, 0)
        receive_until(websocket, "sglang.unit.done")
    assert adapter.chunks == [
        ("motion", "smplx_motion", 120.0, 80.0, b"pose"),
        ("motion", "smplx_motion", 120.0, 80.0, {"pose": [1, 2]}),
    ]


def test_plugin_output_chunks_go_out_as_output_chunk_events() -> None:
    adapter = ScriptedAdapter(
        reply=[
            PluginChunk("resp_0", "motion", "smplx_motion", 600.0, 80.0, 1, b"pose"),
            PluginChunk(None, "image", "image_asset", 0.0, 0.0, None, {"url": "a"}),
        ]
    )
    with build_test_client(adapter, capabilities=PLUGIN_PORTS).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        append_audio(websocket, b"\0" * UNIT_BYTES, 0)
        events = receive_until(websocket, "sglang.unit.done")
    published = events_of_type(events, "sglang.output_chunk")
    assert [
        {
            key: value
            for key, value in event.items()
            if key not in ("event_id", "sglang")
        }
        for event in published
    ] == [
        {
            "type": "sglang.output_chunk",
            "response_id": "resp_0",
            "modality": "motion",
            "format": "smplx_motion",
            "t_start_ms": 600.0,
            "duration_ms": 80.0,
            "turn": 1,
            "payload_b64": base64.b64encode(b"pose").decode(),
        },
        {
            "type": "sglang.output_chunk",
            "response_id": None,
            "modality": "image",
            "format": "image_asset",
            "t_start_ms": 0.0,
            "duration_ms": 0.0,
            "turn": None,
            "payload": {"url": "a"},
        },
    ]
    assert published[0]["sglang"]["unit_id"] == "unit_0"


@pytest.mark.parametrize(
    ("fields", "limits", "code"),
    [
        (
            {"payload": {"a": 1}, "payload_b64": "cG9zZQ=="},
            RuntimeLimits(),
            "invalid_request",
        ),
        ({}, RuntimeLimits(), "invalid_request"),
        ({"payload_b64": "!!!!"}, RuntimeLimits(), "invalid_request"),
        (
            {"payload_b64": "cG9zZXBvc2Vw"},
            RuntimeLimits(max_input_bytes=6),
            "buffer_overflow",
        ),
    ],
)
def test_malformed_input_chunks_are_rejected_without_closing(
    fields: JsonObject, limits: RuntimeLimits, code: str
) -> None:
    with build_test_client(
        ScriptedAdapter(), capabilities=PLUGIN_PORTS, limits=limits
    ).websocket_connect("/v1/realtime") as websocket:
        open_session(websocket)
        send_event(
            websocket,
            "sglang.input_chunk.append",
            modality="motion",
            format="smplx_motion",
            t_ms=0.0,
            duration_ms=0.0,
            **fields,
        )
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == code
        assert error["sglang"]["fatal"] is False
        send_event(websocket, "session.update", session={})
        assert websocket.receive_json()["type"] == "session.updated"


class AudioOnlyAdapter(ScriptedAdapter):
    append_plugin_chunk = InteractionAdapter.append_plugin_chunk


def test_an_adapter_without_plugin_ports_refuses_input_chunks() -> None:
    with build_test_client(AudioOnlyAdapter()).websocket_connect(
        "/v1/realtime"
    ) as websocket:
        open_session(websocket)
        send_event(
            websocket,
            "sglang.input_chunk.append",
            modality="motion",
            format="smplx_motion",
            t_ms=0.0,
            duration_ms=0.0,
            payload={"pose": []},
        )
        error = websocket.receive_json()
        assert (error["error"]["code"], error["sglang"]["fatal"]) == (
            "not_supported",
            False,
        )
