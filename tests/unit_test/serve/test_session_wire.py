# SPDX-License-Identifier: Apache-2.0
"""The Session protocol over /v1/session: msgpack frames over a WebSocket."""


import msgpack
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sglang_omni.admission import QueueFullError
from sglang_omni.proto.session import SessionLimits, TimedChunk
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.types import Capabilities
from sglang_omni.serve.session_wire import register_session_wire
from tests.unit_test.fixtures.session_client import FakeSessionClient


class FakeClient(FakeSessionClient):
    """Hears every append as a partial transcript and refuses the seqs in reject_once once each."""

    def __init__(self, *, reject_once: frozenset[int] = frozenset()) -> None:
        super().__init__()
        self.reject_once = set(reject_once)

    async def append_session(self, session_identity, chunk):
        if chunk.seq in self.reject_once:
            self.reject_once.discard(chunk.seq)
            raise QueueFullError()
        else:
            return await super().append_session(session_identity, chunk)

    def answer(self, chunk):
        self.emit(
            "transcript",
            "transcript",
            {"text": f"heard {chunk.seq}", "is_final": False},
            input_seq=chunk.seq,
            turn=chunk.turn,
        )


def connect(client, limits=SessionLimits()):
    app = FastAPI()
    app.state.client = client
    register_session_wire(app, limits=limits)
    return TestClient(app).websocket_connect("/v1/session")


def send(websocket, message):
    websocket.send_bytes(msgpack.packb(message, use_bin_type=True))


def receive(websocket):
    return msgpack.unpackb(websocket.receive_bytes(), raw=False)


def chunk(seq, *, eos=False, turn=None):
    return TimedChunk(
        "audio", seq * 100.0, 100.0, seq, bytes([seq]) * 4, "pcm16", eos, turn
    ).to_dict()


def open_session(websocket, **fields):
    send(websocket, {"op": "open", "params": {"a": 1}, **fields})
    return receive(websocket)


def test_open_append_outputs_close_round_trip():
    client = FakeClient()
    with connect(client) as websocket:
        opened = open_session(websocket)
        assert opened == {
            "op": "opened",
            "session": {"id": "session", "open_index": 3},
        }
        send(websocket, {"op": "append", "chunk": chunk(0, turn=4)})
        frames = [receive(websocket) for _ in range(3)]
        assert {"op": "accepted", "seq": 0} in frames
        outputs = [frame["chunk"] for frame in frames if frame["op"] == "output"]
        assert [output["kind"] for output in outputs] == ["data", "input_done"]
        assert outputs[0]["payload"] == {"text": "heard 0", "is_final": False}
        assert outputs[0]["turn"] == 4
        assert outputs[0]["session_identity"] == {"id": "session", "open_index": 3}
        send(websocket, {"op": "close"})
        assert receive(websocket) == {"op": "closed"}
    request, limits, _ = client.opened[0]
    assert request.params == {"a": 1}
    assert limits == SessionLimits()
    assert client.appended[0].payload == bytes([0]) * 4
    assert client.appended[0].turn == 4
    assert client.closed.is_set()


def test_rejected_append_keeps_its_seq_for_a_resend():
    client = FakeClient(reject_once={1})
    with connect(client) as websocket:
        open_session(websocket)
        send(websocket, {"op": "append", "chunk": chunk(0)})
        assert {"op": "accepted", "seq": 0} in [receive(websocket) for _ in range(3)]
        send(websocket, {"op": "append", "chunk": chunk(1)})
        rejected = receive(websocket)
        assert rejected == {
            "op": "rejected",
            "seq": 1,
            "reason": QueueFullError.MESSAGE,
        }
        send(websocket, {"op": "append", "chunk": chunk(1, eos=True)})
        assert {"op": "accepted", "seq": 1} in [receive(websocket) for _ in range(3)]
    assert [appended.seq for appended in client.appended] == [0, 1]


def test_output_reader_failure_sends_error_and_closes():
    client = FakeClient()
    with connect(client) as websocket:
        open_session(websocket)
        client.fail(RuntimeError("stage crashed"))
        assert receive(websocket) == {"op": "error", "message": "stage crashed"}
        with pytest.raises(WebSocketDisconnect):
            receive(websocket)
    assert client.closed.is_set()


def test_second_open_is_a_protocol_error():
    client = FakeClient()
    with connect(client) as websocket:
        open_session(websocket)
        send(websocket, {"op": "open"})
        assert receive(websocket) == {
            "op": "error",
            "message": "the session is already open",
        }
    assert client.closed.is_set()


def test_first_frame_must_open():
    client = FakeClient()
    with connect(client) as websocket:
        send(websocket, {"op": "append", "chunk": chunk(0)})
        assert receive(websocket)["op"] == "error"
    assert client.opened == []


def test_client_disconnect_closes_the_session():
    client = FakeClient()
    with connect(client) as websocket:
        open_session(websocket)
        websocket.close()
        assert client.closed.wait(timeout=5)


def test_control_reaches_the_session_and_is_answered():
    client = FakeClient()
    event = TimedChunk("control", 0.0, 0.0, 0, {"type": "interrupt"}, "session", turn=2)
    with connect(client) as websocket:
        open_session(websocket)
        send(
            websocket,
            {
                "op": "control",
                "chunk": event.to_dict(),
                "stages": ["sink"],
                "should_preempt": True,
            },
        )
        assert receive(websocket) == {"op": "controlled", "seq": 0}
        send(websocket, {"op": "control", "chunk": event.to_dict()})
        assert receive(websocket) == {"op": "controlled", "seq": 0}
        send(websocket, {"op": "close"})
        assert receive(websocket) == {"op": "closed"}
    assert client.controlled == [(event, ["sink"], True), (event, None, False)]


def test_a_client_may_not_request_session_limits():
    client = FakeClient()
    with connect(client) as websocket:
        frame = open_session(websocket, limits={"idle_timeout_s": 5.0})
        assert frame["op"] == "error" and "params only" in frame["message"]
    assert client.opened == []


def test_sessions_open_with_the_deployment_limits():
    client = FakeClient()
    with connect(client, SessionLimits(idle_timeout_s=5.0)) as websocket:
        open_session(websocket)
    _, limits, _ = client.opened[0]
    assert limits == SessionLimits(idle_timeout_s=5.0)


def test_the_app_opens_sessions_with_the_realtime_deployment_session_limits():
    client = FakeClient()
    deployment = RealtimeDeployment(
        Capabilities(),
        adapter_factory=lambda: None,
        session_limits=SessionLimits(idle_timeout_s=5.0),
    )
    app = create_app(client, model_name="test", realtime_deployment=deployment)
    with TestClient(app).websocket_connect("/v1/session") as websocket:
        open_session(websocket)
    _, limits, _ = client.opened[0]
    assert limits == SessionLimits(idle_timeout_s=5.0)
