from __future__ import annotations

from fastapi.testclient import TestClient

from sglang_omni_router.python.internal_channel import (
    INTERNAL_TOKEN_HEADER,
    InternalChannelState,
    create_internal_app,
)


def _hello(index: int = 0, generation: int = 1, pid: int = 100) -> dict:
    return {"dp_index": index, "generation": generation, "pid": pid}


def _beat(index: int = 0, generation: int = 1, pid: int = 100, seq: int = 0) -> dict:
    return {
        "dp_index": index,
        "generation": generation,
        "pid": pid,
        "last_applied_seq": seq,
    }


def test_register_then_heartbeat_updates_last_applied_seq() -> None:
    app = create_internal_app()
    with TestClient(app) as client:
        assert client.post("/internal/register", json=_hello()).status_code == 200
        response = client.post("/internal/heartbeat", json=_beat(seq=17))
        assert response.status_code == 200

        listed = client.get("/internal/data_planes").json()["data_planes"]
        assert len(listed) == 1
        assert listed[0]["dp_index"] == 0
        assert listed[0]["generation"] == 1
        assert listed[0]["last_applied_seq"] == 17


def test_heartbeat_before_register_asks_for_registration() -> None:
    # Note (Jiaxin Deng): 428, not 409: an unknown dp (e.g. after a CP restart) must re-
    # register, a fence here would make a whole fleet kill itself on a fast CP restart
    app = create_internal_app()
    with TestClient(app) as client:
        response = client.post("/internal/heartbeat", json=_beat())
        assert response.status_code == 428


def test_stale_generation_is_fenced_out() -> None:
    # Note (Jiaxin Deng): After the supervisor restarts a DP slot with a newer
    # generation, calls from the old process must be rejected, never applied.
    app = create_internal_app()
    with TestClient(app) as client:
        assert (
            client.post("/internal/register", json=_hello(generation=2)).status_code
            == 200
        )
        assert (
            client.post("/internal/register", json=_hello(generation=1)).status_code
            == 409
        )
        assert (
            client.post(
                "/internal/heartbeat", json=_beat(generation=1, seq=99)
            ).status_code
            == 409
        )
        listed = client.get("/internal/data_planes").json()["data_planes"]
        assert listed[0]["generation"] == 2
        assert listed[0]["last_applied_seq"] == 0


def test_newer_generation_register_replaces_the_record() -> None:
    state = InternalChannelState()
    app = create_internal_app(state)
    with TestClient(app) as client:
        client.post("/internal/register", json=_hello(generation=1, pid=100))
        client.post("/internal/register", json=_hello(generation=2, pid=200))
        record = state.data_planes[0]
        assert record.generation == 2
        assert record.pid == 200


def test_internal_token_is_required_when_configured() -> None:
    app = create_internal_app(token="secret-token")
    with TestClient(app) as client:
        assert client.post("/internal/register", json=_hello()).status_code == 403
        assert (
            client.post(
                "/internal/register",
                json=_hello(),
                headers={INTERNAL_TOKEN_HEADER: "wrong"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/internal/register",
                json=_hello(),
                headers={INTERNAL_TOKEN_HEADER: "secret-token"},
            ).status_code
            == 200
        )
        assert client.get("/internal/data_planes").status_code == 403


def test_drop_removes_a_data_plane() -> None:
    state = InternalChannelState()
    app = create_internal_app(state)
    with TestClient(app) as client:
        client.post("/internal/register", json=_hello())
        state.drop(0)
        assert client.get("/internal/data_planes").json()["data_planes"] == []


def test_equal_generation_register_from_another_pid_is_fenced() -> None:
    app = create_internal_app()
    with TestClient(app) as client:
        assert (
            client.post(
                "/internal/register", json=_hello(generation=1, pid=100)
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/internal/register", json=_hello(generation=1, pid=999)
            ).status_code
            == 409
        )


def test_same_pid_reregister_is_idempotent_and_keeps_the_ack() -> None:
    app = create_internal_app()
    with TestClient(app) as client:
        client.post("/internal/register", json=_hello())
        beat = {**_beat(seq=17), "last_applied_epoch": "epoch-x", "serving": False}
        client.post("/internal/heartbeat", json=beat)
        # Note (Jiaxin Deng): dropped heartbeat response: same process registers again.
        # Seq must keep its epoch (a seq under a blank epoch cannot satisfy the ACK
        # barrier) and a shedding DP must not flip back to serving-ready.
        assert client.post("/internal/register", json=_hello()).status_code == 200
        listed = client.get("/internal/data_planes").json()["data_planes"]
        assert listed[0]["last_applied_seq"] == 17
        assert listed[0]["last_applied_epoch"] == "epoch-x"
        assert listed[0]["serving"] is False


def test_heartbeat_with_a_future_generation_requires_registration() -> None:
    app = create_internal_app()
    with TestClient(app) as client:
        client.post("/internal/register", json=_hello(generation=1))
        response = client.post("/internal/heartbeat", json=_beat(generation=2))
        assert response.status_code == 428


def test_heartbeat_from_the_wrong_pid_is_fenced() -> None:
    app = create_internal_app()
    with TestClient(app) as client:
        client.post("/internal/register", json=_hello(pid=100))
        assert (
            client.post("/internal/heartbeat", json=_beat(pid=999)).status_code == 409
        )


def test_replayed_heartbeat_cannot_regress_the_ack() -> None:
    app = create_internal_app()
    with TestClient(app) as client:
        client.post("/internal/register", json=_hello())
        client.post("/internal/heartbeat", json=_beat(seq=17))
        assert client.post("/internal/heartbeat", json=_beat(seq=3)).status_code == 200
        listed = client.get("/internal/data_planes").json()["data_planes"]
        assert listed[0]["last_applied_seq"] == 17


def test_identity_fields_have_lower_bounds() -> None:
    app = create_internal_app()
    with TestClient(app) as client:
        assert (
            client.post("/internal/register", json=_hello(generation=0)).status_code
            == 422
        )
        assert client.post("/internal/register", json=_hello(pid=0)).status_code == 422
