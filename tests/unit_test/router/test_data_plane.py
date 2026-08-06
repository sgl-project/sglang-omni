from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from sglang_omni_router.config import RouterConfig, WorkerConfig
from sglang_omni_router.data_plane import (
    _MAX_FORWARD_BODY_BYTES,
    DataPlaneWorkerView,
    create_data_plane_app,
)
from sglang_omni_router.snapshot import SnapshotWorker, SnapshotWriter


def _config(failure_threshold: int = 1) -> RouterConfig:
    return RouterConfig(
        workers=[WorkerConfig(url="http://worker-a:8101")],
        health_failure_threshold=failure_threshold,
        health_success_threshold=1,
    )


def _entry(url: str = "http://worker-a:8101", **kwargs) -> SnapshotWorker:
    return SnapshotWorker(url=url, worker_id=url.replace("://", "%3A%2F%2F"), **kwargs)


def _snapshot(writer: SnapshotWriter, *entries: SnapshotWorker):
    return writer.publish(list(entries))


class _Recorder:
    def __init__(self, status_for: dict[str, int] | None = None) -> None:
        self.requests: list[tuple[str, bytes]] = []
        self.status_for = status_for or {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.path, request.content))
        status = self.status_for.get(request.url.path, 200)
        return httpx.Response(status, json={"ok": status == 200})


def _dp_app(
    tmp_path: Path,
    upstream: _Recorder,
    internal: _Recorder | None = None,
    failure_threshold: int = 1,
    **kwargs,
):
    snapshot_path = str(tmp_path / "workers.json")
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    internal_client = None
    if internal is not None:
        internal_client = httpx.AsyncClient(
            transport=httpx.MockTransport(internal.handler), base_url="http://cp"
        )
    kwargs.setdefault("dp_refresh_interval_secs", 0.02)
    kwargs.setdefault("heartbeat_interval_secs", 0.02)
    app = create_data_plane_app(
        _config(failure_threshold),
        snapshot_path=snapshot_path,
        dp_index=0,
        generation=1,
        client=client,
        internal_client=internal_client,
        metadata_client=httpx.AsyncClient(
            transport=httpx.MockTransport(upstream.handler)
        ),
        **kwargs,
    )
    return app, snapshot_path


def _wait_for(predicate, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def test_forwarded_admin_routes_publish_the_single_process_schema_identity(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): changing only --router-processes must not break generated
    # clients: the forwarded admin routes keep the single-process operation ids, the
    # worker_id path parameter name, and the Authorization parameter/security shape
    from sglang_omni_router.app import create_app

    single = create_app(
        _config(),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        ),
        admin_api_key="parity-key",
    )
    dp = _dp_app_forwarding_to_cp(tmp_path, _CPRecorder(), admin_api_key="parity-key")

    def _ops(app):
        spec = app.openapi()
        return {
            (path, method): {
                "operationId": op.get("operationId"),
                "parameters": op.get("parameters"),
                "security": op.get("security"),
            }
            for path, item in spec["paths"].items()
            for method, op in item.items()
        }

    single_ops, dp_ops = _ops(single), _ops(dp)
    shared = set(single_ops) & set(dp_ops)
    assert ("/workers/{worker_id}", "get") in shared  # not {worker_path}
    for key in shared:
        assert dp_ops[key] == single_ops[key], key
    authed = [
        params
        for (path, _), op in single_ops.items()
        if path == "/update_weights_from_disk"
        for params in [op["parameters"] or []]
    ]
    assert any(
        param.get("name") == "authorization" for params in authed for param in params
    )

    def _duplicates(ops):
        ids = [op["operationId"] for op in ops.values() if op["operationId"]]
        return {op for op in ids if ids.count(op) > 1}

    # Note (Jiaxin Deng): the split must not introduce a collision the single-process
    # schema does not already have (/weights_checker shares one id in both, pre-
    # existing)
    assert _duplicates(dp_ops) <= _duplicates(single_ops)


def test_replaced_incarnation_stays_reportable_until_its_request_drains() -> None:
    # Note (Jiaxin Deng): DELETE then re-add of the same URL replaces the Worker object;
    # a request still running on the old one records its completion there, so the
    # retired object must keep reaching the CP ledger until it drains
    view = DataPlaneWorkerView()

    class _First:
        seq = 1
        workers = [_entry(incarnation="inc-a")]

    view.apply(_First())
    old = view.workers()[0]
    old.increment_active()  # a request is in flight on incarnation A

    class _Second:
        seq = 2
        workers = [_entry(incarnation="inc-b")]

    view.apply(_Second())
    new = view.workers()[0]
    assert new is not old  # fresh object for the new incarnation
    assert view.workers() == [new]  # routing only ever sees the live one
    assert old in view.reportable_workers()  # but counters still flow
    assert view.drained_retired() == []  # not drained: a request is running

    old.decrement_active()
    old.record_routed_request(status_code=200)
    drained = view.drained_retired()
    assert drained == [old]
    assert old in view.reportable_workers()  # reported once more, then released
    view.release_retired(drained)
    assert view.reportable_workers() == [new]


def test_view_apply_preserves_counters_and_tracks_membership() -> None:
    view = DataPlaneWorkerView()

    class _Snapshot:
        seq = 1
        workers = [_entry(), _entry("http://worker-b:8102")]

    view.apply(_Snapshot())
    assert {w.url for w in view.workers()} == {
        "http://worker-a:8101",
        "http://worker-b:8102",
    }
    worker_a = next(w for w in view.workers() if w.url.endswith("8101"))
    worker_a.increment_active()
    worker_a.record_routed_request(status_code=200)

    class _Second:
        seq = 2
        workers = [_entry(disabled=True)]

    view.apply(_Second())
    assert view.last_applied_seq == 2
    assert [w.url for w in view.workers()] == ["http://worker-a:8101"]
    survivor = view.workers()[0]
    assert survivor is worker_a  # same object: counters survived
    assert survivor.active_requests == 1
    assert survivor.routed_requests == 1
    assert survivor.disabled is True
    assert survivor.is_routable is False


def test_dp_sheds_until_the_first_snapshot_arrives(tmp_path: Path) -> None:
    upstream = _Recorder()
    app, _ = _dp_app(tmp_path, upstream)
    with TestClient(app) as client:
        assert client.get("/live").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["reason"] == "snapshot_stale"
        response = client.post("/generate", json={"prompt": "x"})
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "unavailable_error"
        assert upstream.requests == []


def test_dp_routes_after_snapshot_and_tracks_disabled_workers(
    tmp_path: Path,
) -> None:
    upstream = _Recorder()
    app, snapshot_path = _dp_app(tmp_path, upstream)
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as client:
        _snapshot(writer, _entry())
        _wait_for(lambda: client.get("/ready").status_code == 200)
        assert client.post("/generate", json={"prompt": "x"}).status_code == 200
        assert upstream.requests[-1][0] == "/generate"

        _snapshot(writer, _entry(disabled=True))
        _wait_for(lambda: client.post("/generate", json={}).status_code == 503)
        response = client.post("/generate", json={})
        assert response.json()["error"]["message"] == "no eligible upstream"


def test_dp_sheds_when_the_snapshot_goes_stale_and_recovers(
    tmp_path: Path,
) -> None:
    upstream = _Recorder()
    app, snapshot_path = _dp_app(tmp_path, upstream, snapshot_max_age_secs=0.15)
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as client:
        _snapshot(writer, _entry())
        _wait_for(lambda: client.get("/ready").status_code == 200)

        # Note (Jiaxin Deng): CP goes silent: past max age new requests are shed
        _wait_for(lambda: client.post("/generate", json={}).status_code == 503)
        assert client.get("/ready").json()["reason"] == "snapshot_stale"

        # Note (Jiaxin Deng): CP comes back: one republish restores service
        _snapshot(writer, _entry())
        _wait_for(lambda: client.post("/generate", json={}).status_code == 200)


def test_dp_reports_eviction_relevant_failures_to_the_cp(tmp_path: Path) -> None:
    upstream = _Recorder(status_for={"/generate": 502})
    internal = _Recorder()
    app, snapshot_path = _dp_app(
        tmp_path, upstream, internal=internal, failure_threshold=3
    )
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as client:
        _snapshot(writer, _entry())
        _wait_for(lambda: client.get("/ready").status_code == 200)

        assert client.post("/generate", json={}).status_code == 502
        assert client.post("/generate", json={}).status_code == 502

        _wait_for(
            lambda: any(
                path == "/internal/worker_failure" for path, _ in internal.requests
            )
        )

        def _reports():
            return [
                body
                for path, body in internal.requests
                if path == "/internal/worker_failure"
            ]

        # Note (Jiaxin Deng): every distinct failure is reported: the CP threshold
        # counts events
        reports = _wait_for(lambda: _reports() if len(_reports()) >= 2 else None)
        assert len(reports) == 2
        assert all(
            b'"status_code": 502' in r or b'"status_code":502' in r for r in reports
        )


def test_dp_leaves_the_failure_verdict_to_the_cp(tmp_path: Path) -> None:
    # Note (Jiaxin Deng): the CP owns the health verdict: a snapshot carries state but
    # never resets a DP-local consecutive_failures, so a DP that counted failures too
    # would evict a worker the CP still counts as healthy
    upstream = _Recorder(status_for={"/generate": 502})
    internal = _Recorder()
    app, snapshot_path = _dp_app(
        tmp_path, upstream, internal=internal, failure_threshold=3
    )
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as client:
        view = app.state.worker_view
        _snapshot(writer, _entry())
        _wait_for(lambda: client.get("/ready").status_code == 200)

        for _ in range(2):
            assert client.post("/generate", json={}).status_code == 502

        # Note (Jiaxin Deng): the CP's own health probe succeeded: it reset its counter
        # and republished the worker healthy
        seq = _snapshot(writer, _entry(state="healthy")).seq
        _wait_for(lambda: view.last_applied_seq == seq)

        assert client.post("/generate", json={}).status_code == 502
        assert view.workers()[0].consecutive_failures == 0

        # Note (Jiaxin Deng): the CP has counted one failure of three, so this worker is
        # still routable; a DP-local count would have evicted it here
        upstream.status_for = {}
        assert client.post("/generate", json={}).status_code == 200


def test_dp_heartbeats_the_applied_seq_and_reacts_to_fencing(
    tmp_path: Path,
) -> None:
    upstream = _Recorder()

    class _FencingInternal(_Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.fence_after = 3

        def handler(self, request: httpx.Request) -> httpx.Response:
            self.requests.append((request.url.path, request.content))
            if len(self.requests) > self.fence_after:
                return httpx.Response(409, json={"detail": "stale generation"})
            return httpx.Response(200, json={"status": "ok"})

    internal = _FencingInternal()
    fenced: list[bool] = []
    app, snapshot_path = _dp_app(
        tmp_path, upstream, internal=internal, on_fenced=lambda: fenced.append(True)
    )
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as client:
        published = _snapshot(writer, _entry())
        _wait_for(lambda: client.get("/ready").status_code == 200)
        _wait_for(
            lambda: any(
                path == "/internal/heartbeat" and str(published.seq).encode() in body
                for path, body in internal.requests
            )
        )
        _wait_for(lambda: fenced)
    assert fenced == [True]


def test_a_snapshot_apply_wakes_the_heartbeat_without_a_full_period(
    tmp_path: Path,
) -> None:
    import json as jsonlib

    # Note (Jiaxin Deng): the weight-update ACK barrier learns last_applied_seq
    # only from heartbeats; an apply must reach the CP promptly even when the
    # next periodic beat is a full interval away
    upstream = _Recorder()
    internal = _Recorder()
    app, snapshot_path = _dp_app(
        tmp_path,
        upstream,
        internal=internal,
        heartbeat_interval_secs=600,
    )
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as client:
        _snapshot(writer, _entry())
        _wait_for(lambda: client.get("/ready").status_code == 200)
        published = _snapshot(writer, _entry(), _entry(url="http://worker-b:8102"))

        def _acked() -> bool:
            for path, body in internal.requests:
                if path != "/internal/heartbeat":
                    continue
                if jsonlib.loads(body).get("last_applied_seq") == published.seq:
                    return True
            return False

        _wait_for(_acked)


def test_cp_keepalive_republishes_without_state_changes(tmp_path: Path) -> None:
    from sglang_omni_router.control_plane import create_control_plane_app
    from sglang_omni_router.snapshot import SnapshotReader

    snapshot_path = str(tmp_path / "cp.json")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"status": "healthy"})
        )
    )
    app = create_control_plane_app(
        _config(),
        snapshot_path=snapshot_path,
        cp_epoch="e",
        client=client,
        snapshot_keepalive_secs=0.05,
    )
    with TestClient(app):
        reader = SnapshotReader(snapshot_path)
        _wait_for(lambda: reader.maybe_reload())
        first_seq = reader.snapshot.seq
        _wait_for(lambda: reader.maybe_reload() and reader.snapshot.seq > first_seq)


class _CPRecorder:
    """Records requests the DP forwards to the CP."""

    def __init__(self, status: int = 418) -> None:
        self.requests: list[httpx.Request] = []
        self._status = status

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            self._status, json={"from": "cp"}, headers={"x-cp-extra": "yes"}
        )


def _dp_app_forwarding_to_cp(tmp_path: Path, cp: _CPRecorder, **kwargs):
    from sglang_omni_router.data_plane import create_data_plane_app

    upstream = _Recorder()
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    internal_client = httpx.AsyncClient(
        transport=httpx.MockTransport(cp.handler), base_url="http://cp"
    )
    # NOTE: no snapshot exists, so data routes are shedding; admin and health
    # forwarding must be exempt from that gate and from admission.
    return create_data_plane_app(
        _config(),
        snapshot_path=str(tmp_path / "w2.json"),
        dp_index=0,
        generation=1,
        client=client,
        internal_client=internal_client,
        dp_refresh_interval_secs=0.02,
        heartbeat_interval_secs=600,  # keep heartbeats out of cp.requests
        **kwargs,
    )


def test_dp_forwards_admin_and_health_routes_to_the_cp_with_fidelity(
    tmp_path: Path,
) -> None:
    import json as jsonlib

    cp = _CPRecorder()
    app = _dp_app_forwarding_to_cp(tmp_path, cp)
    with TestClient(app) as tc:
        response = tc.post(
            "/workers?url=http%3A%2F%2Fw%3A1",
            json={"model": "m"},
            headers={"Authorization": "Bearer k"},
        )
        assert response.status_code == 418
        assert response.json() == {"from": "cp"}
        assert response.headers["x-cp-extra"] == "yes"
        forwarded = cp.requests[-1]
        assert forwarded.method == "POST"
        assert forwarded.url.path == "/workers"
        assert b"url=" in forwarded.url.query
        assert forwarded.headers["authorization"] == "Bearer k"
        assert jsonlib.loads(forwarded.content) == {"model": "m"}

        assert tc.get("/health").status_code == 418
        assert cp.requests[-1].url.path == "/health"
        assert cp.requests[-1].method == "GET"

        assert tc.delete("/workers/some%2Fid").status_code == 418
        # Note (Jiaxin Deng): raw_path is the wire form; percent-encoding must survive
        # the hop
        assert cp.requests[-1].url.raw_path == b"/workers/some%2Fid"


def test_dp_strips_hop_by_hop_headers_when_forwarding_to_the_cp(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): the DP re-frames the body as fixed-length bytes, so a client
    # Transfer-Encoding must not ride along (smuggling ambiguity at the CP)
    cp = _CPRecorder(status=200)
    app = _dp_app_forwarding_to_cp(tmp_path, cp)
    with TestClient(app) as tc:
        response = tc.post(
            "/workers?url=http%3A%2F%2Fw%3A1",
            json={"model": "m"},
            headers={
                "Authorization": "Bearer k",
                "Transfer-Encoding": "chunked",
                "TE": "trailers",
                "Upgrade": "h2c",
                "Proxy-Connection": "keep-alive",
            },
        )
        assert response.status_code == 200
        forwarded = cp.requests[-1]
        forwarded_keys = {key.lower() for key in forwarded.headers}
        assert "transfer-encoding" not in forwarded_keys
        assert "te" not in forwarded_keys
        assert "upgrade" not in forwarded_keys
        assert "proxy-connection" not in forwarded_keys
        # Note (Jiaxin Deng): a non-hop-by-hop header (the admin key the CP re-checks)
        # is preserved
        assert forwarded.headers["authorization"] == "Bearer k"


def test_dp_serves_v1_models_locally_from_the_snapshot_view(
    tmp_path: Path,
) -> None:
    class _ModelsUpstream(_Recorder):
        def handler(self, request: httpx.Request) -> httpx.Response:
            self.requests.append((request.url.path, request.content))
            if request.url.path == "/v1/models":
                return httpx.Response(
                    200,
                    json={
                        "object": "list",
                        "data": [{"id": "higgs", "object": "model"}],
                    },
                )
            return httpx.Response(200, json={"ok": True})

    upstream = _ModelsUpstream()
    app, snapshot_path = _dp_app(tmp_path, upstream)
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as client:
        _snapshot(writer, _entry())
        _wait_for(lambda: client.get("/ready").status_code == 200)
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert any(item["id"] == "higgs" for item in response.json()["data"])
        assert any(path == "/v1/models" for path, _ in upstream.requests)


def test_dp_flushes_cumulative_counters_to_the_cp(tmp_path: Path) -> None:
    import json as jsonlib

    upstream = _Recorder()
    internal = _Recorder()
    app, snapshot_path = _dp_app(
        tmp_path, upstream, internal=internal, counter_flush_interval_secs=0.03
    )
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as client:
        _snapshot(writer, _entry())
        _wait_for(lambda: client.get("/ready").status_code == 200)
        assert client.post("/generate", json={}).status_code == 200
        assert client.post("/generate", json={}).status_code == 200

        def _latest_counters():
            reports = [
                jsonlib.loads(body)
                for path, body in internal.requests
                if path == "/internal/counters"
            ]
            for report in reversed(reports):
                if report["workers"] and report["workers"][0]["routed_total"] >= 2:
                    return report
            return None

        report = _wait_for(_latest_counters)
        assert report["dp_index"] == 0
        assert report["generation"] == 1
        assert report["counter_seq"] >= 1
        assert report["workers"][0]["routed_total"] == 2
        assert report["workers"][0]["successful_total"] == 2
        assert report["workers"][0]["current_active"] == 0


def test_failure_reports_retry_with_a_bound_then_give_up(tmp_path: Path) -> None:
    upstream = _Recorder(status_for={"/generate": 502})

    class _FlakyInternal(_Recorder):
        def __init__(self, fail_first: int) -> None:
            super().__init__()
            self.fail_first = fail_first
            self.attempts = 0

        def handler(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/internal/worker_failure":
                self.attempts += 1
                if self.attempts <= self.fail_first:
                    raise httpx.ConnectError("cp down", request=request)
            self.requests.append((request.url.path, request.content))
            return httpx.Response(200, json={"status": "ok"})

    internal = _FlakyInternal(fail_first=2)
    app, snapshot_path = _dp_app(
        tmp_path,
        upstream,
        internal=internal,
        failure_threshold=3,
        counter_flush_interval_secs=600,
        heartbeat_interval_secs=600,
    )
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as client:
        _snapshot(writer, _entry())
        _wait_for(lambda: client.get("/ready").status_code == 200)
        assert client.post("/generate", json={}).status_code == 502
        # Note (Jiaxin Deng): two transport failures, then the bounded third attempt
        # lands
        _wait_for(
            lambda: any(
                path == "/internal/worker_failure" for path, _ in internal.requests
            )
        )
        assert internal.attempts == 3


def test_dp_reregisters_on_428_instead_of_fencing(tmp_path: Path) -> None:
    upstream = _Recorder()

    class _RestartedCP(_Recorder):
        """Heartbeats get 428 until a register arrives (CP lost its state)."""

        def __init__(self) -> None:
            super().__init__()
            self.registered_pids: set[int] = set()

        def handler(self, request: httpx.Request) -> httpx.Response:
            import json as jsonlib

            self.requests.append((request.url.path, request.content))
            if request.url.path == "/internal/register":
                self.registered_pids.add(jsonlib.loads(request.content)["pid"])
                return httpx.Response(200, json={"status": "registered"})
            if request.url.path == "/internal/heartbeat":
                pid = jsonlib.loads(request.content)["pid"]
                if pid not in self.registered_pids:
                    return httpx.Response(428, json={"detail": "register first"})
                # Note (Jiaxin Deng): simulate the CP forgetting us once, after the
                # first beat
                self.registered_pids.discard(pid)
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(200, json={"status": "ok"})

    internal = _RestartedCP()
    fenced: list[bool] = []
    app, snapshot_path = _dp_app(
        tmp_path, upstream, internal=internal, on_fenced=lambda: fenced.append(True)
    )
    with TestClient(app):
        # Note (Jiaxin Deng): register -> heartbeat(200, then forgotten) ->
        # heartbeat(428) -> register again: at least two registrations, and never fenced
        _wait_for(
            lambda: sum(
                1 for path, _ in internal.requests if path == "/internal/register"
            )
            >= 2
        )
    assert fenced == []


def test_dp_fails_closed_after_persistent_registration_rejection(
    tmp_path: Path,
) -> None:
    from sglang_omni_router import data_plane as dp_module

    upstream = _Recorder()
    internal = _Recorder(status_for={"/internal/register": 403})
    fenced: list[bool] = []
    app, snapshot_path = _dp_app(
        tmp_path, upstream, internal=internal, on_fenced=lambda: fenced.append(True)
    )
    with TestClient(app):
        _wait_for(lambda: fenced)
    rejects = [path for path, _ in internal.requests if path == "/internal/register"]
    assert len(rejects) == dp_module._REGISTER_REJECT_LIMIT
    assert fenced == [True]


def test_forwarded_admin_bodies_are_bounded_before_buffering(
    tmp_path: Path,
) -> None:
    upstream = _Recorder()
    internal = _Recorder()
    snapshot_path = str(tmp_path / "w3.json")
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    internal_client = httpx.AsyncClient(
        transport=httpx.MockTransport(internal.handler), base_url="http://cp"
    )
    config = RouterConfig(
        workers=[WorkerConfig(url="http://worker-a:8101")],
        max_payload_size=1024,
    )
    app = create_data_plane_app(
        config,
        snapshot_path=snapshot_path,
        dp_index=0,
        generation=1,
        client=client,
        internal_client=internal_client,
        heartbeat_interval_secs=600,
        counter_flush_interval_secs=600,
    )
    with TestClient(app) as tc:
        # Note (Jiaxin Deng): the admin cap is independent of
        # --max-payload-size, which sizes model traffic; this path buffers
        # before the CP checks the key, so it gets its own small bound.
        oversize = b"x" * (_MAX_FORWARD_BODY_BYTES + 1)
        response = tc.post("/update_weights_from_disk", content=oversize)
        assert response.status_code == 413
        assert response.json()["error"]["type"] == "payload_too_large"
        assert all(path != "/update_weights_from_disk" for path, _ in internal.requests)
        # above max_payload_size (1024 here) but a legitimate admin body
        assert (
            tc.post("/update_weights_from_disk", content=b"x" * 4096).status_code == 200
        )


def test_dp_answers_cors_preflight_like_the_single_process_app(
    tmp_path: Path,
) -> None:
    upstream = _Recorder()
    app, snapshot_path = _dp_app(tmp_path, upstream)
    with TestClient(app) as client:
        response = client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://studio.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


def test_dp_client_limits_split_keepalives_across_processes() -> None:
    from sglang_omni_router.data_plane import dp_client_limits

    config = RouterConfig(
        workers=[WorkerConfig(url="http://worker-a:8101")],
        max_connections=100,
    )
    limits = dp_client_limits(config, total_data_planes=4)
    # Note (Jiaxin Deng): full pool per DP (skew tolerance), keepalives split (idle-fd
    # budget)
    assert limits.max_connections == config.upstream_pool_size
    assert limits.max_keepalive_connections == 25
    single = dp_client_limits(config, total_data_planes=1)
    assert single.max_keepalive_connections == config.upstream_pool_size


def test_dp_uses_the_injected_shared_admission(tmp_path: Path) -> None:
    from sglang_omni_router.admission_shm import SharedAdmission, admission_file_size

    buf = bytearray(admission_file_size(2))
    sibling = SharedAdmission(
        buf, slots=2, own_index=1, max_inflight=1, generation=1, pid=7
    )
    assert sibling.try_acquire() is True  # the sibling holds the whole bound

    own = SharedAdmission(
        buf, slots=2, own_index=0, max_inflight=1, generation=1, pid=8
    )
    upstream = _Recorder()
    snapshot_path = str(tmp_path / "w4.json")
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
    app = create_data_plane_app(
        _config(),
        snapshot_path=snapshot_path,
        dp_index=0,
        generation=1,
        client=client,
        admission=own,
    )
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as tc:
        _snapshot(writer, _entry())
        _wait_for(lambda: tc.get("/ready").status_code == 200)
        response = tc.post("/generate", json={})
        # Note (Jiaxin Deng): the sibling's in-flight fills the GLOBAL bound: this DP
        # sheds
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "overloaded_error"

        sibling.release()
        assert tc.post("/generate", json={}).status_code == 200


def test_selector_rr_offset_staggers_first_picks() -> None:
    from sglang_omni_router.selector import WorkerSelector

    view = DataPlaneWorkerView()

    class _Snapshot:
        seq = 1
        cp_epoch = "e"
        workers = [_entry(), _entry("http://worker-b:8102")]

    view.apply(_Snapshot())
    first_picks = {
        index: WorkerSelector("round_robin", rr_offset=index)
        .select(view.workers(), required_capabilities=set())
        .url
        for index in (0, 1)
    }
    assert first_picks[0] != first_picks[1]


def test_dp_relays_sse_byte_identically(tmp_path: Path) -> None:
    # Note (Jiaxin Deng): the DP app must relay an event stream byte-for-byte through
    # the snapshot-fed relay path
    chunks = [
        b'data: {"choices": [{"delta": {"content": "he"}}]}\n\n',
        b'data: {"choices": [{"delta": {"audio": {"data": "AAAA"}}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    class _EventStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for chunk in chunks:
                yield chunk

    class _SSEUpstream(_Recorder):
        def handler(self, request: httpx.Request) -> httpx.Response:
            self.requests.append((request.url.path, request.content))
            if request.url.path == "/v1/chat/completions":
                return httpx.Response(
                    200,
                    stream=_EventStream(),
                    headers={"content-type": "text/event-stream"},
                    request=request,
                )
            return httpx.Response(200, json={"ok": True})

    upstream = _SSEUpstream()
    app, snapshot_path = _dp_app(tmp_path, upstream)
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as client:
        _snapshot(writer, _entry())
        _wait_for(lambda: client.get("/ready").status_code == 200)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "m", "stream": True},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = b"".join(response.iter_raw())
    assert body == b"".join(chunks)


def test_incarnation_change_replaces_the_worker_object() -> None:
    # Note (Jiaxin Deng): a new incarnation must yield a FRESH Worker object, so an in-
    # flight request holding the old one cannot misattribute a late failure (ABA)
    view = DataPlaneWorkerView()

    class _Snap:
        seq = 1
        cp_epoch = "e"
        workers = [_entry(incarnation="inc-A")]

    view.apply(_Snap())
    old_obj = view.workers()[0]
    assert old_obj.incarnation == "inc-A"

    class _Snap2:
        seq = 2
        cp_epoch = "e"
        workers = [_entry(incarnation="inc-B")]

    view.apply(_Snap2())
    new_obj = view.workers()[0]
    assert new_obj is not old_obj  # replaced, not mutated in place
    assert new_obj.incarnation == "inc-B"
    assert old_obj.incarnation == "inc-A"  # the captured reference is stable


def test_incarnation_unchanged_preserves_the_worker_object_and_counters() -> None:
    view = DataPlaneWorkerView()

    class _Snap:
        seq = 1
        cp_epoch = "e"
        workers = [_entry(incarnation="inc-A")]

    view.apply(_Snap())
    obj = view.workers()[0]
    obj.increment_active()
    obj.record_routed_request(status_code=200)

    class _Snap2:
        seq = 2
        cp_epoch = "e"
        workers = [_entry(incarnation="inc-A", state="unhealthy")]

    view.apply(_Snap2())
    assert view.workers()[0] is obj  # same incarnation: same object
    assert obj.active_requests == 1
    assert obj.routed_requests == 1


def test_dp_does_not_report_non_gateway_failures_to_the_cp(tmp_path: Path) -> None:
    # Note (Jiaxin Deng): eviction is gateway-only (502/504): a 503 is a per-request
    # failure, not an eviction signal, so it must not fire a worker_failure report
    import time as _time

    upstream = _Recorder(status_for={"/generate": 503})
    internal = _Recorder()
    app, snapshot_path = _dp_app(
        tmp_path, upstream, internal=internal, failure_threshold=3
    )
    writer = SnapshotWriter(snapshot_path, cp_epoch="e")
    with TestClient(app) as client:
        _snapshot(writer, _entry())
        _wait_for(lambda: client.get("/ready").status_code == 200)
        assert client.post("/generate", json={}).status_code == 503
        assert client.post("/generate", json={}).status_code == 503
        _time.sleep(0.3)  # let any (erroneous) report task run
        assert not any(
            path == "/internal/worker_failure" for path, _ in internal.requests
        )


def test_control_traffic_survives_a_saturated_forwarding_pool(tmp_path: Path) -> None:
    # Note (Jiaxin Deng): a forwarded admin call can sit behind the CP's update
    # lock for minutes; sharing one pool would starve the heartbeat and ACK
    # inside the liveness window and falsely degrade this DP.
    control_paths: list[str] = []

    async def _forward_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.5)
        return httpx.Response(200, json={"status": "ok"})

    def _control_handler(request: httpx.Request) -> httpx.Response:
        control_paths.append(request.url.path)
        return httpx.Response(200, json={"status": "ok"})

    app = create_data_plane_app(
        _config(),
        snapshot_path=str(tmp_path / "w.json"),
        dp_index=0,
        generation=1,
        client=httpx.AsyncClient(transport=httpx.MockTransport(_Recorder().handler)),
        internal_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_control_handler), base_url="http://cp"
        ),
        forward_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_forward_handler),
            base_url="http://cp",
            limits=httpx.Limits(max_connections=1),
        ),
        dp_refresh_interval_secs=0.02,
        heartbeat_interval_secs=0.05,
    )
    with TestClient(app) as tc:
        occupy = threading.Thread(
            target=lambda: tc.post("/update_weights_from_disk", json={"path": "/m"})
        )
        occupy.start()
        try:
            beats = _wait_for(
                lambda: (
                    [p for p in control_paths if p == "/internal/heartbeat"] or None
                )
            )
            assert len(beats) >= 1
        finally:
            occupy.join(timeout=15)


def test_forwarded_admin_requests_are_bounded_in_concurrency(tmp_path: Path) -> None:
    # Note (Jiaxin Deng): unauthenticated callers reach this path, so an
    # unbounded number of them must not each hold a buffered body.
    from sglang_omni_router.data_plane import _MAX_CONCURRENT_FORWARDS

    async def _cp_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.0)
        return httpx.Response(200, json={"status": "ok"})

    app = create_data_plane_app(
        _config(),
        snapshot_path=str(tmp_path / "w.json"),
        dp_index=0,
        generation=1,
        client=httpx.AsyncClient(transport=httpx.MockTransport(_Recorder().handler)),
        internal_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_cp_handler), base_url="http://cp"
        ),
        heartbeat_interval_secs=600,
        counter_flush_interval_secs=600,
    )
    statuses: list[int] = []
    with TestClient(app) as tc:
        threads = [
            threading.Thread(
                target=lambda: statuses.append(
                    tc.post("/pause_generation", json={}).status_code
                )
            )
            for _ in range(_MAX_CONCURRENT_FORWARDS + 4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)

    assert 503 in statuses
    assert statuses.count(200) <= _MAX_CONCURRENT_FORWARDS
