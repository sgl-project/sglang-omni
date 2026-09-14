# SPDX-License-Identifier: Apache-2.0
"""Data-plane app: a stateless relay fed by the control plane's snapshot.

The DP never mutates the registry. It polls the worker snapshot into a local
view (persistent Worker objects, so per-process counters survive refreshes),
relays the model routes (see register_data_routes) through the standard
ProxyHandler, sheds new requests when the snapshot outlives the stale timeout
(the CP is presumed gone), reports eviction-relevant upstream failures to the
CP, and heartbeats its identity plus last_applied_seq so the CP watchdog and
the weight-update ACK barrier can see it. A 409 heartbeat means this process
was fenced out by a newer generation and must stop serving.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import signal
from contextlib import asynccontextmanager
from typing import Callable

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from sglang_omni import __version__
from sglang_omni.http.admin_auth import resolve_admin_api_key
from sglang_omni_router.python.app import (
    _merge_models,
    _worker_pool_status_response,
    register_data_routes,
)
from sglang_omni_router.python.config import RouterConfig, WorkerConfig
from sglang_omni_router.python.internal_channel import (
    FORWARD_CHANNEL_CONNECTIONS,
    INTERNAL_TOKEN_HEADER,
)
from sglang_omni_router.python.proxy import HOP_BY_HOP_HEADERS, ProxyHandler
from sglang_omni_router.python.selector import WorkerSelector
from sglang_omni_router.python.snapshot import SnapshotReader, WorkerSnapshot
from sglang_omni_router.python.worker import Worker

logger = logging.getLogger("sglang_omni_router.python.data_plane")

DEFAULT_DP_REFRESH_INTERVAL_SECS = 0.2
DEFAULT_SNAPSHOT_MAX_AGE_SECS = 10.0
DEFAULT_HEARTBEAT_INTERVAL_SECS = 2.0
DEFAULT_COUNTER_FLUSH_INTERVAL_SECS = 1.0
# Note (Jiaxin Deng): failure events are bounded-retry best-effort, not
# at-least-once; past the attempts below the CP health probe is the backstop.
_FAILURE_REPORT_MAX_ATTEMPTS = 3
_FAILURE_REPORT_BACKOFF_SECS = 0.2
# Note (Jiaxin Deng): bound pending report tasks so a slow CP cannot pile up
# retry tasks (and their sockets) on the DP.
_FAILURE_REPORT_MAX_PENDING = 64
# Note (Jiaxin Deng): forwarded admin bodies are small JSON, and this path runs
# before the CP checks the admin key.
_MAX_FORWARD_BODY_BYTES = 1024 * 1024
_MAX_CONCURRENT_FORWARDS = 2 * FORWARD_CHANNEL_CONNECTIONS
# Note (Jiaxin Deng): definitive (non-428) registration rejections before the
# DP fails closed; an unregistered DP escapes the weight-update ACK barrier.
_REGISTER_REJECT_LIMIT = 10


class DataPlaneWorkerView:
    """Snapshot-fed worker view with persistent Worker objects.

    Applying a snapshot updates health/disabled/config in place and only
    creates or drops workers on membership changes, so per-process state
    (active_requests, routed counters) survives refreshes.
    """

    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self._retiring: list[Worker] = []
        self.last_applied_seq = 0
        self.last_applied_epoch = ""

    def workers(self) -> list[Worker]:
        return list(self._workers.values())

    def reportable_workers(self) -> list[Worker]:
        # Note (Jiaxin Deng): routing uses workers(); counter flushes use this,
        # so a request still running on a replaced or deleted worker object
        # still reaches the CP ledger when it completes.
        return list(self._workers.values()) + list(self._retiring)

    def drained_retired(self) -> list[Worker]:
        return [worker for worker in self._retiring if worker.active_requests == 0]

    def release_retired(self, reported: list[Worker]) -> None:
        """Drop retired workers whose final counters the CP acknowledged."""
        done = {id(worker) for worker in reported}
        self._retiring = [worker for worker in self._retiring if id(worker) not in done]

    def _retire(self, worker: Worker) -> None:
        if worker.active_requests > 0 or worker.routed_requests > 0:
            self._retiring.append(worker)

    def apply(self, snapshot: WorkerSnapshot) -> None:
        seen: set[str] = set()
        for entry in snapshot.workers:
            seen.add(entry.url)
            config_kwargs: dict = {"url": entry.url, "model": entry.model}
            if entry.capabilities:
                config_kwargs["capabilities"] = set(entry.capabilities)
            config = WorkerConfig(**config_kwargs)
            worker = self._workers.get(entry.url)
            incarnation_changed = (
                worker is not None
                and entry.incarnation
                and worker.incarnation != entry.incarnation
            )
            if worker is None or incarnation_changed:
                # Note (Jiaxin Deng): a new incarnation gets a FRESH Worker, so
                # an in-flight request holding the old object cannot
                # misattribute a late failure to the new worker (ABA).
                if worker is not None:
                    self._retire(worker)
                worker = Worker(config=config)
                if entry.incarnation:
                    worker.incarnation = entry.incarnation
                self._workers[entry.url] = worker
            else:
                if (
                    worker.model != config.model
                    or worker.capabilities != config.capabilities
                ):
                    worker.replace_config(config)
                if entry.incarnation:
                    worker.incarnation = entry.incarnation
            worker.state = entry.state
            worker.disabled = entry.disabled
        for url in list(self._workers):
            if url not in seen:
                self._retire(self._workers.pop(url))
        self.last_applied_seq = snapshot.seq
        self.last_applied_epoch = getattr(snapshot, "cp_epoch", "")


def _default_fence_reaction() -> None:
    # Note (Jiaxin Deng): SIGTERM lets uvicorn finish in-flight responses
    # instead of dropping them.
    os.kill(os.getpid(), signal.SIGTERM)


def dp_client_limits(config: RouterConfig, total_data_planes: int) -> httpx.Limits:
    """Per-DP upstream pool: full-size connections (a skewed client mix can
    pin the whole global bound onto one DP, which must still relay it), but
    keep-alives split across DPs so the cluster's idle-connection total stays
    around one pool's worth."""
    keepalive = -(-config.upstream_pool_size // max(1, total_data_planes))
    return httpx.Limits(
        max_connections=config.upstream_pool_size,
        max_keepalive_connections=keepalive,
    )


def _counter_report_applied(response: httpx.Response) -> bool:
    try:
        return response.json().get("applied", True) is not False
    except ValueError:
        return True


def create_data_plane_app(
    config: RouterConfig,
    *,
    snapshot_path: str,
    dp_index: int,
    generation: int,
    client: httpx.AsyncClient | None = None,
    internal_client: httpx.AsyncClient | None = None,
    forward_client: httpx.AsyncClient | None = None,
    internal_token: str | None = None,
    metadata_client: httpx.AsyncClient | None = None,
    admission=None,
    total_data_planes: int = 1,
    dp_refresh_interval_secs: float = DEFAULT_DP_REFRESH_INTERVAL_SECS,
    snapshot_max_age_secs: float = DEFAULT_SNAPSHOT_MAX_AGE_SECS,
    heartbeat_interval_secs: float = DEFAULT_HEARTBEAT_INTERVAL_SECS,
    counter_flush_interval_secs: float = DEFAULT_COUNTER_FLUSH_INTERVAL_SECS,
    on_fenced: Callable[[], None] = _default_fence_reaction,
    admin_api_key: str | None = None,
) -> FastAPI:
    view = DataPlaneWorkerView()
    reader = SnapshotReader(snapshot_path)

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout_secs),
            limits=dp_client_limits(config, total_data_planes),
        )
    # Note (Jiaxin Deng): /v1/models fans out on its own small pool; metadata
    # is not admission-controlled and must not occupy the sized relay pool.
    owns_metadata_client = metadata_client is None
    if metadata_client is None:
        metadata_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.health_check_timeout_secs),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )

    failure_tasks: set[asyncio.Task] = set()
    failure_seq_counter = itertools.count(1)

    def _report_failure(
        worker: Worker, status_code: int | None, error: str | None
    ) -> None:
        # Note (Jiaxin Deng): each distinct failure gets its own failure_seq
        # (the CP counts events, dedups re-delivery by id; retries reuse it).
        if internal_client is None:
            return
        if len(failure_tasks) >= _FAILURE_REPORT_MAX_PENDING:
            logger.debug("worker_failure report dropped: too many pending")
            return
        loop = asyncio.get_running_loop()
        payload = {
            "worker_id": worker.worker_id,
            "incarnation": worker.incarnation,
            "status_code": status_code,
            "error": error,
            "dp_index": dp_index,
            "generation": generation,
            "failure_seq": next(failure_seq_counter),
        }

        async def _send() -> None:
            last_error: httpx.HTTPError | None = None
            for attempt in range(_FAILURE_REPORT_MAX_ATTEMPTS):
                try:
                    await internal_client.post(
                        "/internal/worker_failure",
                        json=payload,
                        headers=_internal_headers(internal_token),
                    )
                    return
                except httpx.HTTPError as exc:
                    last_error = exc
                    await asyncio.sleep(_FAILURE_REPORT_BACKOFF_SECS * (2**attempt))
            logger.debug(
                f"worker_failure report dropped after "
                f"{_FAILURE_REPORT_MAX_ATTEMPTS} attempts: {last_error}"
            )

        task = loop.create_task(_send())
        failure_tasks.add(task)
        task.add_done_callback(failure_tasks.discard)

    proxy = ProxyHandler(
        config=config,
        workers=[],
        selector=WorkerSelector(config.policy, rr_offset=dp_index),
        client=client,
        worker_provider=view.workers,
        on_worker_failure=_report_failure,
        admission=admission,
    )

    def _stale_gate() -> JSONResponse | None:
        age = reader.age_secs()
        if age is not None and age <= snapshot_max_age_secs:
            return None
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": (
                        "worker snapshot missing or stale; the control plane "
                        "is unreachable and new requests are shed"
                    ),
                    "type": "unavailable_error",
                    "code": 503,
                }
            },
        )

    # Note (Jiaxin Deng): the weight-update ACK barrier waits on
    # last_applied_seq, which the CP only learns from heartbeats. Without this
    # wake-up an apply sits invisible for up to a full heartbeat period, and
    # the barrier holds the pool disabled for the slowest DP's beat phase.
    applied_event = asyncio.Event()

    async def _refresh_loop() -> None:
        while True:
            if reader.maybe_reload():
                before = (view.last_applied_seq, view.last_applied_epoch)
                view.apply(reader.snapshot)
                if (view.last_applied_seq, view.last_applied_epoch) != before:
                    applied_event.set()
            await asyncio.sleep(dp_refresh_interval_secs)

    async def _heartbeat_loop() -> None:
        if internal_client is None:
            return
        identity = {
            "dp_index": dp_index,
            "generation": generation,
            "pid": os.getpid(),
        }
        registered = False
        rejected_registrations = 0
        while True:
            try:
                if not registered:
                    response = await internal_client.post(
                        "/internal/register",
                        json=identity,
                        headers=_internal_headers(internal_token),
                    )
                else:
                    response = await internal_client.post(
                        "/internal/heartbeat",
                        json={
                            **identity,
                            "last_applied_seq": view.last_applied_seq,
                            "last_applied_epoch": view.last_applied_epoch,
                            # Note (Jiaxin Deng): shedding as snapshot_stale
                            # must show up in CP /health.
                            "serving": _stale_gate() is None,
                        },
                        headers=_internal_headers(internal_token),
                    )
                if response.status_code == 409:
                    logger.critical("fenced out by a newer generation; stopping")
                    on_fenced()
                    return
                if response.status_code == 428:
                    # Note (Jiaxin Deng): 428 = CP restarted and lost
                    # registrations; re-register, this is not a fence.
                    registered = False
                elif response.status_code == 200:
                    was_registered = registered
                    registered = True
                    rejected_registrations = 0
                    if not was_registered:
                        # Note (Jiaxin Deng): registration carries no seq, so
                        # the ACK barrier only learns this DP's position on the
                        # next beat; send it now instead of a poll interval
                        # later, which after a timeout would eat most of the
                        # barrier's budget.
                        continue
                else:
                    # Note (Jiaxin Deng): an unregistered DP is invisible to
                    # the weight-update ACK barrier, so a definitively rejected
                    # DP must not keep serving indefinitely.
                    registered = False
                    rejected_registrations += 1
                    logger.warning(
                        f"internal registration rejected with "
                        f"{response.status_code} "
                        f"({rejected_registrations}/{_REGISTER_REJECT_LIMIT})"
                    )
                    if rejected_registrations >= _REGISTER_REJECT_LIMIT:
                        logger.critical(
                            "registration persistently rejected; failing closed"
                        )
                        on_fenced()
                        return
            except httpx.HTTPError:
                # Note (Jiaxin Deng): CP unreachable; the stale-snapshot gate
                # governs serving, keep retrying until the CP is back.
                registered = False
            try:
                await asyncio.wait_for(
                    applied_event.wait(), timeout=heartbeat_interval_secs
                )
            except asyncio.TimeoutError:
                pass
            else:
                applied_event.clear()

    async def _counter_flush_loop() -> None:
        if internal_client is None:
            return
        counter_seq = 0
        while True:
            await asyncio.sleep(counter_flush_interval_secs)
            counter_seq += 1
            # Note (Jiaxin Deng): snapshot before building the payload, so a
            # retiree that drains mid-post survives to the next flush.
            drained = view.drained_retired()
            payload = {
                "dp_index": dp_index,
                "generation": generation,
                "counter_seq": counter_seq,
                "workers": [
                    {
                        "worker_id": worker.worker_id,
                        "incarnation": worker.incarnation,
                        "routed_total": worker.routed_requests,
                        "successful_total": worker.successful_requests,
                        "failed_total": worker.failed_requests,
                        "current_active": worker.active_requests,
                        "routed_requests_by_class": dict(
                            worker.routed_requests_by_class
                        ),
                        "successful_requests_by_class": dict(
                            worker.successful_requests_by_class
                        ),
                        "failed_requests_by_class": dict(
                            worker.failed_requests_by_class
                        ),
                    }
                    for worker in view.reportable_workers()
                ],
            }
            if admission is not None and hasattr(admission, "touch"):
                admission.touch()
            try:
                response = await internal_client.post(
                    "/internal/counters",
                    json=payload,
                    headers=_internal_headers(internal_token),
                )
                if response.status_code == 409:
                    logger.critical("counter report fenced by a newer generation")
                    on_fenced()
                    return
                if response.status_code < 300 and _counter_report_applied(response):
                    # Note (Jiaxin Deng): a retiree's totals live only in this
                    # payload, so release it on acceptance, not on a 200 that
                    # dropped the report as stale.
                    view.release_retired(drained)
            except httpx.HTTPError:
                # Note (Jiaxin Deng): totals are cumulative; the next flush
                # carries everything, so a dropped report loses nothing.
                pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router_config = config
        app.state.worker_view = view
        app.state.snapshot_reader = reader
        app.state.http_client = client
        app.state.proxy = proxy
        app.state.admission_controller = proxy.admission
        tasks = [asyncio.create_task(_refresh_loop())]
        if internal_client is not None:
            tasks.append(asyncio.create_task(_heartbeat_loop()))
            tasks.append(asyncio.create_task(_counter_flush_loop()))
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if owns_client:
                await client.aclose()
            if owns_metadata_client:
                await metadata_client.aclose()
            if internal_client is not None:
                await internal_client.aclose()
            if forward_client is not None and forward_client is not internal_client:
                await forward_client.aclose()

    app = FastAPI(title="sglang-omni-router-dp", version=__version__, lifespan=lifespan)
    # Note (Jiaxin Deng): same external CORS policy as the single-process app;
    # the DP is the public surface.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/live")
    async def live() -> JSONResponse:
        return JSONResponse({"status": "alive"})

    @app.get("/ready")
    async def ready() -> JSONResponse:
        stale = _stale_gate()
        if stale is not None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "snapshot_stale"},
            )
        return _worker_pool_status_response(
            view.workers(),
            available_status="ready",
            unavailable_status="not_ready",
        )

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        return await _merge_models(
            view.workers(),
            metadata_client,
            request,
            timeout_secs=config.health_check_timeout_secs,
        )

    register_data_routes(app, proxy, gate=_stale_gate)
    # Note (Jiaxin Deng): forwarding runs on its own pool; a public admin
    # call can sit behind the CP's update lock for minutes, and sharing the
    # control pool would starve the heartbeat and ACK that keep this DP
    # counted as live.
    forwarding_client = (
        forward_client if forward_client is not None else internal_client
    )
    if forwarding_client is not None:
        _register_cp_forwarding(
            app,
            forwarding_client,
            config,
            admin_api_key=resolve_admin_api_key(admin_api_key),
        )
    return app


# Note (Jiaxin Deng): a DP relays these verbatim and does not check auth; the
# CP owns the admin surface and re-checks it. The route names mirror the
# single-process handlers so the published schema stays identical across
# --router-processes, and a client generated against N=1 still works at N>=2.
# The authed flag mirrors which single-process routes carry the admin auth
# dependency, so the published Authorization parameter stays identical too.
_FORWARDED_CP_ROUTES: tuple[tuple[str, tuple[str, ...], str, bool], ...] = (
    ("/health", ("GET",), "health", False),
    ("/workers", ("GET",), "list_workers", False),
    ("/workers", ("POST",), "create_worker", False),
    ("/workers/{worker_id:path}", ("GET",), "get_worker", False),
    ("/workers/{worker_id:path}", ("PUT",), "update_worker", False),
    ("/workers/{worker_id:path}", ("DELETE",), "delete_worker", False),
    ("/model_info", ("GET",), "model_info", True),
    ("/model_info", ("POST",), "model_info_post", True),
    ("/pause_generation", ("POST",), "pause_generation", True),
    ("/continue_generation", ("POST",), "continue_generation", True),
    ("/update_weights_from_disk", ("POST",), "update_weights_from_disk", True),
    (
        "/weight_update_journal/resolve",
        ("POST",),
        "resolve_weight_update_journal",
        True,
    ),
    ("/update_weights_from_tensor", ("POST",), "update_weights_from_tensor", True),
    ("/init_weights_update_group", ("POST",), "init_weights_update_group", True),
    (
        "/destroy_weights_update_group",
        ("POST",),
        "destroy_weights_update_group",
        True,
    ),
    (
        "/update_weights_from_distributed",
        ("POST",),
        "update_weights_from_distributed",
        True,
    ),
    ("/weights_checker", ("GET", "POST"), "weights_checker", True),
)

# Note (Jiaxin Deng): the body is re-framed as fixed-length bytes, so a client
# transfer-encoding/te must not ride along or the CP sees both framings at once.
_FORWARD_REQUEST_STRIP = HOP_BY_HOP_HEADERS | {
    "host",
    "content-length",
    "accept-encoding",
}
_FORWARD_RESPONSE_STRIP = {
    "content-length",
    "date",
    "server",
    "connection",
    "transfer-encoding",
    "content-encoding",
}


def _register_cp_forwarding(
    app: FastAPI,
    internal_client: httpx.AsyncClient,
    config: RouterConfig,
    *,
    admin_api_key: str | None = None,
) -> None:
    in_flight = 0

    def _payload_too_large() -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "message": (
                        "admin request body exceeds " f"{_MAX_FORWARD_BODY_BYTES} bytes"
                    ),
                    "type": "payload_too_large",
                    "code": 413,
                }
            },
        )

    async def _forward(request: Request) -> Response:
        # Note (Jiaxin Deng): this path buffers before the CP authenticates, so
        # it is bounded on its own terms rather than by --max-payload-size,
        # which sizes model traffic. Admin bodies are small JSON; without both
        # bounds unauthenticated callers could hold N x 512MiB of DP memory.
        nonlocal in_flight
        if in_flight >= _MAX_CONCURRENT_FORWARDS:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "too many concurrent admin requests",
                        "type": "overloaded_error",
                        "code": 503,
                    }
                },
                headers={"Retry-After": "1"},
            )
        in_flight += 1
        try:
            return await _forward_bounded(request)
        finally:
            in_flight -= 1

    async def _forward_bounded(request: Request) -> Response:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > _MAX_FORWARD_BODY_BYTES:
            return _payload_too_large()
        received = bytearray()
        async for chunk in request.stream():
            received.extend(chunk)
            if len(received) > _MAX_FORWARD_BODY_BYTES:
                received.clear()
                return _payload_too_large()
        body = bytes(received)
        received.clear()
        # Note (Jiaxin Deng): raw_path keeps percent-encoding intact (worker
        # ids contain encoded slashes a decode/re-encode round trip loses).
        raw_path = request.scope.get("raw_path")
        url = raw_path.decode("ascii") if raw_path else request.url.path
        query = request.scope.get("query_string", b"")
        if query:
            url = f"{url}?{query.decode('ascii')}"
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _FORWARD_REQUEST_STRIP
        }
        try:
            upstream = await internal_client.request(
                request.method, url, content=body, headers=headers
            )
        except httpx.HTTPError as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"control plane unreachable: {exc}",
                        "type": "control_plane_error",
                        "code": 502,
                    }
                },
            )
        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in _FORWARD_RESPONSE_STRIP
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=response_headers,
        )

    async def _forward_worker(request: Request, worker_id: str) -> Response:
        # Note (Jiaxin Deng): worker_id is unused; it exists so the published
        # schema names the path parameter as the single-process routes do.
        return await _forward(request)

    async def _declare_admin_authorization(
        authorization: str | None = Header(default=None),
    ) -> None:
        # Note (Jiaxin Deng): schema-only, never enforced here; the CP owns
        # the check. Same signature as the single-process auth dependency so
        # the published Authorization parameter is identical across N.
        return

    for path, methods, name, authed in _FORWARDED_CP_ROUTES:
        handler = _forward_worker if "{worker_id" in path else _forward
        dependencies = (
            [Depends(_declare_admin_authorization)]
            if authed and admin_api_key
            else None
        )
        app.add_api_route(
            path,
            handler,
            methods=list(methods),
            name=name,
            dependencies=dependencies,
        )


def _internal_headers(token: str | None) -> dict[str, str]:
    if token:
        return {INTERNAL_TOKEN_HEADER: token}
    return {}


def create_dp_app_from_env() -> FastAPI:
    """uvicorn factory entry point for a supervisor-spawned data plane."""
    import mmap as mmap_module

    from sglang_omni_router.python.admission_shm import (
        SharedAdmission,
        admission_file_size,
    )
    from sglang_omni_router.python.app_factory import load_config_from_env
    from sglang_omni_router.python.internal_channel import (
        CONTROL_CHANNEL_CONNECTIONS,
        CONTROL_CHANNEL_TIMEOUT_SECS,
        FORWARD_CHANNEL_CONNECTIONS,
        INTERNAL_TOKEN_ENV,
    )
    from sglang_omni_router.python.supervisor import (
        ADMISSION_SHM_ENV,
        DP_GENERATION_ENV,
        DP_INDEX_ENV,
        EXPECTED_DPS_ENV,
        INTERNAL_TCP_URL_ENV,
        INTERNAL_UDS_ENV,
        SNAPSHOT_PATH_ENV,
    )

    config = load_config_from_env()
    token = os.environ.get(INTERNAL_TOKEN_ENV)
    timeout = httpx.Timeout(config.request_timeout_secs)
    uds = os.environ.get(INTERNAL_UDS_ENV)
    tcp_url = os.environ.get(INTERNAL_TCP_URL_ENV)
    if not uds and not tcp_url:
        raise RuntimeError(
            "neither the internal UDS nor the internal TCP URL is set; "
            "this factory is only meant to be spawned by the supervisor"
        )

    def _internal_client(connections: int, client_timeout) -> httpx.AsyncClient:
        limits = httpx.Limits(
            max_connections=connections,
            max_keepalive_connections=max(1, connections // 2),
        )
        if uds:
            return httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(uds=uds, limits=limits),
                base_url="http://internal",
                timeout=client_timeout,
            )
        return httpx.AsyncClient(
            base_url=tcp_url, timeout=client_timeout, limits=limits
        )

    internal_client = _internal_client(
        CONTROL_CHANNEL_CONNECTIONS, httpx.Timeout(CONTROL_CHANNEL_TIMEOUT_SECS)
    )
    forward_client = _internal_client(FORWARD_CHANNEL_CONNECTIONS, timeout)

    dp_index = int(os.environ[DP_INDEX_ENV])
    generation = int(os.environ[DP_GENERATION_ENV])
    total = int(os.environ.get(EXPECTED_DPS_ENV, "1"))

    admission = None
    shm_path = os.environ.get(ADMISSION_SHM_ENV)
    admission_file = None
    if shm_path:
        admission_file = open(shm_path, "r+b")
        admission_mmap = mmap_module.mmap(
            admission_file.fileno(), admission_file_size(total)
        )
        admission = SharedAdmission(
            admission_mmap,
            slots=total,
            own_index=dp_index,
            max_inflight=config.effective_max_inflight,
            generation=generation,
            pid=os.getpid(),
            on_fenced=_default_fence_reaction,
        )

    app = create_data_plane_app(
        config,
        snapshot_path=os.environ[SNAPSHOT_PATH_ENV],
        dp_index=dp_index,
        generation=generation,
        internal_client=internal_client,
        forward_client=forward_client,
        internal_token=token,
        admission=admission,
        total_data_planes=total,
    )
    # Note (Jiaxin Deng): keep the shm file object alive for the app's lifetime.
    app.state.admission_shm_file = admission_file
    return app
