# SPDX-License-Identifier: Apache-2.0
"""Control-plane app: the single owner of registry, health, and admin state.

In the multi-process router exactly one CP process runs this app. It owns the
mutable worker registry, the HealthChecker, the admin update lock, and the
privileged admin surface; it publishes the routable-worker snapshot after
every state change (health tick, worker CRUD, disable flips around weight
updates) for the stateless data planes to consume; and it hosts the internal
DP channel, including hot-path worker-failure reports and the weight-update
ACK barrier (all live DPs must acknowledge the disabled-worker snapshot
before a weight update is broadcast; on timeout the update fails closed).
"""

from __future__ import annotations

import asyncio
import logging
import mmap
import os
import time
from contextlib import asynccontextmanager
from typing import cast

import httpx
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import Field as PydanticField

from sglang_omni.http.admin_auth import resolve_admin_api_key
from sglang_omni_router.admission_shm import (
    AdmissionAggregateView,
    SeqlockUnstableError,
    admission_file_size,
)
from sglang_omni_router.app import (
    _error_response,
    _find_worker,
    _pool_summary,
    _worker_pool_status_response,
    recover_worker_pool_from_journal,
    register_admin_routes,
    register_public_metadata_routes,
)
from sglang_omni_router.config import RouterConfig, WorkerConfig
from sglang_omni_router.health import HealthChecker
from sglang_omni_router.internal_channel import (
    InternalChannelState,
    make_internal_token_dependency,
    register_internal_routes,
)
from sglang_omni_router.observability import (
    CounterReport,
    DataPlaneCounterLedger,
    StaleCounterGenerationError,
)
from sglang_omni_router.snapshot import SnapshotReader, SnapshotWorker, SnapshotWriter
from sglang_omni_router.update_journal import build_journal
from sglang_omni_router.worker import Worker, WorkerState, build_workers

logger = logging.getLogger("sglang_omni_router.control_plane")

DEFAULT_DP_ACK_TIMEOUT_SECS = 10.0
DEFAULT_DP_LIVENESS_SECS = 6.0
# Note (Jiaxin Deng): the snapshot doubles as the CP liveness signal for the
# DP stale-timeout, so republish on a fixed cadence, not only on state changes.
DEFAULT_SNAPSHOT_KEEPALIVE_SECS = 2.0
_ACK_POLL_INTERVAL_SECS = 0.05
_ADMISSION_HEALTH_READS = 3
_ADMISSION_HEALTH_READ_DELAY_SECS = 0.01
# Note (Jiaxin Deng): a DP caches its own fail-closed shed for 50ms and then
# re-probes, so the shm design already treats a wedge as self-healing. Declaring
# a total outage faster than that would turn a descheduled writer (cgroup
# throttling runs to a 100ms period) into an LB eviction.
_ADMISSION_UNREADABLE_GRACE_SECS = 0.25


class WorkerFailureReport(BaseModel):
    worker_id: str
    status_code: int | None = None
    error: str | None = None
    # Note (Jiaxin Deng): the incarnation pins the report to the observed
    # worker; a failure from a deleted-and-re-added URL must not evict the new one.
    incarnation: str = ""
    dp_index: int = PydanticField(ge=0)
    generation: int = PydanticField(ge=1)
    # Note (Jiaxin Deng): retries of one failure reuse one failure_seq, so
    # re-delivery cannot count a single failure toward eviction twice.
    failure_seq: int = PydanticField(ge=1)


def snapshot_workers(workers: list[Worker]) -> list[SnapshotWorker]:
    return [
        SnapshotWorker(
            url=worker.url,
            worker_id=worker.worker_id,
            incarnation=worker.incarnation,
            model=worker.model,
            capabilities=sorted(worker.capabilities),
            routable=worker.is_routable,
            state=worker.state,
            disabled=worker.disabled,
        )
        for worker in workers
    ]


def restore_workers(snapshot_path: str, config: RouterConfig) -> list[Worker]:
    if not os.path.exists(snapshot_path):
        return build_workers(config.workers)

    reader = SnapshotReader(snapshot_path)
    if not reader.maybe_reload() or reader.snapshot is None:
        raise RuntimeError(f"cannot recover worker registry from {snapshot_path}")

    workers = []
    for entry in reader.snapshot.workers:
        worker = Worker(
            config=WorkerConfig(
                url=entry.url,
                model=entry.model,
                capabilities=set(entry.capabilities),
            )
        )
        if entry.incarnation:
            worker.incarnation = entry.incarnation
        worker.state = cast(WorkerState, entry.state)
        worker.disabled = entry.disabled
        workers.append(worker)
    return workers


def create_control_plane_app(
    config: RouterConfig,
    *,
    snapshot_path: str,
    cp_epoch: str,
    internal_token: str | None = None,
    client: httpx.AsyncClient | None = None,
    health_client: httpx.AsyncClient | None = None,
    admin_api_key: str | None = None,
    dp_ack_timeout_secs: float = DEFAULT_DP_ACK_TIMEOUT_SECS,
    dp_liveness_secs: float = DEFAULT_DP_LIVENESS_SECS,
    snapshot_keepalive_secs: float = DEFAULT_SNAPSHOT_KEEPALIVE_SECS,
    expected_data_planes: int | None = None,
    admission_shm_path: str | None = None,
    admission_grace_secs: float = _ADMISSION_UNREADABLE_GRACE_SECS,
    journal_path: str | None = None,
) -> FastAPI:
    if expected_data_planes is not None and expected_data_planes < 1:
        raise ValueError("expected_data_planes must be >= 1 when set")
    workers = restore_workers(snapshot_path, config)
    writer = SnapshotWriter(snapshot_path, cp_epoch)
    internal_state = InternalChannelState()
    ledger = DataPlaneCounterLedger()
    # Note (Jiaxin Deng): a durable path (keyed by host:port) outside the
    # per-run workdir; the journal must survive a host reboot, not just a CP
    # respawn, because the remote workers do.
    journal = build_journal(
        config.host, config.port, config.router_state_dir, journal_path
    )

    admission_view: AdmissionAggregateView | None = None
    admission_shm_file = None
    if admission_shm_path and expected_data_planes:
        admission_shm_file = open(admission_shm_path, "rb")
        admission_view = AdmissionAggregateView(
            mmap.mmap(
                admission_shm_file.fileno(),
                admission_file_size(expected_data_planes),
                access=mmap.ACCESS_READ,
            ),
            expected_data_planes,
        )

    # Note (Jiaxin Deng): the CP never relays data traffic, so its pool is
    # sized to the worker count, not to the admission bound.
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout_secs),
            limits=httpx.Limits(max_connections=max(16, 2 * len(workers))),
        )
    owns_health_client = health_client is None and owns_client
    if health_client is None:
        health_client = client
    health_checker = HealthChecker(
        workers=workers,
        config=config,
        client=health_client,
        on_tick=lambda: publish(),
    )

    def publish() -> int:
        writer.publish(snapshot_workers(workers))
        return writer.seq

    async def dp_snapshot_ack_barrier() -> tuple[bool, list[int]]:
        target_seq = writer.seq
        loop = asyncio.get_running_loop()
        deadline = loop.time() + dp_ack_timeout_secs
        while True:
            now = time.time()
            live = {
                record.dp_index: record
                for record in internal_state.data_planes.values()
                if now - record.last_seen_at <= dp_liveness_secs
            }
            # Note (Jiaxin Deng): an ACK must come from this epoch; a seq from
            # a previous CP's numbering must never satisfy the barrier.
            pending = {
                index
                for index, record in live.items()
                if record.last_applied_epoch != cp_epoch
                or record.last_applied_seq < target_seq
            }
            # Note (Jiaxin Deng): fail closed on absent DPs; a DP the CP has
            # not heard from is not an implicit ACK, and with no expected
            # count every DP that ever registered holds the barrier.
            if expected_data_planes is not None:
                expected_indices = range(expected_data_planes)
            else:
                expected_indices = list(internal_state.data_planes)
            pending.update(index for index in expected_indices if index not in live)
            if not pending:
                return True, []
            if loop.time() >= deadline:
                return False, sorted(pending)
            await asyncio.sleep(_ACK_POLL_INTERVAL_SECS)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router_config = config
        app.state.workers = workers
        app.state.http_client = client
        app.state.health_http_client = health_client
        app.state.health_checker = health_checker
        app.state.admin_update_lock = asyncio.Lock()
        app.state.internal_channel = internal_state
        app.state.snapshot_writer = writer
        app.state.counter_ledger = ledger
        app.state.worker_stats_overlay = ledger.overlay
        app.state.on_registry_change = publish
        app.state.dp_snapshot_ack_barrier = dp_snapshot_ack_barrier
        app.state.update_journal = journal
        recover_worker_pool_from_journal(journal, workers)
        publish()
        await health_checker.start()

        async def _keepalive() -> None:
            while True:
                await asyncio.sleep(snapshot_keepalive_secs)
                try:
                    publish()
                except Exception:
                    # Note (Jiaxin Deng): a transient write failure must not
                    # end the cadence; the snapshot is the CP liveness signal.
                    logger.exception("snapshot keepalive publish failed")

        keepalive_task = asyncio.create_task(_keepalive())
        try:
            yield
        finally:
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("snapshot keepalive task ended abnormally")
            await health_checker.stop()
            if admission_shm_file is not None:
                admission_shm_file.close()
            if owns_health_client and health_client is not client:
                await health_client.aclose()
            if owns_client:
                await client.aclose()

    app = FastAPI(title="sglang-omni-router-cp", version="0.1.0", lifespan=lifespan)
    resolved_key = resolve_admin_api_key(admin_api_key)

    @app.get("/live")
    async def live() -> JSONResponse:
        return JSONResponse({"status": "alive"})

    @app.get("/ready")
    async def ready() -> JSONResponse:
        return _worker_pool_status_response(
            workers,
            available_status="ready",
            unavailable_status="not_ready",
            overlay=ledger.overlay,
        )

    admission_unreadable_since: float | None = None

    @app.get("/health")
    async def health() -> JSONResponse:
        now = time.time()
        live_dps = [
            record
            for record in internal_state.data_planes.values()
            if now - record.last_seen_at <= dp_liveness_secs
        ]
        # Note (Jiaxin Deng): serving-ready also requires still owning the
        # admission slot, so a crashed generation drops out immediately
        # instead of lingering as ready.
        slot_generations: dict[int, int] = {}
        admission_error = None
        admission_slots = None
        if admission_view is not None:
            # Note (Jiaxin Deng): a live writer preempted mid-write clears in
            # microseconds, so debounce before believing the slot is wedged.
            # Each read spins a bounded budget rather than sleeping, and the
            # waits between them yield instead of blocking the loop.
            for attempt in range(_ADMISSION_HEALTH_READS):
                try:
                    admission_slots = admission_view.per_slot(now=now)
                    slot_generations = {
                        slot["index"]: slot["generation"] for slot in admission_slots
                    }
                    admission_error = None
                    break
                except SeqlockUnstableError as exc:
                    admission_error = str(exc)
                    if attempt + 1 < _ADMISSION_HEALTH_READS:
                        await asyncio.sleep(_ADMISSION_HEALTH_READ_DELAY_SECS)
        # Note (Jiaxin Deng): a wedge that outlives the grace window is a DP
        # that died mid-write, and every sibling's try_acquire fails closed
        # until the supervisor reclaims it, so nothing can serve. Below the
        # window it is reported but not acted on, because a merely descheduled
        # writer recovers on its own and a 503 costs an LB eviction.
        nonlocal admission_unreadable_since
        if admission_error is None:
            admission_unreadable_since = None
        elif admission_unreadable_since is None:
            admission_unreadable_since = now
        admission_readable = admission_view is None or admission_error is None
        wedged = (
            admission_unreadable_since is not None
            and now - admission_unreadable_since >= admission_grace_secs
        )
        ready_indices = {
            record.dp_index
            for record in live_dps
            if not wedged
            and record.last_applied_epoch == cp_epoch
            and record.serving
            and (
                admission_view is None
                # Note (Jiaxin Deng): inside the grace window the heartbeat is
                # the better signal; a descheduled writer must not read as a
                # DP that stopped owning its slot.
                or admission_error is not None
                or slot_generations.get(record.dp_index) == record.generation
            )
        }
        routable = sum(1 for worker in workers if worker.is_routable)
        if expected_data_planes is not None:
            # Note (Jiaxin Deng): identity, not cardinality; {0, 99} is not a
            # full roster of 2.
            expected_set = set(range(expected_data_planes))
            missing = sorted(expected_set - ready_indices)
            unexpected = sorted({record.dp_index for record in live_dps} - expected_set)
            serving = len(expected_set & ready_indices)
        else:
            missing = []
            unexpected = []
            serving = len(ready_indices)
        # Note (Jiaxin Deng): degraded, not dead, while at least one DP keeps
        # serving; 503 only when nothing can serve at all.
        no_service = (
            routable == 0
            or wedged
            or (expected_data_planes is not None and serving == 0)
        )
        degraded = bool(missing) and serving > 0
        if no_service:
            status = "unhealthy"
        elif degraded:
            status = "degraded"
        else:
            status = "healthy"
        payload = _pool_summary(workers, status=status, overlay=ledger.overlay)
        payload["data_planes"] = [
            record.model_dump()
            for _, record in sorted(internal_state.data_planes.items())
        ]
        payload["live_data_planes"] = len(live_dps)
        payload["serving_ready_data_planes"] = serving
        payload["missing_data_planes"] = missing
        payload["unexpected_data_planes"] = unexpected
        payload["expected_data_planes"] = expected_data_planes
        if admission_view is not None:
            if admission_error is not None:
                payload["admission_error"] = admission_error
            else:
                try:
                    payload["admission"] = admission_view.to_dict(
                        config.effective_max_inflight
                    )
                    payload["admission_slots"] = admission_slots
                except SeqlockUnstableError as exc:
                    payload["admission_error"] = str(exc)
        return JSONResponse(payload, status_code=503 if no_service else 200)

    register_admin_routes(app, workers, config, admin_api_key=resolved_key)
    register_public_metadata_routes(app, workers, config)
    register_internal_routes(app, internal_state, token=internal_token)

    internal_auth = Depends(make_internal_token_dependency(internal_token))

    applied_failure_seqs: dict[tuple[int, int], dict] = {}

    def _failure_already_applied(report: WorkerFailureReport) -> bool:
        # Note (Jiaxin Deng): exact-id dedup (not high-water: events may
        # arrive out of order), with a pruning floor to bound memory.
        state = applied_failure_seqs.setdefault(
            (report.dp_index, report.generation), {"seqs": set(), "floor": 0}
        )
        if report.failure_seq <= state["floor"]:
            return True
        if report.failure_seq in state["seqs"]:
            return True
        state["seqs"].add(report.failure_seq)
        if len(state["seqs"]) > 4096:
            state["floor"] = max(state["seqs"]) - 1024
            state["seqs"] = {seq for seq in state["seqs"] if seq > state["floor"]}
        return False

    def _stale_generation(dp_index: int, generation: int) -> JSONResponse | None:
        # Note (Jiaxin Deng): only an OLDER generation is provably fenced. A
        # NEWER one is a replacement DP racing ahead of its /internal/register;
        # 409ing it would make that valid process treat itself as fenced and exit.
        current = internal_state.data_planes.get(dp_index)
        if current is not None and generation < current.generation:
            return _error_response(
                409,
                f"stale generation {generation}, current is {current.generation}",
            )
        return None

    @app.post("/internal/worker_failure", dependencies=[internal_auth])
    async def worker_failure(report: WorkerFailureReport) -> JSONResponse:
        if error := _stale_generation(report.dp_index, report.generation):
            return error
        worker = _find_worker(workers, report.worker_id)
        if worker is None:
            return _error_response(404, "worker not found")
        if report.incarnation and worker.incarnation != report.incarnation:
            return JSONResponse({"status": "ok", "stale_incarnation": True})
        if _failure_already_applied(report):
            return JSONResponse({"status": "ok", "deduplicated": True})
        if worker.is_dead:
            # Note (Jiaxin Deng): dedup above already consumed this event id,
            # so a retry cannot apply it after a later clear-dead. An operator's
            # dead mark must not be demoted to unhealthy and then probed back.
            return JSONResponse({"status": "ok", "ignored_dead": True})
        before = (worker.is_routable, worker.state)
        worker.record_request_failure(
            failure_threshold=config.health_failure_threshold,
            status_code=report.status_code,
            error=report.error,
        )
        # Note (Jiaxin Deng): publish only on a snapshot-visible transition; a
        # counter bump on an already-unhealthy worker must not serialize a full
        # snapshot per failure (the keepalive covers periodic liveness writes).
        if (worker.is_routable, worker.state) != before:
            publish()
        return JSONResponse(
            {"status": "ok", "worker_state": worker.state, "disabled": worker.disabled}
        )

    @app.post("/internal/counters", dependencies=[internal_auth])
    async def counters(report: CounterReport) -> JSONResponse:
        if error := _stale_generation(report.dp_index, report.generation):
            return error
        try:
            applied = ledger.apply(report)
        except StaleCounterGenerationError as exc:
            return _error_response(409, str(exc))
        return JSONResponse({"status": "ok", "applied": applied})

    return app
