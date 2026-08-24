# SPDX-License-Identifier: Apache-2.0
"""FastAPI application wiring for the external Omni router."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Callable
from urllib.parse import quote, unquote

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from sglang_omni import __version__
from sglang_omni.http.admin_auth import (
    make_admin_auth_dependency,
    resolve_admin_api_key,
)
from sglang_omni.http.favicon import register_favicon
from sglang_omni_router.config import (
    MIN_CONNECTIONS_PER_WORKER,
    RouterConfig,
    WorkerConfig,
    can_own_uploaded_voices,
)
from sglang_omni_router.health import HealthChecker
from sglang_omni_router.proxy import ProxyHandler, filter_request_headers
from sglang_omni_router.selector import WorkerSelector
from sglang_omni_router.update_journal import (
    JournalUnreadableError,
    JournalUnwritableError,
    UpdateJournal,
    build_journal,
)
from sglang_omni_router.voice_routing import VoiceRoutingState
from sglang_omni_router.websocket_proxy import TTSWebSocketProxy
from sglang_omni_router.worker import (
    HEALTH_STATE_UNHEALTHY,
    HEALTH_STATE_UNKNOWN,
    Worker,
    build_workers,
)

logger = logging.getLogger(__name__)

_ADMIN_UPDATE_PATHS = {
    "/pause_generation",
    "/update_weights_from_disk",
    "/update_weights_from_distributed",
    "/init_weights_update_group",
    "/destroy_weights_update_group",
}
_ADMIN_UPDATE_LOCK_TIMEOUT_S = 300.0


def recover_worker_pool_from_journal(
    journal: UpdateJournal, workers: list[Worker]
) -> None:
    """Fail closed on an unresolved weight-update journal at startup.

    Journaled targets stay disabled until an operator verifies weight versions
    and re-enables them (which discards the entry), rather than re-enabling
    potentially mixed weights. An unreadable journal disables the whole pool; a
    journaled worker missing from the registry keeps its entry as a tombstone,
    so re-registering that stable ID creates the worker disabled.
    """
    try:
        unresolved = journal.pending()
    except JournalUnreadableError:
        unresolved = None
    if unresolved is None:
        for worker in workers:
            worker.set_disabled(True)
        logger.critical(
            "weight-update journal is present but unreadable; disabling the "
            "entire worker pool until it is inspected and cleared"
        )
        return
    if not unresolved:
        return
    disabled_count = 0
    for worker_id in unresolved:
        worker = _find_worker(workers, worker_id)
        if worker is not None:
            worker.set_disabled(True)
            disabled_count += 1
    logger.critical(
        f"unresolved weight update in the journal; {disabled_count} target "
        f"worker(s) kept disabled and {len(unresolved) - disabled_count} "
        "absent target(s) kept as tombstones until verified and re-enabled"
    )


def _worker_id_is_journaled(app: FastAPI, worker_id: str) -> bool:
    journal = getattr(app.state, "update_journal", None)
    if journal is None:
        return False
    try:
        return worker_id in journal.pending()
    except JournalUnreadableError:
        return True  # fail closed: an unreadable journal may name this id


def create_app(
    config: RouterConfig,
    *,
    client: httpx.AsyncClient | None = None,
    health_client: httpx.AsyncClient | None = None,
    admin_api_key: str | None = None,
    journal_path: str | None = None,
) -> FastAPI:
    workers = build_workers(config.workers)
    # Note (Jiaxin Deng): the single-process router shares the CP's crash-safe
    # journal, so a crash mid-update fails closed instead of returning success
    # with the pool left disabled.
    journal = build_journal(
        config.host, config.port, config.router_state_dir, journal_path
    )
    timeout = httpx.Timeout(config.request_timeout_secs)
    owns_client = client is None
    if client is None:
        limits = httpx.Limits(max_connections=config.upstream_pool_size)
        client = httpx.AsyncClient(timeout=timeout, limits=limits)
    owns_health_client = health_client is None and owns_client
    if health_client is None:
        if owns_client:
            health_limits = httpx.Limits(
                max_connections=max(1, len(workers)),
                max_keepalive_connections=max(1, len(workers)),
            )
            health_client = httpx.AsyncClient(
                timeout=httpx.Timeout(config.health_check_timeout_secs),
                limits=health_limits,
            )
        else:
            health_client = client
    health_checker = HealthChecker(
        workers=workers,
        config=config,
        client=health_client,
    )
    selector = WorkerSelector(config.policy)
    voice_routing = VoiceRoutingState(
        workers=workers,
        owner_url=config.voice_owner_worker_url,
        client=health_client,
        timeout_secs=config.health_check_timeout_secs,
        retry_interval_secs=config.health_check_interval_secs,
    )
    proxy = ProxyHandler(
        config=config,
        workers=workers,
        selector=selector,
        client=client,
        voice_routing=voice_routing,
    )
    websocket_proxy = TTSWebSocketProxy(
        config=config,
        workers=workers,
        selector=selector,
        admission=proxy.admission,
        voice_routing=voice_routing,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router_config = config
        app.state.workers = workers
        app.state.http_client = client
        app.state.health_http_client = health_client
        app.state.health_checker = health_checker
        app.state.proxy = proxy
        app.state.voice_routing = voice_routing
        app.state.websocket_proxy = websocket_proxy
        app.state.admission_controller = proxy.admission
        app.state.admin_update_lock = asyncio.Lock()
        app.state.update_journal = journal
        recover_worker_pool_from_journal(journal, workers)
        async with AsyncExitStack() as resources:
            if owns_client:
                resources.push_async_callback(client.aclose)
            if owns_health_client:
                resources.push_async_callback(health_client.aclose)
            await health_checker.start()
            resources.push_async_callback(health_checker.stop)
            await voice_routing.start()
            resources.push_async_callback(voice_routing.stop)
            yield

    resolved_key = resolve_admin_api_key(admin_api_key)

    app = FastAPI(title="sglang-omni-router", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_routes(
        app,
        workers,
        proxy,
        websocket_proxy,
        voice_routing,
        config,
        admin_api_key=resolved_key,
    )
    register_favicon(app)
    return app


def register_routes(
    app: FastAPI,
    workers: list[Worker],
    proxy: ProxyHandler,
    websocket_proxy: TTSWebSocketProxy,
    voice_routing: VoiceRoutingState,
    config: RouterConfig,
    *,
    admin_api_key: str | None = None,
) -> None:
    register_health_routes(app, workers, proxy, voice_routing)
    register_admin_routes(
        app,
        workers,
        config,
        voice_routing=voice_routing,
        admin_api_key=admin_api_key,
    )
    register_public_metadata_routes(app, workers, config)
    register_data_routes(app, proxy)
    register_tts_routes(app, proxy, websocket_proxy)


def register_health_routes(
    app: FastAPI,
    workers: list[Worker],
    proxy: ProxyHandler,
    voice_routing: VoiceRoutingState,
) -> None:
    @app.get("/live")
    async def live() -> JSONResponse:
        return JSONResponse({"status": "alive"})

    @app.get("/ready")
    async def ready() -> JSONResponse:
        return _worker_pool_status_response(
            workers,
            available_status="ready",
            unavailable_status="not_ready",
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        return _worker_pool_status_response(
            workers,
            available_status="healthy",
            unavailable_status="unhealthy",
            extra={
                "admission": proxy.admission.to_dict(),
                "voice_routing": voice_routing.to_dict(),
            },
        )


def _registry_lock_or_reject(app: FastAPI):
    """The admin update lock, or a 409 when an update owns or is next in line.

    Note (Jiaxin Deng): the caller must acquire the returned lock with no await
    in between, or locked() plus acquire stops being atomic. locked() alone is
    not enough either: release() wakes the first waiter without marking the
    lock held, so in that window a caller would be admitted and then queue
    behind an update it never saw.
    """
    lock = getattr(app.state, "admin_update_lock", None)
    if lock is None:
        return None, None
    # Note (Jiaxin Deng): a woken waiter is already resolved but has not
    # resumed yet, so "not done" would filter out exactly the case this
    # guards; any entry means an update is about to take the lock.
    queued = bool(getattr(lock, "_waiters", None))
    if lock.locked() or queued:
        return None, _error_response(
            409, "a weight update is in progress; retry when it completes"
        )
    return lock, None


def register_admin_routes(
    app: FastAPI,
    workers: list[Worker],
    config: RouterConfig,
    *,
    voice_routing: VoiceRoutingState | None = None,
    admin_api_key: str | None = None,
) -> None:
    _auth = make_admin_auth_dependency(admin_api_key)

    def _not_implemented_response() -> JSONResponse:
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": (
                        "This weight update path is not yet implemented. "
                        "Use /update_weights_from_disk for the disk-based update path."
                    ),
                    "code": "not_implemented",
                }
            },
        )

    @app.post("/workers")
    async def create_worker(request: Request) -> JSONResponse:
        payload, error = await _read_json_object(request)
        if error is not None:
            return error
        allowed_fields = {"url", "worker_url", "capabilities", "model"}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            return _error_response(
                400, f"unsupported fields: {', '.join(unknown_fields)}"
            )
        worker_url = request.query_params.get("url") or request.query_params.get(
            "worker_url"
        )
        worker_url = worker_url or _string_or_none(
            payload.get("url") or payload.get("worker_url")
        )
        if worker_url is None:
            return _error_response(400, "worker url is required")
        worker_config_kwargs: dict[str, Any] = {
            "url": worker_url,
            "model": payload.get("model"),
        }
        if "capabilities" in payload:
            worker_config_kwargs["capabilities"] = payload["capabilities"]
        try:
            worker_config = WorkerConfig(**worker_config_kwargs)
        except ValidationError as exc:
            return _error_response(400, str(exc))
        # Note (Jiaxin Deng): probe the staged worker BEFORE taking the registry
        # lock. That lock also excludes weight updates, so holding it across an
        # arbitrary worker's /health would let one blackholed candidate stall
        # every other CRUD call and any RL update for the full health timeout.
        worker = Worker(config=worker_config)
        await app.state.health_checker.check_worker_health(worker)

        lock, rejected = _registry_lock_or_reject(app)
        if rejected is not None:
            return rejected
        if lock is not None:
            await lock.acquire()
        try:
            # Note (Jiaxin Deng): membership and the journal can both have
            # changed while the probe ran unlocked.
            if any(existing.url == worker_config.url for existing in workers):
                return _error_response(409, "worker already registered")

            if _worker_id_is_journaled(app, worker.worker_id):
                # Note (Jiaxin Deng): this stable ID has an unresolved weight
                # update (tombstone); it may carry mixed weights, so it starts
                # disabled until an authenticated re-enable resolves it.
                worker.set_disabled(True)
                logger.warning(
                    f"worker {worker.display_id} registered disabled: its id "
                    "is journaled by an unresolved weight update"
                )
            workers.append(worker)
            if config.max_connections < MIN_CONNECTIONS_PER_WORKER * len(workers):
                logger.warning(
                    f"max_connections={config.max_connections} is below "
                    f"{MIN_CONNECTIONS_PER_WORKER} x {len(workers)} workers after "
                    "registration; the upstream client is sized at startup and can "
                    "under-feed the grown pool"
                )
            logger.info(
                f"worker_registered worker={worker.display_id} url={worker.url} "
                f"model={worker.model or '-'} "
                f"capabilities={','.join(sorted(worker.capabilities))} "
                f"health_state={worker.state} disabled={worker.disabled}",
            )
            _notify_registry_change(app)
            return JSONResponse({"status": "ok", "worker": worker.to_dict()})
        finally:
            if lock is not None:
                lock.release()

    @app.get("/workers")
    async def list_workers() -> JSONResponse:
        return JSONResponse(
            _pool_summary(
                workers,
                status="ok",
                include_workers=True,
                overlay=getattr(app.state, "worker_stats_overlay", None),
            )
        )

    @app.get("/workers/{worker_id:path}")
    async def get_worker(worker_id: str) -> JSONResponse:
        worker = _find_worker(workers, worker_id)
        if worker is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": "worker not found"}},
            )
        # Note (Jiaxin Deng): same DP-counter overlay as the /workers listing;
        # on the CP the local Worker never handles data traffic, so its own
        # counters are not the ones a caller wants.
        overlay = getattr(app.state, "worker_stats_overlay", None)
        payload = worker.to_dict()
        if overlay is not None:
            payload.update(overlay(worker))
        return JSONResponse(payload)

    @app.put("/workers/{worker_id:path}")
    async def update_worker(worker_id: str, request: Request) -> JSONResponse:
        payload, error = await _read_json_object(request)
        if error is not None:
            return error
        allowed_fields = {"is_dead", "disabled", "capabilities", "model"}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            return _error_response(
                400, f"unsupported fields: {', '.join(unknown_fields)}"
            )
        if not payload:
            return _error_response(400, "at least one worker field is required")

        requested_is_dead: bool | None = None
        requested_disabled: bool | None = None

        if "is_dead" in payload:
            requested_is_dead = payload["is_dead"]
            if not isinstance(requested_is_dead, bool):
                return _error_response(400, "is_dead must be a boolean")

        if "disabled" in payload:
            requested_disabled = payload["disabled"]
            if not isinstance(requested_disabled, bool):
                return _error_response(400, "disabled must be a boolean")

        lock, rejected = _registry_lock_or_reject(app)
        if rejected is not None:
            return rejected
        if lock is not None:
            await lock.acquire()
        try:
            response, reprobe = await _apply_worker_update(
                worker_id,
                payload,
                requested_is_dead,
                requested_disabled,
                request,
            )
        finally:
            if lock is not None:
                lock.release()

        if reprobe is None:
            return response
        await app.state.health_checker.check_worker_health(reprobe)
        _notify_registry_change(app)
        return JSONResponse({"status": "ok", "worker": reprobe.to_dict()})

    def _discard_needs_admin_auth(resolved_worker_id: str) -> bool:
        # Note (Jiaxin Deng): discarding a journal entry asserts the weights
        # are verified; admin-sensitive even though ordinary worker CRUD is not.
        journal = getattr(app.state, "update_journal", None)
        if journal is None or not admin_api_key:
            return False
        try:
            return resolved_worker_id in journal.pending()
        except Exception:
            return True  # unreadable journal: require auth to touch it

    async def _apply_worker_update(
        worker_id: str,
        payload: dict,
        requested_is_dead: bool | None,
        requested_disabled: bool | None,
        request: Request,
    ) -> tuple[JSONResponse, Worker | None]:
        """Returns the response and, when set, a worker to re-probe unlocked."""
        worker = _find_worker(workers, worker_id)
        if (
            requested_disabled is False
            and worker is not None
            and _discard_needs_admin_auth(worker.worker_id)
        ):
            try:
                await _auth(authorization=request.headers.get("authorization"))
            except HTTPException as exc:
                return _error_response(exc.status_code, str(exc.detail)), None
        if worker is None:
            return _error_response(404, "worker not found"), None
        next_config = worker.config

        if "capabilities" in payload or "model" in payload:
            try:
                next_config = WorkerConfig(
                    url=worker.url,
                    model=(
                        payload.get("model") if "model" in payload else worker.model
                    ),
                    capabilities=(
                        payload.get("capabilities")
                        if "capabilities" in payload
                        else worker.capabilities
                    ),
                )
            except ValidationError as exc:
                return _error_response(400, str(exc)), None

        # Note (Jiaxin Deng): every fallible precondition is checked before any
        # state is committed, so a rejected request cannot leave a half-applied
        # config for the CP keepalive to publish.
        if requested_disabled is False:
            journal = getattr(app.state, "update_journal", None)
            if journal is not None and not journal.discard(worker.worker_id):
                # Note (Jiaxin Deng): reporting success here would leave every
                # weight update blocked behind the 409 gate.
                return (
                    _error_response(
                        503,
                        "cannot re-enable: the weight-update journal at "
                        f"{journal.path} could not be durably resolved; inspect "
                        "and remove or replace it, then retry",
                    ),
                    None,
                )

        voice_owner = (
            voice_routing.ensure_owner() if voice_routing is not None else None
        )
        if voice_owner is worker and not can_own_uploaded_voices(
            next_config.capabilities
        ):
            return (
                _error_response(
                    409,
                    "voice owner worker must retain speech and audio_input capabilities",
                ),
                None,
            )

        worker.replace_config(next_config)

        if requested_disabled is not None:
            worker.set_disabled(requested_disabled)

        reprobe: Worker | None = None
        if requested_is_dead is not None:
            if requested_is_dead:
                worker.mark_dead()
            else:
                worker.clear_dead()
                # Note (Jiaxin Deng): the refresh probe runs after the lock is
                # released; it is a network call, and this lock also excludes
                # weight updates.
                reprobe = worker

        logger.info(
            f"worker_updated worker={worker.display_id} url={worker.url} "
            f"model={worker.model or '-'} "
            f"capabilities={','.join(sorted(worker.capabilities))} "
            f"health_state={worker.state} disabled={worker.disabled}",
        )
        _notify_registry_change(app)
        return JSONResponse({"status": "ok", "worker": worker.to_dict()}), reprobe

    @app.delete("/workers/{worker_id:path}")
    async def delete_worker(worker_id: str) -> JSONResponse:
        lock, rejected = _registry_lock_or_reject(app)
        if rejected is not None:
            return rejected
        if lock is not None:
            await lock.acquire()
        try:
            return _apply_worker_delete(worker_id)
        finally:
            if lock is not None:
                lock.release()

    def _apply_worker_delete(worker_id: str) -> JSONResponse:
        worker = _find_worker(workers, worker_id)
        if worker is None:
            return _error_response(404, "worker not found")
        voice_owner = (
            voice_routing.ensure_owner() if voice_routing is not None else None
        )
        if voice_owner is worker:
            return _error_response(409, "voice owner worker cannot be deleted")
        workers.remove(worker)
        logger.info(
            f"worker_deleted worker={worker.display_id} url={worker.url} "
            f"model={worker.model or '-'}",
        )
        _notify_registry_change(app)
        return JSONResponse({"status": "ok", "worker_id": worker.worker_id})

    @app.post("/weight_update_journal/resolve", dependencies=[Depends(_auth)])
    async def resolve_weight_update_journal(request: Request) -> JSONResponse:
        """Drop the crash-recovery record after an operator verified weights.

        The journal blocks every weight update until the targets of an
        interrupted one are re-enabled, and a journal that cannot be read can
        no longer be resolved that way. Without this the only way out is
        deleting the file on the host, so a new fail-closed mechanism would
        ship with no in-band recovery path.
        """
        payload, error = await _read_json_object(request)
        if error is not None:
            return error
        if payload.get("acknowledge") is not True:
            return _error_response(
                422,
                "resolving the journal discards the record that keeps workers "
                "with uncertain weight versions disabled; send "
                '{"acknowledge": true} to confirm those versions were verified',
            )
        journal = getattr(app.state, "update_journal", None)
        if journal is None:
            return _error_response(503, "no weight-update journal is configured")
        # Note (Jiaxin Deng): reject rather than queue behind a running update;
        # clearing mid-broadcast would erase the record of a transaction whose
        # outcome is still unknown.
        lock, rejected = _registry_lock_or_reject(app)
        if rejected is not None:
            return rejected
        if lock is not None:
            await lock.acquire()
        try:
            try:
                journaled = journal.pending()
                readable = True
            except JournalUnreadableError:
                journaled = []
                readable = False
            try:
                journal.clear()
            except JournalUnwritableError as exc:
                # Note (Jiaxin Deng): the unlink may have succeeded and only
                # its directory sync failed, so do not assert the file is
                # still there; the operator has to look.
                return _error_response(
                    503,
                    f"the weight-update journal at {journal.path} could not be "
                    f"durably resolved ({exc}); inspect it before assuming "
                    "updates are unblocked",
                )
        finally:
            if lock is not None:
                lock.release()
        logger.warning(
            f"weight_update_journal_resolved readable={readable} "
            f"worker_ids={journaled}"
        )
        return JSONResponse(
            {
                "status": "ok",
                "journal_readable": readable,
                "resolved_worker_ids": journaled,
            }
        )

    @app.get("/model_info", dependencies=[Depends(_auth)])
    async def model_info(request: Request) -> JSONResponse:
        return await _broadcast_admin_request(app, request, "/model_info")

    @app.post("/model_info", dependencies=[Depends(_auth)])
    async def model_info_post(request: Request) -> JSONResponse:
        return await _broadcast_admin_request(app, request, "/model_info")

    @app.post("/pause_generation", dependencies=[Depends(_auth)])
    async def pause_generation(request: Request) -> JSONResponse:
        return await _broadcast_admin_request(app, request, "/pause_generation")

    @app.post("/continue_generation", dependencies=[Depends(_auth)])
    async def continue_generation(request: Request) -> JSONResponse:
        return await _broadcast_admin_request(app, request, "/continue_generation")

    @app.post("/update_weights_from_disk", dependencies=[Depends(_auth)])
    async def update_weights_from_disk(request: Request) -> JSONResponse:
        return await _broadcast_admin_request(
            app,
            request,
            "/update_weights_from_disk",
        )

    @app.post("/update_weights_from_tensor", dependencies=[Depends(_auth)])
    async def update_weights_from_tensor(request: Request) -> JSONResponse:
        return _not_implemented_response()

    @app.post("/init_weights_update_group", dependencies=[Depends(_auth)])
    async def init_weights_update_group(request: Request) -> JSONResponse:
        return await _broadcast_admin_request(
            app,
            request,
            "/init_weights_update_group",
        )

    @app.post("/destroy_weights_update_group", dependencies=[Depends(_auth)])
    async def destroy_weights_update_group(request: Request) -> JSONResponse:
        return await _broadcast_admin_request(
            app,
            request,
            "/destroy_weights_update_group",
        )

    @app.post("/update_weights_from_distributed", dependencies=[Depends(_auth)])
    async def update_weights_from_distributed(request: Request) -> JSONResponse:
        return await _broadcast_admin_request(
            app,
            request,
            "/update_weights_from_distributed",
        )

    @app.api_route(
        "/weights_checker",
        methods=["GET", "POST"],
        dependencies=[Depends(_auth)],
    )
    async def weights_checker(request: Request) -> JSONResponse:
        return await _broadcast_admin_request(app, request, "/weights_checker")


def register_public_metadata_routes(
    app: FastAPI,
    workers: list[Worker],
    config: RouterConfig,
) -> None:
    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        return await _merge_models(
            workers,
            app.state.http_client,
            request,
            timeout_secs=config.health_check_timeout_secs,
        )


def register_data_routes(
    app: FastAPI,
    proxy: ProxyHandler,
    *,
    gate: Callable[[], Response | None] | None = None,
) -> None:
    # Note (Jiaxin Deng): gate is the data-plane stale-snapshot shed check;
    # unset in single-process mode.
    async def _forward(request: Request, path: str) -> Response:
        if gate is not None:
            gated = gate()
            if gated is not None:
                return gated
        return await proxy.forward_model_request(request, path)

    @app.post("/generate")
    async def generate(request: Request) -> Response:
        return await _forward(request, "/generate")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _forward(request, "/v1/chat/completions")

    @app.post("/v1/audio/speech")
    async def audio_speech(request: Request) -> Response:
        return await _forward(request, "/v1/audio/speech")

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(request: Request) -> Response:
        return await _forward(request, "/v1/audio/transcriptions")

    @app.post("/v1/audio/translations")
    async def audio_translations(request: Request) -> Response:
        return await _forward(request, "/v1/audio/translations")


def register_tts_routes(
    app: FastAPI,
    proxy: ProxyHandler,
    websocket_proxy: TTSWebSocketProxy,
) -> None:
    @app.post("/v1/audio/speech/batch")
    async def audio_speech_batch(request: Request) -> Response:
        return await proxy.forward_model_request(request, "/v1/audio/speech/batch")

    @app.websocket("/v1/audio/speech/stream")
    async def audio_speech_stream(websocket: WebSocket) -> None:
        await websocket_proxy.forward(websocket)

    @app.get("/v1/audio/voices")
    async def audio_voices(request: Request) -> Response:
        return await proxy.forward_model_request(request, "/v1/audio/voices")

    @app.post("/v1/audio/voices")
    async def upload_audio_voice(request: Request) -> Response:
        return await proxy.forward_model_request(request, "/v1/audio/voices")

    @app.delete("/v1/audio/voices/{name}")
    async def delete_audio_voice(name: str, request: Request) -> Response:
        path = f"/v1/audio/voices/{quote(name, safe='')}"
        return await proxy.forward_model_request(request, path)


def _pool_summary(
    workers: list[Worker],
    *,
    status: str,
    include_workers: bool = True,
    overlay: Callable[[Worker], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    healthy = sum(1 for worker in workers if worker.is_healthy)
    dead = sum(1 for worker in workers if worker.is_dead)
    unhealthy = sum(1 for worker in workers if worker.state == HEALTH_STATE_UNHEALTHY)
    unknown = sum(1 for worker in workers if worker.state == HEALTH_STATE_UNKNOWN)
    disabled = sum(1 for worker in workers if worker.disabled)
    routable = sum(1 for worker in workers if worker.is_routable)
    payload: dict[str, Any] = {
        "status": status,
        "healthy_workers": healthy,
        "dead_workers": dead,
        "disabled_workers": disabled,
        "routable_workers": routable,
        "unhealthy_workers": unhealthy,
        "unknown_workers": unknown,
        "total_workers": len(workers),
    }
    if include_workers:
        if overlay is None:
            payload["workers"] = [worker.to_dict() for worker in workers]
        else:
            payload["workers"] = [
                {**worker.to_dict(), **overlay(worker)} for worker in workers
            ]
    return payload


def _worker_pool_status_response(
    workers: list[Worker],
    *,
    available_status: str,
    unavailable_status: str,
    extra: dict[str, Any] | None = None,
    overlay: Callable[[Worker], dict[str, Any]] | None = None,
) -> JSONResponse:
    routable = sum(1 for worker in workers if worker.is_routable)
    status_code = 200 if routable > 0 else 503
    status = available_status if routable > 0 else unavailable_status
    payload = _pool_summary(workers, status=status, overlay=overlay)
    if extra:
        payload.update(extra)
    return JSONResponse(payload, status_code=status_code)


def _notify_registry_change(app: FastAPI) -> None:
    voice_routing = getattr(app.state, "voice_routing", None)
    if voice_routing is not None:
        voice_routing.request_refresh()
    # Note (Jiaxin Deng): CP hook, republishes the snapshot after a registry
    # mutation; unset (single-process mode) is a no-op.
    callback = getattr(app.state, "on_registry_change", None)
    if callback is not None:
        callback()


async def _await_dp_snapshot_ack(app: FastAPI) -> JSONResponse | None:
    # Note (Jiaxin Deng): CP hook, waits until all live DPs acknowledged the
    # disabled-worker snapshot; unset (single-process mode) is a no-op.
    barrier = getattr(app.state, "dp_snapshot_ack_barrier", None)
    if barrier is None:
        return None
    acked, pending = await barrier()
    if acked:
        return None
    return _error_response(
        503,
        f"weight update aborted: data planes {pending} did not acknowledge "
        "the disabled-worker snapshot within the ack timeout; no broadcast "
        "was sent",
    )


async def _broadcast_admin_request(
    app: FastAPI,
    request: Request,
    path: str,
) -> JSONResponse:
    workers: list[Worker] = app.state.workers
    target_workers = [worker for worker in workers if not worker.is_dead]
    if not target_workers:
        return _error_response(503, "no live upstream workers")

    # Note (Xuesong): distributed-init assigns each worker an NCCL rank from a
    # single shared rank_offset (sglang: rank = rank_offset + tp_rank).
    # Broadcasting the same body to multiple replicas makes them join with
    # colliding ranks and hang the rendezvous. Reject until the trainer assigns
    # a distinct rank_offset per replica (genuine multi-replica support is a
    # larger design).
    if path == "/init_weights_update_group" and len(target_workers) > 1:
        return _error_response(
            422,
            "distributed weight-update init currently supports a single-replica "
            f"target stage, but {len(target_workers)} live workers were targeted; "
            "multi-replica refit needs a distinct rank_offset per replica.",
        )

    if path in _ADMIN_UPDATE_PATHS:
        try:
            await asyncio.wait_for(
                app.state.admin_update_lock.acquire(),
                timeout=_ADMIN_UPDATE_LOCK_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return _error_response(
                503,
                f"admin update lock not acquired within {_ADMIN_UPDATE_LOCK_TIMEOUT_S:.0f}s; "
                "another update operation may be in progress",
            )
        try:
            journal = getattr(app.state, "update_journal", None)
            if journal is not None and journal.has_pending():
                return _error_response(
                    409,
                    "an earlier weight update did not complete; verify the "
                    "journaled workers' weight versions, re-enable them "
                    "(admin-authenticated PUT /workers {disabled: false}), "
                    "then retry. A journaled worker no longer registered must "
                    "be re-added under its original URL (it starts disabled) "
                    "and re-enabled the same way",
                )
            # Note (Jiaxin Deng): re-resolve the target set under the lock;
            # membership may have changed while waiting for a previous update.
            target_workers = [worker for worker in workers if not worker.is_dead]
            if not target_workers:
                return _error_response(503, "no live upstream workers")
            if path == "/init_weights_update_group" and len(target_workers) > 1:
                return _error_response(
                    422,
                    "distributed weight-update init currently supports a "
                    "single-replica target stage, but "
                    f"{len(target_workers)} live workers were targeted; "
                    "multi-replica refit needs a distinct rank_offset per "
                    "replica.",
                )
            return await _broadcast_admin_request_locked(
                app,
                request,
                path,
                target_workers,
                disable_targets=True,
            )
        finally:
            app.state.admin_update_lock.release()

    return await _broadcast_admin_request_locked(
        app,
        request,
        path,
        target_workers,
        disable_targets=False,
    )


async def _broadcast_admin_request_locked(
    app: FastAPI,
    request: Request,
    path: str,
    workers: list[Worker],
    *,
    disable_targets: bool,
) -> JSONResponse:
    body = await request.body()
    headers = filter_request_headers(request)
    previous_disabled = {worker.worker_id: worker.disabled for worker in workers}
    # Note (Jiaxin Deng): `results` drives the journal/restore logic. None =
    # crashed after the broadcast started (fail closed); [] = aborted before
    # anything was sent; list = completed (restore only if all succeeded).
    results: list[dict[str, Any]] | None = None
    journal_error: str | None = None
    journal = getattr(app.state, "update_journal", None)
    if disable_targets and journal is not None:
        # Note (Jiaxin Deng): journal before disabling/publishing so a crash in
        # that window fails closed on recovery; an update that cannot journal
        # must not run, or a host crash re-enables a mixed-weight pool.
        try:
            journal.begin(path, [worker.worker_id for worker in workers])
        except JournalUnwritableError as exc:
            logger.error(f"weight_update_refused journal_not_durable error={exc}")
            return _error_response(
                503,
                "cannot start the weight update: the journal at "
                f"{journal.path} could not be durably written ({exc}); the "
                "update is refused so a crash cannot re-enable a "
                "mixed-weight pool",
            )
    if disable_targets:
        for worker in workers:
            worker.set_disabled(True)
        _notify_registry_change(app)
    try:
        if disable_targets:
            ack_error = await _await_dp_snapshot_ack(app)
            if ack_error is not None:
                results = []  # nothing was sent
                return ack_error
        results = await asyncio.gather(
            *[
                _send_admin_to_worker(
                    app.state.http_client,
                    worker,
                    request,
                    path,
                    body,
                    headers,
                )
                for worker in workers
            ]
        )
    finally:
        if disable_targets:
            outcome_safe = results is not None and (
                not results or all(item["success"] for item in results)
            )
            _restore_admin_disabled_state(workers, previous_disabled, outcome_safe)
            if journal is not None:
                # Note (Jiaxin Deng): resolve the journal BEFORE publishing;
                # the publish can raise (snapshot write), and a crash there
                # must not leave the outcome unrecorded.
                try:
                    if outcome_safe:
                        journal.clear()
                    else:
                        # Note (Jiaxin Deng): once the broadcast started every
                        # target's weight version is uncertain, not just the
                        # ones that failed.
                        journal.keep([worker.worker_id for worker in workers])
                except JournalUnwritableError as exc:
                    # Note (Jiaxin Deng): the in-memory outcome already stands,
                    # so this is not a 500; but a surviving entry blocks every
                    # later update behind the 409 gate, which the caller cannot
                    # see from a 200 alone.
                    journal_error = (
                        f"the journal at {journal.path} could not be resolved "
                        f"({exc}); later updates stay blocked until it is "
                        "cleared"
                    )
                    logger.error(f"journal_not_durable path={journal.path} {exc}")
            _notify_registry_change(app)

    success = all(item["success"] for item in results)
    if path == "/model_info":
        return _model_info_broadcast_response(results, success=success)

    payload = {
        "success": success,
        "message": "ok" if success else "one or more workers failed admin request",
        "path": path,
        "worker_count": len(results),
        "results": results,
    }
    if journal_error is not None:
        payload["journal_error"] = journal_error
    return JSONResponse(payload, status_code=200 if success else 502)


def _restore_admin_disabled_state(
    workers: list[Worker],
    previous_disabled: dict[str, bool],
    outcome_safe: bool,
) -> None:
    if not outcome_safe:
        # Note (Jiaxin Deng): the weight version is uncertain for the pool,
        # so none may be re-enabled (fail closed).
        for worker in workers:
            worker.set_disabled(True)
        return
    for worker in workers:
        worker.set_disabled(previous_disabled[worker.worker_id])


def _model_info_broadcast_response(
    results: list[dict[str, Any]],
    *,
    success: bool,
) -> JSONResponse:
    if not success:
        payload = {
            "success": False,
            "message": "one or more workers failed model_info request",
            "path": "/model_info",
            "worker_count": len(results),
            "workers": results,
            "results": results,
        }
        return JSONResponse(payload, status_code=502)

    worker_infos = _extract_worker_model_infos(results)
    weight_version = _common_worker_model_info_value(
        worker_infos,
        "weight_version",
        mixed_status_code=409,
        results=results,
    )
    payload = {
        "success": True,
        "message": "ok",
        "path": "/model_info",
        "worker_count": len(results),
        "weight_version": weight_version,
        "model_path": _common_worker_model_info_value(worker_infos, "model_path"),
        "load_format": _common_worker_model_info_value(worker_infos, "load_format"),
        "workers": results,
        "results": results,
    }
    return JSONResponse(payload)


def _extract_worker_model_infos(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for result in results:
        body = result.get("body")
        if not isinstance(body, dict):
            continue
        worker = result.get("worker")
        top_level = {
            key: body.get(key)
            for key in ("weight_version", "model_path", "load_format")
            if body.get(key) is not None
        }
        if top_level:
            top_level["worker"] = worker
            infos.append(top_level)
        for stage in body.get("stages") or body.get("results") or []:
            if not isinstance(stage, dict):
                continue
            data = stage.get("data")
            if not isinstance(data, dict):
                continue
            if data.get("skipped") or data.get("unsupported"):
                continue
            info = dict(data)
            info.setdefault("worker", worker)
            info.setdefault("stage", stage.get("stage"))
            infos.append(info)
    return infos


def _common_worker_model_info_value(
    worker_infos: list[dict[str, Any]],
    key: str,
    *,
    mixed_status_code: int | None = None,
    results: list[dict[str, Any]] | None = None,
) -> Any:
    values = [info[key] for info in worker_infos if info.get(key) is not None]
    if not values:
        return None
    unique: dict[str, Any] = {}
    for value in values:
        unique.setdefault(json.dumps(value, sort_keys=True, default=str), value)
    if len(unique) == 1:
        return next(iter(unique.values()))
    if mixed_status_code is not None:
        payload = {
            "success": False,
            "message": f"mixed worker {key}",
            "path": "/model_info",
            "mixed_state": {key: list(unique.values())},
            "workers": results or [],
            "results": results or [],
        }
        raise HTTPException(status_code=mixed_status_code, detail=payload)
    return None


async def _send_admin_to_worker(
    client: httpx.AsyncClient,
    worker: Worker,
    request: Request,
    path: str,
    body: bytes,
    headers: dict[str, str],
) -> dict[str, Any]:
    upstream_url = f"{worker.url}{path}"
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"
    try:
        response = await client.request(
            request.method,
            upstream_url,
            content=body,
            headers=headers,
        )
    except httpx.HTTPError as exc:
        return {
            "worker": worker.url,
            "success": False,
            "error": type(exc).__name__,
        }

    body_payload = _decode_response_payload(response)
    body_success = (
        body_payload.get("success", True) if isinstance(body_payload, dict) else True
    )
    success = 200 <= response.status_code < 300 and body_success is not False
    return {
        "worker": worker.url,
        "success": success,
        "status_code": response.status_code,
        "body": body_payload,
    }


def _decode_response_payload(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


def _find_worker(workers: list[Worker], worker_id: str) -> Worker | None:
    decoded = unquote(worker_id)
    for worker in workers:
        if worker.worker_id == worker_id or worker.url == decoded:
            return worker
    return None


async def _read_json_object(
    request: Request,
) -> tuple[dict[str, Any], JSONResponse | None]:
    body = await request.body()
    if not body:
        return {}, None
    try:
        payload = await request.json()
    except Exception:
        return {}, _error_response(400, "invalid JSON body")
    if not isinstance(payload, dict):
        return {}, _error_response(400, "request body must be a JSON object")
    return payload, None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message}},
    )


async def _merge_models(
    workers: list[Worker],
    client: httpx.AsyncClient,
    request: Request,
    *,
    timeout_secs: int,
) -> JSONResponse:
    routable_workers = [worker for worker in workers if worker.is_routable]
    if not routable_workers:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "no routable upstream"}},
        )

    request_headers = filter_request_headers(request)
    query = request.url.query
    cards_by_id: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    worker_results = await asyncio.gather(
        *(
            _fetch_worker_models(
                worker,
                client,
                request_headers,
                query,
                timeout_secs=timeout_secs,
            )
            for worker in routable_workers
        )
    )
    for worker, data, error in worker_results:
        if error is not None:
            errors[worker.url] = error
            continue
        if data is None:
            errors[worker.url] = "invalid models payload"
            continue
        for card in data:
            if not isinstance(card, dict):
                continue
            model_id = card.get("id") or card.get("model")
            dedupe_key = (
                model_id
                if isinstance(model_id, str) and model_id
                else json.dumps(card, sort_keys=True)
            )
            cards_by_id.setdefault(dedupe_key, card)

    if not cards_by_id:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "failed to fetch models from workers",
                    "details": errors,
                }
            },
        )

    return JSONResponse({"object": "list", "data": list(cards_by_id.values())})


async def _fetch_worker_models(
    worker: Worker,
    client: httpx.AsyncClient,
    request_headers: dict[str, str],
    query: bytes,
    *,
    timeout_secs: int,
) -> tuple[Worker, list[Any] | None, str | None]:
    url = f"{worker.url}/v1/models" if not query else f"{worker.url}/v1/models?{query}"
    try:
        response = await client.get(
            url,
            headers=request_headers,
            timeout=timeout_secs,
        )
    except Exception as exc:
        return worker, None, type(exc).__name__
    if not 200 <= response.status_code < 300:
        return worker, None, f"status={response.status_code}"
    try:
        payload = response.json()
    except Exception as exc:
        return worker, None, type(exc).__name__
    data = payload.get("data")
    if not isinstance(data, list):
        return worker, None, "invalid models payload"
    return worker, data, None
