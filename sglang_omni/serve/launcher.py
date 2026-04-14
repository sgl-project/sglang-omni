# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server from a PipelineConfig.

Usage (programmatic)::

    from sglang_omni.serve.launcher import launch_server
    launch_server(pipeline_config, host="0.0.0.0", port=8000)

Usage (CLI — with config file)::

    sglang-omni-server --config pipeline.json --port 8000

Usage (CLI — built-in pipeline, no JSON needed)::

    sglang-omni-server \\
        --pipeline qwen3-omni \\
        --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --port 8000

Export a config to JSON::

    sglang-omni-server --pipeline qwen3-omni --model-id ... --export-config out.json
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from typing import Any

import uvicorn
from fastapi import APIRouter
from pydantic import BaseModel

from sglang_omni.client import Client
from sglang_omni.config import PipelineConfig, build_pipeline_runner
from sglang_omni.profiler.profiler_control import ProfilerControlClient
from sglang_omni.serve.openai_api import create_app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in pipeline registry
# ---------------------------------------------------------------------------


def _find_available_port(host: str, port: int) -> int:
    """Return *port* if available, otherwise find a free port and warn."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return port
    except OSError:
        pass
    logger.warning(f"Port {port} is already in use on {host}.")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        free_port = s.getsockname()[1]
    logger.warning(f"Using port {free_port} instead.")
    return free_port


def _default_run_id() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def _default_template(profiler_dir: str, run_id: str) -> str:
    return os.path.join(profiler_dir, run_id, "trace")


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _collect_stage_control_endpoints(stages) -> dict[str, str]:
    """Derive {stage_name: control_plane_recv_endpoint} from runtime Stage objects."""
    out: dict[str, str] = {}
    for st in stages:
        cp = getattr(st, "control_plane", None)
        ep = getattr(cp, "recv_endpoint", None) if cp is not None else None

        if not ep:
            ep = getattr(st, "recv_endpoint", None)

        if not ep:
            raise RuntimeError(
                f"Cannot resolve control endpoint for stage={getattr(st, 'name', st)}"
            )
        out[st.name] = ep
    return out


class StartReq(BaseModel):
    run_id: str | None = None
    trace_path_template: str | None = None
    config: dict[str, Any] | None = None


class StopReq(BaseModel):
    run_id: str | None = None


def _mount_profiler_routes(
    app, profiler_ctl: ProfilerControlClient, profiler_dir: str
) -> None:
    router = APIRouter()

    @router.post("/start_profile")
    async def start(req: StartReq):
        run_id = req.run_id or _default_run_id()
        tpl = req.trace_path_template or _default_template(profiler_dir, run_id)
        await profiler_ctl.broadcast_start(
            run_id=run_id,
            trace_path_template=tpl,
            config=req.config,
        )
        return {"run_id": run_id, "trace_path_template": tpl}

    @router.post("/stop_profile")
    async def stop(req: StopReq):
        run_id = req.run_id or "default"
        await profiler_ctl.broadcast_stop(run_id=run_id)
        return {"run_id": run_id}

    app.include_router(router)


async def _build_app_and_serve(
    coordinator,
    *,
    model_name: str,
    host: str,
    port: int,
    log_level: str,
    client_kwargs: dict[str, Any] | None = None,
    stage_endpoints: dict[str, str] | None = None,
) -> None:
    """Create Client + FastAPI app + uvicorn server."""
    cl_kwargs = client_kwargs or {}
    client = Client(coordinator, **cl_kwargs)
    app = create_app(client, model_name=model_name)

    if stage_endpoints is not None:
        profiler_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR")
        profiler_ctl = ProfilerControlClient(stage_endpoints)
        _mount_profiler_routes(app, profiler_ctl, profiler_dir)

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)
    await server.serve()


async def _run_server(
    pipeline_config: PipelineConfig,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    model_name: str | None = None,
    log_level: str = "info",
    client_kwargs: dict[str, Any] | None = None,
) -> None:
    """Compile the pipeline, start stages, and run the OpenAI server.

    This is the async entry point.  For a blocking call use :func:`launch_server`.
    """
    # 0. Check port availability before loading models
    port = _find_available_port(host, port)

    gpu_ids = set(pipeline_config.gpu_placement.values())
    need_multi_process = len(gpu_ids) > 1
    resolved_name = model_name or pipeline_config.name
    logger.info(
        f"GPU placement: {dict(pipeline_config.gpu_placement)} → "
        f"{'multi-process' if need_multi_process else 'single-process'}"
    )

    if need_multi_process:
        from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

        mp_runner = MultiProcessPipelineRunner(pipeline_config)
        startup_timeout = float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "600"))
        await mp_runner.start(timeout=startup_timeout)
        coordinator = mp_runner.coordinator
        logger.info(
            f"Pipeline '{pipeline_config.name}' started "
            f"(multi-process, {len(gpu_ids)} GPU(s))"
        )

        try:

            # Note (Chenyang): Profiler is not enabled in multi-process mode
            # because stages run in separate processes and their control planes
            # are not directly reachable from the HTTP server process.

            await _build_app_and_serve(
                coordinator,
                model_name=resolved_name,
                host=host,
                port=port,
                log_level=log_level,
                client_kwargs=client_kwargs,
            )
        finally:
            logger.info(f"Shutting down pipeline …")
            await mp_runner.stop()
            logger.info(f"Pipeline stopped.")
    else:
        runner = build_pipeline_runner(pipeline_config)
        stage_endpoints = _collect_stage_control_endpoints(runner.stages)
        await runner.start()
        logger.info(
            f"Pipeline '{pipeline_config.name}' started ({len(runner.stages)} stages)"
        )

        try:
            await _build_app_and_serve(
                runner.coordinator,
                model_name=resolved_name,
                host=host,
                port=port,
                log_level=log_level,
                client_kwargs=client_kwargs,
                stage_endpoints=stage_endpoints,
            )
        finally:
            logger.info(f"Shutting down pipeline …")
            await runner.stop()
            logger.info(f"Pipeline stopped.")


def launch_server(
    pipeline_config: PipelineConfig,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    model_name: str | None = None,
    log_level: str = "info",
    client_kwargs: dict[str, Any] | None = None,
) -> None:
    """Blocking helper: compile pipeline and start the OpenAI-compatible server.

    Args:
        pipeline_config: Declarative pipeline configuration.
        host: Bind address for the HTTP server.
        port: Bind port for the HTTP server.
        model_name: Model name reported in ``/v1/models`` responses.
            Defaults to the pipeline name.
        log_level: Uvicorn log level.
        client_kwargs: Extra keyword arguments forwarded to
            :class:`~sglang_omni.client.Client`.
    """
    asyncio.run(
        _run_server(
            pipeline_config,
            host=host,
            port=port,
            model_name=model_name,
            log_level=log_level,
            client_kwargs=client_kwargs,
        )
    )
