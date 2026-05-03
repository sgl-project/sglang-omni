# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server from a PipelineConfig.

Usage (programmatic)::

    from sglang_omni_v1.serve.launcher import launch_server
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

from sglang_omni_v1.client import Client
from sglang_omni_v1.config import PipelineConfig, compile_pipeline
from sglang_omni_v1.profiler.profiler_control import ProfilerControlClient
from sglang_omni_v1.serve.openai_api import create_app

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
    logger.warning("Port %d is already in use on %s.", port, host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        free_port = s.getsockname()[1]
    logger.warning("Using port %d instead.", free_port)
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
        ep = st.control_plane.recv_endpoint
        if not ep:
            raise RuntimeError(f"Cannot resolve control endpoint for stage={st.name}")
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

    # Determine whether we need multi-process mode.  This is true when
    # stages span more than one GPU *or* any stage uses tensor parallelism.
    gpu_ids: set[int] = set()
    for v in pipeline_config.gpu_placement.values():
        if isinstance(v, list):
            gpu_ids.update(v)
        else:
            gpu_ids.add(v)
    any_tp = any(s.tp_size > 1 for s in pipeline_config.stages)
    needs_mp = len(gpu_ids) > 1 or any_tp
    logger.info(
        "GPU placement: %s → %s",
        dict(pipeline_config.gpu_placement),
        "multi-process" if needs_mp else "single-process",
    )

    if needs_mp:
        from sglang_omni_v1.pipeline.mp_runner import MultiProcessPipelineRunner

        mp_runner = MultiProcessPipelineRunner(pipeline_config)
        startup_timeout = float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "600"))
        await mp_runner.start(timeout=startup_timeout)
        coordinator = mp_runner.coordinator
        logger.info(
            "Pipeline '%s' started (multi-process, %d GPU(s))",
            pipeline_config.name,
            len(gpu_ids),
        )

        try:
            cl_kwargs = client_kwargs or {}
            client = Client(coordinator, **cl_kwargs)
            app = create_app(
                client,
                model_name=model_name or pipeline_config.name,
            )

            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level=log_level,
                timeout_keep_alive=120,
            )
            server = uvicorn.Server(config)
            await server.serve()
        finally:
            logger.info("Shutting down pipeline …")
            await mp_runner.stop()
            logger.info("Pipeline stopped.")
    else:
        coordinator, stages = compile_pipeline(pipeline_config)
        stage_endpoints = _collect_stage_control_endpoints(stages)

        # Start coordinator + all stages as async tasks
        await coordinator.start()
        completion_task = asyncio.create_task(coordinator.run_completion_loop())
        stage_tasks = [asyncio.create_task(s.run()) for s in stages]
        logger.info(
            "Pipeline '%s' started (%d stages)",
            pipeline_config.name,
            len(stages),
        )

        try:
            cl_kwargs = client_kwargs or {}
            client = Client(coordinator, **cl_kwargs)
            app = create_app(
                client,
                model_name=model_name or pipeline_config.name,
            )

            profiler_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR")
            profiler_ctl = ProfilerControlClient(stage_endpoints)
            _mount_profiler_routes(app, profiler_ctl, profiler_dir)

            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level=log_level,
                timeout_keep_alive=120,
            )
            server = uvicorn.Server(config)
            await server.serve()
        finally:
            logger.info("Shutting down pipeline …")
            for t in stage_tasks:
                t.cancel()
            completion_task.cancel()
            await coordinator.stop()
            logger.info("Pipeline stopped.")


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
            :class:`~sglang_omni_v1.client.Client`.
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
