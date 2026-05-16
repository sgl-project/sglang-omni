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
from contextlib import suppress
from typing import Any

import uvicorn
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from sglang_omni.client import Client
from sglang_omni.config import (
    PipelineConfig,
    build_stage_placement_plan,
    compile_pipeline_core,
    resolve_pipeline_process_mode,
)
from sglang_omni.profiler.profiler_control import ProfilerControlClient
from sglang_omni.serve.openai_api import create_app
from sglang_omni.utils.gpu_memory import (
    GpuDeviceInfo,
    format_bytes_gib,
    get_gpu_device_info,
)

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


def _stage_runtime_log_summary(pipeline_config: PipelineConfig) -> dict[str, Any]:
    """Build stage placement and runtime budget fields for startup logs."""

    summary: dict[str, Any] = {}
    for stage in pipeline_config.stages:
        resources = stage.runtime.resources
        mem_fraction = stage.runtime.sglang_server_args.mem_fraction_static
        if stage.gpu is None and resources.total_gpu_memory_fraction is None:
            continue
        summary[stage.name] = {
            "gpu": stage.gpu,
            "total_gpu_memory_fraction": resources.total_gpu_memory_fraction,
            "mem_fraction_static": mem_fraction,
        }
    return summary


def _format_gpu_device_info(info: GpuDeviceInfo) -> dict[str, Any]:
    return {
        "device_id": info.device_id,
        "name": info.name or "unknown",
        "total_memory": (
            format_bytes_gib(info.total_memory_bytes)
            if info.total_memory_bytes is not None
            else "unknown"
        ),
    }


def _placement_log_summary(
    placement_plan,
    pipeline_config: PipelineConfig,
) -> dict[str, Any]:
    """Build the resolved startup placement summary.

    The summary includes topology, stage placement, stage budgets, per-GPU
    totals, and best-effort hardware metadata.
    """

    hardware = {
        gpu_id: _format_gpu_device_info(get_gpu_device_info(gpu_id))
        for gpu_id in sorted(placement_plan.gpus)
    }
    return {
        "topology": pipeline_config.config_cls or type(pipeline_config).__name__,
        "pipeline": pipeline_config.name,
        "stage_runtime": _stage_runtime_log_summary(pipeline_config),
        "gpus": {
            gpu_id: {
                "hardware": hardware[gpu_id],
                "stages": list(gpu.stage_names),
                "total_gpu_memory_fraction": round(gpu.total_gpu_memory_fraction, 3),
                "missing_fraction_stages": list(gpu.missing_fraction_stage_names),
            }
            for gpu_id, gpu in placement_plan.gpus.items()
        },
    }


class StartReq(BaseModel):
    run_id: str | None = None
    trace_path_template: str | None = None
    config: dict[str, Any] | None = None


class StopReq(BaseModel):
    run_id: str | None = None


def _mount_profiler_routes(
    app, profiler_ctl: ProfilerControlClient, profiler_dir: str | None
) -> None:
    router = APIRouter()

    @router.post("/start_profile")
    async def start(req: StartReq):
        run_id = req.run_id or _default_run_id()
        if req.trace_path_template is not None:
            tpl = req.trace_path_template
        elif profiler_dir is not None:
            tpl = _default_template(profiler_dir, run_id)
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "trace_path_template is required when "
                    "SGLANG_TORCH_PROFILER_DIR is not set"
                ),
            )
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

    placement_plan = build_stage_placement_plan(pipeline_config)
    needs_mp = resolve_pipeline_process_mode(pipeline_config, placement_plan)
    gpu_ids = set(placement_plan.gpus)
    process_mode = "multi-process" if needs_mp else "single-process"
    placement_summary = _placement_log_summary(placement_plan, pipeline_config)
    logger.info(
        f"Resolved placement plan: process_mode={process_mode} "
        f"placement={placement_summary}"
    )

    if needs_mp:
        from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

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
            profiler_dir = os.environ.get("SGLANG_TORCH_PROFILER_DIR")
            profiler_ctl = ProfilerControlClient(mp_runner.stage_control_endpoints)
            _mount_profiler_routes(app, profiler_ctl, profiler_dir)

            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level=log_level,
                timeout_keep_alive=120,
            )
            server = uvicorn.Server(config)
            await _serve_with_failure_watch(server, [mp_runner.wait_failed()])
        finally:
            logger.info("Shutting down pipeline …")
            await mp_runner.stop()
            logger.info("Pipeline stopped.")
    else:
        compiled = compile_pipeline_core(pipeline_config)
        coordinator = compiled.coordinator
        stages = compiled.stages
        runtime_dir = compiled.runtime_dir
        completion_task = None
        stage_tasks = []
        coordinator_started = False

        try:
            stage_endpoints = _collect_stage_control_endpoints(stages)
            await coordinator.start()
            coordinator_started = True
            completion_task = asyncio.create_task(coordinator.run_completion_loop())
            stage_tasks = [asyncio.create_task(s.run()) for s in stages]
            logger.info(
                "Pipeline '%s' started (%d stages)",
                pipeline_config.name,
                len(stages),
            )

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
            runtime_tasks = [completion_task, *stage_tasks]
            await _serve_with_failure_watch(server, runtime_tasks)
        finally:
            logger.info("Shutting down pipeline …")
            for t in stage_tasks:
                t.cancel()
            if completion_task is not None:
                completion_task.cancel()
                with suppress(asyncio.CancelledError):
                    await completion_task
            if stage_tasks:
                await asyncio.gather(*stage_tasks, return_exceptions=True)
            try:
                if coordinator_started:
                    await coordinator.stop()
                else:
                    with suppress(Exception):
                        await coordinator.stop()
            finally:
                if runtime_dir is not None:
                    runtime_dir.close()
            logger.info("Pipeline stopped.")


async def _serve_with_failure_watch(
    server: uvicorn.Server,
    runtime_watchers,
) -> None:
    server_task = asyncio.create_task(server.serve())
    watcher_tasks = [
        watcher if isinstance(watcher, asyncio.Task) else asyncio.create_task(watcher)
        for watcher in runtime_watchers
        if watcher is not None
    ]
    try:
        done, _ = await asyncio.wait(
            [server_task, *watcher_tasks],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if server_task in done:
            await server_task
            return

        server.should_exit = True
        with suppress(asyncio.CancelledError):
            await server_task

        for task in done:
            if task is server_task:
                continue
            if task.cancelled():
                raise RuntimeError("Pipeline runtime task was cancelled")
            exc = task.exception()
            if exc is not None:
                raise exc
            raise RuntimeError("Pipeline runtime task exited unexpectedly")
    finally:
        for task in watcher_tasks:
            if not task.done():
                task.cancel()


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
        model_name: Model name reported in /v1/models responses.
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
