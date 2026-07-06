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
import json
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
from sglang_omni.config import PipelineConfig
from sglang_omni.models.model_capabilities import get_model_capabilities
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.profiler.event_recorder import get_recorder as _get_event_recorder
from sglang_omni.profiler.profiler_control import ProfilerControlClient
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.protocol import DEFAULT_TTS_BATCH_MAX_ITEMS
from sglang_omni.utils.gpu_compat import apply_gpu_compat_env_defaults
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
    process_plan,
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
        "process_groups": {
            group.name: {"stages": list(group.stage_names), "gpu": group.gpu_id}
            for group in process_plan.groups
        },
        "tp_process_groups": {
            stage_name: list(process_names)
            for stage_name, process_names in process_plan.tp_stage_to_processes.items()
        },
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


def _model_capabilities_log_summary(
    pipeline_config: PipelineConfig,
) -> dict[str, Any] | None:
    architecture = getattr(type(pipeline_config), "architecture", None)
    if architecture is None:
        return None
    capabilities = get_model_capabilities(architecture)
    if capabilities is None:
        return None
    return {
        "architecture": architecture,
        "reference_audio": capabilities.supports_reference_audio,
        "batch_vocoder": capabilities.supports_batch_vocoder,
        "streaming_vocoder": capabilities.supports_streaming_vocoder,
        "cuda_graph": capabilities.supports_cuda_graph,
        "torch_compile": capabilities.supports_torch_compile,
    }


def _log_model_capabilities(pipeline_config: PipelineConfig) -> None:
    try:
        summary = _model_capabilities_log_summary(pipeline_config)
    except Exception:
        logger.warning(
            "Failed to resolve model capabilities for startup log",
            exc_info=True,
        )
        return
    if summary is not None:
        logger.info("Model capabilities: %s", json.dumps(summary, sort_keys=True))


class StartReq(BaseModel):
    run_id: str | None = None
    trace_path_template: str | None = None
    config: dict[str, Any] | None = None
    event_dir: str | None = None
    enable_torch: bool = True


class StopReq(BaseModel):
    run_id: str | None = None


class StartRequestProfileReq(BaseModel):
    run_id: str | None = None
    event_dir: str | None = None


def _default_event_dir(profiler_dir: str, run_id: str) -> str:
    return os.path.join(profiler_dir, run_id, "events")


def _mount_profiler_routes(
    app, profiler_ctl: ProfilerControlClient, profiler_dir: str | None
) -> None:
    router = APIRouter()

    @router.post("/start_profile")
    async def start(req: StartReq):
        run_id = req.run_id or _default_run_id()
        event_dir = req.event_dir
        if event_dir is None and profiler_dir is not None:
            event_dir = _default_event_dir(profiler_dir, run_id)
        if req.enable_torch:
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
        else:
            if event_dir is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "event_dir is required when enable_torch=false and "
                        "SGLANG_TORCH_PROFILER_DIR is not set"
                    ),
                )
            tpl = req.trace_path_template or ""
        if event_dir is not None:
            try:
                _get_event_recorder().start(
                    run_id=run_id, event_dir=event_dir, stage="coordinator"
                )
            except Exception:
                logger.warning(
                    "Failed to start coordinator request event recorder",
                    exc_info=True,
                )
        await profiler_ctl.broadcast_start(
            run_id=run_id,
            trace_path_template=tpl,
            config=req.config,
            event_dir=event_dir,
            enable_torch=req.enable_torch,
        )
        return {
            "run_id": run_id,
            "trace_path_template": tpl,
            "event_dir": event_dir,
            "enable_torch": req.enable_torch,
        }

    @router.post("/start_request_profile")
    async def start_request(req: StartRequestProfileReq):
        """Start request-level (JSONL) event profiling only (no torch trace)."""
        run_id = req.run_id or _default_run_id()
        event_dir = req.event_dir
        if event_dir is None:
            if profiler_dir is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "event_dir is required when "
                        "SGLANG_TORCH_PROFILER_DIR is not set"
                    ),
                )
            event_dir = _default_event_dir(profiler_dir, run_id)
        try:
            _get_event_recorder().start(
                run_id=run_id, event_dir=event_dir, stage="coordinator"
            )
        except Exception:
            logger.warning(
                "Failed to start coordinator request event recorder",
                exc_info=True,
            )
        await profiler_ctl.broadcast_start(
            run_id=run_id,
            trace_path_template="",
            event_dir=event_dir,
            enable_torch=False,
        )
        return {"run_id": run_id, "event_dir": event_dir}

    @router.post("/stop_profile")
    async def stop(req: StopReq):
        # run_id=None is a wildcard (stop whatever's active).
        run_id = req.run_id
        recorder = _get_event_recorder()
        active = recorder.active_run_id() if recorder.is_active() else None
        if recorder.is_active() and (run_id is None or active == run_id):
            recorder.stop(run_id=active)
        await profiler_ctl.broadcast_stop(run_id=run_id)
        return {"run_id": run_id or active}

    @router.post("/stop_request_profile")
    async def stop_request(req: StopReq):
        """Stop request-level event profiling."""
        run_id = req.run_id
        recorder = _get_event_recorder()
        active = recorder.active_run_id() if recorder.is_active() else None
        if recorder.is_active() and (run_id is None or active == run_id):
            recorder.stop(run_id=active)
        await profiler_ctl.broadcast_stop(run_id=run_id)
        return {"run_id": run_id or active}

    app.include_router(router)


async def _run_server(
    pipeline_config: PipelineConfig,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    model_name: str | None = None,
    log_level: str = "info",
    client_kwargs: dict[str, Any] | None = None,
    enable_realtime: bool = False,
    allowed_local_media_path: str | None = None,
    allowed_media_domains: list[str] | None = None,
    tts_batch_max_items: int = DEFAULT_TTS_BATCH_MAX_ITEMS,
) -> None:
    """Start the pipeline and run the OpenAI server.

    This is the async entry point.  For a blocking call use :func:`launch_server`.
    """
    # 0. Check port availability before loading models
    port = _find_available_port(host, port)

    mp_runner = MultiProcessPipelineRunner(pipeline_config)
    startup_timeout = float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "600"))
    await mp_runner.start(timeout=startup_timeout)
    coordinator = mp_runner.coordinator

    # Plans are resolved once inside ``mp_runner.start()`` (which applies
    # stage fusion); read them back from the runner for logging rather than
    # recomputing on the un-fused config.
    placement_plan = mp_runner.prep.placement_plan
    process_plan = mp_runner.prep.process_plan
    gpu_ids = set(placement_plan.gpus)
    placement_summary = _placement_log_summary(
        placement_plan,
        process_plan,
        pipeline_config,
    )
    logger.info(
        f"Resolved placement/topology plan: placement={placement_summary}",
    )
    _log_model_capabilities(pipeline_config)
    logger.info(
        "Pipeline '%s' started (%d GPU(s))",
        pipeline_config.name,
        len(gpu_ids),
    )

    try:
        cl_kwargs = client_kwargs or {}
        client = Client(coordinator, **cl_kwargs)
        app = create_app(
            client,
            model_name=model_name or pipeline_config.name,
            requires_uploaded_voice_for_named_voice=(
                pipeline_config.requires_uploaded_voice_for_named_voice()
            ),
            supports_uploaded_voice_references=(
                pipeline_config.supports_uploaded_voice_references()
            ),
            enable_realtime=enable_realtime,
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
            tts_batch_max_items=tts_batch_max_items,
            architectures=[pipeline_config.architecture],
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
    enable_realtime: bool = False,
    allowed_local_media_path: str | None = None,
    allowed_media_domains: list[str] | None = None,
    tts_batch_max_items: int = DEFAULT_TTS_BATCH_MAX_ITEMS,
) -> None:
    """Blocking helper: start the pipeline and OpenAI-compatible server.

    Args:
        pipeline_config: Declarative pipeline configuration.
        host: Bind address for the HTTP server.
        port: Bind port for the HTTP server.
        model_name: Model name reported in /v1/models responses.
            Defaults to the pipeline name.
        log_level: Uvicorn log level.
        client_kwargs: Extra keyword arguments forwarded to
            :class:`~sglang_omni.client.Client`.
        enable_realtime: If True, mount the WebSocket ``/v1/realtime``
            endpoint (OpenAI Realtime API).
        allowed_local_media_path: Directory allowed for ``file://`` media
            references in TTS requests.
        allowed_media_domains: Domains allowed for remote TTS reference audio.
        tts_batch_max_items: Maximum items accepted by
            ``/v1/audio/speech/batch``.
    """
    apply_gpu_compat_env_defaults()
    asyncio.run(
        _run_server(
            pipeline_config,
            host=host,
            port=port,
            model_name=model_name,
            log_level=log_level,
            client_kwargs=client_kwargs,
            enable_realtime=enable_realtime,
            allowed_local_media_path=allowed_local_media_path,
            allowed_media_domains=allowed_media_domains,
            tts_batch_max_items=tts_batch_max_items,
        )
    )
