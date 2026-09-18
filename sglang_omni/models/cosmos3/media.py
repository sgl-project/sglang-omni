# SPDX-License-Identifier: Apache-2.0
"""Construct the native SGLang media application for Cosmos3."""

from __future__ import annotations

import inspect
import socket
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from sglang_omni.config import PipelineConfig
from sglang_omni.models.cosmos3.checkpoint import resolve_native_checkpoint
from sglang_omni.models.cosmos3.stages import native_server_kwargs


def _unused_port(exclude: set[int]) -> int:
    for _ in range(100):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in exclude and port + 1 not in exclude:
            return port
    raise RuntimeError("Could not select distinct native runtime ports")


def prepare_native_media_app(
    config: PipelineConfig, *, host: str, port: int
) -> FastAPI | None:
    """Resolve one native runtime before its owning stage starts."""
    if config.native_media_stage is None:
        return None
    stage = next(s for s in config.stages if s.name == config.native_media_stage)
    if not stage.allow_child_processes or stage.tp_size != 1:
        raise ValueError("The native media stage must own its child processes")
    if not isinstance(stage.gpu, int):
        raise ValueError("The native media stage requires one explicit GPU")
    kwargs: dict[str, Any] = dict(
        getattr(stage.factory, "server_args_overrides", None) or {}
    )
    # A stage-local model_path is also supported by the SDK factory resolver.
    model_path = getattr(stage.factory, "model_path", config.model_path)
    native_kwargs = native_server_kwargs(
        model_path, stage.gpu, kwargs, stage.runtime_gpu_ids
    )
    from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
    from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    if (
        "worker_failure"
        not in inspect.signature(AsyncSchedulerClient.initialize).parameters
    ):
        raise RuntimeError(
            "Cosmos3 native media requires SGLang scheduler owner-failure "
            "support. Install the native lifecycle prerequisites before serving."
        )

    excluded = {port, port + 1}
    kwargs.setdefault("scheduler_port", _unused_port(excluded))
    excluded.update({kwargs["scheduler_port"], kwargs["scheduler_port"] + 1})
    kwargs.setdefault("master_port", _unused_port(excluded))
    excluded.add(kwargs["master_port"])
    kwargs.setdefault("nccl_port", _unused_port(excluded))
    output_dir = str(Path(getattr(stage.factory, "output_dir", "outputs")).resolve())
    kwargs.update(host=host, port=port, strict_ports=True, output_path=output_dir)
    native_kwargs.update(kwargs)
    native_kwargs = resolve_native_checkpoint(native_kwargs)
    native_args = ServerArgs.from_kwargs(**native_kwargs)
    # Explicit ports make startup fail if another process takes them.
    # Both clients must address the same scheduler, never silently choose another.
    set_global_server_args(native_args)
    app = create_app(native_args)
    # Publish the snapshot and runtime addresses together only after frontend
    # construction succeeds. A moving Hub branch must not resolve to different
    # checkpoints in the frontend and the generation worker.
    kwargs["served_model_name"] = native_kwargs["served_model_name"]
    stage.factory.model_path = native_kwargs["model_path"]
    stage.factory.server_args_overrides = kwargs
    return app
