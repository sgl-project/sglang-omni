# SPDX-License-Identifier: Apache-2.0
"""Construct the native SGLang media application for Cosmos3."""

from __future__ import annotations

import inspect
import socket
from functools import partial
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI

from sglang_omni.config import PipelineConfig
from sglang_omni.models.cosmos3.stages import native_server_kwargs


def unused_port(exclude: set[int]) -> int:
    for _ in range(100):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in exclude and port + 1 not in exclude:
            return port
        else:
            pass
    raise RuntimeError("Could not select distinct native runtime ports")


def prepare_native_media_app(
    config: PipelineConfig, *, host: str, port: int
) -> Callable[[], FastAPI] | None:
    """Resolve shared runtime settings before importing the native frontend."""
    if config.native_media_stage is None:
        return None
    else:
        pass
    stage = next(s for s in config.stages if s.name == config.native_media_stage)
    if not stage.allow_child_processes or stage.tp_size != 1:
        raise ValueError("The native media stage must own its child processes")
    else:
        pass
    if not isinstance(stage.gpu, int):
        raise ValueError("The native media stage requires one explicit GPU")
    else:
        pass

    kwargs: dict[str, Any] = dict(
        getattr(stage.factory, "server_args_overrides", None) or {}
    )
    excluded = {port, port + 1}
    kwargs.setdefault("scheduler_port", unused_port(excluded))
    excluded.update({kwargs["scheduler_port"], kwargs["scheduler_port"] + 1})
    kwargs.setdefault("master_port", unused_port(excluded))
    excluded.add(kwargs["master_port"])
    kwargs.setdefault("nccl_port", unused_port(excluded))
    output_dir = str(Path(getattr(stage.factory, "output_dir", "outputs")).resolve())
    kwargs.update(host=host, port=port, strict_ports=True, output_path=output_dir)
    server_kwargs = native_server_kwargs(
        config.model_path, stage.gpu, kwargs, stage.runtime_gpu_ids
    )
    # Explicit ports make startup fail if another process takes them.
    # Both clients must address the same scheduler, never silently choose another.
    stage.factory.server_args_overrides = kwargs
    return partial(build_native_media_app, server_kwargs)


def build_native_media_app(server_kwargs: dict[str, Any]) -> FastAPI:
    """Import the native runtime and construct its application."""
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
            "Cosmos3 native media requires SGLang scheduler owner-failure support "
            "(worker_failure in AsyncSchedulerClient.initialize), which the "
            "installed SGLang does not provide. Without it, a request in flight "
            "when the generation worker dies waits for its client deadline. "
            "Install an SGLang build with this capability, then restart."
        )
    else:
        pass
    native_args = ServerArgs.from_kwargs(**server_kwargs)
    set_global_server_args(native_args)
    return create_app(native_args)
