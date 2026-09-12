# SPDX-License-Identifier: Apache-2.0
"""Construct the native SGLang media application for Cosmos3."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from sglang_omni.config import PipelineConfig


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
    from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )

    from sglang_omni.models.cosmos3.stages import native_server_kwargs

    kwargs: dict[str, Any] = dict(
        getattr(stage.factory, "server_args_overrides", None) or {}
    )
    excluded = {port}
    kwargs.setdefault("scheduler_port", _unused_port(excluded))
    excluded.update({kwargs["scheduler_port"], kwargs["scheduler_port"] + 1})
    kwargs.setdefault("master_port", _unused_port(excluded))
    excluded.add(kwargs["master_port"])
    kwargs.setdefault("nccl_port", _unused_port(excluded))
    output_dir = str(Path(getattr(stage.factory, "output_dir", "outputs")).resolve())
    kwargs.update(host=host, port=port, strict_ports=True, output_path=output_dir)
    native_args = ServerArgs.from_kwargs(
        **native_server_kwargs(
            config.model_path, stage.gpu, kwargs, stage.runtime_gpu_ids
        )
    )
    # Explicit ports make startup fail if another process takes them.
    # Both clients must address the same scheduler, never silently choose another.
    stage.factory.server_args_overrides = kwargs
    set_global_server_args(native_args)
    return create_app(native_args)
