# SPDX-License-Identifier: Apache-2.0
"""Compose a model's native media application with the Omni frontend."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from sglang_omni.config import PipelineConfig
from sglang_omni.utils.imports import import_string


def prepare_native_media_app(
    config: PipelineConfig, *, host: str, port: int
) -> FastAPI | None:
    """Construct the native frontend declared by the model configuration."""
    if config.native_media_stage is None:
        return None
    factory_path = config.native_media_factory_path
    if factory_path is None:
        raise ValueError(
            "native_media_stage requires a model native_media_factory_path"
        )
    factory = import_string(factory_path)
    native_app = factory(config, host=host, port=port)
    if not isinstance(native_app, FastAPI):
        raise TypeError("native_media_factory_path must return a FastAPI application")
    return native_app


def mount_native_media_app(
    app: FastAPI,
    native_app: FastAPI,
    *,
    runtime_failure: asyncio.Future | None = None,
    stop_runtime: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Preserve native routing and run its lifespan before worker shutdown."""
    if runtime_failure is not None:
        native_app.state.scheduler_failure = runtime_failure
    if stop_runtime is not None:
        native_app.state.stop_runtime = stop_runtime
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(parent: FastAPI):
        async with previous_lifespan(parent):
            async with native_app.router.lifespan_context(native_app):
                yield

    app.router.lifespan_context = lifespan
    app.mount("/", native_app)
