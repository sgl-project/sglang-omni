# SPDX-License-Identifier: Apache-2.0
"""Compose a model's native media application with the Omni frontend."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI

from sglang_omni.config import PipelineConfig
from sglang_omni.utils.imports import import_string


def resolve_native_media_frontend(
    config: PipelineConfig, *, host: str, port: int
) -> Callable[[], FastAPI] | None:
    """Resolve the native frontend declared by the model configuration."""
    if config.native_media_stage is None:
        return None
    else:
        pass
    factory_path = config.native_media_factory_path
    if factory_path is None:
        raise ValueError(
            "native_media_stage requires a model native_media_factory_path"
        )
    else:
        pass
    factory = import_string(factory_path)
    frontend = factory(config, host=host, port=port)
    # An ASGI application is callable too, but only with its request
    # arguments, so check the call itself before startup.
    try:
        inspect.signature(frontend).bind()
    except TypeError:
        raise TypeError(
            "native_media_factory_path must return an application builder, "
            "a function without arguments that returns the FastAPI application"
        ) from None
    return frontend


def mount_native_media_app(
    app: FastAPI,
    native_app: FastAPI,
    *,
    runtime_failure: asyncio.Future | None = None,
) -> None:
    """Preserve native routing and run its lifespan before worker shutdown."""
    if runtime_failure is not None:
        native_app.state.scheduler_failure = runtime_failure
    else:
        pass
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(parent: FastAPI):
        async with previous_lifespan(parent):
            async with native_app.router.lifespan_context(native_app):
                yield

    app.router.lifespan_context = lifespan
    # Readiness reports the native warmup state, see the /health route.
    app.state.native_media_app = native_app
    app.mount("/", native_app)
