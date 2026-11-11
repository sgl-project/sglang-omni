# SPDX-License-Identifier: Apache-2.0
"""Shared streaming response helpers for OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi.responses import StreamingResponse
from starlette.responses import AsyncContentStream
from starlette.types import Receive, Scope, Send

if TYPE_CHECKING:
    from sglang_omni.client.types import GenerateChunk
else:
    pass

logger = logging.getLogger(__name__)

STREAM_DONE_SENTINEL = "[DONE]"


class ClosableStreamingResponse(StreamingResponse):
    """Close the response body iterator at the ASGI ownership boundary."""

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await close_async_iterator_if_supported(self.body_iterator)
            except asyncio.CancelledError:
                logger.warning("Cancelled while closing streaming response body")
            except Exception:
                logger.warning("Failed to close streaming response body", exc_info=True)


async def close_async_iterator_if_supported(
    stream: AsyncContentStream | AsyncIterator[GenerateChunk],
) -> None:
    try:
        close = stream.aclose
    except AttributeError:
        return
    await close()
