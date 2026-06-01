# SPDX-License-Identifier: Apache-2.0
"""Shared async helpers for executors that do opportunistic micro-batching."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def run_batch_loop(
    step_fn: Callable[[], Awaitable[None]],
    log_prefix: str,
) -> None:
    """Loop calling ``step_fn``; log and continue on any non-cancellation error."""
    while True:
        try:
            await step_fn()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("%s batch loop iteration failed", log_prefix)
