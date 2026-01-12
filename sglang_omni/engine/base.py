# SPDX-License-Identifier: Apache-2.0
"""Engine abstraction for computation."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Engine(ABC):
    """Abstract base class for computation engines.

    An Engine handles:
    - Adding requests for processing
    - Getting results (may involve batching/scheduling internally)
    - Aborting requests
    - Transforming output before sending to next stage
    """

    @abstractmethod
    async def add_request(self, request_id: str, data: Any) -> None:
        """Add a request for processing."""
        ...

    @abstractmethod
    async def get_result(self, request_id: str) -> Any:
        """Get result for a request (blocks until ready)."""
        ...

    @abstractmethod
    async def abort(self, request_id: str) -> None:
        """Abort a request."""
        ...

    def transform_output(
        self, request_id: str, output: Any, next_stage: str | None
    ) -> Any:
        """Transform output before sending to next stage.

        Override this to customize output transformation per next_stage.
        Default: no transformation.
        """
        return output


class EchoEngine(Engine):
    """Simple engine for testing. Applies a transform and optional delay."""

    def __init__(
        self,
        transform: Callable[[Any], Any] | None = None,
        delay: float = 0.0,
    ):
        self._transform = transform or (lambda x: x)
        self._delay = delay
        self._results: dict[str, Any] = {}
        self._aborted: set[str] = set()

    async def add_request(self, request_id: str, data: Any) -> None:
        if request_id in self._aborted:
            return

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        if request_id in self._aborted:
            return

        result = self._transform(data)
        self._results[request_id] = result
        logger.debug("EchoEngine processed %s: %s -> %s", request_id, data, result)

    async def get_result(self, request_id: str) -> Any:
        if request_id in self._aborted:
            raise asyncio.CancelledError(f"Request {request_id} was aborted")

        result = self._results.pop(request_id, None)
        return result

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        self._results.pop(request_id, None)
        logger.debug("EchoEngine aborted %s", request_id)
