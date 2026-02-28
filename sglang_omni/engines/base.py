# SPDX-License-Identifier: Apache-2.0
"""Engine abstraction for computation."""

import logging
from abc import ABC, abstractmethod
from typing import Any

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
