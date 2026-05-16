# SPDX-License-Identifier: Apache-2.0
"""Input handlers for different input patterns."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


class InputHandler(ABC):
    """Abstract base class for input handling."""

    @abstractmethod
    def receive(
        self, request_id: str, from_stage: str, data: Any
    ) -> StagePayload | None:
        """Receive data from a stage.
        Returns merged payload if ready, None if still waiting.
        """
        ...

    @abstractmethod
    def cancel(self, request_id: str) -> None: ...


class DirectInput(InputHandler):
    """Direct pass-through. Single input, no aggregation."""

    def receive(self, request_id: str, from_stage: str, data: Any) -> StagePayload:
        return data

    def cancel(self, request_id: str) -> None:
        pass


class AggregatedInput(InputHandler):
    """Fan-in: wait for inputs from multiple sources then merge."""

    def __init__(
        self,
        sources: set[str],
        merge: Callable[[dict[str, StagePayload]], StagePayload],
    ):
        self._sources = sources
        self._merge = merge
        self._pending: dict[str, dict[str, Any]] = {}

    def receive(
        self, request_id: str, from_stage: str, data: Any
    ) -> StagePayload | None:
        if from_stage not in self._sources:
            logger.warning(
                "AggregatedInput: unexpected source %s for request %s",
                from_stage,
                request_id,
            )
            return None

        if request_id not in self._pending:
            self._pending[request_id] = {}
        self._pending[request_id][from_stage] = data

        if set(self._pending[request_id].keys()) == self._sources:
            inputs = self._pending.pop(request_id)
            return self._merge(inputs)

        return None

    def cancel(self, request_id: str) -> None:
        self._pending.pop(request_id, None)
