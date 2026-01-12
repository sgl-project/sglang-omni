# SPDX-License-Identifier: Apache-2.0
"""Input handlers for different input patterns."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

logger = logging.getLogger(__name__)


class InputHandler(ABC):
    """Abstract base class for input handling.

    Handles input aggregation patterns like:
    - Direct pass-through (single input)
    - Fan-in (wait for multiple inputs then merge)
    """

    @abstractmethod
    def receive(self, request_id: str, from_stage: str, data: Any) -> Any | None:
        """Receive data from a stage.

        Returns:
            Merged data if ready for processing, None if still waiting.
        """
        ...

    @abstractmethod
    def cancel(self, request_id: str) -> None:
        """Cancel a pending request (e.g., on abort)."""
        ...


class DirectInput(InputHandler):
    """Direct pass-through. Single input, no aggregation."""

    def receive(self, request_id: str, from_stage: str, data: Any) -> Any:
        return data

    def cancel(self, request_id: str) -> None:
        pass  # Nothing to clean up


class AggregatedInput(InputHandler):
    """Fan-in pattern. Wait for inputs from multiple sources then merge.

    Example use case: A stage that receives from both encoder and decoder,
    waits for both, then merges them.
    """

    def __init__(
        self,
        sources: set[str],
        merge: Callable[[dict[str, Any]], Any],
    ):
        """Initialize aggregated input handler.

        Args:
            sources: Set of stage names we expect input from
            merge: Function to merge inputs. Receives dict {stage_name: data}
        """
        self._sources = sources
        self._merge = merge
        self._pending: dict[str, dict[str, Any]] = (
            {}
        )  # request_id -> {from_stage: data}

    def receive(self, request_id: str, from_stage: str, data: Any) -> Any | None:
        if from_stage not in self._sources:
            logger.warning(
                "AggregatedInput: unexpected source %s for request %s",
                from_stage,
                request_id,
            )
            return None

        # Initialize pending dict for this request
        if request_id not in self._pending:
            self._pending[request_id] = {}

        # Store data
        self._pending[request_id][from_stage] = data

        # Check if all sources received
        if set(self._pending[request_id].keys()) == self._sources:
            # All inputs received, merge and return
            inputs = self._pending.pop(request_id)
            merged = self._merge(inputs)
            logger.debug(
                "AggregatedInput: merged inputs for %s from %s",
                request_id,
                list(inputs.keys()),
            )
            return merged

        # Still waiting for more inputs
        logger.debug(
            "AggregatedInput: waiting for %s (got %s, need %s)",
            request_id,
            list(self._pending[request_id].keys()),
            self._sources,
        )
        return None

    def cancel(self, request_id: str) -> None:
        self._pending.pop(request_id, None)
