# SPDX-License-Identifier: Apache-2.0
"""Input handlers for different input patterns."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Callable

from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

ExpectedSourcesFn = Callable[[str, str, StagePayload], str | Iterable[str] | None]


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
        expected_sources_fn: ExpectedSourcesFn | None = None,
    ):
        self._sources = sources
        self._merge = merge
        self._expected_sources_fn = expected_sources_fn
        self._pending: dict[str, dict[str, Any]] = {}
        self._expected_sources: dict[str, set[str]] = {}

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

        expected_sources = self._expected_sources.get(request_id)
        if expected_sources is None and self._expected_sources_fn is not None:
            resolved = self._expected_sources_fn(request_id, from_stage, data)
            if resolved is not None:
                expected_sources = self._normalize_expected_sources(
                    request_id,
                    resolved,
                )
                self._expected_sources[request_id] = expected_sources
        elif expected_sources is None:
            expected_sources = self._sources

        if expected_sources is None:
            return None

        pending_sources = set(self._pending[request_id])
        unexpected = pending_sources - expected_sources
        if unexpected:
            raise ValueError(
                "AggregatedInput: received sources outside expected fan-in for "
                f"request {request_id}: {sorted(unexpected)}. "
                f"Expected: {sorted(expected_sources)}"
            )

        if pending_sources == expected_sources:
            inputs = self._pending.pop(request_id)
            self._expected_sources.pop(request_id, None)
            return self._merge(inputs)

        return None

    def _normalize_expected_sources(
        self,
        request_id: str,
        sources: str | Iterable[str],
    ) -> set[str]:
        if isinstance(sources, str):
            expected = {sources}
        else:
            try:
                expected = set(sources)
            except TypeError as exc:
                raise ValueError(
                    "AggregatedInput: dynamic fan-in resolver returned unsupported "
                    f"sources for request {request_id}: {sources!r}"
                ) from exc
        if not expected:
            raise ValueError(
                "AggregatedInput: dynamic fan-in resolver returned no sources "
                f"for request {request_id}"
            )
        non_string = [source for source in expected if not isinstance(source, str)]
        if non_string:
            raise ValueError(
                "AggregatedInput: dynamic fan-in resolver returned non-string "
                f"sources for request {request_id}: {non_string!r}"
            )
        unknown = expected - self._sources
        if unknown:
            raise ValueError(
                "AggregatedInput: dynamic fan-in resolver returned sources "
                f"outside static wait_for for request {request_id}: "
                f"{sorted(unknown)}. Allowed: {sorted(self._sources)}"
            )
        return expected

    def cancel(self, request_id: str) -> None:
        self._pending.pop(request_id, None)
        self._expected_sources.pop(request_id, None)
