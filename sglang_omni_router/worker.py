# SPDX-License-Identifier: Apache-2.0
"""Worker state tracked by the external Omni router."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Literal
from urllib.parse import quote, urlsplit

from sglang_omni_router.config import Capability, WorkerConfig

WorkerState = Literal["dead", "healthy", "unknown", "unhealthy"]
HEALTH_STATE_DEAD: WorkerState = "dead"
HEALTH_STATE_HEALTHY: WorkerState = "healthy"
HEALTH_STATE_UNKNOWN: WorkerState = "unknown"
HEALTH_STATE_UNHEALTHY: WorkerState = "unhealthy"

logger = logging.getLogger(__name__)


def worker_id_from_url(url: str) -> str:
    return quote(url, safe="")


def display_id_from_url(url: str) -> str:
    parsed = urlsplit(url)
    return parsed.netloc or url


@dataclass
class Worker:
    config: WorkerConfig
    worker_id: str = field(init=False)
    display_id: str = field(init=False)
    active_requests: int = 0
    state: WorkerState = HEALTH_STATE_UNKNOWN
    disabled: bool = False
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_status_code: int | None = None
    last_error: str | None = None
    last_checked_at: datetime | None = None

    def __post_init__(self) -> None:
        self.worker_id = worker_id_from_url(self.url)
        self.display_id = display_id_from_url(self.url)

    @property
    def url(self) -> str:
        return self.config.url

    @property
    def model(self) -> str | None:
        return self.config.model

    @property
    def capabilities(self) -> set[Capability]:
        return self.config.capabilities

    @property
    def is_healthy(self) -> bool:
        return self.state == HEALTH_STATE_HEALTHY

    @property
    def is_dead(self) -> bool:
        return self.state == HEALTH_STATE_DEAD

    @property
    def is_routable(self) -> bool:
        return self.is_healthy and not self.disabled

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def replace_config(self, config: WorkerConfig) -> None:
        if config.url != self.url:
            raise ValueError("worker URL cannot be changed")
        self.config = config

    def mark_dead(self, *, error: str | None = None) -> None:
        previous_state = self.state
        self.state = HEALTH_STATE_DEAD
        if error is not None:
            self.last_error = error
        self._log_state_transition(previous_state, self.state)

    def clear_dead(self) -> None:
        previous_state = self.state
        if self.is_dead:
            self.state = HEALTH_STATE_UNKNOWN
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self._log_state_transition(previous_state, self.state)

    def set_disabled(self, disabled: bool) -> None:
        if self.disabled != disabled:
            logger.info(
                f"Worker {self.display_id} disabled={disabled}",
            )
        self.disabled = disabled

    def increment_active(self) -> None:
        self.active_requests += 1

    def decrement_active(self) -> None:
        assert self.active_requests > 0, "active request count cannot be negative"
        self.active_requests -= 1

    @contextmanager
    def request_guard(self) -> Iterator[None]:
        self.increment_active()
        try:
            yield
        finally:
            self.decrement_active()

    def record_health_result(
        self,
        *,
        ok: bool,
        failure_threshold: int,
        success_threshold: int,
        status_code: int | None = None,
        error: str | None = None,
        checked_at: datetime | None = None,
    ) -> None:
        previous_state = self.state
        self.last_checked_at = checked_at or datetime.now(timezone.utc)
        self.last_status_code = status_code
        self.last_error = error

        if ok:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            if self.consecutive_successes >= success_threshold:
                self.state = HEALTH_STATE_HEALTHY
            self._log_state_transition(previous_state, self.state)
            return

        self._record_failure(
            previous_state=previous_state,
            failure_threshold=failure_threshold,
        )

    def record_request_failure(
        self,
        *,
        failure_threshold: int,
        status_code: int | None = None,
        error: str | None = None,
        observed_at: datetime | None = None,
    ) -> None:
        previous_state = self.state
        self.last_checked_at = observed_at or datetime.now(timezone.utc)
        self.last_status_code = status_code
        self.last_error = error
        self._record_failure(
            previous_state=previous_state,
            failure_threshold=failure_threshold,
        )

    def _record_failure(
        self,
        *,
        previous_state: WorkerState,
        failure_threshold: int,
    ) -> None:
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        failure_threshold_reached = self.consecutive_failures >= failure_threshold
        failure_threshold_crossed = self.consecutive_failures == failure_threshold
        if failure_threshold_reached:
            self.state = HEALTH_STATE_UNHEALTHY
        self._log_state_transition(
            previous_state,
            self.state,
            warn=failure_threshold_crossed,
        )
        if failure_threshold_crossed and previous_state == self.state:
            logger.warning(
                f"Worker {self.display_id} marked unhealthy after "
                f"{self.consecutive_failures} consecutive failures "
                f"(threshold={failure_threshold})",
            )

    def _log_state_transition(
        self,
        previous_state: WorkerState,
        next_state: WorkerState,
        *,
        warn: bool = False,
    ) -> None:
        if previous_state == next_state:
            return
        if previous_state == HEALTH_STATE_UNKNOWN:
            if next_state == HEALTH_STATE_HEALTHY:
                logger.info(f"Worker {self.display_id} is Healthy")
            else:
                logger.debug(
                    f"Worker {self.display_id} health state changed "
                    f"from {_format_state_for_log(previous_state)} "
                    f"to {_format_state_for_log(next_state)}",
                )
            return

        log = (
            logger.warning
            if next_state == HEALTH_STATE_DEAD
            or (warn and previous_state == HEALTH_STATE_HEALTHY)
            else logger.info
        )
        log(
            f"Worker {self.display_id} health state changed "
            f"from {_format_state_for_log(previous_state)} "
            f"to {_format_state_for_log(next_state)}",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_id": self.worker_id,
            "display_id": self.display_id,
            "url": self.url,
            "model": self.model,
            "capabilities": sorted(self.capabilities),
            "active_requests": self.active_requests,
            "health_state": self.state,
            "disabled": self.disabled,
            "routable": self.is_routable,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_status_code": self.last_status_code,
            "last_error": self.last_error,
            "last_checked_at": (
                self.last_checked_at.isoformat() if self.last_checked_at else None
            ),
        }


def build_workers(configs: list[WorkerConfig]) -> list[Worker]:
    return [Worker(config=config) for config in configs]


def _format_state_for_log(state: WorkerState) -> str:
    return state.replace("_", " ").title()
