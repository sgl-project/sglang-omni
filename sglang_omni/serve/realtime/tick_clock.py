from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING

from sglang_omni.proto.session import Cohort
from sglang_omni.serve.realtime.types import MS_PER_SECOND

if TYPE_CHECKING:
    from sglang_omni.serve.realtime.runtime import SessionRuntime
else:
    pass

logger = logging.getLogger(__name__)
NS_PER_SECOND = 1_000_000_000
# Note (Junnan Li): Headroom for the step-cost spread where the group count decides.
STEP_COST_MARGIN = 1.25


class TickClock:
    """Boundary k is at epoch + k * period, so the grid does not drift."""

    def __init__(
        self,
        unit_ms: int,
        *,
        growth_margin: float,
        groups: int = 1,
        openings_per_unit: bool = False,
        follow_arrivals: bool = False,
        arrivals_quiet_s: float = 0.0,
    ) -> None:
        self.unit_ms = unit_ms
        self.growth_margin = growth_margin
        self.max_groups = groups
        self.groups = groups
        self.opening_waiters: dict[SessionRuntime, asyncio.Future[Cohort]] = {}
        self.openings_per_unit = openings_per_unit
        self.last_opening_s: float | None = None
        self.step_ms: float | None = None
        self.epoch_s = time.monotonic()
        self.releases = 0
        self.waiters: dict[SessionRuntime, asyncio.Future[Cohort]] = {}
        self.ticker: asyncio.Task[None] | None = None
        self.follow_arrivals = follow_arrivals
        self.arrivals_quiet_s = arrivals_quiet_s
        self.first_arrival_s: float | None = None
        self.last_arrival_s = 0.0
        self.last_release_s = -math.inf

    @property
    def period_s(self) -> float:
        return self.unit_ms / MS_PER_SECOND / self.groups

    def observe_step_ms(self, step_ms: float) -> None:
        self.step_ms = step_ms

    def fitted_groups(self) -> int:
        step_ms = self.step_ms
        if step_ms is None:
            return self.max_groups
        else:
            pass
        groups = self.groups
        while groups > 1 and step_ms * STEP_COST_MARGIN * groups > self.unit_ms:
            groups -= 1
        while (
            groups < self.max_groups
            and step_ms * self.growth_margin * (groups + 1) < self.unit_ms
        ):
            groups += 1
        return groups

    def session_count_changed(self) -> None:
        groups = self.fitted_groups()
        if groups != self.groups:
            logger.info(
                "Unit grid: %d phase groups for a step cost of %s ms",
                groups,
                None if self.step_ms is None else f"{self.step_ms:.1f}",
            )
            self.groups = groups
        else:
            pass

    async def wait(self, waiter: SessionRuntime, is_opening: bool = False) -> Cohort:
        waiters = self.opening_waiters if is_opening else self.waiters
        if not is_opening:
            self.last_arrival_s = time.monotonic()
            if self.first_arrival_s is None:
                self.first_arrival_s = self.last_arrival_s
            else:
                pass
        else:
            pass
        future: asyncio.Future[Cohort] = asyncio.get_running_loop().create_future()
        waiters[waiter] = future
        if self.ticker is None or self.ticker.done():
            self.ticker = asyncio.create_task(self.tick())
        else:
            pass
        try:
            return await future
        finally:
            if waiters.get(waiter) is future:
                del waiters[waiter]
            else:
                pass

    def take_openings(
        self, tick_s: float
    ) -> dict[SessionRuntime, asyncio.Future[Cohort]]:
        if (
            self.openings_per_unit
            and self.last_opening_s is not None
            # Note (Junnan Li): Half a period below a unit, so a boundary a unit
            # after the last release is not missed by rounding.
            and tick_s - self.last_opening_s
            < (self.unit_ms / MS_PER_SECOND) - self.period_s / 2
        ):
            return {}
        elif self.opening_waiters:
            self.last_opening_s = tick_s
        else:
            pass
        released, self.opening_waiters = self.opening_waiters, {}
        return released

    def next_grid_boundary_s(self, now_s: float) -> float:
        index = math.floor((now_s - self.epoch_s) / self.period_s) + 1
        return self.epoch_s + index * self.period_s

    def next_release_s(self, grid_s: float) -> float:
        if not self.follow_arrivals or self.first_arrival_s is None:
            return grid_s
        else:
            return max(
                min(
                    self.last_arrival_s + self.arrivals_quiet_s,
                    self.first_arrival_s + self.period_s,
                ),
                self.last_release_s + self.period_s,
            )

    async def wait_for_release(self) -> float:
        """Arrivals move the release, so with follow_arrivals the sleep is cut short."""
        grid_s = self.next_grid_boundary_s(time.monotonic())
        while True:
            now_s = time.monotonic()
            release_s = self.next_release_s(grid_s)
            if release_s <= now_s:
                return release_s
            elif not self.follow_arrivals:
                await asyncio.sleep(release_s - now_s)
                return release_s
            else:
                await asyncio.sleep(min(release_s - now_s, self.arrivals_quiet_s))

    async def tick(self) -> None:
        while self.waiters or self.opening_waiters:
            tick_s = await self.wait_for_release()
            self.last_release_s = tick_s
            self.first_arrival_s = None
            released, self.waiters = self.waiters, {}
            self.releases += 1
            # Note (Junnan Li): A first unit leaves without the slot, so the slot's
            # size counts only the units that carry it.
            slot = Cohort(
                index=self.releases,
                size=len(released),
                due_ns=round((tick_s + self.period_s) * NS_PER_SECOND),
            )
            released.update(self.take_openings(tick_s))
            for future in released.values():
                if not future.done():
                    future.set_result(slot)
                else:
                    pass


__all__ = ["TickClock"]
