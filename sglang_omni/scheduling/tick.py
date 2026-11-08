from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.proto.session import Cohort


@dataclass(kw_only=True)
class TickCohort:
    size: int
    due_ns: int
    arrived: int = 0


class TickCohorts:
    """A tick's batch waits for its late appends no longer than the tick lasts."""

    def __init__(self) -> None:
        self.cohorts: dict[int, TickCohort] = {}
        self.ran_due_ns: dict[int, int] = {}

    def arrive(self, tick: Cohort) -> None:
        if tick.index in self.ran_due_ns:
            return
        else:
            pass
        cohort = self.cohorts.get(tick.index)
        if cohort is None:
            cohort = TickCohort(size=tick.size, due_ns=tick.due_ns)
            self.cohorts[tick.index] = cohort
        else:
            pass
        cohort.arrived += 1

    def forget_elapsed(self, now_ns: int) -> None:
        for index in [
            index for index, cohort in self.cohorts.items() if cohort.due_ns <= now_ns
        ]:
            del self.cohorts[index]
        for index in [
            index for index, due_ns in self.ran_due_ns.items() if due_ns <= now_ns
        ]:
            del self.ran_due_ns[index]

    def hold_until_ns(self, now_ns: int) -> int | None:
        self.forget_elapsed(now_ns)
        due = [
            cohort.due_ns
            for cohort in self.cohorts.values()
            if cohort.arrived < cohort.size
        ]
        return min(due) if due else None

    def is_collecting(self, now_ns: int) -> bool:
        self.forget_elapsed(now_ns)
        return bool(self.cohorts)

    def latest_due_ns(self) -> int:
        return max((cohort.due_ns for cohort in self.cohorts.values()), default=0)

    def ran(self, ticks: list[Cohort]) -> None:
        for tick in ticks:
            self.cohorts.pop(tick.index, None)
            self.ran_due_ns[tick.index] = tick.due_ns


__all__ = ["TickCohort", "TickCohorts"]
