# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

import sglang_omni.serve.realtime.runtime as runtime_module
import sglang_omni.serve.realtime.tick_clock as tick_clock_module
from sglang_omni.proto.session import Cohort
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.tick_clock import NS_PER_SECOND, TickClock
from sglang_omni.serve.realtime.types import (
    Capabilities,
    InteractionAdapter,
    OutputSink,
    RuntimeLimits,
    Unit,
)
from tests.unit_test.fixtures.realtime_fakes import FakeClock

MODEL_NAME = "duplex-test"
SAMPLE_RATE = 16000
NATIVE_UNIT_MS = 20
UNIT_BYTES = SAMPLE_RATE * NATIVE_UNIT_MS // 1000 * 2
PERIOD_S = NATIVE_UNIT_MS / 1000
PADDED_CAPABILITIES = Capabilities(tail_policy="pad")
TEST_WAIT_TIMEOUT_SECONDS = 5.0
GROWTH_MARGIN = 1.5


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(runtime_module, "time", clock)
    monkeypatch.setattr(tick_clock_module, "time", clock)
    monkeypatch.setattr(asyncio, "sleep", clock.sleep)
    return clock


class RecordingAdapter(InteractionAdapter):
    def __init__(self) -> None:
        self.units: list[Unit] = []

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        pass

    async def process(self, unit: Unit) -> int:
        self.units.append(unit)
        return unit.real_samples

    async def close(self) -> None:
        pass


class HoldingAdapter(RecordingAdapter):
    def __init__(self, held_index: int) -> None:
        super().__init__()
        self.held_index = held_index
        self.release = asyncio.Event()

    async def process(self, unit: Unit) -> int:
        self.units.append(unit)
        if unit.index == self.held_index:
            await self.release.wait()
        else:
            pass
        return unit.real_samples


@asynccontextmanager
async def open_runtime(
    adapter: RecordingAdapter,
    tick_clock: TickClock | None,
    capabilities: Capabilities = PADDED_CAPABILITIES,
) -> AsyncIterator[SessionRuntime]:
    runtime = SessionRuntime(
        MODEL_NAME, capabilities, lambda: adapter, RuntimeLimits(), tick_clock
    )
    runtime.start()
    try:
        await runtime.update({}, "client_update")
        yield runtime
    finally:
        await asyncio.wait_for(
            runtime.close("client_closed"), TEST_WAIT_TIMEOUT_SECONDS
        )


def slot_at(index: int, size: int) -> Cohort:
    return Cohort(
        index=index, size=size, due_ns=round((index + 1) * PERIOD_S * NS_PER_SECOND)
    )


@pytest.mark.asyncio
async def test_first_partial_unit_is_head_padded_and_leaves_on_the_tick(
    fake_clock: FakeClock,
) -> None:
    adapter = RecordingAdapter()
    async with open_runtime(
        adapter, TickClock(NATIVE_UNIT_MS, growth_margin=GROWTH_MARGIN)
    ) as runtime:
        await fake_clock.advance(PERIOD_S / 4)
        real_pcm = b"\1\0" * (UNIT_BYTES // 4)
        await runtime.append(real_pcm, 0, None, "append_0")
        await fake_clock.advance(PERIOD_S / 4)
        assert not adapter.units

        await fake_clock.advance(PERIOD_S / 2)
        [first] = adapter.units
        assert first.pcm == b"\0" * (UNIT_BYTES // 2) + real_pcm
        assert first.real_samples == UNIT_BYTES // 4
        assert first.tick is None
        assert runtime.padding_samples == UNIT_BYTES // 4


async def append(
    fake_clock: FakeClock, runtime: SessionRuntime, units: float, sequence: int
) -> None:
    await runtime.append(
        b"\1\0" * int(units * UNIT_BYTES / 2), sequence, None, f"append_{sequence}"
    )
    await fake_clock.advance(0)


@pytest.mark.asyncio
async def test_a_whole_unit_waits_for_the_tick_and_carries_its_slot(
    fake_clock: FakeClock,
) -> None:
    adapter = RecordingAdapter()
    async with open_runtime(
        adapter, TickClock(NATIVE_UNIT_MS, growth_margin=GROWTH_MARGIN)
    ) as runtime:
        await append(fake_clock, runtime, 1, 0)
        await fake_clock.advance(PERIOD_S)
        await fake_clock.advance(PERIOD_S / 4)
        await append(fake_clock, runtime, 1, 1)
        await fake_clock.advance(PERIOD_S / 4)
        assert len(adapter.units) == 1

        await fake_clock.advance(PERIOD_S / 2)
        assert [unit.tick for unit in adapter.units] == [None, slot_at(2, 1)]


@pytest.mark.asyncio
async def test_two_buffered_units_leave_as_one_unit_on_the_tick(
    fake_clock: FakeClock,
) -> None:
    adapter = RecordingAdapter()
    async with open_runtime(
        adapter, TickClock(NATIVE_UNIT_MS, growth_margin=GROWTH_MARGIN)
    ) as runtime:
        await append(fake_clock, runtime, 1, 0)
        await fake_clock.advance(PERIOD_S)
        await append(fake_clock, runtime, 3, 1)
        await fake_clock.advance(PERIOD_S / 2)
        assert len(adapter.units) == 1

        await fake_clock.advance(PERIOD_S / 2)
        assert [unit.index for unit in adapter.units] == [0, 1]
        assert adapter.units[1].tick == slot_at(2, 1)
        assert adapter.units[1].real_samples == 2 * UNIT_BYTES // 2
        await fake_clock.advance(PERIOD_S)
        assert adapter.units[2].tick == slot_at(3, 1)
        assert adapter.units[2].real_samples == UNIT_BYTES // 2


@pytest.mark.asyncio
async def test_a_session_that_missed_a_tick_catches_up_on_the_next(
    fake_clock: FakeClock,
) -> None:
    adapter = HoldingAdapter(held_index=1)
    async with open_runtime(
        adapter, TickClock(NATIVE_UNIT_MS, growth_margin=GROWTH_MARGIN)
    ) as runtime:
        await append(fake_clock, runtime, 1, 0)
        await fake_clock.advance(PERIOD_S)
        await append(fake_clock, runtime, 1, 1)
        await fake_clock.advance(PERIOD_S)
        assert adapter.units[-1].tick == slot_at(2, 1)

        await fake_clock.advance(PERIOD_S / 2)
        await append(fake_clock, runtime, 1, 2)
        await fake_clock.advance(3 * PERIOD_S / 4)
        await append(fake_clock, runtime, 1, 3)
        assert len(adapter.units) == 2
        adapter.release.set()
        await fake_clock.advance(0)
        assert len(adapter.units) == 2

        await fake_clock.advance(PERIOD_S)
        assert [unit.index for unit in adapter.units] == [0, 1, 2]
        assert adapter.units[2].tick == Cohort(
            index=3, size=1, due_ns=round(5 * PERIOD_S * NS_PER_SECOND)
        )
        assert adapter.units[2].real_samples == 2 * UNIT_BYTES // 2


@pytest.mark.asyncio
async def test_a_session_stalled_for_ticks_catches_up_one_unit_a_tick(
    fake_clock: FakeClock,
) -> None:
    adapter = HoldingAdapter(held_index=1)
    async with open_runtime(
        adapter, TickClock(NATIVE_UNIT_MS, growth_margin=GROWTH_MARGIN)
    ) as runtime:
        await append(fake_clock, runtime, 1, 0)
        await fake_clock.advance(PERIOD_S)
        await append(fake_clock, runtime, 1, 1)
        await fake_clock.advance(PERIOD_S)
        for sequence in (2, 3, 4):
            await fake_clock.advance(PERIOD_S)
            await append(fake_clock, runtime, 1, sequence)
        await fake_clock.advance(PERIOD_S / 2)
        adapter.release.set()
        await fake_clock.advance(0)
        assert len(adapter.units) == 2

        await fake_clock.advance(PERIOD_S * 0.6)
        assert adapter.units[-1].real_samples == 2 * UNIT_BYTES // 2
        await append(fake_clock, runtime, 1, 5)
        await fake_clock.advance(PERIOD_S)
        assert adapter.units[-1].real_samples == 2 * UNIT_BYTES // 2
        await append(fake_clock, runtime, 1, 6)
        await fake_clock.advance(PERIOD_S)
        assert adapter.units[-1].real_samples == UNIT_BYTES // 2
        assert [unit.index for unit in adapter.units] == [0, 1, 2, 3, 4]
        assert runtime.pending_pcm == b""


@pytest.mark.asyncio
async def test_a_unit_arriving_after_the_tick_waits_without_holding_the_others(
    fake_clock: FakeClock,
) -> None:
    clock = TickClock(NATIVE_UNIT_MS, growth_margin=GROWTH_MARGIN)
    early, late = RecordingAdapter(), RecordingAdapter()
    async with (
        open_runtime(early, clock) as early_runtime,
        open_runtime(late, clock) as late_runtime,
    ):
        for runtime in (early_runtime, late_runtime):
            await append(fake_clock, runtime, 1, 0)
        await fake_clock.advance(PERIOD_S)
        await append(fake_clock, early_runtime, 1, 1)
        await fake_clock.advance(PERIOD_S)
        assert early.units[-1].tick == slot_at(2, 1)
        assert len(late.units) == 1

        await fake_clock.advance(PERIOD_S / 4)
        await append(fake_clock, late_runtime, 1, 1)
        await append(fake_clock, early_runtime, 1, 2)
        await fake_clock.advance(PERIOD_S / 4)
        assert len(late.units) == 1

        await fake_clock.advance(PERIOD_S / 2)
        assert early.units[-1].tick == slot_at(3, 2)
        assert late.units[-1].tick == slot_at(3, 2)


@pytest.mark.asyncio
async def test_without_the_grid_a_unit_leaves_as_it_arrives(
    fake_clock: FakeClock,
) -> None:
    adapter = RecordingAdapter()
    async with open_runtime(adapter, None) as runtime:
        await fake_clock.advance(PERIOD_S / 4)
        await runtime.append(b"\1\0" * (UNIT_BYTES // 2), 0, None, "append_0")
        await fake_clock.advance(0)
        [unit] = adapter.units
        assert unit.tick is None and unit.pcm == b"\1\0" * (UNIT_BYTES // 2)


def test_the_grid_needs_native_sessions() -> None:
    with pytest.raises(ValueError, match="native"):
        RealtimeDeployment(
            capabilities=Capabilities(interaction="turn_based"),
            adapter_factory=RecordingAdapter,
            tick_groups=1,
        )


@pytest.mark.asyncio
async def test_phase_groups_release_sessions_on_their_own_boundary(
    fake_clock: FakeClock,
) -> None:
    clock = TickClock(NATIVE_UNIT_MS, growth_margin=GROWTH_MARGIN, groups=4)
    boundary_s = PERIOD_S / 4
    first, second = RecordingAdapter(), RecordingAdapter()
    async with (
        open_runtime(first, clock) as first_runtime,
        open_runtime(second, clock) as second_runtime,
    ):
        await append(fake_clock, first_runtime, 1, 0)
        await fake_clock.advance(boundary_s)
        assert len(first.units) == 1
        await fake_clock.advance(2 * boundary_s)
        await append(fake_clock, second_runtime, 1, 0)
        await fake_clock.advance(boundary_s)
        assert len(second.units) == 1

        await append(fake_clock, first_runtime, 1, 1)
        await fake_clock.advance(boundary_s)
        assert first.units[1].tick == Cohort(
            index=3, size=1, due_ns=round(6 * boundary_s * NS_PER_SECOND)
        )
        await fake_clock.advance(2 * boundary_s)
        await append(fake_clock, second_runtime, 1, 1)
        await fake_clock.advance(boundary_s)
        assert second.units[1].tick == Cohort(
            index=4, size=1, due_ns=round(9 * boundary_s * NS_PER_SECOND)
        )


def test_the_group_count_follows_the_step_cost_up_to_the_configured_groups() -> None:
    clock = TickClock(80, growth_margin=GROWTH_MARGIN, groups=4)
    clock.session_count_changed()
    assert clock.groups == 4
    clock.observe_step_ms(25.0)
    assert clock.groups == 4
    clock.session_count_changed()
    assert clock.groups == 2 and clock.period_s == pytest.approx(0.04)
    clock.observe_step_ms(12.0)
    clock.session_count_changed()
    assert clock.groups == 4
    clock.observe_step_ms(100.0)
    clock.session_count_changed()
    assert clock.groups == 1


def test_a_step_cost_near_a_counts_limit_does_not_move_the_count_back_and_forth() -> (
    None
):
    clock = TickClock(80, growth_margin=GROWTH_MARGIN, groups=4)
    counts = []
    # Note (Junnan Li): 4 groups fit a step of 16 ms (16 x 1.25 x 4 = 80) and not
    # of 17.6 ms (88); back at 16 ms, a fourth group needs 16 x 1.5 x 4 = 96 to fit,
    # so 3 stay.
    for step_ms in (16.0, 17.6, 16.0, 15.4, 16.3, 13.0):
        clock.observe_step_ms(step_ms)
        clock.session_count_changed()
        counts.append(clock.groups)
    assert counts == [4, 3, 3, 3, 3, 4]


@pytest.mark.asyncio
async def test_a_units_step_cost_reaches_the_clock(fake_clock: FakeClock) -> None:
    adapter = RecordingAdapter()
    adapter.model_step_ms = 21.5
    clock = TickClock(NATIVE_UNIT_MS, growth_margin=GROWTH_MARGIN, groups=2)
    async with open_runtime(adapter, clock) as runtime:
        await append(fake_clock, runtime, 1, 0)
        await fake_clock.advance(PERIOD_S)
        assert len(adapter.units) == 1
        assert clock.step_ms == 21.5


@pytest.mark.asyncio
async def test_with_openings_per_unit_first_units_leave_together_once_a_unit(
    fake_clock: FakeClock,
) -> None:
    clock = TickClock(
        NATIVE_UNIT_MS, growth_margin=GROWTH_MARGIN, groups=4, openings_per_unit=True
    )
    boundary_s = PERIOD_S / 4
    adapters = [RecordingAdapter() for _ in range(3)]
    async with (
        open_runtime(adapters[0], clock) as first,
        open_runtime(adapters[1], clock) as second,
        open_runtime(adapters[2], clock) as third,
    ):
        await append(fake_clock, first, 1, 0)
        await fake_clock.advance(boundary_s)
        assert [len(adapter.units) for adapter in adapters] == [1, 0, 0]
        await append(fake_clock, second, 1, 0)
        await fake_clock.advance(boundary_s)
        await append(fake_clock, third, 1, 0)
        await fake_clock.advance(2 * boundary_s)
        assert [len(adapter.units) for adapter in adapters] == [1, 0, 0]
        await fake_clock.advance(boundary_s)
        assert [len(adapter.units) for adapter in adapters] == [1, 1, 1]


@pytest.mark.asyncio
async def test_a_slot_counts_only_the_units_that_carry_it(
    fake_clock: FakeClock,
) -> None:
    clock = TickClock(NATIVE_UNIT_MS, growth_margin=GROWTH_MARGIN)
    live, joining = RecordingAdapter(), RecordingAdapter()
    async with (
        open_runtime(live, clock) as live_runtime,
        open_runtime(joining, clock) as joining_runtime,
    ):
        await append(fake_clock, live_runtime, 1, 0)
        await fake_clock.advance(PERIOD_S)
        await append(fake_clock, live_runtime, 1, 1)
        await append(fake_clock, joining_runtime, 1, 0)
        await fake_clock.advance(PERIOD_S)
        assert [unit.tick for unit in joining.units] == [None]
        assert live.units[1].tick is not None and live.units[1].tick.size == 1


MS = 0.001
ARRIVALS_QUIET_S = 2 * MS


async def open_on_the_grid(
    fake_clock: FakeClock, runtimes: list[SessionRuntime]
) -> None:
    for runtime in runtimes:
        await append(fake_clock, runtime, 1, 0)
    await fake_clock.advance(PERIOD_S)


@pytest.mark.asyncio
async def test_following_arrivals_a_group_leaves_together_once_it_stopped_arriving(
    fake_clock: FakeClock,
) -> None:
    clock = TickClock(
        NATIVE_UNIT_MS,
        growth_margin=GROWTH_MARGIN,
        follow_arrivals=True,
        arrivals_quiet_s=ARRIVALS_QUIET_S,
    )
    adapters = [RecordingAdapter() for _ in range(3)]
    async with (
        open_runtime(adapters[0], clock) as first,
        open_runtime(adapters[1], clock) as second,
        open_runtime(adapters[2], clock) as third,
    ):
        await open_on_the_grid(fake_clock, [first, second, third])
        assert [len(adapter.units) for adapter in adapters] == [1, 1, 1]
        # The group straddles where the grid's next boundary would be.
        await fake_clock.advance(PERIOD_S - 1 * MS)
        await append(fake_clock, first, 1, 1)
        await fake_clock.advance(1 * MS)
        await append(fake_clock, second, 1, 1)
        await fake_clock.advance(1.5 * MS)
        await append(fake_clock, third, 1, 1)
        await fake_clock.advance(1.5 * MS)
        assert [len(adapter.units) for adapter in adapters] == [1, 1, 1]
        await fake_clock.advance(0.5 * MS)
        assert [len(adapter.units) for adapter in adapters] == [2, 2, 2]
        assert {adapter.units[1].tick for adapter in adapters} == {
            adapters[0].units[1].tick
        }
        assert adapters[0].units[1].tick.size == 3


@pytest.mark.asyncio
async def test_following_arrivals_no_unit_waits_longer_than_a_period(
    fake_clock: FakeClock,
) -> None:
    clock = TickClock(
        NATIVE_UNIT_MS,
        growth_margin=GROWTH_MARGIN,
        follow_arrivals=True,
        arrivals_quiet_s=ARRIVALS_QUIET_S,
    )
    adapters = [RecordingAdapter() for _ in range(12)]
    async with contextlib.AsyncExitStack() as stack:
        runtimes = [
            await stack.enter_async_context(open_runtime(adapter, clock))
            for adapter in adapters
        ]
        await open_on_the_grid(fake_clock, runtimes)
        await fake_clock.advance(PERIOD_S)
        # A unit arrives every 1.9 ms, so they never stop arriving.
        for runtime in runtimes:
            await append(fake_clock, runtime, 1, 1)
            await fake_clock.advance(1.9 * MS)
        left = [len(adapter.units) == 2 for adapter in adapters]
        # The first 11 arrived within a period of the oldest, at 0 .. 19 ms.
        assert left == [True] * 11 + [False]
        assert adapters[0].units[1].tick.size == 11


@pytest.mark.asyncio
async def test_following_arrivals_a_release_waits_a_period_after_the_previous(
    fake_clock: FakeClock,
) -> None:
    clock = TickClock(
        NATIVE_UNIT_MS,
        growth_margin=GROWTH_MARGIN,
        follow_arrivals=True,
        arrivals_quiet_s=ARRIVALS_QUIET_S,
    )
    first_adapter, second_adapter = RecordingAdapter(), RecordingAdapter()
    async with (
        open_runtime(first_adapter, clock) as first,
        open_runtime(second_adapter, clock) as second,
    ):
        await open_on_the_grid(fake_clock, [first, second])
        await fake_clock.advance(PERIOD_S)
        await append(fake_clock, first, 1, 1)
        await fake_clock.advance(2 * MS)
        assert len(first_adapter.units) == 2
        # Stopped arriving at 44 ms, but the previous release was at 42 ms.
        await append(fake_clock, second, 1, 1)
        await fake_clock.advance(19.5 * MS)
        assert len(second_adapter.units) == 1
        await fake_clock.advance(0.5 * MS)
        assert len(second_adapter.units) == 2
