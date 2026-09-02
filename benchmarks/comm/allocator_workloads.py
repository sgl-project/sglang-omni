# SPDX-License-Identifier: Apache-2.0
"""Fragmentation workload driver for the CUDA-IPC contiguous slot allocator.

Replays synthetic (and, later, GPU-captured) allocation schedules against
``_ContiguousSlotAllocator`` on a virtual tick clock -- no GPU, no wall-clock
sleeps -- and reports occupancy, fragmentation, and contiguity-induced wait
metrics. Alternative allocators (buddy, size-class) replay the same schedule
for offline comparison.

Wait classification:
- capacity wait: the pool simply had fewer free slots than requested.
- contiguity wait: enough free slots existed, but no contiguous run fit the
  request -- this is the fragmentation failure mode debated in RFC #287.

Usage:
    python -m benchmarks.comm.allocator_workloads --schedule bimodal \
        --pool-slots 256 --slot-kb 64 --seeds 0 1 2 --format json --out out.json
    python -m benchmarks.comm.allocator_workloads --sweep-large-ratio \
        --pool-slots 256 --slot-kb 64 --seeds 0 1 2 --format md
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol

from sglang_omni.relay.cuda_ipc import (
    _ContiguousSlotAllocator,
    _SlotAllocation,
    _slots_for_size,
)

# Rounds of cooperative yielding per tick so blocked acquires can observe
# releases made earlier in the same tick.
SETTLE_ROUNDS = 8


class SlotAllocator(Protocol):
    slot_count: int
    slot_size: int

    async def acquire_async(
        self, num_slots: int, *, capture_layout: bool = False
    ) -> _SlotAllocation: ...

    def release(self, offset: int, num_slots: int) -> None: ...

    def layout_snapshot(self) -> dict[str, int]: ...


# --------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    request_id: str
    submit_tick: int
    size_bytes: int
    hold_ticks: int


@dataclass(frozen=True)
class Schedule:
    name: str
    params: dict[str, Any]
    requests: list[Request]


def fixed_schedule(
    *,
    count: int = 200,
    size_bytes: int = 3 * 2**20,
    arrival_period: int = 1,
    hold_ticks: int = 8,
) -> Schedule:
    requests = [
        Request(f"fixed-{i}", i * arrival_period, size_bytes, hold_ticks)
        for i in range(count)
    ]
    return Schedule(
        "fixed",
        {
            "count": count,
            "size_bytes": size_bytes,
            "arrival_period": arrival_period,
            "hold_ticks": hold_ticks,
        },
        requests,
    )


def bimodal_schedule(
    *,
    count: int = 400,
    small_bytes: int = 128 * 2**10,
    large_bytes: int = 8 * 2**20,
    large_ratio: float = 0.1,
    jitter_pct: float = 0.2,
    arrival_period: int = 1,
    small_hold: int = 4,
    large_hold: int = 16,
    seed: int = 0,
) -> Schedule:
    rng = random.Random(seed)
    requests = []
    for i in range(count):
        is_large = rng.random() < large_ratio
        base = large_bytes if is_large else small_bytes
        jitter = 1.0 + rng.uniform(-jitter_pct, jitter_pct)
        size = max(1, int(base * jitter))
        hold = large_hold if is_large else small_hold
        requests.append(Request(f"bimodal-{i}", i * arrival_period, size, hold))
    return Schedule(
        "bimodal",
        {
            "count": count,
            "small_bytes": small_bytes,
            "large_bytes": large_bytes,
            "large_ratio": large_ratio,
            "jitter_pct": jitter_pct,
            "arrival_period": arrival_period,
            "small_hold": small_hold,
            "large_hold": large_hold,
            "seed": seed,
        },
        requests,
    )


def adversarial_schedule(
    *,
    slot_count: int,
    slot_size: int,
    large_run: int = 8,
    large_count: int = 12,
    comb_hold_short: int = 2,
    comb_hold_long: int = 200,
) -> Schedule:
    """Deterministically provoke a comb: fill the pool with 1-slot holds where
    even slots persist and odd slots expire, then request large contiguous
    runs while free capacity is ~half the pool with no run longer than 1.
    """
    requests = []
    for i in range(slot_count):
        hold = comb_hold_long if i % 2 == 0 else comb_hold_short
        requests.append(Request(f"comb-{i}", 0, slot_size, hold))
    for i in range(large_count):
        requests.append(
            Request(
                f"large-{i}",
                comb_hold_short + 1 + i,
                large_run * slot_size,
                4,
            )
        )
    return Schedule(
        "adversarial",
        {
            "slot_count": slot_count,
            "slot_size": slot_size,
            "large_run": large_run,
            "large_count": large_count,
            "comb_hold_short": comb_hold_short,
            "comb_hold_long": comb_hold_long,
        },
        requests,
    )


def replay_schedule(path: str) -> Schedule:
    """Replay a JSONL trace: {"id": str, "tick": int, "size_bytes": int,
    "hold_ticks": int} per line (e.g. converted from GPU comm-trace logs)."""
    requests = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            requests.append(
                Request(
                    str(row.get("id", f"replay-{line_no}")),
                    int(row["tick"]),
                    int(row["size_bytes"]),
                    int(row["hold_ticks"]),
                )
            )
    requests.sort(key=lambda r: r.submit_tick)
    return Schedule("replay", {"path": path, "count": len(requests)}, requests)


# --------------------------------------------------------------------------
# Virtual-tick driver
# --------------------------------------------------------------------------


@dataclass
class RequestResult:
    request_id: str
    submit_tick: int
    grant_tick: int | None
    num_slots: int
    size_bytes: int
    wait_rounds: int
    last_failed_free_slots: int
    last_failed_largest_free_run: int
    last_failed_free_runs: int

    @property
    def wait_ticks(self) -> int | None:
        if self.grant_tick is None:
            return None
        return self.grant_tick - self.submit_tick

    @property
    def waited(self) -> bool:
        return self.wait_rounds > 0

    @property
    def contiguity_wait(self) -> bool:
        """Free capacity was sufficient at the last failed probe, yet no
        contiguous run fit: fragmentation, not load, caused the wait."""
        return self.waited and self.last_failed_free_slots >= self.num_slots

    @property
    def capacity_wait(self) -> bool:
        return self.waited and not self.contiguity_wait


@dataclass
class RunResult:
    schedule: str
    params: dict[str, Any]
    slot_count: int
    slot_size: int
    ticks: int
    requests: list[RequestResult] = field(default_factory=list)
    occupancy_series: list[float] = field(default_factory=list)
    largest_free_run_series: list[int] = field(default_factory=list)
    free_runs_series: list[int] = field(default_factory=list)
    unsatisfiable: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        granted = [r for r in self.requests if r.grant_tick is not None]
        waits = [r.wait_ticks for r in granted if r.wait_ticks is not None]
        contiguity = [r for r in granted if r.contiguity_wait]
        capacity = [r for r in granted if r.capacity_wait]
        internal_waste = sum(
            r.num_slots * self.slot_size - r.size_bytes for r in self.requests
        )
        payload_bytes = sum(r.size_bytes for r in self.requests)
        return {
            "schedule": self.schedule,
            "params": self.params,
            "slot_count": self.slot_count,
            "slot_size": self.slot_size,
            "ticks": self.ticks,
            "requests": len(self.requests),
            "granted": len(granted),
            "unsatisfiable": len(self.unsatisfiable),
            "occupancy": _percentiles(self.occupancy_series),
            "largest_free_run": _percentiles(self.largest_free_run_series),
            "wait_ticks": _percentiles(waits),
            "waited_requests": sum(1 for r in granted if r.waited),
            "contiguity_waits": len(contiguity),
            "contiguity_wait_ticks": _percentiles(
                [r.wait_ticks for r in contiguity if r.wait_ticks is not None]
            ),
            "capacity_waits": len(capacity),
            "internal_waste_bytes": internal_waste,
            "internal_waste_ratio": (
                internal_waste / payload_bytes if payload_bytes else 0.0
            ),
        }


def _percentiles(values: list[float] | list[int]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    ordered = sorted(values)

    def pick(q: float) -> float:
        index = min(len(ordered) - 1, int(q * len(ordered)))
        return float(ordered[index])

    return {
        "p50": pick(0.50),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


async def run_schedule(
    allocator: SlotAllocator,
    schedule: Schedule,
    *,
    max_ticks: int = 100_000,
) -> RunResult:
    slot_size = allocator.slot_size
    result = RunResult(
        schedule=schedule.name,
        params=schedule.params,
        slot_count=allocator.slot_count,
        slot_size=slot_size,
        ticks=0,
    )
    pending_requests = sorted(schedule.requests, key=lambda r: r.submit_tick)
    next_request = 0
    releases: list[tuple[int, int, int, int]] = []  # (tick, seq, offset, slots)
    release_seq = 0
    inflight: dict[asyncio.Task, tuple[Request, int]] = {}

    tick = 0
    while tick < max_ticks:
        # 1) releases due this tick wake blocked acquires via _changed.set()
        while releases and releases[0][0] <= tick:
            _, _, offset, num_slots = heapq.heappop(releases)
            allocator.release(offset, num_slots)

        # 2) submit arrivals due this tick
        while (
            next_request < len(pending_requests)
            and pending_requests[next_request].submit_tick <= tick
        ):
            request = pending_requests[next_request]
            next_request += 1
            num_slots = _slots_for_size(request.size_bytes, slot_size)
            task = asyncio.create_task(
                allocator.acquire_async(num_slots, capture_layout=True)
            )
            inflight[task] = (request, num_slots)

        # 3) settle: let waiters contend and completed acquires finish
        for _ in range(SETTLE_ROUNDS):
            await asyncio.sleep(0)
            done = [task for task in inflight if task.done()]
            for task in done:
                request, num_slots = inflight.pop(task)
                if task.exception() is not None:
                    # e.g. request larger than any size class can admit
                    result.unsatisfiable.append(request.request_id)
                    result.requests.append(
                        RequestResult(
                            request_id=request.request_id,
                            submit_tick=request.submit_tick,
                            grant_tick=None,
                            num_slots=num_slots,
                            size_bytes=request.size_bytes,
                            wait_rounds=-1,
                            last_failed_free_slots=-1,
                            last_failed_largest_free_run=-1,
                            last_failed_free_runs=-1,
                        )
                    )
                    continue
                allocation = task.result()
                result.requests.append(
                    RequestResult(
                        request_id=request.request_id,
                        submit_tick=request.submit_tick,
                        grant_tick=tick,
                        num_slots=num_slots,
                        size_bytes=request.size_bytes,
                        wait_rounds=allocation.wait_rounds,
                        last_failed_free_slots=allocation.last_failed_free_slots,
                        last_failed_largest_free_run=(
                            allocation.last_failed_largest_free_run
                        ),
                        last_failed_free_runs=allocation.last_failed_free_runs,
                    )
                )
                release_seq += 1
                heapq.heappush(
                    releases,
                    (
                        tick + request.hold_ticks,
                        release_seq,
                        allocation.offset,
                        num_slots,
                    ),
                )

        # 4) per-tick layout snapshot (single-threaded tick boundary: safe)
        layout = allocator.layout_snapshot()
        result.occupancy_series.append(
            1.0 - layout["free_slots"] / allocator.slot_count
        )
        result.largest_free_run_series.append(layout["largest_free_run"])
        result.free_runs_series.append(layout["free_runs"])

        tick += 1
        if next_request >= len(pending_requests) and not inflight and not releases:
            break
        # Real traces span millions of ticks that are mostly idle; stepping
        # through them one at a time is both pointless and O(slot_count) per
        # tick. When nothing is in flight, jump straight to the next event.
        if not inflight:
            candidates = []
            if next_request < len(pending_requests):
                candidates.append(pending_requests[next_request].submit_tick)
            if releases:
                candidates.append(releases[0][0])
            if candidates:
                tick = max(tick, min(candidates))

    result.ticks = tick
    for task, (request, num_slots) in inflight.items():
        task.cancel()
        result.unsatisfiable.append(request.request_id)
        result.requests.append(
            RequestResult(
                request_id=request.request_id,
                submit_tick=request.submit_tick,
                grant_tick=None,
                num_slots=num_slots,
                size_bytes=request.size_bytes,
                wait_rounds=-1,
                last_failed_free_slots=-1,
                last_failed_largest_free_run=-1,
                last_failed_free_runs=-1,
            )
        )
    if inflight:
        await asyncio.gather(*inflight, return_exceptions=True)
    return result


# --------------------------------------------------------------------------
# Allocators under comparison
# --------------------------------------------------------------------------


class FirstFitAllocator(_ContiguousSlotAllocator):
    """The production allocator, plus a driver-facing layout probe."""

    def layout_snapshot(self) -> dict[str, int]:
        layout = self._find_contiguous_with_layout(1)
        return {
            "free_slots": layout.free_slots,
            "largest_free_run": layout.largest_free_run,
            "free_runs": layout.free_runs,
        }


class _WaitingAllocatorBase:
    """Shared event-driven waiting shell mirroring _ContiguousSlotAllocator's
    acquire/wait/release contract for the comparison allocators."""

    def __init__(self, *, slot_count: int, slot_size: int) -> None:
        self.slot_count = slot_count
        self.slot_size = slot_size
        self._changed = asyncio.Event()
        self._changed.set()

    def _try_acquire(self, num_slots: int) -> int | None:
        raise NotImplementedError

    def _do_release(self, offset: int, num_slots: int) -> None:
        raise NotImplementedError

    def _free_slot_count(self) -> int:
        raise NotImplementedError

    def _largest_free_block(self) -> int:
        raise NotImplementedError

    def _free_block_count(self) -> int:
        raise NotImplementedError

    async def acquire_async(
        self, num_slots: int, *, capture_layout: bool = False
    ) -> _SlotAllocation:
        del capture_layout  # layout fields are always cheap here
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        wait_rounds = 0
        last_failed_free = 0
        last_failed_largest = 0
        last_failed_runs = 0
        while True:
            free_before = self._free_slot_count()
            slot_index = self._try_acquire(num_slots)
            if slot_index is not None:
                return _SlotAllocation(
                    offset=slot_index * self.slot_size,
                    wait_rounds=wait_rounds,
                    free_slots_before=free_before,
                    largest_free_run_before=-1,
                    free_runs_before=-1,
                    last_failed_free_slots=last_failed_free,
                    last_failed_largest_free_run=last_failed_largest,
                    last_failed_free_runs=last_failed_runs,
                )
            last_failed_free = free_before
            last_failed_largest = self._largest_free_block()
            last_failed_runs = self._free_block_count()
            wait_rounds += 1
            self._changed.clear()
            await self._changed.wait()

    def release(self, offset: int, num_slots: int) -> None:
        if offset % self.slot_size != 0:
            raise ValueError("offset must be slot aligned")
        self._do_release(offset // self.slot_size, num_slots)
        self._changed.set()

    def layout_snapshot(self) -> dict[str, int]:
        return {
            "free_slots": self._free_slot_count(),
            "largest_free_run": self._largest_free_block(),
            "free_runs": self._free_block_count(),
        }


def _round_up_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


class BuddyAllocator(_WaitingAllocatorBase):
    """Classic buddy allocator over slot indices. Requests round up to the
    next power of two (higher internal waste, near-zero external
    fragmentation for aligned blocks)."""

    def __init__(self, *, slot_count: int, slot_size: int) -> None:
        super().__init__(slot_count=slot_count, slot_size=slot_size)
        if slot_count & (slot_count - 1):
            raise ValueError("BuddyAllocator requires power-of-two slot_count")
        self._max_order = slot_count.bit_length() - 1
        self._free_lists: dict[int, set[int]] = {
            order: set() for order in range(self._max_order + 1)
        }
        self._free_lists[self._max_order].add(0)
        self._allocated: dict[int, int] = {}  # slot_index -> order
        self._free_slots = slot_count

    def _try_acquire(self, num_slots: int) -> int | None:
        order = (_round_up_pow2(num_slots)).bit_length() - 1
        source = next(
            (
                candidate
                for candidate in range(order, self._max_order + 1)
                if self._free_lists[candidate]
            ),
            None,
        )
        if source is None:
            return None
        index = min(self._free_lists[source])
        self._free_lists[source].remove(index)
        while source > order:
            source -= 1
            self._free_lists[source].add(index + (1 << source))
        self._allocated[index] = order
        self._free_slots -= 1 << order
        return index

    def _do_release(self, slot_index: int, num_slots: int) -> None:
        del num_slots  # buddy tracks its own rounded order
        order = self._allocated.pop(slot_index)
        self._free_slots += 1 << order
        index = slot_index
        while order < self._max_order:
            buddy = index ^ (1 << order)
            if buddy not in self._free_lists[order]:
                break
            self._free_lists[order].remove(buddy)
            index = min(index, buddy)
            order += 1
        self._free_lists[order].add(index)

    def _free_slot_count(self) -> int:
        return self._free_slots

    def _largest_free_block(self) -> int:
        for order in range(self._max_order, -1, -1):
            if self._free_lists[order]:
                return 1 << order
        return 0

    def _free_block_count(self) -> int:
        return sum(len(entries) for entries in self._free_lists.values())


class SizeClassAllocator(_WaitingAllocatorBase):
    """Strictly partitioned power-of-two size classes. Each class owns a
    dedicated region sized by ``class_fractions``; classes never borrow, so
    a hot class hits capacity waits while other regions sit idle -- the
    classic isolation trade-off."""

    def __init__(
        self,
        *,
        slot_count: int,
        slot_size: int,
        class_fractions: dict[int, float] | None = None,
    ) -> None:
        super().__init__(slot_count=slot_count, slot_size=slot_size)
        fractions = class_fractions or {1: 0.25, 2: 0.125}
        self._regions: list[dict[str, Any]] = []
        cursor = 0
        for block_slots, fraction in sorted(fractions.items()):
            region_slots = int(slot_count * fraction)
            blocks = region_slots // block_slots
            if blocks == 0:
                continue
            self._regions.append(
                {
                    "block_slots": block_slots,
                    "start": cursor,
                    "free": set(range(blocks)),
                    "blocks": blocks,
                }
            )
            cursor += blocks * block_slots
        if cursor > slot_count:
            raise ValueError("class fractions exceed the pool")
        # Whatever is left becomes a single large-object block, so any request
        # up to the leftover region size is admissible.
        leftover = slot_count - cursor
        if leftover > 0:
            self._regions.append(
                {
                    "block_slots": leftover,
                    "start": cursor,
                    "free": {0},
                    "blocks": 1,
                }
            )
        self._block_of: dict[int, tuple[dict[str, Any], int]] = {}

    def _region_for(self, num_slots: int) -> dict[str, Any] | None:
        for region in self._regions:
            if region["block_slots"] >= num_slots:
                return region
        return None

    def _try_acquire(self, num_slots: int) -> int | None:
        region = self._region_for(num_slots)
        if region is None:
            raise ValueError(f"no size class can hold {num_slots} slots")
        if not region["free"]:
            return None
        block = min(region["free"])
        region["free"].remove(block)
        slot_index = region["start"] + block * region["block_slots"]
        self._block_of[slot_index] = (region, block)
        return slot_index

    def _do_release(self, slot_index: int, num_slots: int) -> None:
        del num_slots
        region, block = self._block_of.pop(slot_index)
        if block in region["free"]:
            raise RuntimeError("size-class block released twice")
        region["free"].add(block)

    def _free_slot_count(self) -> int:
        return sum(
            len(region["free"]) * region["block_slots"] for region in self._regions
        )

    def _largest_free_block(self) -> int:
        largest = 0
        for region in self._regions:
            if region["free"]:
                largest = max(largest, region["block_slots"])
        return largest

    def _free_block_count(self) -> int:
        return sum(len(region["free"]) for region in self._regions)


ALLOCATORS = {
    "first_fit": FirstFitAllocator,
    "buddy": BuddyAllocator,
    "size_class": SizeClassAllocator,
}


# --------------------------------------------------------------------------
# Report entry points
# --------------------------------------------------------------------------


def build_schedule(
    name: str,
    *,
    slot_count: int,
    slot_size: int,
    seed: int = 0,
    large_ratio: float = 0.1,
    count: int | None = None,
    replay_path: str | None = None,
) -> Schedule:
    if name == "fixed":
        return fixed_schedule(count=count or 200)
    if name == "bimodal":
        return bimodal_schedule(count=count or 400, seed=seed, large_ratio=large_ratio)
    if name == "adversarial":
        return adversarial_schedule(slot_count=slot_count, slot_size=slot_size)
    if name == "replay":
        if replay_path is None:
            raise SystemExit(
                "--schedule replay requires --replay-path (JSONL captured from "
                "the GPU comm trace; see benchmarks/comm/cuda_ipc_delay_sweep.py)"
            )
        return replay_schedule(replay_path)
    raise SystemExit(f"unknown schedule {name!r}")


def run_one(
    allocator_name: str,
    schedule: Schedule,
    *,
    slot_count: int,
    slot_size: int,
    max_ticks: int = 100_000,
) -> RunResult:
    allocator = ALLOCATORS[allocator_name](slot_count=slot_count, slot_size=slot_size)
    return asyncio.run(run_schedule(allocator, schedule, max_ticks=max_ticks))


def sweep_large_ratio(
    *,
    slot_count: int,
    slot_size: int,
    seeds: list[int],
    ratios: list[float] | None = None,
    allocator_name: str = "first_fit",
) -> list[dict[str, Any]]:
    """Find where contiguity waits switch on as the large-object share grows:
    the "safe below X" boundary for the RFC response."""
    ratios = ratios or [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    rows = []
    for ratio in ratios:
        contiguity = 0
        capacity = 0
        wait_p99 = 0.0
        occupancy_p99 = 0.0
        for seed in seeds:
            schedule = bimodal_schedule(seed=seed, large_ratio=ratio)
            run = run_one(
                allocator_name,
                schedule,
                slot_count=slot_count,
                slot_size=slot_size,
            )
            summary = run.summary()
            contiguity += summary["contiguity_waits"]
            capacity += summary["capacity_waits"]
            wait_p99 = max(wait_p99, summary["wait_ticks"]["p99"])
            occupancy_p99 = max(occupancy_p99, summary["occupancy"]["p99"])
        rows.append(
            {
                "large_ratio": ratio,
                "seeds": len(seeds),
                "contiguity_waits": contiguity,
                "capacity_waits": capacity,
                "wait_ticks_p99": wait_p99,
                "occupancy_p99": round(occupancy_p99, 4),
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return f"## {title}\n\n(no data)\n"
    headers = list(rows[0])
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule", choices=["fixed", "bimodal", "adversarial", "replay"]
    )
    parser.add_argument(
        "--allocators", nargs="+", default=["first_fit"], choices=list(ALLOCATORS)
    )
    parser.add_argument("--pool-slots", type=int, default=256)
    parser.add_argument("--slot-kb", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--large-ratio", type=float, default=0.1)
    parser.add_argument("--replay-path", default=None)
    parser.add_argument("--sweep-large-ratio", action="store_true")
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=100_000,
        help="virtual-tick budget; real traces can span millions of ticks, "
        "so raise this when replaying one (the driver skips idle ticks)",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny fast run")
    parser.add_argument("--format", choices=["json", "md"], default="json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    slot_size = args.slot_kb * 1024
    count = 40 if args.smoke else args.count

    if args.sweep_large_ratio:
        rows = sweep_large_ratio(
            slot_count=args.pool_slots,
            slot_size=slot_size,
            seeds=args.seeds,
        )
        payload: Any = {
            "kind": "sweep_large_ratio",
            "pool_slots": args.pool_slots,
            "slot_kb": args.slot_kb,
            "seeds": args.seeds,
            "rows": rows,
        }
        text = (
            json.dumps(payload, indent=2)
            if args.format == "json"
            else render_markdown(rows, "contiguity-wait onset vs large_ratio")
        )
    else:
        if args.schedule is None:
            raise SystemExit("pass --schedule or --sweep-large-ratio")
        reports = []
        for allocator_name in args.allocators:
            for seed in args.seeds:
                schedule = build_schedule(
                    args.schedule,
                    slot_count=args.pool_slots,
                    slot_size=slot_size,
                    seed=seed,
                    large_ratio=args.large_ratio,
                    count=count,
                    replay_path=args.replay_path,
                )
                run = run_one(
                    allocator_name,
                    schedule,
                    slot_count=args.pool_slots,
                    slot_size=slot_size,
                    max_ticks=args.max_ticks,
                )
                summary = run.summary()
                summary["allocator"] = allocator_name
                summary["seed"] = seed
                reports.append(summary)
        payload = {
            "kind": "schedule_run",
            "pool_slots": args.pool_slots,
            "slot_kb": args.slot_kb,
            "reports": reports,
        }
        if args.format == "json":
            text = json.dumps(payload, indent=2)
        else:
            flat = [
                {
                    "allocator": r["allocator"],
                    "seed": r["seed"],
                    "granted": r["granted"],
                    "unsatisfiable": r["unsatisfiable"],
                    "occ_p99": round(r["occupancy"]["p99"], 3),
                    "wait_p99": r["wait_ticks"]["p99"],
                    "contiguity_waits": r["contiguity_waits"],
                    "capacity_waits": r["capacity_waits"],
                    "waste_ratio": round(r["internal_waste_ratio"], 4),
                }
                for r in reports
            ]
            text = render_markdown(flat, f"{args.schedule} schedule")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
