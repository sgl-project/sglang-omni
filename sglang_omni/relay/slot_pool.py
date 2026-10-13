# SPDX-License-Identifier: Apache-2.0
"""Contiguous slot bookkeeping shared by the CUDA and Level-Zero IPC relay pools."""
from __future__ import annotations

import asyncio
from typing import NamedTuple


class SlotLayout(NamedTuple):
    slot_index: int | None
    free_slots: int
    largest_free_run: int
    free_runs: int


class SlotAllocation(NamedTuple):
    offset: int
    wait_rounds: int
    free_slots_before: int
    largest_free_run_before: int
    free_runs_before: int
    last_failed_free_slots: int
    last_failed_largest_free_run: int
    last_failed_free_runs: int


def slots_for_size(size: int, slot_size: int) -> int:
    if size < 0:
        raise ValueError("transfer size must be non-negative")
    else:
        pass
    return max(1, (size + slot_size - 1) // slot_size)


class ContiguousSlotAllocator:
    def __init__(self, *, slot_count: int, slot_size: int) -> None:
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        else:
            pass
        if slot_size <= 0:
            raise ValueError("slot_size must be positive")
        else:
            pass
        self.slot_count = slot_count
        self.slot_size = slot_size
        self.free = [True] * slot_count
        self.free_slots = slot_count
        self.lock = asyncio.Lock()
        self.changed = asyncio.Event()
        self.changed.set()

    async def acquire_async(
        self, num_slots: int, *, capture_layout: bool = False
    ) -> SlotAllocation:
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        else:
            pass
        if num_slots > self.slot_count:
            raise ValueError(
                f"allocation requires {num_slots} slots, but pool has {self.slot_count}"
            )
        else:
            pass

        wait_rounds = 0
        last_failed_free_slots = 0
        last_failed_largest_free_run = 0
        last_failed_free_runs = 0
        while True:
            async with self.lock:
                slot_index = self.find_contiguous(num_slots)
                free_slots_before = self.free_slots
                if slot_index is not None:
                    for index in range(slot_index, slot_index + num_slots):
                        self.free[index] = False
                    self.free_slots -= num_slots
                    if self.free_slots == 0:
                        self.changed.clear()
                    else:
                        pass
                    return SlotAllocation(
                        offset=slot_index * self.slot_size,
                        wait_rounds=wait_rounds,
                        free_slots_before=free_slots_before,
                        largest_free_run_before=-1,
                        free_runs_before=-1,
                        last_failed_free_slots=last_failed_free_slots,
                        last_failed_largest_free_run=last_failed_largest_free_run,
                        last_failed_free_runs=last_failed_free_runs,
                    )
                else:
                    pass
                if capture_layout:
                    layout = self.find_contiguous_with_layout(num_slots)
                    last_failed_free_slots = free_slots_before
                    last_failed_largest_free_run = layout.largest_free_run
                    last_failed_free_runs = layout.free_runs
                else:
                    pass
                wait_rounds += 1
                self.changed.clear()
            await self.changed.wait()

    def release(self, offset: int, num_slots: int) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        else:
            pass
        if offset % self.slot_size != 0:
            raise ValueError("offset must be slot aligned")
        else:
            pass
        slot_index = offset // self.slot_size
        if slot_index < 0 or slot_index + num_slots > self.slot_count:
            raise ValueError("slot range is outside the pool")
        else:
            pass
        for index in range(slot_index, slot_index + num_slots):
            if self.free[index]:
                raise RuntimeError("slot released twice")
            else:
                pass
        for index in range(slot_index, slot_index + num_slots):
            self.free[index] = True
        self.free_slots += num_slots
        self.changed.set()

    def find_contiguous(self, num_slots: int) -> int | None:
        run_start = 0
        run_len = 0
        for index, is_free in enumerate(self.free):
            if is_free:
                if run_len == 0:
                    run_start = index
                else:
                    pass
                run_len += 1
                if run_len == num_slots:
                    return run_start
                else:
                    pass
            else:
                run_len = 0
        return None

    def find_contiguous_with_layout(self, num_slots: int) -> SlotLayout:
        run_start = 0
        run_len = 0
        free_slots = 0
        free_runs = 0
        largest_free_run = 0
        slot_index: int | None = None
        in_run = False
        for index, is_free in enumerate(self.free):
            if is_free:
                free_slots += 1
                if not in_run:
                    in_run = True
                    free_runs += 1
                    run_start = index
                    run_len = 0
                else:
                    pass
                run_len += 1
                largest_free_run = max(largest_free_run, run_len)
                if slot_index is None and run_len >= num_slots:
                    slot_index = run_start
                else:
                    pass
            else:
                in_run = False
                run_len = 0
        return SlotLayout(
            slot_index=slot_index,
            free_slots=free_slots,
            largest_free_run=largest_free_run,
            free_runs=free_runs,
        )

    def snapshot_layout(self) -> SlotLayout:
        """Return the current pool layout for failure-path diagnostics."""
        return self.find_contiguous_with_layout(1)
