# SPDX-License-Identifier: Apache-2.0
"""Shared-memory admission: one soft global in-flight bound across N DPs.

Layout: one 64-byte slot per DP in a supervisor-owned file, mmap'd by every
process. Each slot is single-writer (its DP); readers (sibling DPs summing
in-flight, the CP aggregating /health, the supervisor) use a per-slot
seqlock: the writer bumps seq to an odd value, writes all fields, then bumps
seq to the next even value; a reader rereads while seq is odd or changed
mid-read. Correctness relies on the seqlock protocol, not on aligned 8-byte
stores being atomic.

Platform restriction (documented, enforced with a warning): the pure-Python
seqlock relies on x86-64 total-store-order semantics plus CPython performing
pack_into as a C-level copy under the GIL; the sequence word is not a native
atomic. Deployment targets are x86-64 Linux only; other architectures are
unsupported for multi-process admission and log a warning at construction.

The slot array carries one extra supervisor-owned RETIRED slot after the DP
slots: on reclaim the dying DP's rejected_total and peak are folded into it,
so those aggregate values survive DP restarts. This is best-effort across
crashes, not exact: a DP killed mid-write leaves its slot unreadable and its
counters are skipped at reclaim (logged), and a shed taken while the DP's own
slot is mid-fold cannot be recorded at all, so the aggregates can lose (never
invent) rejections around a DP crash.

Documented semantics (the bound is a shedding knob, not an invariant):
- the global bound is SOFT: concurrent check-then-increment across processes
  can overshoot by at most N-1 requests;
- inflight is an instantaneous sum of per-slot snapshots, not a linearizable
  value;
- rejected_total is a sum of single-writer counters: exact while DPs are
  live, best-effort across DP crashes (see the retired-slot note above);
- peak_inflight is best-effort: each DP records the highest SUM it observed
  at its own acquire time and the aggregate takes the max; there is no
  global atomic max;
- fencing: acquire/release verify the slot still carries this process's
  (generation, pid); a mismatch means the supervisor reclaimed the slot and
  the process must stop serving (on_fenced is invoked, the operation is a
  no-op);
- slot reclaim is supervisor-only and happens strictly after the owning
  process was reaped (waitpid), so the single-writer discipline holds.
"""

from __future__ import annotations

import logging
import platform
import struct
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger("sglang_omni_router.admission_shm")

_PLATFORM_WARNED = False


def _warn_if_unverified_platform() -> None:
    global _PLATFORM_WARNED
    if _PLATFORM_WARNED:
        return
    _PLATFORM_WARNED = True
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        logger.warning(
            f"shared admission's seqlock is only validated on x86-64 (TSO "
            f"store ordering); machine={machine} is unsupported for "
            "multi-process admission"
        )


SLOT_SIZE = 64
_SLOT_FORMAT = "<qqqqqqdq"
_FIELDS_FORMAT = "<qqqqqdq"
_SEQ_FORMAT = "<q"
# Note (Jiaxin Deng): a live writer finishes in microseconds; a slot left odd
# by a dead writer resolves at supervisor reclaim, so readers eventually raise.
_READ_SPIN_LIMIT = 1_000
_READ_BACKOFF_SECS = 0.0001
_READ_BACKOFF_LIMIT = 20_000
# Note (Jiaxin Deng): a slot stays odd until supervisor reclaim, so probing it
# per request would burn the spin budget (and a log line) on every request for
# a whole poll interval; cache the fail-closed shed and re-probe at this rate.
_UNSTABLE_REPROBE_SECS = 0.05

assert struct.calcsize(_SLOT_FORMAT) == SLOT_SIZE


@dataclass
class SlotView:
    seq: int
    inflight: int
    peak_sum: int
    rejected_total: int
    generation: int
    pid: int
    heartbeat_ts: float


def admission_file_size(slots: int) -> int:
    # Note (Jiaxin Deng): the extra slot is the supervisor-owned retired fold.
    return (slots + 1) * SLOT_SIZE


def retired_slot_index(slots: int) -> int:
    return slots


def create_admission_file(path: str, slots: int) -> None:
    with open(path, "wb") as f:
        f.write(b"\x00" * admission_file_size(slots))


class SeqlockUnstableError(RuntimeError):
    """A slot did not stabilize within the fail-fast spin budget."""


class SlotCodec:
    """Seqlock accessor for one slot; write() may only be called by the
    slot's single writer (its DP, or the supervisor after reaping it)."""

    def __init__(self, buf, index: int) -> None:
        self._buf = buf
        self._offset = index * SLOT_SIZE

    def read(self, *, fail_fast: bool = False) -> SlotView:
        # Note (Jiaxin Deng): fail_fast spins without sleeping; request event
        # loops must not block on a slot left odd by a crashed writer.
        budget = (
            _READ_SPIN_LIMIT if fail_fast else _READ_SPIN_LIMIT + _READ_BACKOFF_LIMIT
        )
        for attempt in range(budget):
            seq_before = struct.unpack_from(_SEQ_FORMAT, self._buf, self._offset)[0]
            if seq_before % 2:
                if not fail_fast and attempt >= _READ_SPIN_LIMIT:
                    time.sleep(_READ_BACKOFF_SECS)
                continue  # writer mid-flight
            fields = struct.unpack_from(_SLOT_FORMAT, self._buf, self._offset)
            seq_after = struct.unpack_from(_SEQ_FORMAT, self._buf, self._offset)[0]
            if seq_before == seq_after:
                return SlotView(*fields[:7])
        raise SeqlockUnstableError("seqlock read did not stabilize")

    def write(
        self,
        *,
        inflight: int,
        peak_sum: int,
        rejected_total: int,
        generation: int,
        pid: int,
        heartbeat_ts: float,
    ) -> None:
        # Note (Jiaxin Deng): seq | 1 tolerates a slot left odd by a writer
        # that died mid-write; the final value is the next even number
        # strictly above what any reader saw.
        seq = struct.unpack_from(_SEQ_FORMAT, self._buf, self._offset)[0]
        writing_marker = seq | 1
        struct.pack_into(_SEQ_FORMAT, self._buf, self._offset, writing_marker)
        struct.pack_into(
            _FIELDS_FORMAT,
            self._buf,
            self._offset + 8,
            inflight,
            peak_sum,
            rejected_total,
            generation,
            pid,
            heartbeat_ts,
            0,
        )
        struct.pack_into(_SEQ_FORMAT, self._buf, self._offset, writing_marker + 1)

    def begin_write(self) -> int:
        """Mark this slot mid-write (odd seq) and return the marker.

        Lets the supervisor hold a slot in the in-progress state across a
        multi-slot operation (the retired-fold) so aggregate readers retry
        until the whole operation completes. Pair with write_fields + end_write.
        """
        seq = struct.unpack_from(_SEQ_FORMAT, self._buf, self._offset)[0]
        marker = seq | 1
        struct.pack_into(_SEQ_FORMAT, self._buf, self._offset, marker)
        return marker

    def write_fields(
        self,
        *,
        inflight: int,
        peak_sum: int,
        rejected_total: int,
        generation: int,
        pid: int,
        heartbeat_ts: float,
    ) -> None:
        struct.pack_into(
            _FIELDS_FORMAT,
            self._buf,
            self._offset + 8,
            inflight,
            peak_sum,
            rejected_total,
            generation,
            pid,
            heartbeat_ts,
            0,
        )

    def end_write(self, marker: int) -> None:
        struct.pack_into(_SEQ_FORMAT, self._buf, self._offset, marker + 1)

    def reclaim(self, *, now: float | None = None) -> None:
        """Supervisor-only, and only after the owner was reaped."""
        self.write(
            inflight=0,
            peak_sum=0,
            rejected_total=0,
            generation=0,
            pid=0,
            heartbeat_ts=now if now is not None else time.time(),
        )


class SharedAdmission:
    """Drop-in for AdmissionController backed by the shared slot array.

    try_acquire keeps the check and its own-slot increment inside one
    synchronous section (no await), mirroring the single-process contract.
    """

    def __init__(
        self,
        buf,
        *,
        slots: int,
        own_index: int,
        max_inflight: int,
        generation: int,
        pid: int,
        on_fenced: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 0 <= own_index < slots:
            raise ValueError(f"own_index {own_index} outside {slots} slots")
        _warn_if_unverified_platform()
        self._codecs = [SlotCodec(buf, index) for index in range(slots)]
        self._retired_codec = SlotCodec(buf, retired_slot_index(slots))
        self._own = self._codecs[own_index]
        self._max_inflight = max_inflight
        self._generation = generation
        self._pid = pid
        self._on_fenced = on_fenced
        self._clock = clock
        self._inflight = 0
        self._peak_sum = 0
        self._rejected_total = 0
        self._sibling_unstable = False
        self._sibling_probed_at = 0.0
        self._write_own()  # claim the slot

    def _write_own(self) -> None:
        self._own.write(
            inflight=self._inflight,
            peak_sum=self._peak_sum,
            rejected_total=self._rejected_total,
            generation=self._generation,
            pid=self._pid,
            heartbeat_ts=self._clock(),
        )

    def _still_owner(self) -> bool:
        view = self._own.read(fail_fast=True)
        return view.generation == self._generation and view.pid == self._pid

    def _fenced(self) -> None:
        if self._on_fenced is not None:
            self._on_fenced()

    def _total_inflight(self) -> int:
        return sum(codec.read(fail_fast=True).inflight for codec in self._codecs)

    def _shed_from_cached_unstable(self) -> bool:
        return (
            self._sibling_unstable
            and self._clock() - self._sibling_probed_at < _UNSTABLE_REPROBE_SECS
        )

    def _mark_sibling_unstable(self) -> None:
        self._sibling_probed_at = self._clock()
        if not self._sibling_unstable:
            self._sibling_unstable = True
            logger.warning("sibling admission slot unstable; shedding until reclaim")

    def _mark_sibling_stable(self) -> None:
        if self._sibling_unstable:
            self._sibling_unstable = False
            logger.warning("sibling admission slots readable again; resuming admission")

    @property
    def inflight(self) -> int:
        return self._total_inflight()

    def try_acquire(self) -> bool:
        try:
            if not self._still_owner():
                self._fenced()
                return False
        except SeqlockUnstableError:
            # Note (Jiaxin Deng): own slot mid-fold by the supervisor; writing
            # now would race the fold, so this shed cannot be counted.
            logger.warning("own admission slot unstable; shedding until reclaim")
            return False
        # Note (Jiaxin Deng): a sibling slot is stuck mid-write (writer
        # crashed); shed instead of blocking the event loop. The own slot is
        # intact, so the rejection is counted like any other shed.
        if self._shed_from_cached_unstable():
            self._rejected_total += 1
            self._write_own()
            return False
        try:
            total = self._total_inflight()
        except SeqlockUnstableError:
            self._mark_sibling_unstable()
            self._rejected_total += 1
            self._write_own()
            return False
        self._mark_sibling_stable()
        if total >= self._max_inflight:
            self._rejected_total += 1
            self._write_own()
            return False
        self._inflight += 1
        observed = total + 1
        if observed > self._peak_sum:
            self._peak_sum = observed
        self._write_own()
        return True

    def release(self) -> None:
        try:
            if not self._still_owner():
                self._fenced()
                return
        except SeqlockUnstableError:
            logger.warning("own admission slot unstable on release; skipping")
            return
        if self._inflight <= 0:
            # Note (Jiaxin Deng): not an assert (stripped under python -O); a
            # stray release must not drive the shared count negative.
            logger.error("admission slot released with no in-flight request; ignoring")
            return
        self._inflight -= 1
        self._write_own()

    def touch(self) -> None:
        """Refresh the slot heartbeat (idle DPs otherwise never write)."""
        if self._still_owner():
            self._write_own()

    def to_dict(self) -> dict[str, int]:
        return _aggregate_to_dict(self._codecs, self._retired_codec, self._max_inflight)


def _aggregate_to_dict(
    codecs: list[SlotCodec], retired_codec: SlotCodec, max_inflight: int
) -> dict[str, int]:
    # Note (Jiaxin Deng): a fold is a two-slot transfer the per-slot seqlocks
    # cannot make atomic; use the retired slot's seq as a fold version and
    # retry, fail-fast so event-loop readers never sleep.
    for _ in range(4):
        version_before = retired_codec.read(fail_fast=True).seq
        views = [codec.read(fail_fast=True) for codec in codecs]
        retired = retired_codec.read(fail_fast=True)
        if retired.seq == version_before:
            return {
                "inflight": sum(view.inflight for view in views),
                "max_inflight": max_inflight,
                "peak_inflight": max(
                    (view.peak_sum for view in views + [retired]), default=0
                ),
                "rejected_total": (
                    sum(view.rejected_total for view in views) + retired.rejected_total
                ),
            }
    raise SeqlockUnstableError("aggregate read did not stabilize")


class AdmissionAggregateView:
    """Read-only aggregate over the slot array (CP /health, supervisor)."""

    def __init__(self, buf, slots: int) -> None:
        self._codecs = [SlotCodec(buf, index) for index in range(slots)]
        self._retired_codec = SlotCodec(buf, retired_slot_index(slots))

    def per_slot(self, *, now: float | None = None) -> list[dict]:
        current = now if now is not None else time.time()
        result = []
        for index, codec in enumerate(self._codecs):
            # Note (Jiaxin Deng): /health runs on the event loop; a slot left
            # odd by a dead writer must not block it on a backoff sleep.
            view = codec.read(fail_fast=True)
            result.append(
                {
                    "index": index,
                    "generation": view.generation,
                    "pid": view.pid,
                    "inflight": view.inflight,
                    "rejected_total": view.rejected_total,
                    "heartbeat_age_secs": (
                        round(current - view.heartbeat_ts, 3)
                        if view.heartbeat_ts > 0
                        else None
                    ),
                }
            )
        return result

    def to_dict(self, max_inflight: int) -> dict[str, int]:
        return _aggregate_to_dict(self._codecs, self._retired_codec, max_inflight)
