from __future__ import annotations

import struct
import threading
import time

import pytest

from sglang_omni_router.python.admission_shm import (
    SLOT_SIZE,
    AdmissionAggregateView,
    SharedAdmission,
    SlotCodec,
    admission_file_size,
)
from sglang_omni_router.python.proxy import AdmissionController


def _buf(slots: int = 3) -> bytearray:
    return bytearray(admission_file_size(slots))


def _admission(buf, index: int, *, bound: int = 4, generation: int = 1, **kwargs):
    return SharedAdmission(
        buf,
        slots=len(buf) // SLOT_SIZE - 1,  # minus the retired slot
        own_index=index,
        max_inflight=bound,
        generation=generation,
        pid=100 + index,
        **kwargs,
    )


def test_bound_is_enforced_across_processes() -> None:
    buf = _buf()
    a = _admission(buf, 0, bound=3)
    b = _admission(buf, 1, bound=3)

    assert a.try_acquire() and a.try_acquire()
    assert b.try_acquire()
    # Note (Jiaxin Deng): bound reached across slots: both sides shed
    assert b.try_acquire() is False
    assert a.try_acquire() is False

    a.release()
    assert b.try_acquire() is True
    assert b.to_dict()["rejected_total"] == 2


def test_soft_bound_overshoot_is_at_most_n_minus_one() -> None:
    # Note (Jiaxin Deng): simulate the stale-read race: B captured the sum before A's
    # increment
    buf = _buf()
    a = _admission(buf, 0, bound=3)
    b = _admission(buf, 1, bound=3)
    c = _admission(buf, 2, bound=3)

    assert a.try_acquire() and a.try_acquire()  # sum = 2 = bound - 1
    stale_total = b._total_inflight()
    assert a.try_acquire() is True  # sum = 3 = bound
    b._total_inflight = lambda: stale_total  # type: ignore[method-assign]
    assert b.try_acquire() is True  # stale check admits: sum = 4 = bound + 1

    # Note (Jiaxin Deng): overshoot is bounded by N-1 and a fresh reader immediately
    # sheds
    assert AdmissionAggregateView(buf, 3).to_dict(3)["inflight"] == 4  # 3 + (2-1)
    assert c.try_acquire() is False


def test_seqlock_reader_never_returns_a_torn_slot() -> None:
    buf = _buf(1)
    codec = SlotCodec(buf, 0)
    codec.write(
        inflight=1,
        peak_sum=1,
        rejected_total=0,
        generation=7,
        pid=42,
        heartbeat_ts=1.0,
    )
    # Note (Jiaxin Deng): simulate a writer dying mid-write: odd seq + garbage fields
    seq = struct.unpack_from("<q", buf, 0)[0]
    struct.pack_into("<q", buf, 0, seq + 1)  # odd: write in progress
    struct.pack_into("<q", buf, 8, 999_999)  # garbage inflight

    results: list[int] = []
    reader = threading.Thread(target=lambda: results.append(codec.read().inflight))
    reader.start()
    time.sleep(0.05)
    assert results == []  # reader is retrying, not returning garbage

    struct.pack_into("<q", buf, 8, 2)
    struct.pack_into("<q", buf, 0, seq + 2)
    reader.join(timeout=3.0)
    assert results == [2]


def test_shed_on_a_sibling_stuck_slot_counts_as_a_rejection() -> None:
    # Note (Jiaxin Deng): the client received a rejection, so /health must count it: the
    # own slot is intact even though a sibling's writer died mid-write
    buf = _buf(2)
    stuck = SlotCodec(buf, 0)
    stuck.write(
        inflight=0, peak_sum=0, rejected_total=0, generation=1, pid=1, heartbeat_ts=1.0
    )
    seq = struct.unpack_from("<q", buf, 0)[0]
    struct.pack_into("<q", buf, 0, seq + 1)  # slot 0 left odd: writer died

    b = _admission(buf, 1)
    assert b.try_acquire() is False
    assert SlotCodec(buf, 1).read(fail_fast=True).rejected_total == 1


def test_unstable_sibling_is_cached_not_respun_on_every_request(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Note (Jiaxin Deng): a DP killed mid-write leaves its slot odd until the supervisor
    # reclaims it (up to one poll interval); every request in that window must shed
    # without re-spinning the seqlock budget and without its own log line, while still
    # being counted, and must admit again once the slot is healed
    import logging

    import sglang_omni_router.python.admission_shm as shm

    buf = _buf(2)
    stuck = SlotCodec(buf, 0)
    stuck.write(
        inflight=0, peak_sum=0, rejected_total=0, generation=1, pid=1, heartbeat_ts=1.0
    )
    seq = struct.unpack_from("<q", buf, 0)[0]
    struct.pack_into("<q", buf, 0, seq + 1)  # slot 0 left odd: writer died

    now = [1000.0]
    b = _admission(buf, 1, clock=lambda: now[0])

    failed_probes: list[int] = []
    original_read = shm.SlotCodec.read

    def counting_read(self, *, fail_fast: bool = False):
        try:
            return original_read(self, fail_fast=fail_fast)
        except shm.SeqlockUnstableError:
            failed_probes.append(self._offset)
            raise

    monkeypatch.setattr(shm.SlotCodec, "read", counting_read)

    burst = 200
    with caplog.at_level(logging.WARNING):
        for _ in range(burst):
            assert b.try_acquire() is False

        assert len(failed_probes) == 1  # probed once, the rest shed from cache
        unstable_logs = [
            record
            for record in caplog.records
            if "sibling admission slot unstable" in record.getMessage()
        ]
        assert len(unstable_logs) == 1  # one transition, not one per request
        assert SlotCodec(buf, 1).read(fail_fast=True).rejected_total == burst

        # Note (Jiaxin Deng): supervisor reclaims the dead slot: service resumes on the
        # next probe
        SlotCodec(buf, 0).reclaim(now=0.0)
        now[0] += 1.0
        assert b.try_acquire() is True
        assert len(failed_probes) == 1


def test_cp_aggregate_reads_fail_fast_and_never_block_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Note (Jiaxin Deng): a DP killed mid-write leaves its slot odd until reclaim;
    # event-loop aggregate readers must fail fast (spin, never sleep) instead of
    # blocking.
    import sglang_omni_router.python.admission_shm as shm

    sleeps: list[float] = []
    monkeypatch.setattr(shm.time, "sleep", lambda secs: sleeps.append(secs))

    buf = _buf(2)
    codec = SlotCodec(buf, 0)
    codec.write(
        inflight=1, peak_sum=1, rejected_total=0, generation=1, pid=1, heartbeat_ts=1.0
    )
    seq = struct.unpack_from("<q", buf, 0)[0]
    struct.pack_into("<q", buf, 0, seq + 1)  # slot 0 left odd: writer died

    view = AdmissionAggregateView(buf, 2)
    with pytest.raises(shm.SeqlockUnstableError):
        view.per_slot()
    with pytest.raises(shm.SeqlockUnstableError):
        view.to_dict(4)
    # Note (Jiaxin Deng): the DP-side aggregate read (DP /health) must be fail-fast too
    admission = _admission(buf, 1)
    with pytest.raises(shm.SeqlockUnstableError):
        admission.to_dict()
    assert sleeps == []  # never slept: the event loop was not blocked


def test_fenced_process_cannot_acquire_or_release() -> None:
    buf = _buf()
    fenced: list[bool] = []
    a = _admission(buf, 0, on_fenced=lambda: fenced.append(True))
    assert a.try_acquire() is True

    # Note (Jiaxin Deng): supervisor reclaims the slot (only ever after reaping the
    # owner)
    SlotCodec(buf, 0).reclaim(now=0.0)

    assert a.try_acquire() is False
    a.release()  # must be a no-op, not corrupt the reclaimed slot
    assert fenced == [True, True]
    view = SlotCodec(buf, 0).read()
    assert view.generation == 0
    assert view.inflight == 0
    # Note (Jiaxin Deng): fence rejections are not admission rejections
    assert AdmissionAggregateView(buf, 3).to_dict(4)["rejected_total"] == 0


def test_slot_reclaim_frees_a_dead_dps_inflight() -> None:
    buf = _buf()
    a = _admission(buf, 0, bound=2)
    b = _admission(buf, 1, bound=2)
    assert a.try_acquire() and a.try_acquire()
    assert b.try_acquire() is False

    SlotCodec(buf, 0).reclaim()  # A crashed and was reaped

    assert b.try_acquire() is True  # capacity is back
    assert b.try_acquire() is True
    assert b.try_acquire() is False


def test_peak_is_the_best_effort_max_observed_sum() -> None:
    buf = _buf()
    a = _admission(buf, 0, bound=10)
    b = _admission(buf, 1, bound=10)
    a.try_acquire()
    b.try_acquire()
    b.try_acquire()  # b observes sum 3 at acquire time
    a.release()
    b.release()
    assert b.to_dict()["peak_inflight"] == 3


def test_to_dict_matches_the_single_process_surface() -> None:
    buf = _buf()
    shared_keys = set(_admission(buf, 0).to_dict())
    local_keys = set(AdmissionController(4).to_dict())
    assert shared_keys == local_keys


def test_release_below_zero_is_clamped_not_asserted() -> None:
    # Note (Jiaxin Deng): not an assert (stripped under python -O): a stray release is
    # clamped and logged, never driving the shared in-flight count negative
    buf = _buf()
    a = _admission(buf, 0)
    a.release()  # no in-flight: ignored, not fatal
    assert AdmissionAggregateView(buf, 3).to_dict(4)["inflight"] == 0


def test_aggregate_view_reports_per_slot_details() -> None:
    buf = _buf()
    a = _admission(buf, 0, generation=3)
    a.try_acquire()
    per_slot = AdmissionAggregateView(buf, 3).per_slot(now=time.time())
    assert per_slot[0]["generation"] == 3
    assert per_slot[0]["inflight"] == 1
    assert per_slot[0]["heartbeat_age_secs"] is not None
    assert per_slot[1]["generation"] == 0  # unclaimed slot
    assert per_slot[1]["heartbeat_age_secs"] is None


def test_reclaim_normalizes_a_slot_left_odd_by_a_mid_write_crash() -> None:
    # Note (Jiaxin Deng): a DP that died between the two seq bumps leaves an odd seq;
    # the supervisor's reclaim must restore an even, readable slot
    buf = _buf(1)
    codec = SlotCodec(buf, 0)
    codec.write(
        inflight=5,
        peak_sum=5,
        rejected_total=0,
        generation=1,
        pid=42,
        heartbeat_ts=1.0,
    )
    seq = struct.unpack_from("<q", buf, 0)[0]
    struct.pack_into("<q", buf, 0, seq + 1)  # crash mid-write: odd forever

    SlotCodec(buf, 0).reclaim(now=0.0)

    view = codec.read()  # must not hang or raise
    assert view.seq % 2 == 0
    assert view.inflight == 0
    assert view.generation == 0


def test_fail_fast_read_sheds_instead_of_blocking() -> None:
    import time as time_module

    buf = _buf()
    a = _admission(buf, 0, bound=4)
    # Note (Jiaxin Deng): sibling slot 1 is stuck odd (its writer crashed mid-write)
    seq = struct.unpack_from("<q", buf, SLOT_SIZE)[0]
    struct.pack_into("<q", buf, SLOT_SIZE, seq + 1)

    started = time_module.monotonic()
    assert a.try_acquire() is False  # shed, not admitted on unknown state
    elapsed = time_module.monotonic() - started
    assert elapsed < 0.5  # no sleep-backoff on the request path


def test_retired_slot_keeps_rejected_and_peak_across_reclaim() -> None:
    from sglang_omni_router.python.admission_shm import retired_slot_index

    buf = _buf()
    a = _admission(buf, 0, bound=1)
    assert a.try_acquire() is True
    assert a.try_acquire() is False  # rejected_total = 1, peak = 1

    # Note (Jiaxin Deng): supervisor folds before reclaiming (what
    # _fold_and_reclaim_slot does)
    view = SlotCodec(buf, 0).read()
    retired = SlotCodec(buf, retired_slot_index(3))
    accumulated = retired.read()
    retired.write(
        inflight=0,
        peak_sum=max(accumulated.peak_sum, view.peak_sum),
        rejected_total=accumulated.rejected_total + view.rejected_total,
        generation=0,
        pid=0,
        heartbeat_ts=0.0,
    )
    SlotCodec(buf, 0).reclaim()

    aggregate = AdmissionAggregateView(buf, 3).to_dict(1)
    assert aggregate["rejected_total"] == 1  # not reset by the crash
    assert aggregate["peak_inflight"] == 1
    assert aggregate["inflight"] == 0


def test_platform_warning_fires_off_x86(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from sglang_omni_router.python import admission_shm

    monkeypatch.setattr(admission_shm, "_PLATFORM_WARNED", False)
    monkeypatch.setattr(admission_shm.platform, "machine", lambda: "aarch64")
    with caplog.at_level(logging.WARNING):
        _admission(_buf(), 0)
    assert "unsupported" in caplog.text


def test_fold_is_atomic_across_the_two_slot_transfer() -> None:
    # Note (Jiaxin Deng): a reader between retired.write_fields and dying.reclaim must
    # not double count: the retired slot stays mid-write (odd) across both, so it
    # retries
    from sglang_omni_router.python.admission_shm import (
        SeqlockUnstableError,
        retired_slot_index,
    )

    buf = _buf()  # 3 DP slots + retired
    dying = SlotCodec(buf, 0)
    dying.write(
        inflight=0, peak_sum=1, rejected_total=7, generation=1, pid=5, heartbeat_ts=1.0
    )
    retired = SlotCodec(buf, retired_slot_index(3))

    view = dying.read()
    accumulated = retired.read()
    marker = retired.begin_write()  # fold in progress: retired seq is odd
    retired.write_fields(
        inflight=0,
        peak_sum=max(accumulated.peak_sum, view.peak_sum),
        rejected_total=accumulated.rejected_total + view.rejected_total,
        generation=0,
        pid=0,
        heartbeat_ts=0.0,
    )
    # Note (Jiaxin Deng): a reader hitting the aggregate now must not see 7 (dying) + 7
    # (retired)
    with pytest.raises(SeqlockUnstableError):
        AdmissionAggregateView(buf, 3).to_dict(10)

    dying.reclaim()
    retired.end_write(marker)
    assert AdmissionAggregateView(buf, 3).to_dict(10)["rejected_total"] == 7
