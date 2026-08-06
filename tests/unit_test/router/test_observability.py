from __future__ import annotations

import pytest

from sglang_omni_router.observability import (
    CounterReport,
    DataPlaneCounterLedger,
    StaleCounterGenerationError,
    WorkerCounters,
)


def _report(
    seq: int,
    routed: int,
    *,
    dp: int = 0,
    generation: int = 1,
    active: int = 0,
    worker_id: str = "w0",
    incarnation: str = "",
) -> CounterReport:
    return CounterReport(
        dp_index=dp,
        generation=generation,
        counter_seq=seq,
        workers=[
            WorkerCounters(
                worker_id=worker_id,
                incarnation=incarnation,
                routed_total=routed,
                successful_total=routed,
                failed_total=0,
                current_active=active,
            )
        ],
    )


def test_first_contact_establishes_the_baseline() -> None:
    # Note (Jiaxin Deng): since-CP-start: the first report only sets the baseline;
    # growth after first contact is what gets displayed
    ledger = DataPlaneCounterLedger()
    assert ledger.apply(_report(1, 20), now=0.0) is True
    assert ledger.totals("w0")["routed_total"] == 0
    ledger.apply(_report(2, 25), now=0.0)
    assert ledger.totals("w0")["routed_total"] == 5


def test_cp_restart_restarts_the_window_coherently() -> None:
    # Note (Jiaxin Deng): a fresh ledger (CP restart) takes a surviving DP's cumulative
    # as baseline: the display restarts at zero
    old_cp = DataPlaneCounterLedger()
    old_cp.apply(_report(1, 0), now=0.0)
    old_cp.apply(_report(9, 120), now=0.0)
    assert old_cp.totals("w0")["routed_total"] == 120

    new_cp = DataPlaneCounterLedger()
    new_cp.apply(_report(10, 120), now=0.0)
    assert new_cp.totals("w0")["routed_total"] == 0
    new_cp.apply(_report(11, 130), now=0.0)
    assert new_cp.totals("w0")["routed_total"] == 10


def test_cumulative_reports_are_idempotent_and_ordered() -> None:
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 0), now=0.0)
    assert ledger.apply(_report(2, 5), now=0.0) is True
    # Note (Jiaxin Deng): duplicate delivery of the same seq: dropped, totals unchanged
    assert ledger.apply(_report(2, 5), now=0.0) is False
    # Note (Jiaxin Deng): out-of-order older seq: dropped even with different numbers
    assert ledger.apply(_report(1, 3), now=0.0) is False
    assert ledger.totals("w0")["routed_total"] == 5


def test_a_regressed_cumulative_cannot_lower_the_display() -> None:
    # Note (Jiaxin Deng): e.g. a worker URL deleted and re-added on the DP resets its
    # counters; the display clamps to the high-water mark instead of falling
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 0), now=0.0)
    ledger.apply(_report(2, 10), now=0.0)
    assert ledger.apply(_report(3, 3), now=0.0) is True  # newer seq, lower value
    assert ledger.totals("w0")["routed_total"] == 10


def test_workers_missing_from_a_report_keep_their_contribution() -> None:
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 0), now=0.0)
    ledger.apply(_report(2, 7), now=0.0)
    # Note (Jiaxin Deng): next report omits w0 entirely (e.g. deleted from the registry)
    ledger.apply(
        CounterReport(dp_index=0, generation=1, counter_seq=3, workers=[]),
        now=0.0,
    )
    assert ledger.totals("w0")["routed_total"] == 7


def test_generation_bump_retires_the_old_contribution() -> None:
    # Note (Jiaxin Deng): displayed totals must never move backwards across a DP restart
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 2, generation=1), now=0.0)  # baseline 2
    ledger.apply(_report(7, 10, generation=1), now=0.0)  # contribution 8
    # Note (Jiaxin Deng): the respawned process starts its counters at zero, so
    # its first report counts in full; baselining it there would silently drop
    # everything it served before that flush.
    ledger.apply(_report(1, 3, generation=2), now=0.0)
    assert ledger.totals("w0")["routed_total"] == 11
    ledger.apply(_report(2, 4, generation=2), now=0.0)
    assert ledger.totals("w0")["routed_total"] == 12


def test_only_the_first_report_from_a_data_plane_is_a_baseline() -> None:
    # Note (Jiaxin Deng): a fresh ledger is a CP restart, where the DP has been
    # running a while, so its accumulated cumulative is the baseline.
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(9, 100, generation=4), now=0.0)
    assert ledger.totals("w0")["routed_total"] == 0
    ledger.apply(_report(10, 105, generation=4), now=0.0)
    assert ledger.totals("w0")["routed_total"] == 5


def test_older_generation_reports_are_fenced() -> None:
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 1, generation=2), now=0.0)
    with pytest.raises(StaleCounterGenerationError):
        ledger.apply(_report(9, 9, generation=1), now=0.0)


def test_active_gauge_sums_only_live_entries() -> None:
    ledger = DataPlaneCounterLedger(liveness_secs=3.0)
    ledger.apply(_report(1, 1, dp=0, active=2), now=100.0)
    ledger.apply(_report(1, 1, dp=1, active=3), now=102.0)
    assert ledger.active_gauge("w0", now=102.5) == 5
    # Note (Jiaxin Deng): dp 0's report is now stale: its in-flight work is gone with it
    assert ledger.active_gauge("w0", now=103.5) == 3


def test_incarnation_change_retires_the_old_segment_and_counts_the_new() -> None:
    # Note (Jiaxin Deng): DELETE then POST of the same URL on the DP is a fresh worker
    # object whose cumulatives restart at zero; without retiring the old segment the
    # high-water clamp would freeze the display until the new object caught up
    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 0, incarnation="inc-a"), now=0.0)  # baseline 0
    ledger.apply(_report(2, 10, incarnation="inc-a"), now=1.0)
    assert ledger.totals("w0")["routed_total"] == 10

    # Note (Jiaxin Deng): same stable id, new incarnation, cumulative restarted at 3
    ledger.apply(_report(3, 3, incarnation="inc-b"), now=2.0)
    totals = ledger.totals("w0")
    # Note (Jiaxin Deng): old segment retired (10) + new segment fully counted (3), not
    # clamped
    assert totals["routed_total"] == 13
    ledger.apply(_report(4, 5, incarnation="inc-b"), now=3.0)
    assert ledger.totals("w0")["routed_total"] == 15


def test_overlay_renders_counter_fields() -> None:
    class _Worker:
        worker_id = "w0"

    ledger = DataPlaneCounterLedger()
    ledger.apply(_report(1, 0, active=0))
    ledger.apply(_report(2, 6, active=1))  # real clock: entry counts as live
    overlay = ledger.overlay(_Worker())
    assert overlay == {
        "active_requests": 1,
        "routed_requests": 6,
        "successful_requests": 6,
        "failed_requests": 0,
    }


def _two_incarnation_report(seq: int, rows: list[tuple[str, int]]) -> CounterReport:
    return CounterReport(
        dp_index=0,
        generation=1,
        counter_seq=seq,
        workers=[
            WorkerCounters(
                worker_id="w0",
                incarnation=incarnation,
                routed_total=routed,
                successful_total=routed,
                failed_total=0,
                current_active=0,
            )
            for incarnation, routed in rows
        ],
    )


def test_a_report_carrying_both_incarnations_does_not_double_count() -> None:
    # Note (Jiaxin Deng): during a replacement one report lists the new object
    # and the draining old one; keyed by stable id alone each row retires the
    # other, so a single request is counted once per report.
    ledger = DataPlaneCounterLedger()
    ledger.apply(_two_incarnation_report(1, [("inc-a", 1)]), now=0.0)
    assert ledger.totals("w0")["routed_total"] == 0  # first contact is baseline

    ledger.apply(_two_incarnation_report(2, [("inc-b", 1), ("inc-a", 1)]), now=1.0)
    assert ledger.totals("w0")["routed_total"] == 1

    ledger.apply(_two_incarnation_report(3, [("inc-b", 2)]), now=2.0)
    assert ledger.totals("w0")["routed_total"] == 2


def test_a_drained_incarnation_is_folded_and_dropped() -> None:
    # Note (Jiaxin Deng): a report is a full snapshot, so retaining every
    # incarnation that ever appeared would grow the map without bound.
    ledger = DataPlaneCounterLedger()
    ledger.apply(_two_incarnation_report(1, [("inc-a", 0)]), now=0.0)
    ledger.apply(_two_incarnation_report(2, [("inc-b", 3), ("inc-a", 2)]), now=1.0)
    ledger.apply(_two_incarnation_report(3, [("inc-b", 3)]), now=2.0)

    assert ledger.totals("w0")["routed_total"] == 5
    entry = ledger._entries[0]
    assert {wid: list(seg) for wid, seg in entry.per_worker.items()} == {
        "w0": ["inc-b"]
    }
