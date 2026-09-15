# SPDX-License-Identifier: Apache-2.0
"""Tests the shared-lock contention profiler.

The profiler measures a hot serving path, so the properties that matter are:
reentrancy must not deadlock or double-count, timings must be attributed to the
right call site, and the disabled path must not record anything.
"""

from __future__ import annotations

import threading
import time

import pytest

from sglang_omni.utils.lock_profile import (
    UNLABELED,
    ProfiledRLock,
    labeled,
    lock_profiling_enabled,
)


def test_disabled_lock_records_nothing() -> None:
    lock = ProfiledRLock(enabled=False)
    with lock.labeled("a"):
        pass
    with lock:
        pass
    assert lock.enabled is False
    assert lock.stats() == {}


def test_labeled_acquisition_is_recorded() -> None:
    lock = ProfiledRLock(enabled=True)
    with lock.labeled("reference_encode"):
        time.sleep(0.01)
    stats = lock.stats()
    assert set(stats) == {"reference_encode"}
    site = stats["reference_encode"]
    assert site["acquisitions"] == 1
    assert site["hold_s"] >= 0.008
    assert site["max_hold_s"] >= 0.008


def test_plain_context_manager_uses_unlabeled_site() -> None:
    lock = ProfiledRLock(enabled=True)
    with lock:
        pass
    assert set(lock.stats()) == {UNLABELED}


def test_reentrant_acquisition_does_not_deadlock_or_double_count() -> None:
    lock = ProfiledRLock(enabled=True)
    with lock.labeled("outer"):
        with lock.labeled("inner"):
            with lock.labeled("innermost"):
                pass
    stats = lock.stats()
    # Only the outermost frame is measured; nested frames cannot contend.
    assert set(stats) == {"outer"}
    assert stats["outer"]["acquisitions"] == 1


def test_exception_inside_the_lock_still_releases_and_records() -> None:
    lock = ProfiledRLock(enabled=True)
    with pytest.raises(RuntimeError, match="boom"):
        with lock.labeled("failing"):
            raise RuntimeError("boom")
    assert lock.stats()["failing"]["acquisitions"] == 1
    # The lock must be usable afterwards.
    with lock.labeled("after"):
        pass
    assert lock.stats()["after"]["acquisitions"] == 1


def test_contention_between_threads_is_attributed_to_the_waiter() -> None:
    """A blocked call site must show wait time, the holder must not."""
    lock = ProfiledRLock(enabled=True, contended_threshold_s=1e-4)
    holder_entered = threading.Event()
    release_holder = threading.Event()

    def holder() -> None:
        with lock.labeled("holder"):
            holder_entered.set()
            release_holder.wait(timeout=5)

    def waiter() -> None:
        with lock.labeled("waiter"):
            pass

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert holder_entered.wait(timeout=5)

    waiter_thread = threading.Thread(target=waiter)
    waiter_thread.start()
    time.sleep(0.05)
    release_holder.set()
    holder_thread.join(timeout=5)
    waiter_thread.join(timeout=5)

    stats = lock.stats()
    assert stats["waiter"]["wait_s"] >= 0.03
    assert stats["waiter"]["contended"] == 1
    # The holder never waited, so it must not be charged for contention.
    assert stats["holder"]["contended"] == 0


def test_stats_accumulate_across_acquisitions_and_reset_clears() -> None:
    lock = ProfiledRLock(enabled=True)
    for _ in range(3):
        with lock.labeled("stream_step"):
            pass
    assert lock.stats()["stream_step"]["acquisitions"] == 3
    lock.reset()
    assert lock.stats() == {}


def test_summary_lists_sites_and_handles_empty() -> None:
    lock = ProfiledRLock(enabled=True)
    assert "disabled or no acquisitions" in ProfiledRLock(enabled=False).summary()
    with lock.labeled("stream_step"):
        pass
    summary = lock.summary()
    assert "stream_step" in summary
    assert "wait_s" in summary


def test_env_flag_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("SGLANG_OMNI_PROFILE_LOCKS", value)
        assert lock_profiling_enabled() is True
    for value in ("0", "false", "", "no"):
        monkeypatch.setenv("SGLANG_OMNI_PROFILE_LOCKS", value)
        assert lock_profiling_enabled() is False
    monkeypatch.delenv("SGLANG_OMNI_PROFILE_LOCKS", raising=False)
    assert lock_profiling_enabled() is False


def test_default_enablement_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SGLANG_OMNI_PROFILE_LOCKS", raising=False)
    assert ProfiledRLock().enabled is False
    monkeypatch.setenv("SGLANG_OMNI_PROFILE_LOCKS", "1")
    assert ProfiledRLock().enabled is True


def test_mutual_exclusion_is_preserved_under_load() -> None:
    """Profiling must not weaken the lock: no two threads may overlap."""
    lock = ProfiledRLock(enabled=True)
    overlaps = []
    active = 0
    guard = threading.Lock()

    def worker() -> None:
        nonlocal active
        for _ in range(50):
            with lock.labeled("worker"):
                with guard:
                    active += 1
                    if active > 1:
                        overlaps.append(active)
                time.sleep(0.0002)
                with guard:
                    active -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert overlaps == []
    assert lock.stats()["worker"]["acquisitions"] == 200


def test_labeled_helper_accepts_a_plain_lock() -> None:
    """labeled() falls back to plain lock acquisition when needed."""
    plain = threading.RLock()

    with labeled(plain, "reference_encode"):
        assert plain.acquire(blocking=False) is True
        plain.release()


def test_labeled_helper_still_profiles_a_profiled_lock() -> None:
    lock = ProfiledRLock(enabled=True)

    with labeled(lock, "stream_step"):
        pass

    assert lock.stats()["stream_step"]["acquisitions"] == 1
