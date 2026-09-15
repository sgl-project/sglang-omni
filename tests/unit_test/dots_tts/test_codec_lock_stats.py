# SPDX-License-Identifier: Apache-2.0
"""Tests the dots.tts codec's shared-lock contention reporting surface.

The reporter runs on the streaming hot path, so it must stay silent and cheap
when profiling is off, must not serialize on the lock it is measuring, and must
tolerate a codec whose lock was replaced with a plain one.
"""

from __future__ import annotations

import logging
import threading

import torch

from sglang_omni.models.dots_tts.codec import DotsAudioCodec
from sglang_omni.utils.lock_profile import ProfiledRLock, labeled


def _make_codec(lock: object) -> DotsAudioCodec:
    codec = object.__new__(DotsAudioCodec)
    codec.device = torch.device("cpu")
    codec.lock = lock
    codec._lock_log_guard = threading.Lock()
    codec._last_lock_log_time = 0.0
    return codec


def test_lock_stats_are_empty_when_profiling_is_disabled() -> None:
    codec = _make_codec(ProfiledRLock(enabled=False))

    with labeled(codec.lock, "stream_step"):
        pass

    assert codec.lock_stats() == {}


def test_lock_stats_report_per_site_totals_when_enabled() -> None:
    codec = _make_codec(ProfiledRLock(enabled=True))

    with labeled(codec.lock, "stream_step"):
        pass
    with labeled(codec.lock, "reference_encode"):
        pass

    stats = codec.lock_stats()
    assert stats["stream_step"]["acquisitions"] == 1
    assert stats["reference_encode"]["acquisitions"] == 1


def test_lock_stats_tolerate_a_plain_lock() -> None:
    """Embedders and tests substitute a bare RLock for the codec lock."""
    codec = _make_codec(threading.RLock())

    assert codec.lock_stats() == {}
    codec.maybe_log_lock_stats()  # must not raise


def test_disabled_profiling_logs_nothing(caplog) -> None:
    codec = _make_codec(ProfiledRLock(enabled=False))

    with caplog.at_level(logging.INFO):
        codec.maybe_log_lock_stats()

    assert "lock contention" not in caplog.text


def test_logging_is_rate_limited(caplog) -> None:
    codec = _make_codec(ProfiledRLock(enabled=True))
    with labeled(codec.lock, "stream_step"):
        pass

    with caplog.at_level(logging.INFO):
        codec.maybe_log_lock_stats()
        codec.maybe_log_lock_stats()
        codec.maybe_log_lock_stats()

    assert caplog.text.count("codec lock contention") == 1


def test_reporting_does_not_acquire_the_lock_it_measures() -> None:
    """Taking the codec lock to log would both contend with and skew the data."""
    codec = _make_codec(ProfiledRLock(enabled=True))
    with labeled(codec.lock, "stream_step"):
        pass
    before = codec.lock_stats()["stream_step"]["acquisitions"]

    # Hold the codec lock from another thread; the reporter must not block.
    holder_has_lock = threading.Event()
    release_holder = threading.Event()

    def _hold() -> None:
        with labeled(codec.lock, "holder"):
            holder_has_lock.set()
            release_holder.wait(timeout=10)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert holder_has_lock.wait(timeout=10)

    finished = threading.Event()

    def _report() -> None:
        codec.maybe_log_lock_stats()
        finished.set()

    reporter = threading.Thread(target=_report, daemon=True)
    reporter.start()
    reported_without_waiting = finished.wait(timeout=5)

    release_holder.set()
    holder.join(timeout=10)
    reporter.join(timeout=10)

    assert reported_without_waiting
    assert codec.lock_stats()["stream_step"]["acquisitions"] == before
