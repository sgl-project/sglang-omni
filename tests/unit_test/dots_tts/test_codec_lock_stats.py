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
from sglang_omni.utils.gpu_timing import CudaSpanTimer
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
    codec.maybe_log_contention()  # must not raise


def test_codec_without_a_span_timer_still_runs() -> None:
    from sglang_omni.utils.gpu_timing import span

    codec = _make_codec(threading.RLock())

    assert codec.spans is None
    with span(codec.spans, "reference_encode"):
        pass


def test_disabled_profiling_logs_nothing(caplog) -> None:
    codec = _make_codec(ProfiledRLock(enabled=False))

    with caplog.at_level(logging.INFO):
        codec.maybe_log_contention()

    assert "lock contention" not in caplog.text


def test_logging_is_rate_limited(caplog) -> None:
    codec = _make_codec(ProfiledRLock(enabled=True))
    with labeled(codec.lock, "stream_step"):
        pass

    with caplog.at_level(logging.INFO):
        codec.maybe_log_contention()
        codec.maybe_log_contention()
        codec.maybe_log_contention()

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
        codec.maybe_log_contention()
        finished.set()

    reporter = threading.Thread(target=_report, daemon=True)
    reporter.start()
    reported_without_waiting = finished.wait(timeout=5)

    release_holder.set()
    holder.join(timeout=10)
    reporter.join(timeout=10)

    assert reported_without_waiting
    assert codec.lock_stats()["stream_step"]["acquisitions"] == before


class _FakeStream:
    def __init__(self) -> None:
        self.now_ms = 0.0

    def synchronize(self) -> None:
        pass


class _FakeEvent:
    def __init__(self, stream: _FakeStream) -> None:
        self._stream = stream
        self.at_ms = 0.0

    def record(self) -> None:
        self.at_ms = self._stream.now_ms

    def query(self) -> bool:
        return True

    def elapsed_time(self, other: "_FakeEvent") -> float:
        return other.at_ms - self.at_ms


def _span_timer(enabled: bool) -> CudaSpanTimer:
    stream = _FakeStream()
    return CudaSpanTimer(
        enabled=enabled,
        event_factory=lambda: _FakeEvent(stream),
        synchronize=stream.synchronize,
    )


def test_gpu_span_stats_are_empty_without_a_timer() -> None:
    codec = _make_codec(ProfiledRLock(enabled=False))

    assert codec.gpu_span_stats() == {}


def test_gpu_span_stats_report_per_site_totals() -> None:
    from sglang_omni.utils.gpu_timing import span

    codec = _make_codec(ProfiledRLock(enabled=False))
    codec.spans = _span_timer(enabled=True)

    with span(codec.spans, "stream_step"):
        pass

    assert codec.gpu_span_stats()["stream_step"]["spans"] == 1


def test_gpu_spans_log_even_when_lock_profiling_is_off(caplog) -> None:
    """The two profilers are gated independently.

    Stream queueing is visible with GPU spans alone, so gating the report on the
    lock profiler would hide exactly the case the follow-up is looking for.
    """
    from sglang_omni.utils.gpu_timing import span

    codec = _make_codec(ProfiledRLock(enabled=False))
    codec.spans = _span_timer(enabled=True)
    with span(codec.spans, "stream_step"):
        pass

    with caplog.at_level(logging.INFO):
        codec.maybe_log_contention()

    assert "GPU spans" in caplog.text
    assert "lock contention" not in caplog.text


def test_lock_logs_even_when_gpu_spans_are_off(caplog) -> None:
    codec = _make_codec(ProfiledRLock(enabled=True))
    codec.spans = _span_timer(enabled=False)
    with labeled(codec.lock, "stream_step"):
        pass

    with caplog.at_level(logging.INFO):
        codec.maybe_log_contention()

    assert "lock contention" in caplog.text
    assert "GPU spans" not in caplog.text
