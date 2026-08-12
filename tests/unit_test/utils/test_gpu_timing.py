# SPDX-License-Identifier: Apache-2.0
"""Tests GPU span timing bookkeeping without a GPU.

The interesting logic is not the CUDA calls but the accounting around them:
queueing delay must be recovered separately from execution time, the hot path
must never synchronize, and a wrapped ring must not report timings read from
events that were already overwritten. A fake event backend models a stream that
runs queued work in order, so all of that is testable on CPU.
"""

from __future__ import annotations

import threading

import pytest

from sglang_omni.utils.gpu_timing import CudaSpanTimer, gpu_span_profiling_enabled


class FakeStream:
    """An in-order GPU stream on a virtual clock, in milliseconds."""

    def __init__(self) -> None:
        self.now_ms = 0.0
        self.sync_count = 0

    def advance(self, ms: float) -> None:
        """Simulate queued kernels occupying the stream."""
        self.now_ms += ms

    def synchronize(self) -> None:
        self.sync_count += 1


class FakeEvent:
    """Records the stream position at which it was reached."""

    def __init__(self, stream: FakeStream) -> None:
        self._stream = stream
        self.at_ms = 0.0
        self.ready = True

    def record(self) -> None:
        self.at_ms = self._stream.now_ms
        self.ready = True

    def query(self) -> bool:
        """Whether the GPU has reached this event yet."""
        return self.ready

    def elapsed_time(self, other: "FakeEvent") -> float:
        return other.at_ms - self.at_ms


@pytest.fixture()
def stream() -> FakeStream:
    return FakeStream()


def _timer(stream: FakeStream, **kwargs: object) -> CudaSpanTimer:
    cpu_clock = kwargs.pop("clock", None)
    return CudaSpanTimer(
        enabled=True,
        event_factory=lambda: FakeEvent(stream),
        synchronize=stream.synchronize,
        clock=cpu_clock or (lambda: stream.now_ms / 1000.0),
        **kwargs,  # type: ignore[arg-type]
    )


def test_disabled_timer_records_nothing_and_never_synchronizes(
    stream: FakeStream,
) -> None:
    timer = CudaSpanTimer(
        enabled=False,
        event_factory=lambda: FakeEvent(stream),
        synchronize=stream.synchronize,
    )

    with timer.span("stream_step"):
        stream.advance(50.0)

    assert timer.stats() == {}
    assert stream.sync_count == 0


def test_requested_profiling_is_off_when_cuda_is_unavailable() -> None:
    timer = CudaSpanTimer(enabled=True, available=lambda: False)

    assert timer.enabled is False
    with timer.span("stream_step"):
        pass
    assert timer.stats() == {}


def test_execution_time_is_measured_per_site(stream: FakeStream) -> None:
    timer = _timer(stream)

    with timer.span("stream_step"):
        stream.advance(3.0)

    stats = timer.stats()
    assert stats["stream_step"]["spans"] == 1
    assert stats["stream_step"]["gpu_ms"] == pytest.approx(3.0)


def test_work_inside_the_span_counts_as_execution_not_queueing(
    stream: FakeStream,
) -> None:
    """Time after the start event is reached belongs to execution."""
    cpu_now = {"s": 0.0}
    timer = _timer(stream, clock=lambda: cpu_now["s"])

    with timer.span("stream_step"):
        stream.advance(54.0)

    site = timer.stats()["stream_step"]
    assert site["gpu_ms"] == pytest.approx(54.0)
    assert site["max_queue_ms"] == pytest.approx(0.0, abs=1e-6)


def test_stream_busy_before_the_span_shows_up_as_queue_delay(
    stream: FakeStream,
) -> None:
    """The delay this whole module exists to catch.

    Reference encode is already in the stream, so streaming decode's *start*
    event is only reached 51 ms after the CPU enqueued it. ``elapsed_time``
    spans only start-to-end and reports a harmless 3 ms, so the wait has to come
    from correlating the CPU enqueue time against the GPU timeline instead.
    """
    cpu_now = {"s": 0.0}
    timer = _timer(stream, clock=lambda: cpu_now["s"])

    # Someone else's kernels occupy the stream after the CPU enqueued ours.
    stream.advance(51.0)
    with timer.span("stream_step"):
        stream.advance(3.0)

    site = timer.stats()["stream_step"]
    # Execution alone would call this healthy...
    assert site["gpu_ms"] == pytest.approx(3.0)
    # ...while the real cost sat ahead of the start event.
    assert site["max_queue_ms"] == pytest.approx(51.0)


def test_hot_path_does_not_synchronize(stream: FakeStream) -> None:
    """Synchronizing inside a span would serialize the very path being measured."""
    timer = _timer(stream)

    for _ in range(5):
        with timer.span("stream_step"):
            stream.advance(1.0)

    assert stream.sync_count == 1  # the epoch only
    timer.drain()
    assert stream.sync_count == 1  # periodic drain must not synchronize either

    with timer.span("stream_step"):
        stream.advance(1.0)
    timer.drain(block=True)
    assert stream.sync_count == 2  # only an explicit blocking drain syncs


def test_wrapped_ring_drops_stale_records_instead_of_lying(
    stream: FakeStream,
) -> None:
    timer = _timer(stream, capacity=4)

    for _ in range(10):
        with timer.span("stream_step"):
            stream.advance(1.0)

    stats = timer.stats()
    assert stats["stream_step"]["spans"] == 4
    assert timer.dropped == 6


def test_stats_accumulate_and_reset_clears(stream: FakeStream) -> None:
    timer = _timer(stream)

    with timer.span("stream_step"):
        stream.advance(2.0)
    with timer.span("stream_step"):
        stream.advance(4.0)

    assert timer.stats()["stream_step"]["spans"] == 2
    assert timer.stats()["stream_step"]["gpu_ms"] == pytest.approx(6.0)

    timer.reset()
    assert timer.stats() == {}


def test_exception_inside_a_span_still_records_the_end_event(
    stream: FakeStream,
) -> None:
    timer = _timer(stream)

    with pytest.raises(RuntimeError):
        with timer.span("stream_step"):
            stream.advance(2.0)
            raise RuntimeError("kernel blew up")

    assert timer.stats()["stream_step"]["spans"] == 1


def test_sites_are_attributed_separately(stream: FakeStream) -> None:
    timer = _timer(stream)

    with timer.span("reference_encode"):
        stream.advance(51.0)
    with timer.span("stream_step"):
        stream.advance(3.0)

    stats = timer.stats()
    assert stats["reference_encode"]["gpu_ms"] == pytest.approx(51.0)
    assert stats["stream_step"]["gpu_ms"] == pytest.approx(3.0)


def test_concurrent_spans_do_not_lose_records(stream: FakeStream) -> None:
    timer = _timer(stream, capacity=512)

    def worker() -> None:
        for _ in range(50):
            with timer.span("worker"):
                pass

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert timer.stats()["worker"]["spans"] == 200
    assert timer.dropped == 0


def test_summary_reports_and_flags_drops(stream: FakeStream) -> None:
    timer = _timer(stream, capacity=2)
    assert "no GPU spans recorded" in timer.summary()

    for _ in range(5):
        with timer.span("stream_step"):
            stream.advance(1.0)

    summary = timer.summary()
    assert "stream_step" in summary
    assert "dropped 3" in summary


def test_env_flag_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("SGLANG_OMNI_PROFILE_GPU_SPANS", value)
        assert gpu_span_profiling_enabled() is True
    for value in ("", "0", "false", "off"):
        monkeypatch.setenv("SGLANG_OMNI_PROFILE_GPU_SPANS", value)
        assert gpu_span_profiling_enabled() is False


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        CudaSpanTimer(enabled=False, capacity=0)


def test_in_flight_spans_are_left_for_a_later_drain(stream: FakeStream) -> None:
    """A non-blocking drain must not read events the GPU has not reached.

    ``elapsed_time`` on an incomplete event is invalid, so an unfinished span is
    deferred rather than guessed at.
    """
    timer = _timer(stream)

    with timer.span("stream_step"):
        stream.advance(2.0)
    with timer.span("stream_step"):
        stream.advance(3.0)

    # Mark the second span's end event as still executing on the GPU.
    timer._ends[1].ready = False

    assert timer.stats()["stream_step"]["spans"] == 1
    assert stream.sync_count == 1  # no synchronize was used to find that out

    timer._ends[1].ready = True
    assert timer.stats()["stream_step"]["spans"] == 2
