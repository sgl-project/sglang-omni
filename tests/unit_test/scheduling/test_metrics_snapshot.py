# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from prometheus_client import CollectorRegistry, Counter

from sglang_omni.scheduling.metrics_snapshot import backend_metrics_snapshot


def test_radix_bridge_retains_deduplication_and_defers_storage():
    registry = CollectorRegistry()
    eviction = Counter("sglang:evicted_tokens_total", "Eviction", registry=registry)
    storage = Counter("sglang:prefetched_tokens_total", "Storage", registry=registry)
    unrelated = Counter("other:requests_total", "Other", registry=registry)
    eviction.inc(4)
    storage.inc(7)
    scheduler = SimpleNamespace(
        enable_metrics=True,
        tree_cache=SimpleNamespace(
            metrics_collector=SimpleNamespace(
                eviction=eviction, alias=eviction, unrelated=unrelated
            ),
            storage_metrics_collector=SimpleNamespace(storage=storage),
        ),
    )
    families = backend_metrics_snapshot(scheduler)
    assert [family["name"] for family in families] == ["sglang:evicted_tokens"]
    assert families[0]["samples"][0]["value"] == 4
    scheduler.enable_metrics = False
    assert backend_metrics_snapshot(scheduler) == []


def test_threaded_snapshot_counts_queue_and_aborted_running_work():
    from concurrent.futures import Future

    from sglang_omni.scheduling.messages import IncomingMessage
    from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

    scheduler = ThreadedSimpleScheduler(lambda payload: payload)
    try:
        future = Future()
        future.set_running_or_notify_cancel()
        scheduler._pending["running"] = future
        scheduler.enqueue(IncomingMessage("queued", "new_request", None))
        assert scheduler.metrics_snapshot() == {
            "num_running_reqs": 1,
            "num_queue_reqs": 1,
        }
        scheduler.abort("queued")
        scheduler.abort("running")
        assert scheduler.metrics_snapshot() == {
            "num_running_reqs": 1,
            "num_queue_reqs": 0,
        }
    finally:
        scheduler._executor.shutdown(wait=False, cancel_futures=True)


def test_threaded_compute_metrics_clean_state_after_failure():
    import pytest

    from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

    def fail(payload):
        raise ValueError("failure")

    scheduler = ThreadedSimpleScheduler(fail)
    try:
        scheduler.metrics_queued["r"] = 0
        with pytest.raises(ValueError):
            scheduler._run_one_tracked("r", None)
        events = [scheduler.outbox.get_nowait().data for _ in range(3)]
        assert [event["event"] for event in events] == ["queue", "batch", "execution"]
        assert events[-1]["outcome"] == "error"
        assert not scheduler.metrics_queued
        assert not scheduler.metrics_active
    finally:
        scheduler._executor.shutdown(wait=False, cancel_futures=True)
