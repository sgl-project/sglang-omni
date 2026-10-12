# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from sglang_omni.scheduling.metrics_snapshot import (
    backend_metrics_snapshot,
    engine_metrics_snapshot,
)


def send_backend_snapshot(connection):
    from prometheus_client import CollectorRegistry, Counter

    from sglang_omni.pipeline.control_plane import serialize_message
    from sglang_omni.proto import AdminResult, AdminResultMessage

    counter = Counter(
        "sglang:spec_verify_calls_total", "Verify", registry=CollectorRegistry()
    )
    counter.inc(7)
    scheduler = SimpleNamespace(
        enable_metrics=True,
        metrics_reporter=SimpleNamespace(
            metrics_collector=SimpleNamespace(counter=counter)
        ),
    )
    result = AdminResult(
        "metrics-op",
        "decode",
        "metrics",
        True,
        data={"backend_metrics": backend_metrics_snapshot(scheduler)},
    )
    connection.send_bytes(serialize_message(AdminResultMessage(result)))
    connection.close()


def test_backend_snapshot_crosses_spawned_process_control_serialization():
    import multiprocessing

    from sglang_omni.pipeline.control_plane import deserialize_message

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=send_backend_snapshot, args=(sender,))
    process.start()
    sender.close()
    try:
        assert receiver.poll(15), "Worker did not return metrics"
        result = deserialize_message(receiver.recv_bytes()).result
        assert result.stage == "decode"
        family = result.data["backend_metrics"][0]
        assert family["type"] == "counter"
        assert family["samples"][0]["value"] == 7
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_engine_snapshot_uses_pool_and_backend_reporter():
    pool = SimpleNamespace(
        get_kv_token_stats=lambda: (20, 0.2),
        full_token_usage=0.2,
        full_available_size=70,
        full_evictable_size=10,
        swa_token_usage=None,
        mamba_usage=None,
    )
    scheduler = SimpleNamespace(
        pool_stats_observer=SimpleNamespace(get_pool_stats=lambda: pool),
        max_total_num_tokens=100,
        enable_metrics=True,
        metrics_reporter=SimpleNamespace(
            _current_gen_throughput=lambda now: 42,
            stats=SimpleNamespace(cache_hit_rate=0.5),
        ),
    )
    assert engine_metrics_snapshot(scheduler) == {
        "num_used_tokens": 20,
        "token_usage": 0.2,
        "full_token_usage": 0.2,
        "kv_available_tokens": 70,
        "kv_evictable_tokens": 10,
        "max_total_num_tokens": 100,
        "gen_throughput": 42,
        "cache_hit_rate": 0.5,
    }


def test_non_kv_scheduler_does_not_publish_synthetic_zeros():
    assert engine_metrics_snapshot(SimpleNamespace()) == {}


def test_backend_snapshot_preserves_native_types_labels_and_samples():
    from prometheus_client import CollectorRegistry, Counter, Histogram

    registry = CollectorRegistry()
    count = Counter(
        "sglang:spec_verify_calls_total", "Verifications", ["mode"], registry=registry
    )
    count.labels("decode").inc(3)
    latency = Histogram("sglang:grammar_latency_seconds", "Grammar", registry=registry)
    latency.observe(0.2)
    scheduler = SimpleNamespace(
        enable_metrics=True,
        metrics_reporter=SimpleNamespace(
            metrics_collector=SimpleNamespace(count=count, alias=count, latency=latency)
        ),
    )
    families = backend_metrics_snapshot(scheduler)
    assert len(families) == 2
    assert families[0]["type"] == "counter"
    assert families[0]["samples"][0] == {
        "name": "sglang:spec_verify_calls_total",
        "labels": {"mode": "decode"},
        "value": 3,
    }
    assert families[1]["type"] == "histogram"
    scheduler.enable_metrics = False
    assert backend_metrics_snapshot(scheduler) == []


def test_radix_collector_exports_without_reporter_but_storage_is_deferred():
    from prometheus_client import CollectorRegistry, Counter

    registry = CollectorRegistry()
    eviction = Counter(
        "sglang:radix_evicted_tokens_total", "Eviction", registry=registry
    )
    storage = Counter(
        "sglang:storage_prefetched_tokens_total", "Storage", registry=registry
    )
    unrelated = Counter("other:requests_total", "Other", registry=registry)
    eviction.inc(4)
    storage.inc(7)
    scheduler = SimpleNamespace(
        enable_metrics=True,
        tree_cache=SimpleNamespace(
            metrics_collector=SimpleNamespace(
                eviction=eviction, alias=eviction, unrelated=unrelated
            ),
            storage_metrics_collector=SimpleNamespace(storage=storage, alias=eviction),
        ),
    )
    families = backend_metrics_snapshot(scheduler)
    assert [family["name"] for family in families] == [
        "sglang:radix_evicted_tokens",
    ]
    assert [family["samples"][0]["value"] for family in families] == [4]
    scheduler.tree_cache = None
    assert backend_metrics_snapshot(scheduler) == []


def test_threaded_snapshot_counts_queued_and_running_futures():
    from concurrent.futures import Future

    from sglang_omni.scheduling.message import IncomingMessage
    from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

    scheduler = ThreadedSimpleScheduler(lambda value: value)
    running = Future()
    running.set_running_or_notify_cancel()
    scheduler.pending["running"] = running
    scheduler.enqueue(IncomingMessage("queued", "new_request", None))
    assert scheduler.metrics_snapshot() == {"num_running_reqs": 1, "num_queue_reqs": 1}
    scheduler.abort("queued")
    assert scheduler.metrics_snapshot() == {"num_running_reqs": 1, "num_queue_reqs": 0}
    scheduler.abort("running")
    assert scheduler.metrics_snapshot() == {"num_running_reqs": 1, "num_queue_reqs": 0}
    scheduler.executor.shutdown(wait=False, cancel_futures=True)


def test_threaded_compute_events_clean_state_and_report_failure():
    import pytest

    from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

    def fail(payload):
        raise ValueError("failure")

    scheduler = ThreadedSimpleScheduler(fail)
    scheduler.metrics_queued["r"] = 0
    with pytest.raises(ValueError):
        scheduler.run_one_tracked("r", None)
    events = [scheduler.outbox.get_nowait().data for _ in range(3)]
    assert [event["event"] for event in events] == ["queue", "batch", "execution"]
    assert events[-1]["outcome"] == "error"
    assert not scheduler.metrics_queued
    assert not scheduler.metrics_active
    scheduler.executor.shutdown(wait=False, cancel_futures=True)
