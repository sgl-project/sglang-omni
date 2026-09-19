# SPDX-License-Identifier: Apache-2.0

import json
import logging

from sglang_omni.profiler.runtime_stats import RuntimeStats


def _report(caplog):
    return json.loads(caplog.records[-1].message.removeprefix("stage_stats "))


def test_stats_count_forwards_not_requests_and_reset_each_window(caplog):
    caplog.set_level(logging.INFO)
    stats = RuntimeStats("talker", interval_s=10)
    stats.record_batch("decode", 8, graph=True)
    stats.record_batch("decode", 2, graph=False)
    stats.record_batch("prefill", 3)
    stats.record_queue_wait(0.01)
    stats.record_queue_wait(0.03)
    stats.maybe_log()
    assert not caplog.records
    stats.maybe_log(force=True)
    report = _report(caplog)
    assert report["batches"]["decode"] == {
        "count": 2,
        "mean_size": 5,
        "graph_replays": 1,
        "graph_forwards": 2,
        "graph_replay_rate": 0.5,
    }
    assert report["batches"]["prefill"]["graph_replay_rate"] is None
    assert report["queue_wait_mean_ms"] == 20
    assert report["queue_wait_max_ms"] == 30
    stats.record_batch("decode", 1, graph=False)
    stats.maybe_log(force=True)
    report = _report(caplog)
    assert report["batches"]["decode"]["count"] == 1
    assert report["batches"]["decode"]["graph_replay_rate"] == 0
    assert report["queue_wait_mean_ms"] is None


def test_disabled_stats_do_not_accumulate_or_log(caplog):
    caplog.set_level(logging.INFO)
    stats = RuntimeStats("talker", interval_s=0)
    stats.record_batch("decode", 64, graph=True)
    stats.record_queue_wait(1)
    stats.maybe_log(force=True)
    assert not caplog.records
    assert not stats._batches
    assert stats._queue_count == 0
