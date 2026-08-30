# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

BENCHMARK_PATH = (
    Path(__file__).parents[3]
    / "benchmarks"
    / "eval"
    / "benchmark_nemotron_voicechat.py"
)
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "benchmark_nemotron_voicechat", BENCHMARK_PATH
)
assert BENCHMARK_SPEC is not None and BENCHMARK_SPEC.loader is not None
benchmark = importlib.util.module_from_spec(BENCHMARK_SPEC)
BENCHMARK_SPEC.loader.exec_module(benchmark)


def audio_event(arrival_ms):
    return {
        "type": "response.output_audio.delta",
        "arrival_ms": arrival_ms,
    }


def test_aggregate_report_pairs_80ms_events_and_excludes_close_flush_gap():
    report = {
        "runs": [
            {
                "first_audio_ms": 100.0,
                "session_close_sent_ms": 500.0,
                "audio_event_count": 6,
                "output_samples": 11_520,
                "text": "I am your AI voice assistant.",
                "events": [
                    audio_event(100),
                    audio_event(101),
                    audio_event(260),
                    audio_event(261),
                    audio_event(700),
                    audio_event(701),
                ],
            }
        ]
    }

    summary = benchmark.aggregate_report(report)

    assert summary["first_audio_ms"]["mean"] == 100.0
    assert summary["native_audio_event_interval_ms"]["count"] == 5
    assert summary["paired_160ms_interval_ms"] == {
        "count": 1,
        "mean": 160.0,
        "p50": 160.0,
        "p95": 160.0,
        "p99": 160.0,
        "min": 160.0,
        "max": 160.0,
    }
    assert summary["transcripts"] == ["I am your AI voice assistant."]
    assert summary["audio_event_counts"] == [6]
    assert summary["output_sample_counts"] == [11_520]
