# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

if importlib.util.find_spec("datasets") is None:
    datasets_stub = types.ModuleType("datasets")

    def _missing_load_dataset(*args, **kwargs):
        raise RuntimeError("datasets is not installed in this test environment")

    def _missing_concatenate_datasets(*args, **kwargs):
        raise RuntimeError("datasets is not installed in this test environment")

    datasets_stub.load_dataset = _missing_load_dataset
    datasets_stub.concatenate_datasets = _missing_concatenate_datasets
    sys.modules["datasets"] = datasets_stub

if importlib.util.find_spec("huggingface_hub") is None:
    hub_stub = types.ModuleType("huggingface_hub")

    def _missing_snapshot_download(*args, **kwargs):
        raise RuntimeError("huggingface_hub is not installed in this test environment")

    hub_stub.snapshot_download = _missing_snapshot_download
    sys.modules["huggingface_hub"] = hub_stub

if importlib.util.find_spec("aiohttp") is None:
    aiohttp_stub = types.ModuleType("aiohttp")

    class _MissingAiohttpError(Exception):
        pass

    class _MissingClientTimeout:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("aiohttp is not installed in this test environment")

    class _MissingClientSession:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("aiohttp is not installed in this test environment")

    aiohttp_stub.ClientError = _MissingAiohttpError
    aiohttp_stub.ClientTimeout = _MissingClientTimeout
    aiohttp_stub.ClientSession = _MissingClientSession
    sys.modules["aiohttp"] = aiohttp_stub

if importlib.util.find_spec("requests") is None:
    requests_stub = types.ModuleType("requests")

    class _MissingRequestsError(Exception):
        pass

    def _missing_request(*args, **kwargs):
        raise RuntimeError("requests is not installed in this test environment")

    requests_stub.RequestException = _MissingRequestsError
    requests_stub.get = _missing_request
    requests_stub.post = _missing_request
    sys.modules["requests"] = requests_stub

if importlib.util.find_spec("tqdm") is None:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_asyncio_stub = types.ModuleType("tqdm.asyncio")

    class _DummyProgress:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

        def close(self):
            pass

    tqdm_asyncio_stub.tqdm = _DummyProgress
    sys.modules["tqdm"] = tqdm_stub
    sys.modules["tqdm.asyncio"] = tqdm_asyncio_stub

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.videomme import VideoAMMESample
from benchmarks.eval.benchmark_omni_videoamme_talker_stress import (
    StressRequestRecord,
    _bucket_records,
    _parse_positive_int_list,
    _parse_request_rate,
    _repeat_samples,
    _repeat_source_sample_ids,
    _scheduled_offsets,
    _stress_summary,
)


def _sample(sample_id: str = "sample/one") -> VideoAMMESample:
    return VideoAMMESample(
        sample_id=sample_id,
        video_path="/tmp/video.mp4",
        audio_path="/tmp/audio.wav",
        question="Question?",
        options=["A", "B", "C", "D"],
        answer="A",
        question_id=sample_id,
        prompt="prompt",
        all_choices=["A", "B", "C", "D"],
        index2ans={"A": "A", "B": "B", "C": "C", "D": "D"},
    )


def _record(
    *,
    index: int,
    finish_offset_s: float,
    latency_s: float,
    rtf: float,
    queue_wait_s: float = 0.0,
    success: bool = True,
) -> StressRequestRecord:
    return StressRequestRecord(
        stage_id="c2",
        index=index,
        request_id=f"req-{index}",
        source_sample_id=f"src-{index}",
        scheduled_offset_s=float(index),
        start_offset_s=float(index),
        finish_offset_s=finish_offset_s,
        scheduled_lag_s=0.01 * index,
        client_queue_wait_s=queue_wait_s,
        result=RequestResult(
            request_id=f"req-{index}",
            is_success=success,
            latency_s=latency_s,
            audio_duration_s=1.0 if success else 0.0,
            rtf=rtf,
            completion_tokens=8 if success else 0,
            engine_time_s=latency_s,
        ),
    )


def test_parse_positive_int_list() -> None:
    assert _parse_positive_int_list("8,16, 32") == [8, 16, 32]

    with pytest.raises(ValueError):
        _parse_positive_int_list("8,nope")
    with pytest.raises(Exception, match="positive"):
        _parse_positive_int_list("8,0")


def test_parse_request_rate_accepts_inf_and_positive_values() -> None:
    assert _parse_request_rate("inf") == float("inf")
    assert _parse_request_rate("2.5") == 2.5

    with pytest.raises(Exception, match="positive"):
        _parse_request_rate("0")


def test_repeat_samples_uses_unique_safe_ids() -> None:
    repeated = _repeat_samples([_sample()], request_count=3, stage_id="c16")

    assert [sample.sample_id for sample in repeated] == [
        "c16_00000_sample-one",
        "c16_00001_sample-one",
        "c16_00002_sample-one",
    ]
    assert {sample.audio_path for sample in repeated} == {"/tmp/audio.wav"}
    assert {sample.video_path for sample in repeated} == {"/tmp/video.mp4"}


def test_repeat_source_sample_ids_preserves_original_ids() -> None:
    sources = [_sample("sample/one_part"), _sample("sample/two_part")]

    repeated_source_ids = _repeat_source_sample_ids(sources, request_count=5)

    assert repeated_source_ids == [
        "sample/one_part",
        "sample/two_part",
        "sample/one_part",
        "sample/two_part",
        "sample/one_part",
    ]


def test_scheduled_offsets_support_closed_and_open_loop() -> None:
    assert _scheduled_offsets(3, float("inf")) == [0.0, 0.0, 0.0]
    assert _scheduled_offsets(4, 2.0) == [0.0, 0.5, 1.0, 1.5]


def test_bucket_records_reports_queue_wait_and_audio_rate() -> None:
    records = [
        _record(index=0, finish_offset_s=3.0, latency_s=3.0, rtf=1.0),
        _record(index=1, finish_offset_s=8.0, latency_s=4.0, rtf=1.5),
        _record(
            index=2,
            finish_offset_s=13.0,
            latency_s=5.0,
            rtf=2.0,
            queue_wait_s=0.7,
        ),
    ]

    buckets = _bucket_records(records, bucket_seconds=10)

    assert len(buckets) == 2
    assert buckets[0]["completed_requests"] == 2
    assert buckets[0]["audio_returned"] == 2
    assert buckets[0]["throughput_qps"] == 0.2
    assert buckets[1]["client_queue_wait_s"]["p95"] == 0.7


def test_stress_summary_reports_latency_and_rtf_drift() -> None:
    records = [
        _record(index=0, finish_offset_s=2.0, latency_s=2.0, rtf=1.0),
        _record(index=1, finish_offset_s=4.0, latency_s=2.0, rtf=1.0),
        _record(index=2, finish_offset_s=8.0, latency_s=5.0, rtf=2.0),
        _record(index=3, finish_offset_s=12.0, latency_s=7.0, rtf=3.0),
    ]

    summary = _stress_summary(records, bucket_seconds=10)

    assert summary["audio_return_rate"] == 1.0
    assert summary["latency_mean_drift_s"]["first_half_mean"] == 2.0
    assert summary["latency_mean_drift_s"]["last_half_mean"] == 6.0
    assert summary["latency_mean_drift_s"]["delta"] == 4.0
    assert summary["rtf_mean_drift"]["delta"] == 1.5
