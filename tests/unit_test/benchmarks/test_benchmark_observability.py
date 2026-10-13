# SPDX-License-Identifier: Apache-2.0
"""Contracts for comparable benchmark speed measurements."""

from __future__ import annotations

import argparse
import base64
import logging
import time
from dataclasses import replace

import httpx
import pytest

from benchmarks.benchmarker.conditions import (
    SWEEP_METRIC_NAMES,
    aggregate_numbers,
    aggregate_repeats,
    fingerprint_fields,
    sampling_seed_field,
    warn_if_tail_percentile_is_thin,
)
from benchmarks.dataset.mmmu import MMMUSample
from benchmarks.dataset.mmsu import MmsuSample
from benchmarks.dataset.videomme import VideoMMESample
from benchmarks.eval.benchmark_omni_rollout_stress import (
    _make_rollout_send_fn,
    rollout_request_seeds,
)
from benchmarks.eval.benchmark_omni_streaming_ttft import (
    DEFAULT_STREAMING_TTFT_SEED,
    PROMPTS,
    _run,
    streaming_ttft_payload,
)
from benchmarks.realtime_asr.client import SessionTrace, _sender
from benchmarks.tasks.audio_understanding import _build_request_payload
from benchmarks.tasks.tts import TalkerSamplingParams, talker_sampling_params
from benchmarks.tasks.video_understanding import make_video_send_fn
from benchmarks.tasks.visual_understand import make_mmmu_send_fn


class AppendSocket:
    async def send(self, message: str) -> None:
        return None


@pytest.mark.asyncio
async def test_sender_encodes_audio_before_first_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encode_times: list[float] = []
    real_encode = base64.b64encode

    def tracking_encode(packet: bytes) -> bytes:
        encode_times.append(time.perf_counter())
        return real_encode(packet)

    monkeypatch.setattr(
        "benchmarks.realtime_asr.client.base64.b64encode",
        tracking_encode,
    )
    trace = SessionTrace(url="ws://localhost/v1/realtime")
    await _sender(
        trace,
        AppendSocket(),
        [b"\x01\x00" * 8, b"\x02\x00" * 8],
        packet_ms=200,
        paced=False,
        manual_commit=False,
    )
    assert encode_times
    assert trace.first_send_s is not None
    assert max(encode_times) <= trace.first_send_s


def test_sampling_seed_field_keeps_zero_and_omits_unset() -> None:
    assert sampling_seed_field(0) == {"seed": 0}
    assert sampling_seed_field(None) == {}


def test_aggregate_numbers_empty_and_populated() -> None:
    assert aggregate_numbers([]) == {"mean": None, "min": None, "max": None, "n": 0}
    assert aggregate_numbers([1.0, None, 3.0]) == {
        "mean": 2.0,
        "min": 1.0,
        "max": 3.0,
        "n": 2,
    }


def test_sweep_aggregate_uses_the_shared_metric_names() -> None:
    row = aggregate_repeats(
        1,
        [
            {
                "repeat": 1,
                "output_dir": "c1",
                "completed_requests": 1,
                "failed_requests": 0,
            }
        ],
    )
    assert row["repeats"] == 1
    assert len(row["per_repeat"]) == 1
    for metric_name in SWEEP_METRIC_NAMES:
        assert set(row[metric_name]) == {"mean", "min", "max", "n"}


def test_sweep_aggregate_records_warmup_and_time_to_first_audio() -> None:
    row = aggregate_repeats(
        1,
        [
            {
                "repeat": 1,
                "output_dir": "c1_r1",
                "completed_requests": 1,
                "failed_requests": 0,
                "warmup": 2,
                "audio_ttfp_mean_s": 0.25,
                "audio_ttfp_p95_s": 0.5,
            },
            {
                "repeat": 2,
                "output_dir": "c1_r2",
                "completed_requests": 1,
                "failed_requests": 0,
                "warmup": 2,
                "audio_ttfp_mean_s": 0.75,
                "audio_ttfp_p95_s": None,
            },
        ],
    )
    assert row["warmup"] == {"mean": 2.0, "min": 2.0, "max": 2.0, "n": 2}
    assert row["audio_ttfp_mean_s"] == {"mean": 0.5, "min": 0.25, "max": 0.75, "n": 2}
    assert row["audio_ttfp_p95_s"] == {"mean": 0.5, "min": 0.5, "max": 0.5, "n": 1}


def test_fingerprint_fields_skip_collection_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_client(model_path: str | None = None) -> dict[str, str]:
        raise AssertionError(model_path)

    def fail_server(base_url: str) -> dict[str, str]:
        raise AssertionError(base_url)

    monkeypatch.setattr(
        "benchmarks.benchmarker.conditions.collect_environment_fingerprint",
        fail_client,
    )
    monkeypatch.setattr(
        "benchmarks.benchmarker.conditions.collect_server_identity",
        fail_server,
    )
    assert fingerprint_fields(False, "http://localhost:8000") == {}


def test_tail_warning_fires_below_one_hundred_samples(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        warn_if_tail_percentile_is_thin(2)
    assert "two slowest of 2 requests" in caplog.text


def test_rollout_request_seeds_offset_by_index() -> None:
    assert rollout_request_seeds(None, ["a", "b"]) == {}
    assert rollout_request_seeds(10, ["a", "b"]) == {"a": 10, "b": 11}


def test_streaming_ttft_payload_pins_seed_and_talker() -> None:
    payload = streaming_ttft_payload(
        model="qwen3-omni",
        prompt="hello",
        seed=DEFAULT_STREAMING_TTFT_SEED,
        request_id_hint="baseline-short-0",
        talker_params=talker_sampling_params(
            talker_temperature=0.0,
            talker_top_p=None,
            talker_top_k=1,
            talker_repetition_penalty=None,
        ),
    )
    assert payload["seed"] == DEFAULT_STREAMING_TTFT_SEED
    assert payload["talker_temperature"] == 0.0
    assert payload["talker_top_k"] == 1
    assert "talker_top_p" not in payload


def streaming_args() -> argparse.Namespace:
    return argparse.Namespace(
        label="baseline",
        base_url="http://localhost:8000",
        model="qwen3-omni",
        warmup=1,
        repeats=2,
        timeout_s=1.0,
        seed=DEFAULT_STREAMING_TTFT_SEED,
        talker_temperature=None,
        talker_top_p=None,
        talker_top_k=None,
        talker_repetition_penalty=None,
    )


@pytest.mark.asyncio
async def test_streaming_ttft_reuses_one_seed_for_warmup_and_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_seeds: list[int] = []

    async def fake_measure(
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        prompt: str,
        *,
        request_id_hint: str,
        seed: int,
        timeout_s: float,
        talker_params: TalkerSamplingParams,
    ) -> tuple[float, float, int, int]:
        assert isinstance(client, httpx.AsyncClient)
        assert base_url
        assert model
        assert prompt
        assert request_id_hint
        assert timeout_s > 0
        assert talker_params == {}
        seen_seeds.append(seed)
        return 0.1, 0.2, 1, 200

    monkeypatch.setattr(
        "benchmarks.eval.benchmark_omni_streaming_ttft._measure_one",
        fake_measure,
    )
    summary = await _run(streaming_args())
    calls_per_prompt = 1 + 2
    assert seen_seeds == [DEFAULT_STREAMING_TTFT_SEED] * (
        len(PROMPTS) * calls_per_prompt
    )
    assert summary.seed == DEFAULT_STREAMING_TTFT_SEED


def mmsu_sample() -> MmsuSample:
    return MmsuSample(
        sample_id="s0",
        audio_path="clip.wav",
        question="Which note?",
        choices=["A", "B"],
        answer_text="A",
        answer_index=0,
        task_name="pitch",
        category="music",
        sub_category="",
        sub_sub_category="",
        linguistics_sub_discipline="",
    )


def test_mmsu_payload_forwards_seed() -> None:
    seeded = _build_request_payload(
        mmsu_sample(),
        model_name="qwen3-omni",
        prompt="Answer",
        modalities=["text"],
        max_tokens=8,
        temperature=0.0,
        seed=7,
    )
    omitted = _build_request_payload(
        mmsu_sample(),
        model_name="qwen3-omni",
        prompt="Answer",
        modalities=["text"],
        max_tokens=8,
        temperature=0.0,
    )
    assert seeded["seed"] == 7
    assert "seed" not in omitted


class JsonResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self.status = 200
        self.body = body

    async def json(self) -> dict[str, object]:
        return self.body

    def raise_for_status(self) -> None:
        return None

    async def __aenter__(self) -> "JsonResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class RecordingSession:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def post(self, url: str, json: dict[str, object]) -> JsonResponse:
        self.payloads.append(json)
        return JsonResponse(
            {"choices": [{"message": {"content": "Answer: A"}}], "usage": {}}
        )


def mmmu_sample() -> MMMUSample:
    return MMMUSample(
        sample_id="m0",
        question="q",
        options=["A"],
        answer="A",
        image_data_uris=("data:image/png;base64,YQ==",),
        subject="math",
        prompt="prompt",
    )


@pytest.mark.asyncio
async def test_mmmu_request_includes_seed_only_when_set() -> None:
    session = RecordingSession()
    seeded = make_mmmu_send_fn("qwen3-omni", "http://localhost/v1", seed=4)
    omitted = make_mmmu_send_fn("qwen3-omni", "http://localhost/v1")
    await seeded(session, mmmu_sample())
    await omitted(session, mmmu_sample())
    assert session.payloads[0]["seed"] == 4
    assert "seed" not in session.payloads[1]


def video_sample() -> VideoMMESample:
    return VideoMMESample(
        sample_id="v0",
        video_path="clip.mp4",
        question="q",
        options=["A"],
        answer="A",
        prompt="prompt",
    )


@pytest.mark.asyncio
async def test_video_request_includes_seed_only_when_set() -> None:
    session = RecordingSession()
    seeded = make_video_send_fn("qwen3-omni", "http://localhost/v1", seed=5)
    omitted = make_video_send_fn("qwen3-omni", "http://localhost/v1")
    await seeded(session, video_sample())
    await omitted(session, video_sample())
    assert session.payloads[0]["seed"] == 5
    assert "seed" not in session.payloads[1]


@pytest.mark.asyncio
async def test_rollout_request_uses_per_index_seed_and_talker() -> None:
    session = RecordingSession()
    sample = replace(mmmu_sample(), sample_id="m0:rollout-n1-0")
    send_fn = _make_rollout_send_fn(
        model_name="qwen3-omni",
        api_url="http://localhost/v1",
        rollout_group_id="group",
        max_tokens=8,
        temperature=0.8,
        enable_audio=False,
        talker_max_new_tokens=None,
        seeds_by_sample_id={sample.sample_id: 11},
        talker_params=talker_sampling_params(
            talker_temperature=0.2,
            talker_top_p=None,
            talker_top_k=None,
            talker_repetition_penalty=None,
        ),
    )
    await send_fn(session, sample)
    assert session.payloads[0]["seed"] == 11
    assert session.payloads[0]["talker_temperature"] == 0.2
