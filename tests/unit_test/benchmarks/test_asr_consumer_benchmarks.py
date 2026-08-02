# SPDX-License-Identifier: Apache-2.0
"""Consumer-model ASR benchmark and stability-harness contracts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import aiohttp
import pytest

from benchmarks.eval import benchmark_asr_seedtts, benchmark_asr_stability
from benchmarks.tasks.asr import (
    FUN_ASR_MODEL_PATH,
    OMNI_WHISPER_MODEL_PATH,
    QWEN3_ASR_MODEL_PATH,
)
from benchmarks.tts_serving.http_contracts import ResponseBodyTooLarge

EXPECTED_MODEL_REVISIONS = {
    QWEN3_ASR_MODEL_PATH: "7278e1e70fe206f11671096ffdd38061171dd6e5",
    FUN_ASR_MODEL_PATH: "854d88f94205cd17d2afdb24332130d86fbe654a",
    OMNI_WHISPER_MODEL_PATH: "06f233fe06e710322aca913c1bc4249a0d71fce1",
}


def test_consumer_model_revision_table_is_exact() -> None:
    assert benchmark_asr_seedtts.PINNED_MODEL_REVISIONS == EXPECTED_MODEL_REVISIONS


def _stability_args(
    model_path: str,
    **overrides,
) -> SimpleNamespace:
    values = {
        "concurrencies": "1",
        "duration_s": 1.0,
        "samples_per_language": 1,
        "request_timeout_s": 1.0,
        "max_audio_duration_s": 30.0,
        "monitor_interval_s": 0.2,
        "chaos_interval_s": 30.0,
        "cooldown_s": 0.0,
        "min_free_memory_mib": 0.0,
        "max_retained_memory_mib": 0.0,
        "gpu_index": 0,
        "model_path": model_path,
        "model_revision": None,
        "include_translation": None,
        "translation_source_language": "zh",
        "check_audio_boundary": None,
        "meta": "dataset",
        "dataset_revision": "revision",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_seedtts_repeat_keeps_per_sample_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    per_sample = [
        {
            "id": "sample-1",
            "is_success": True,
            "audio_duration_s": 4.0,
            "text_ttft_s": 0.1,
            "inter_chunk_s": [0.02],
        }
    ]

    async def fake_run(*_args, **_kwargs):
        return {
            "summary": {
                "evaluated": 1,
                "total_samples": 1,
                "skipped": 0,
                "corpus_wer": 0.0,
                "wer_per_sample_max": 0.0,
            },
            "speed": {
                "throughput_samples_per_s": 0.5,
                "rtfx": 2.0,
                "latency_mean_s": 0.2,
                "latency_p95_s": 0.2,
                "latency_p99_s": 0.2,
                "rtf_mean": 0.05,
                "rtf_p95": 0.05,
            },
            "wall_clock_s": 2.0,
            "worker": {},
            "per_sample": per_sample,
        }

    monkeypatch.setattr(
        benchmark_asr_seedtts,
        "run_asr_seedtts_once",
        fake_run,
    )
    args = SimpleNamespace(
        disable_resource_monitor=True,
        gpu_index=0,
        monitor_interval_s=0.2,
        host="127.0.0.1",
        port=8000,
        model_path=FUN_ASR_MODEL_PATH,
        lang="en",
        stream=False,
    )

    result = await benchmark_asr_seedtts._run_repeat(args, [], 1, 1)

    assert result["rtfx"] == 2.0
    assert result["audio_seconds_per_s"] == 2.0
    assert result["per_sample"] == per_sample


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", [1]), ("1,4,16", [1, 4, 16])],
)
def test_stability_concurrency_parser_accepts_positive_values(
    raw: str,
    expected: list[int],
) -> None:
    assert benchmark_asr_stability._parse_concurrencies(raw) == expected


@pytest.mark.parametrize("raw", ["", "0", "-1", "1,", "one"])
def test_stability_concurrency_parser_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        benchmark_asr_stability._parse_concurrencies(raw)


@pytest.mark.parametrize(
    "raw",
    ["nan", "inf", "-inf", "0"],
)
def test_seedtts_monitor_interval_parser_requires_finite_positive_value(
    raw: str,
) -> None:
    with pytest.raises(
        benchmark_asr_seedtts.argparse.ArgumentTypeError,
        match="finite and greater than zero",
    ):
        benchmark_asr_seedtts._positive_float(raw)


@pytest.mark.parametrize(
    (
        "model_path",
        "expected_translation",
        "expected_boundary",
    ),
    [
        (QWEN3_ASR_MODEL_PATH, False, False),
        (FUN_ASR_MODEL_PATH, False, True),
        (OMNI_WHISPER_MODEL_PATH, False, False),
    ],
)
def test_stability_model_defaults_are_capability_specific(
    model_path: str,
    expected_translation: bool,
    expected_boundary: bool,
) -> None:
    args = _stability_args(model_path)

    benchmark_asr_stability._validate_args(args)

    assert args.include_translation is expected_translation
    assert args.check_audio_boundary is expected_boundary
    assert args.model_revision is None


def test_stability_memory_retention_requires_free_and_cooldown_headroom() -> None:
    checkpoints = [
        {
            "label": "before_functional",
            "gpu_memory_used_mib": 100.0,
            "gpu_memory_free_mib": 23000.0,
        },
        {
            "label": "after_cooldown",
            "gpu_memory_used_mib": 300.0,
            "gpu_memory_free_mib": 22800.0,
        },
    ]
    resources = {"gpu_memory_free_mib": {"min": 2200.0}}

    passed = benchmark_asr_stability._validate_memory_retention(
        checkpoints,
        resources,
        min_free_memory_mib=2048.0,
        max_retained_memory_mib=256.0,
    )
    low_free = benchmark_asr_stability._validate_memory_retention(
        checkpoints,
        resources,
        min_free_memory_mib=4096.0,
        max_retained_memory_mib=256.0,
    )
    retained = benchmark_asr_stability._validate_memory_retention(
        checkpoints,
        resources,
        min_free_memory_mib=2048.0,
        max_retained_memory_mib=128.0,
    )

    assert passed["passed"] is True
    assert passed["retained_after_cooldown_mib"] == 200.0
    assert low_free["passed"] is False
    assert retained["passed"] is False


def test_stability_memory_retention_fails_closed_without_samples() -> None:
    result = benchmark_asr_stability._validate_memory_retention(
        [],
        {"available": False},
        min_free_memory_mib=0.0,
        max_retained_memory_mib=0.0,
    )

    assert result["passed"] is False
    assert result["error"] == "required GPU memory samples are unavailable"


def test_stability_memory_retention_rejects_nonfinite_samples() -> None:
    result = benchmark_asr_stability._validate_memory_retention(
        [
            {
                "label": "before_functional",
                "gpu_memory_used_mib": 100.0,
                "gpu_memory_free_mib": 23000.0,
            },
            {
                "label": "after_cooldown",
                "gpu_memory_used_mib": float("nan"),
                "gpu_memory_free_mib": 23000.0,
            },
        ],
        {"gpu_memory_free_mib": {"min": 23000.0}},
        min_free_memory_mib=2048.0,
        max_retained_memory_mib=256.0,
    )

    assert result["passed"] is False
    assert result["error"] == "required GPU memory samples are unavailable"


def test_stability_memory_retention_rejects_monitor_errors() -> None:
    checkpoints = [
        {
            "label": "before_functional",
            "gpu_memory_used_mib": 100.0,
            "gpu_memory_free_mib": 23000.0,
            "error": None,
        },
        {
            "label": "after_cooldown",
            "gpu_memory_used_mib": 100.0,
            "gpu_memory_free_mib": 23000.0,
            "error": None,
        },
    ]
    resource_error = benchmark_asr_stability._validate_memory_retention(
        checkpoints,
        {
            "gpu_memory_free_mib": {"min": 23000.0},
            "error": "NVML sampling failed",
        },
        min_free_memory_mib=2048.0,
        max_retained_memory_mib=256.0,
    )
    checkpoints[-1]["error"] = "checkpoint sampling failed"
    checkpoint_error = benchmark_asr_stability._validate_memory_retention(
        checkpoints,
        {"gpu_memory_free_mib": {"min": 23000.0}, "error": None},
        min_free_memory_mib=2048.0,
        max_retained_memory_mib=256.0,
    )

    assert resource_error["passed"] is False
    assert checkpoint_error["passed"] is False


@pytest.mark.asyncio
async def test_stability_transport_error_becomes_request_evidence() -> None:
    class FailingSession:
        def post(self, *_args, **_kwargs):
            raise aiohttp.ClientConnectionError("connection refused")

    args = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model_path=FUN_ASR_MODEL_PATH,
    )

    result = await benchmark_asr_stability._post_raw_audio(
        FailingSession(),
        args,
        b"RIFF",
        "sample.wav",
        "en",
    )

    assert result["status"] == 0
    assert result["text"] == ""
    assert "connection refused" in result["error"]


def test_stability_translation_requires_english_output() -> None:
    english = benchmark_asr_stability._expect_translation_success(
        "translation",
        {"status": 200, "text": "hello world", "error": None},
    )
    untranslated = benchmark_asr_stability._expect_translation_success(
        "translation",
        {"status": 200, "text": "你好世界", "error": None},
    )

    assert english["passed"] is True
    assert untranslated["passed"] is False


@pytest.mark.asyncio
async def test_stability_stream_uses_route_header_and_terminal_contract() -> None:
    captured: dict = {}

    async def content():
        yield b'data: {"type":"transcript.text.delta","delta":"hi"}\n'
        yield b'data: {"type":"transcript.text.done","text":"hi"}\n'
        yield b"data: [DONE]\n"

    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self.content = content()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class FakeSession:
        def post(self, _url, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    args = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model_path=FUN_ASR_MODEL_PATH,
    )
    sample = benchmark_asr_stability.PreparedSample(
        sample_id="sample",
        language="en",
        audio_bytes=b"RIFF",
        duration_s=1.0,
    )

    result = await benchmark_asr_stability._post_streaming_transcription(
        FakeSession(),
        args,
        sample,
    )

    assert captured["headers"] == {"x-sglang-omni-route-stream": "true"}
    assert result["status"] == 200
    assert result["done"] is True
    assert result["delta_events"] == 1
    assert result["first_event_latency_s"] is not None
    assert result["text"] == "hi"


@pytest.mark.asyncio
async def test_stability_cancel_stream_is_python_310_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def content():
        yield b'data: {"type":"transcript.text.delta","delta":"hi"}\n'

    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self.content = content()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    response = FakeResponse()

    class FakeSession:
        async def post(self, *_args, **_kwargs):
            return response

    monkeypatch.delattr(
        benchmark_asr_stability.asyncio,
        "timeout",
        raising=False,
    )
    args = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model_path=FUN_ASR_MODEL_PATH,
        request_timeout_s=1.0,
    )
    sample = benchmark_asr_stability.PreparedSample(
        sample_id="sample",
        language="en",
        audio_bytes=b"RIFF",
        duration_s=1.0,
    )

    result = await benchmark_asr_stability._cancel_stream(
        FakeSession(),
        args,
        sample,
    )

    assert result["received_event"] is True
    assert response.closed is True


@pytest.mark.asyncio
async def test_stability_cancel_stream_rejects_terminal_first_response() -> None:
    async def content():
        yield b'data: {"type":"transcript.text.done","text":"complete"}\n'
        yield b"data: [DONE]\n"

    class FakeResponse:
        status = 200

        def __init__(self) -> None:
            self.content = content()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    response = FakeResponse()

    class FakeSession:
        async def post(self, *_args, **_kwargs):
            return response

    args = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model_path=FUN_ASR_MODEL_PATH,
        request_timeout_s=1.0,
    )
    sample = benchmark_asr_stability.PreparedSample(
        sample_id="sample",
        language="en",
        audio_bytes=b"RIFF",
        duration_s=1.0,
    )

    result = await benchmark_asr_stability._cancel_stream(
        FakeSession(),
        args,
        sample,
    )

    assert result["received_event"] is False
    assert response.closed is True


@pytest.mark.asyncio
async def test_stability_stops_resource_monitor_when_functional_phase_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped = False

    class FakeMonitor:
        def __init__(self, **_kwargs) -> None:
            self.interval_s = 0.2
            self.samples = [object()]
            self.error = None

        def start(self):
            return self

        def stop(self):
            nonlocal stopped
            stopped = True
            return {"available": False, "error": "fake"}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    async def fail_functional(*_args, **_kwargs):
        raise RuntimeError("functional failed")

    sample = benchmark_asr_stability.PreparedSample(
        sample_id="sample",
        language="en",
        audio_bytes=b"RIFF",
        duration_s=1.0,
    )
    monkeypatch.setattr(benchmark_asr_stability, "ResourceMonitor", FakeMonitor)
    monkeypatch.setattr(
        benchmark_asr_stability,
        "_load_prepared_samples",
        lambda _args, _revision: [sample],
    )
    monkeypatch.setattr(
        benchmark_asr_stability,
        "_evaluation_input_sha256",
        lambda _samples: "hash",
    )
    monkeypatch.setattr(
        benchmark_asr_stability,
        "_memory_checkpoint",
        lambda *_args: {"label": "checkpoint"},
    )
    monkeypatch.setattr(
        benchmark_asr_stability.aiohttp,
        "TCPConnector",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        benchmark_asr_stability.aiohttp,
        "ClientSession",
        lambda **_kwargs: FakeSession(),
    )
    monkeypatch.setattr(
        benchmark_asr_stability,
        "_run_functional_checks",
        fail_functional,
    )
    args = _stability_args(
        FUN_ASR_MODEL_PATH,
        include_translation=False,
        check_audio_boundary=False,
    )

    with pytest.raises(RuntimeError, match="functional failed"):
        await benchmark_asr_stability.main_async(args)

    assert stopped is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration_s", float("inf")),
        ("request_timeout_s", float("nan")),
        ("max_audio_duration_s", float("-inf")),
        ("monitor_interval_s", float("inf")),
        ("chaos_interval_s", float("nan")),
        ("cooldown_s", float("inf")),
        ("min_free_memory_mib", float("nan")),
        ("max_retained_memory_mib", float("inf")),
    ],
)
def test_stability_validation_rejects_nonfinite_operational_values(
    field: str,
    value: float,
) -> None:
    args = _stability_args(FUN_ASR_MODEL_PATH)
    setattr(args, field, value)

    with pytest.raises(ValueError, match="finite"):
        benchmark_asr_stability._validate_args(args)


def test_stability_rejects_one_local_meta_for_bilingual_soak() -> None:
    args = _stability_args(
        FUN_ASR_MODEL_PATH,
        meta="local-seedtts.lst",
    )

    with pytest.raises(ValueError, match="distinct en and zh splits"):
        benchmark_asr_stability._validate_args(args)


def test_stability_prepared_samples_are_language_interleaved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_splits: list[str] = []

    def fake_load(
        _meta,
        *,
        max_samples,
        split,
        revision,
    ):
        requested_splits.append(split)
        return [
            SimpleNamespace(sample_id=f"{split}-{index}")
            for index in range(max_samples)
        ]

    def fake_prepare(sample, language):
        return benchmark_asr_stability.PreparedSample(
            sample_id=sample.sample_id,
            language=language,
            audio_bytes=b"RIFF",
            duration_s=1.0,
        )

    monkeypatch.setattr(
        benchmark_asr_stability,
        "load_seedtts_samples",
        fake_load,
    )
    monkeypatch.setattr(
        benchmark_asr_stability,
        "_prepare_sample",
        fake_prepare,
    )
    args = _stability_args(
        FUN_ASR_MODEL_PATH,
        samples_per_language=2,
    )

    samples = benchmark_asr_stability._load_prepared_samples(
        args,
        "revision",
    )

    assert requested_splits == ["en", "zh"]
    assert [sample.sample_id for sample in samples] == [
        "en-0",
        "zh-0",
        "en-1",
        "zh-1",
    ]


@pytest.mark.asyncio
async def test_stability_response_cap_becomes_request_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def too_large(_response):
        raise ResponseBodyTooLarge(bytes_read=65, max_bytes=64)

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class FakeSession:
        def post(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        benchmark_asr_stability,
        "read_response_body",
        too_large,
    )
    args = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model_path=FUN_ASR_MODEL_PATH,
    )

    result = await benchmark_asr_stability._post_raw_audio(
        FakeSession(),
        args,
        b"RIFF",
        "sample.wav",
        "en",
    )

    assert result["status"] == 200
    assert result["text"] == ""
    assert "read cap" in result["error"]


def test_stability_sse_caps_line_and_cumulative_bytes() -> None:
    with pytest.raises(ResponseBodyTooLarge, match="read cap"):
        benchmark_asr_stability._checked_stream_bytes(
            0,
            b"x" * (benchmark_asr_stability.MAX_SSE_LINE_BYTES + 1),
        )
    with pytest.raises(ResponseBodyTooLarge, match="read cap"):
        benchmark_asr_stability._checked_stream_bytes(
            benchmark_asr_stability.MAX_HTTP_RESPONSE_BYTES,
            b"x",
        )


@pytest.mark.asyncio
async def test_stability_soak_bounds_total_concurrency_and_global_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0

    async def pulse() -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.002)
        finally:
            active -= 1

    async def successful_request(*_args):
        await pulse()
        return {"status": 200, "text": "ok", "latency_s": 0.002}

    async def malformed_request(*_args):
        await pulse()
        return {"status": 400, "error": None}

    async def cancel_request(*_args):
        await pulse()
        return {"status": 200, "received_event": True, "error": None}

    async def reconnect_request(*_args):
        await pulse()
        return {"status": 200, "done": True, "error": None}

    monkeypatch.setattr(
        benchmark_asr_stability,
        "_post_transcription",
        successful_request,
    )
    monkeypatch.setattr(
        benchmark_asr_stability,
        "_post_translation",
        successful_request,
    )
    monkeypatch.setattr(
        benchmark_asr_stability,
        "_post_raw_audio",
        malformed_request,
    )
    monkeypatch.setattr(
        benchmark_asr_stability,
        "_cancel_stream",
        cancel_request,
    )
    monkeypatch.setattr(
        benchmark_asr_stability,
        "_post_streaming_transcription",
        reconnect_request,
    )
    monkeypatch.setattr(
        benchmark_asr_stability,
        "_memory_checkpoint",
        lambda *_args: {},
    )
    samples = [
        benchmark_asr_stability.PreparedSample(
            sample_id=f"{language}-{index}",
            language=language,
            audio_bytes=b"RIFF",
            duration_s=1.0,
        )
        for index, language in enumerate(("en", "zh", "en", "zh"))
    ]
    args = SimpleNamespace(
        seed=7,
        duration_s=0.04,
        include_translation=True,
        chaos_interval_s=0.005,
    )

    stages, chaos = await benchmark_asr_stability._run_soak(
        object(),
        args,
        samples,
        concurrencies=[4],
        monitor=object(),
    )

    stage = stages[0]
    assert active == 0
    assert max_active <= 4
    assert stage["max_in_flight_observed"] == max_active
    assert stage["chaos_events"] == len(chaos)
    assert stage["chaos_events"] > 0
    assert stage["total_http_requests"] == (stage["requests"] + stage["chaos_requests"])
    assert stage["translations"] == (stage["zh_requests_issued"] + 3) // 4
    assert stage["translation_observed_ratio"] == (stage["translations"] / stage["zh"])


@pytest.mark.asyncio
async def test_stability_soak_cancellation_cleans_up_all_child_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    active = 0

    async def blocking_request(*_args):
        nonlocal active
        active += 1
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1

    for name in (
        "_post_transcription",
        "_post_translation",
        "_post_raw_audio",
        "_cancel_stream",
        "_post_streaming_transcription",
    ):
        monkeypatch.setattr(
            benchmark_asr_stability,
            name,
            blocking_request,
        )
    monkeypatch.setattr(
        benchmark_asr_stability,
        "_memory_checkpoint",
        lambda *_args: {},
    )
    samples = [
        benchmark_asr_stability.PreparedSample(
            sample_id="en",
            language="en",
            audio_bytes=b"RIFF",
            duration_s=1.0,
        ),
        benchmark_asr_stability.PreparedSample(
            sample_id="zh",
            language="zh",
            audio_bytes=b"RIFF",
            duration_s=1.0,
        ),
    ]
    args = SimpleNamespace(
        seed=7,
        duration_s=10.0,
        include_translation=False,
        chaos_interval_s=30.0,
    )
    tasks_before = set(asyncio.all_tasks())
    soak_task = asyncio.create_task(
        benchmark_asr_stability._run_soak(
            object(),
            args,
            samples,
            concurrencies=[2],
            monitor=object(),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)

    soak_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await soak_task
    await asyncio.sleep(0)

    assert active == 0
    assert set(asyncio.all_tasks()) <= tasks_before
