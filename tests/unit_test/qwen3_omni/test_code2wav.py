# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.qwen3_omni.components import code2wav_scheduler
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavRunResult,
    GraphKey,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel, make_qwen_payload

_DEFAULT_GRAPH_KEYS = tuple(
    GraphKey(batch_size=1, frames=frames) for frames in (10, 20, 30, 35)
)


class _FactoryModel(FakeCode2WavModel):
    def __init__(self, *, num_quantizers: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_quantizers=num_quantizers)
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self


class _FakeCudaGraphRunner:
    def __init__(self, model, *, replay_error: Exception | None = None) -> None:
        self.model = model
        self.replay_error = replay_error
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def run(self, codes: torch.Tensor, *, eligible: bool) -> Code2WavRunResult:
        self.calls.append((tuple(codes.shape), eligible))
        if self.replay_error is not None:
            raise self.replay_error
        output = self.model(codes)
        key = GraphKey(batch_size=int(codes.shape[0]), frames=int(codes.shape[-1]))
        if eligible and key in _DEFAULT_GRAPH_KEYS:
            return Code2WavRunResult(output, "cuda_graph", key, None)
        if eligible:
            return Code2WavRunResult(output, "eager", key, "key_miss")
        return Code2WavRunResult(output, "eager", None, "ineligible")


def _activate_event_capture(monkeypatch) -> list[dict]:
    events: list[dict] = []

    class _ActiveRecorder:
        @staticmethod
        def is_active() -> bool:
            return True

    monkeypatch.setattr(
        code2wav_scheduler, "_get_event_recorder", lambda: _ActiveRecorder()
    )
    monkeypatch.setattr(
        code2wav_scheduler, "_emit_event", lambda **event: events.append(event)
    )
    return events


def _seed_stream_state(
    scheduler: Code2WavScheduler,
    request_id: str = "req-1",
) -> None:
    scheduler._stream_payloads[request_id] = make_qwen_payload(request_id=request_id)
    scheduler._get_or_create_stream_state(request_id)


def test_qwen_load_code2wav_model_returns_eval_model(monkeypatch) -> None:
    from transformers import AutoConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeCode2Wav,
    )

    from sglang_omni.models import weight_loader

    model = _FactoryModel()
    config = SimpleNamespace(code2wav_config=object())
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: config),
    )
    monkeypatch.setattr(
        Qwen3OmniMoeCode2Wav,
        "_from_config",
        staticmethod(lambda code2wav_config: model),
    )
    monkeypatch.setattr(weight_loader, "resolve_dtype", lambda dtype: torch.float32)
    monkeypatch.setattr(
        weight_loader,
        "load_module",
        lambda loaded_model, *args, **kwargs: loaded_model,
    )

    loaded = code2wav_scheduler.load_code2wav_model("dummy", device="cpu")

    assert loaded is model
    assert model.eval_calls == 1


def test_qwen_code2wav_factory_default_does_not_build_cuda_graphs(monkeypatch) -> None:
    model = _FactoryModel()
    monkeypatch.setattr(
        code2wav_scheduler, "load_code2wav_model", lambda *a, **k: model
    )

    def _unexpected_build(*args, **kwargs):
        raise AssertionError("disabled default must not build CUDA graphs")

    monkeypatch.setattr(
        code2wav_scheduler.Code2WavCudaGraphRunner,
        "build",
        staticmethod(_unexpected_build),
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "dummy",
        device="cpu",
    )

    assert scheduler._cuda_graph_runner is None


def test_qwen_code2wav_enabled_factory_rejects_missing_typed_budget_before_load(
    monkeypatch,
) -> None:
    load_calls = 0

    def _load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return _FactoryModel()

    monkeypatch.setattr(code2wav_scheduler, "load_code2wav_model", _load)

    with pytest.raises(ValueError, match="total_gpu_memory_fraction"):
        code2wav_scheduler.create_code2wav_scheduler(
            "dummy",
            device="cuda:0",
            enable_cuda_graph=True,
        )

    assert load_calls == 0


def test_qwen_code2wav_factory_rejects_batching_with_cuda_graph_before_load(
    monkeypatch,
) -> None:
    load_calls = 0

    def _load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return _FactoryModel()

    monkeypatch.setattr(code2wav_scheduler, "load_code2wav_model", _load)

    with pytest.raises(ValueError, match="cannot be enabled together"):
        code2wav_scheduler.create_code2wav_scheduler(
            "dummy",
            device="cuda:0",
            enable_batching=True,
            enable_cuda_graph=True,
            total_gpu_memory_fraction=0.02,
        )

    assert load_calls == 0


@pytest.mark.parametrize(
    ("stream_chunk_size", "left_context_size", "expected_frames"),
    [
        (10, 25, (10, 20, 30, 35)),
        (20, 25, (20, 40, 45)),
        (6, 0, (6,)),
    ],
)
def test_qwen_code2wav_serial_threshold_graph_keys_follow_scheduler_windows(
    stream_chunk_size: int,
    left_context_size: int,
    expected_frames: tuple[int, ...],
) -> None:
    assert code2wav_scheduler._serial_threshold_graph_keys(
        stream_chunk_size,
        left_context_size,
    ) == tuple(GraphKey(batch_size=1, frames=frames) for frames in expected_frames)


def test_qwen_code2wav_enabled_factory_normalizes_device_and_derives_graph_keys(
    monkeypatch,
    caplog,
) -> None:
    model = _FactoryModel(num_quantizers=12)
    expected_graph_keys = tuple(
        GraphKey(batch_size=1, frames=frames) for frames in (20, 40, 45)
    )
    runner = SimpleNamespace(
        stats=lambda: {
            "enabled": True,
            "disable_reason": None,
            "graph_contract": {
                "keys": [
                    {"batch_size": key.batch_size, "frames": key.frames}
                    for key in expected_graph_keys
                ]
            },
            "build": {"published_graph_count": 3},
        }
    )
    load_call: dict = {}
    build_call: dict = {}
    monkeypatch.setattr(
        code2wav_scheduler.torch.cuda,
        "current_device",
        lambda: 3,
    )

    def _load(*args, **kwargs):
        load_call.update(args=args, **kwargs)
        return model.eval()

    def _build(built_model, **kwargs):
        build_call.update(model=built_model, **kwargs)
        return runner

    monkeypatch.setattr(code2wav_scheduler, "load_code2wav_model", _load)
    monkeypatch.setattr(
        code2wav_scheduler.Code2WavCudaGraphRunner,
        "build",
        staticmethod(_build),
    )

    with caplog.at_level(logging.INFO, logger=code2wav_scheduler.__name__):
        scheduler = code2wav_scheduler.create_code2wav_scheduler(
            "dummy",
            enable_cuda_graph=True,
            total_gpu_memory_fraction=0.02,
            stream_chunk_size=20,
            left_context_size=25,
        )

    assert model.eval_calls == 1
    assert load_call == {
        "args": ("dummy",),
        "device": "cuda:3",
        "dtype": None,
    }
    assert build_call == {
        "model": model,
        "device": torch.device("cuda:3"),
        "num_quantizers": 12,
        "total_gpu_memory_fraction": 0.02,
        "graph_keys": expected_graph_keys,
    }
    assert scheduler._device == torch.device("cuda:3")
    assert scheduler._stream_chunk_size == 20
    assert scheduler._left_context_size == 25
    assert scheduler._cuda_graph_runner is runner
    stats_record = next(
        record
        for record in caplog.records
        if "CUDA graph startup stats=" in record.message
    )
    assert json.loads(stats_record.message.split("stats=", 1)[1]) == runner.stats()


def test_qwen_code2wav_enabled_factory_logs_disabled_build_reason(
    monkeypatch,
    caplog,
) -> None:
    model = _FactoryModel(num_quantizers=12)
    runner = SimpleNamespace(
        stats=lambda: {
            "enabled": False,
            "disable_reason": "capture_failed: RuntimeError: capture failed",
        }
    )
    monkeypatch.setattr(
        code2wav_scheduler, "load_code2wav_model", lambda *a, **k: model.eval()
    )
    monkeypatch.setattr(
        code2wav_scheduler.Code2WavCudaGraphRunner,
        "build",
        staticmethod(lambda *args, **kwargs: runner),
    )

    with caplog.at_level(logging.INFO, logger=code2wav_scheduler.__name__):
        scheduler = code2wav_scheduler.create_code2wav_scheduler(
            "dummy",
            gpu_id=3,
            enable_cuda_graph=True,
            total_gpu_memory_fraction=0.02,
        )

    assert scheduler._cuda_graph_runner is runner
    stats_record = next(
        record
        for record in caplog.records
        if "CUDA graph startup stats=" in record.message
    )
    assert json.loads(stats_record.message.split("stats=", 1)[1]) == {
        "disable_reason": "capture_failed: RuntimeError: capture failed",
        "enabled": False,
    }


def test_qwen_code2wav_threshold_context_windows_hit_cuda_graph(monkeypatch) -> None:
    model = FakeCode2WavModel(total_upsample=2)
    runner = _FakeCudaGraphRunner(model)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=25,
        enable_cuda_graph=True,
        _cuda_graph_runner=runner,
    )
    _seed_stream_state(scheduler)
    events = _activate_event_capture(monkeypatch)

    for chunk_id in range(40):
        scheduler._on_chunk(
            "req-1",
            StreamItem(
                chunk_id,
                torch.tensor([chunk_id + 1, 10]),
                "talker",
                metadata={"stream": True},
            ),
        )

    assert runner.calls == [
        ((1, 2, 10), True),
        ((1, 2, 20), True),
        ((1, 2, 30), True),
        ((1, 2, 35), True),
    ]
    decode_ends = [
        event for event in events if event["event_name"] == "code2wav_decode_end"
    ]
    assert [event["metadata"]["execution_mode"] for event in decode_ends] == [
        "cuda_graph"
    ] * 4
    assert [event["metadata"]["graph_key"]["frames"] for event in decode_ends] == [
        10,
        20,
        30,
        35,
    ]


def test_qwen_code2wav_stream_done_tail_is_eager_when_shape_matches_graph(
    monkeypatch,
) -> None:
    model = FakeCode2WavModel(total_upsample=2)
    runner = _FakeCudaGraphRunner(model)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=6,
        left_context_size=5,
        enable_cuda_graph=True,
        _cuda_graph_runner=runner,
    )
    _seed_stream_state(scheduler)
    events = _activate_event_capture(monkeypatch)

    for chunk_id in range(11):
        scheduler._on_chunk(
            "req-1",
            StreamItem(
                chunk_id,
                torch.tensor([chunk_id + 1, 10]),
                "talker",
                metadata={"stream": False},
            ),
        )
    scheduler._on_done("req-1")

    assert runner.calls == [((1, 2, 6), True), ((1, 2, 10), False)]
    decode_ends = [
        event for event in events if event["event_name"] == "code2wav_decode_end"
    ]
    tail_metadata = decode_ends[-1]["metadata"]
    assert tail_metadata["trigger"] == "stream_done"
    assert tail_metadata["window_frames"] == 10
    assert tail_metadata["execution_mode"] == "eager"
    assert tail_metadata["graph_key"] is None
    assert tail_metadata["fallback_reason"] == "ineligible"


def test_qwen_code2wav_request_events_are_symmetric_and_keep_start_metadata(
    monkeypatch,
) -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )
    _seed_stream_state(scheduler)
    events = _activate_event_capture(monkeypatch)

    for chunk_id, codes in enumerate(([1, 10], [2, 20], [3, 30])):
        scheduler._on_chunk(
            "req-1",
            StreamItem(
                chunk_id,
                torch.tensor(codes),
                "talker",
                metadata={"stream": False},
            ),
        )
    scheduler._on_done("req-1")

    decode_events = [
        event for event in events if event["event_name"].startswith("code2wav_decode_")
    ]
    assert [event["event_name"] for event in decode_events] == [
        "code2wav_decode_start",
        "code2wav_decode_end",
        "code2wav_decode_start",
        "code2wav_decode_end",
    ]
    first = decode_events[0]["metadata"]
    assert first == {
        "trigger": "threshold",
        "start_frame": 0,
        "end_frame": 2,
        "new_frames": 2,
        "context_frames": 0,
        "window_frames": 2,
        "active_request_count": 1,
        "threshold_ready_request_count": 1,
        "inbox_depth": 0,
        "pending_message_depth": 0,
    }
    assert decode_events[1]["metadata"] == {
        **first,
        "audio_samples": 4,
        "execution_mode": "eager",
        "graph_key": None,
        "fallback_reason": None,
    }
    tail = decode_events[2]["metadata"]
    assert tail["trigger"] == "stream_done"
    assert tail["new_frames"] == 1
    assert tail["context_frames"] == 1
    assert tail["window_frames"] == 2
    assert tail["threshold_ready_request_count"] == 0
    assert decode_events[3]["metadata"] == {
        **tail,
        "audio_samples": 2,
        "execution_mode": "eager",
        "graph_key": None,
        "fallback_reason": None,
    }
    assert any(event["event_name"] == "code2wav_first_audio" for event in events)


def test_qwen_code2wav_eligible_key_miss_has_json_safe_fallback_metadata(
    monkeypatch,
) -> None:
    model = FakeCode2WavModel(total_upsample=2)
    runner = _FakeCudaGraphRunner(model)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=6,
        left_context_size=0,
        enable_cuda_graph=True,
        _cuda_graph_runner=runner,
    )
    _seed_stream_state(scheduler)
    events = _activate_event_capture(monkeypatch)

    for chunk_id in range(6):
        scheduler._on_chunk(
            "req-1",
            StreamItem(
                chunk_id,
                torch.tensor([chunk_id + 1, 10]),
                "talker",
                metadata={"stream": True},
            ),
        )

    decode_start = next(
        event for event in events if event["event_name"] == "code2wav_decode_start"
    )
    decode_end = next(
        event for event in events if event["event_name"] == "code2wav_decode_end"
    )
    assert "execution_mode" not in decode_start["metadata"]
    assert "graph_key" not in decode_start["metadata"]
    assert "fallback_reason" not in decode_start["metadata"]
    assert decode_end["metadata"]["execution_mode"] == "eager"
    assert decode_end["metadata"]["graph_key"] == {
        "batch_size": 1,
        "frames": 6,
    }
    assert decode_end["metadata"]["fallback_reason"] == "key_miss"


def _run_code2wav_stream(*, cuda_graph: bool) -> tuple[list[tuple], object]:
    model = FakeCode2WavModel(total_upsample=2)
    runner = _FakeCudaGraphRunner(model) if cuda_graph else None
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=1,
        enable_cuda_graph=cuda_graph,
        _cuda_graph_runner=runner,
    )
    _seed_stream_state(scheduler)
    for chunk_id in range(11):
        scheduler._on_chunk(
            "req-1",
            StreamItem(
                chunk_id,
                torch.tensor([chunk_id + 1, 10]),
                "talker",
                metadata={"stream": True},
            ),
        )
    scheduler._on_done("req-1")

    messages = [scheduler.outbox.get_nowait() for _ in range(scheduler.outbox.qsize())]
    snapshot: list[tuple] = []
    for message in messages:
        if message.type == "stream":
            snapshot.append(
                (
                    message.type,
                    message.data["audio_waveform"],
                    message.data["sample_rate"],
                    message.metadata,
                )
            )
        else:
            snapshot.append((message.type, message.data.data))
    return snapshot, runner


def test_qwen_code2wav_graph_output_protocol_matches_eager_exactly() -> None:
    eager_snapshot, _ = _run_code2wav_stream(cuda_graph=False)
    graph_snapshot, runner = _run_code2wav_stream(cuda_graph=True)

    assert graph_snapshot == eager_snapshot
    assert [item[0] for item in graph_snapshot] == ["stream", "stream", "result"]
    assert [
        len(np.frombuffer(item[1], dtype=np.float32))
        for item in graph_snapshot
        if item[0] == "stream"
    ] == [20, 2]
    assert runner.calls == [((1, 2, 10), True), ((1, 2, 2), False)]


def test_qwen_code2wav_consumes_borrowed_output_under_state_lock() -> None:
    class _BorrowedOutputRunner:
        def __init__(self) -> None:
            self.scheduler = None
            self.static_output = torch.zeros((1, 1, 2), dtype=torch.float32)
            self.lock_was_held: list[bool] = []
            self.replays = 0

        def run(self, codes: torch.Tensor, *, eligible: bool) -> Code2WavRunResult:
            assert eligible
            self.lock_was_held.append(self.scheduler._state_lock._is_owned())
            self.replays += 1
            self.static_output.fill_(float(self.replays))
            return Code2WavRunResult(
                self.static_output,
                "cuda_graph",
                GraphKey(1, int(codes.shape[-1])),
                None,
            )

    model = FakeCode2WavModel(total_upsample=2)
    runner = _BorrowedOutputRunner()
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=1,
        left_context_size=0,
        enable_cuda_graph=True,
        _cuda_graph_runner=runner,
    )
    runner.scheduler = scheduler
    _seed_stream_state(scheduler)

    for chunk_id in range(2):
        scheduler._on_chunk(
            "req-1",
            StreamItem(
                chunk_id,
                torch.tensor([chunk_id + 1, 10]),
                "talker",
                metadata={"stream": True},
            ),
        )

    assert runner.lock_was_held == [True, True]
    assert [
        chunk.tolist() for chunk in scheduler._stream_states["req-1"].audio_parts
    ] == [
        [1.0, 1.0],
        [2.0, 2.0],
    ]


def test_qwen_code2wav_replay_error_reaches_base_abort_without_eager_retry() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    replay_error = RuntimeError("replay failed")
    runner = _FakeCudaGraphRunner(model, replay_error=replay_error)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=1,
        left_context_size=0,
        enable_cuda_graph=True,
        _cuda_graph_runner=runner,
    )
    _seed_stream_state(scheduler)
    scheduler.inbox.put(
        IncomingMessage(
            request_id="req-1",
            type="stream_chunk",
            data=StreamItem(
                0,
                torch.tensor([1, 10]),
                "talker",
                metadata={"stream": True},
            ),
        )
    )

    worker = threading.Thread(target=scheduler.start)
    worker.start()
    try:
        message = scheduler.outbox.get(timeout=2.0)
    finally:
        scheduler.stop()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert runner.calls == [((1, 2, 1), True)]
    assert model.calls == []
    assert message.request_id == "req-1"
    assert message.type == "error"
    assert message.data is replay_error
    assert scheduler._is_aborted("req-1")
    assert "req-1" not in scheduler._stream_states


def _make_scheduler(model: FakeCode2WavModel) -> Code2WavScheduler:
    return Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )


def _feed(
    scheduler: Code2WavScheduler,
    request_id: str,
    codes: tuple[int, ...],
    *,
    stream: bool,
) -> None:
    meta = {"stream": stream}
    for i, code in enumerate(codes):
        scheduler._handle_stream_chunk(
            request_id,
            StreamItem(i, torch.tensor([code, code * 10]), "talker", metadata=meta),
        )


def test_qwen_code2wav_streams_incrementally_and_abort_clears_state() -> None:
    """Preserves incremental waveform windows and request-state cleanup on abort."""
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = _make_scheduler(model)
    scheduler._stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    _feed(scheduler, "req-1", (1, 2, 3), stream=False)
    scheduler._on_done("req-1")

    message = scheduler.outbox.get_nowait()
    audio = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 2, 2), (1, 2, 2)]
    assert audio.shape == (6,)

    scheduler._stream_payloads["req-2"] = make_qwen_payload(request_id="req-2")
    scheduler._get_or_create_stream_state("req-2")
    scheduler.abort("req-2")
    assert "req-2" not in scheduler._stream_states


def test_streaming_client_gets_stream_chunks_and_metadata_final() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = _make_scheduler(model)
    scheduler._stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    _feed(scheduler, "req-1", (1, 2, 3), stream=True)
    scheduler._on_done("req-1")

    first = scheduler.outbox.get_nowait()
    assert first.type == "stream"
    first_audio = np.frombuffer(first.data["audio_waveform"], dtype=np.float32)
    assert first_audio.shape == (4,)

    remainder = scheduler.outbox.get_nowait()
    assert remainder.type == "stream"
    remainder_audio = np.frombuffer(remainder.data["audio_waveform"], dtype=np.float32)
    assert remainder_audio.shape == (2,)

    result = scheduler.outbox.get_nowait()
    assert result.type == "result"
    assert result.data.data == {"modality": "audio", "sample_rate": 24000}
    assert model.calls == [(1, 2, 2), (1, 2, 2)]


def test_eos_chunk_is_skipped_and_never_decoded() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = _make_scheduler(model)
    scheduler._stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    _feed(scheduler, "req-1", (1, 2), stream=False)
    assert model.calls == [(1, 2, 2)]

    scheduler._handle_stream_chunk(
        "req-1",
        StreamItem(2, torch.tensor([2150, 0]), "talker", metadata={"stream": False}),
    )
    assert model.calls == [(1, 2, 2)]

    scheduler._on_done("req-1")
    message = scheduler.outbox.get_nowait()
    audio = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 2, 2)]
    assert audio.shape == (4,)
