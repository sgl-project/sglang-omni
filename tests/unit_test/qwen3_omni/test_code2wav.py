# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)

from sglang_omni.config.schema import StageConfig
from sglang_omni.models.qwen3_omni.components import code2wav_scheduler
from sglang_omni.models.qwen3_omni.components.code2wav import (
    Qwen3OmniCode2Wav,
    Qwen3OmniIncrementalCodec,
)
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavCudaGraphRunner,
    Code2WavRunResult,
    GraphKey,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.models.qwen3_omni.components.codec_state_pool import CodecStatePool
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.utils import snake_beta
from tests.unit_test.fixtures.qwen_fakes import (
    FakeCode2WavModel,
    deliver_code2wav_chunk,
    make_qwen_payload,
)

DEFAULT_GRAPH_KEYS = tuple(
    GraphKey(batch_size=1, frames=frames) for frames in (10, 20, 30, 35)
)


class FactoryModel(FakeCode2WavModel):
    def __init__(self, *, num_quantizers: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_quantizers=num_quantizers)
        self.eval_calls = 0

    def parameters(self) -> list[torch.Tensor]:
        return [torch.zeros(3, dtype=torch.float32)]

    def buffers(self) -> list[torch.Tensor]:
        return [torch.zeros(2, dtype=torch.long)]

    def eval(self):
        self.eval_calls += 1
        return self


def pin_cuda_platform(monkeypatch) -> None:
    import sglang_omni.platforms as platforms

    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )


class FakeDecodeStream:
    def __init__(self, *, device: torch.device, priority: int) -> None:
        self.device = device
        self.priority = priority

    @staticmethod
    def priority_range() -> tuple[int, int]:
        return (0, -3)


class FakeCudaGraphRunner:
    def __init__(self, model, *, replay_error: Exception | None = None) -> None:
        self.model = model
        self.replay_error = replay_error
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def available_batch_sizes(self, frames: int, decode: str) -> tuple[int, ...]:
        return tuple(
            key.batch_size
            for key in DEFAULT_GRAPH_KEYS
            if key.frames == int(frames) and key.decode == decode
        )

    def run(self, codes: torch.Tensor, *, eligible: bool) -> Code2WavRunResult:
        self.calls.append((tuple(codes.shape), eligible))
        if self.replay_error is not None:
            raise self.replay_error
        output = self.model(codes)
        key = GraphKey(batch_size=int(codes.shape[0]), frames=int(codes.shape[-1]))
        if eligible and key in DEFAULT_GRAPH_KEYS:
            return Code2WavRunResult(output, "cuda_graph", key, None)
        if eligible:
            return Code2WavRunResult(output, "eager", key, "key_miss")
        return Code2WavRunResult(output, "eager", None, "ineligible")


def activate_event_capture(monkeypatch) -> list[dict]:
    events: list[dict] = []

    class ActiveRecorder:
        @staticmethod
        def is_active() -> bool:
            return True

        @staticmethod
        def active_run_id() -> str:
            return "test-run"

    monkeypatch.setattr(
        code2wav_scheduler, "_get_event_recorder", lambda: ActiveRecorder()
    )
    monkeypatch.setattr(
        code2wav_scheduler, "_emit_event", lambda **event: events.append(event)
    )
    return events


def seed_stream_state(
    scheduler: Code2WavScheduler,
    request_id: str = "req-1",
) -> None:
    scheduler.stream_payloads[request_id] = make_qwen_payload(request_id=request_id)
    scheduler.get_or_create_stream_state(request_id)


def test_qwen_code2wav_factory_default_does_not_build_cuda_graphs(monkeypatch) -> None:
    model = FactoryModel()
    monkeypatch.setattr(
        code2wav_scheduler, "load_code2wav_model", lambda *a, **k: model
    )

    def unexpected_build(*args, **kwargs):
        raise AssertionError("disabled default must not build device graphs")

    monkeypatch.setattr(
        code2wav_scheduler.Code2WavCudaGraphRunner,
        "build",
        staticmethod(unexpected_build),
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "dummy",
        device="cpu",
    )

    assert scheduler.cuda_graph_runner is None


def test_the_code2wav_stage_takes_its_graph_flag_from_the_platform() -> None:
    """config.py must wire the hook through, since the factory no longer guards."""
    from sglang_omni.config import resolve_stage_factory_args
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
    from sglang_omni.platforms import current_platform

    config = Qwen3OmniSpeechPipelineConfig(model_path="unused")
    stage = next(s for s in config.stages if s.name == "code2wav")
    args = resolve_stage_factory_args(stage, config)

    assert args["enable_cuda_graph"] is current_platform.enable_code2wav_graph()


def test_qwen_code2wav_enabled_factory_rejects_missing_typed_budget_before_load(
    monkeypatch,
) -> None:
    load_calls = 0

    def load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return FactoryModel()

    monkeypatch.setattr(code2wav_scheduler, "load_code2wav_model", load)

    with pytest.raises(ValueError, match="gpu_memory_fraction") as excinfo:
        code2wav_scheduler.create_code2wav_scheduler(
            "dummy",
            device="cuda",
            gpu_id=0,
            enable_cuda_graph=True,
        )

    assert load_calls == 0
    # The message aborts startup, so the key it names has to be settable.
    named = {word.strip("'\".,:") for word in str(excinfo.value).split()}
    assert named & set(StageConfig.model_fields), str(excinfo.value)


def test_qwen_code2wav_factory_captures_every_walk_length_at_every_row_count(
    monkeypatch,
) -> None:
    pin_cuda_platform(monkeypatch)
    captured: list[dict] = []

    class RecordingRunner:
        @staticmethod
        def build(model, **kwargs):
            captured.append(kwargs)
            runner = object.__new__(code2wav_scheduler.Code2WavCudaGraphRunner)
            runner.stats = lambda: {"enabled": True, "disable_reason": None}
            return runner

    monkeypatch.setattr(
        code2wav_scheduler,
        "load_code2wav_model",
        lambda *args, **kwargs: FactoryModel(),
    )
    monkeypatch.setattr(
        code2wav_scheduler,
        "Code2WavCudaGraphRunner",
        RecordingRunner,
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "dummy",
        device="cuda",
        gpu_id=0,
        max_replay_rows=3,
        enable_cuda_graph=True,
        total_gpu_memory_fraction=0.02,
    )

    assert scheduler.max_replay_rows == 3
    (build,) = captured
    walk = (10, 20, 30, 35)
    # every row count of every walk length, so any number of queued windows up to the cap
    # replays one graph; single rows first, they are the atomic tier
    assert build["graph_keys"] == tuple(
        GraphKey(batch_size=rows, frames=frames)
        for rows in (1, 2, 3)
        for frames in walk
    )
    # final windows stay one row each: every length a stream can end on, short streams
    # included, minus the walk's own lengths
    assert build["best_effort_keys"] == tuple(
        GraphKey(batch_size=1, frames=frames)
        for frames in range(1, 35)
        if frames not in (10, 20, 30)
    )


@pytest.mark.parametrize(
    ("stream_chunk_size", "left_context_size", "initial_frames", "expected_frames"),
    [
        (10, 25, 0, (10, 20, 30, 35)),
        (20, 25, 0, (20, 40, 45)),
        (6, 0, 0, (6,)),
        # a short first window offsets every window until the context saturates
        (10, 25, 2, (2, 12, 22, 32, 35)),
    ],
)
def test_qwen_code2wav_window_graph_keys_are_the_windows_the_scheduler_decodes(
    stream_chunk_size: int,
    left_context_size: int,
    initial_frames: int,
    expected_frames: tuple[int, ...],
) -> None:
    model = FakeCode2WavModel(total_upsample=1)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
        initial_codec_chunk_frames=initial_frames,
        enable_output_overlap=False,
    )
    scheduler.stream_payloads["r"] = make_qwen_payload(request_id="r")
    for index in range(left_context_size + 3 * stream_chunk_size):
        deliver_code2wav_chunk(
            scheduler,
            "r",
            StreamItem(
                index, torch.tensor([1, 2]), "talker", metadata={"stream": True}
            ),
        )
    decoded = tuple(dict.fromkeys(shape[-1] for shape in model.calls))
    assert decoded == expected_frames
    assert code2wav_scheduler.window_graph_keys(
        stream_chunk_size, left_context_size, 2, initial_frames
    ) == tuple(
        GraphKey(batch_size=rows, frames=frames)
        for rows in (1, 2)
        for frames in expected_frames
    )


@pytest.mark.parametrize(
    ("stream_chunk_size", "left_context_size", "initial_chunk_frames"),
    [(10, 25, 0), (20, 25, 0), (10, 25, 4), (6, 0, 0), (10, 0, 4), (10, 1, 0)],
)
def test_qwen_code2wav_factory_captures_the_last_window_of_every_stream_length(
    monkeypatch,
    stream_chunk_size: int,
    left_context_size: int,
    initial_chunk_frames: int,
) -> None:
    """Streams of every length end on a window the factory captures, and its final keys hold no other length."""
    final_lengths: set[int] = set()
    for stream_frames in range(1, left_context_size + 3 * stream_chunk_size + 1):
        emitted = 0
        while True:
            step = (
                initial_chunk_frames
                if emitted == 0 and initial_chunk_frames
                else stream_chunk_size
            )
            if stream_frames - emitted >= step:
                emitted += step
            else:
                break
        if stream_frames > emitted:
            final_lengths.add(min(left_context_size, emitted) + stream_frames - emitted)
        else:
            pass

    pin_cuda_platform(monkeypatch)
    captured: dict[str, tuple[GraphKey, ...]] = {}

    def build(built_model, *, graph_keys, best_effort_keys, **kwargs):
        captured.update(graph_keys=graph_keys, best_effort_keys=best_effort_keys)
        return SimpleNamespace(stats=lambda: {"enabled": True, "disable_reason": None})

    monkeypatch.setattr(
        code2wav_scheduler, "load_code2wav_model", lambda *a, **k: FactoryModel()
    )
    monkeypatch.setattr(
        code2wav_scheduler.Code2WavCudaGraphRunner, "build", staticmethod(build)
    )

    code2wav_scheduler.create_code2wav_scheduler(
        "dummy",
        device="cuda",
        gpu_id=0,
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
        initial_codec_chunk_frames=initial_chunk_frames,
        enable_cuda_graph=True,
        total_gpu_memory_fraction=0.02,
    )

    threshold_frames = {key.frames for key in captured["graph_keys"]}
    final_keys = captured["best_effort_keys"]
    assert all(key.batch_size == 1 for key in final_keys)
    assert len(final_keys) == len(set(final_keys))
    assert {key.frames for key in final_keys} == final_lengths - threshold_frames


def test_qwen_code2wav_enabled_factory_normalizes_device_and_derives_graph_keys(
    monkeypatch,
    caplog,
) -> None:
    model = FactoryModel(num_quantizers=12)
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
    import sglang_omni.platforms as platforms

    monkeypatch.setattr(
        platforms.current_platform, "device_type", "cuda", raising=False
    )
    monkeypatch.setattr(
        platforms.current_platform,
        "get_device",
        lambda index: torch.device("cuda", index),
        raising=False,
    )
    monkeypatch.setattr(
        code2wav_scheduler.torch,
        "get_device_module",
        lambda *args: SimpleNamespace(
            current_device=lambda: 3, Stream=FakeDecodeStream
        ),
    )

    def load(*args, **kwargs):
        load_call.update(args=args, **kwargs)
        return model.eval()

    def build(built_model, **kwargs):
        build_call.update(model=built_model, **kwargs)
        return runner

    monkeypatch.setattr(code2wav_scheduler, "load_code2wav_model", load)
    monkeypatch.setattr(
        code2wav_scheduler.Code2WavCudaGraphRunner,
        "build",
        staticmethod(build),
    )

    with caplog.at_level(logging.INFO, logger=code2wav_scheduler.__name__):
        scheduler = code2wav_scheduler.create_code2wav_scheduler(
            "dummy",
            enable_cuda_graph=True,
            total_gpu_memory_fraction=0.02,
            stream_chunk_size=20,
            left_context_size=25,
            max_replay_rows=1,
            talker_in_process=True,
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
        "best_effort_keys": tuple(
            GraphKey(batch_size=1, frames=frames)
            for frames in range(1, 45)
            if frames not in (20, 40, 45)
        ),
        "model_footprint_bytes": 3 * 4 + 2 * 8,
        "decode_stream": scheduler.decode_stream,
        "incremental_codec": None,
        "codec_pool": None,
    }
    assert scheduler.decode_stream.device == torch.device("cuda:3")
    assert scheduler.decode_stream.priority == -2
    assert scheduler.device == torch.device("cuda:3")
    assert scheduler.stream_chunk_size == 20
    assert scheduler.left_context_size == 25
    assert scheduler.cuda_graph_runner is runner
    stats_record = next(
        record
        for record in caplog.records
        if "device graph startup stats=" in record.message
    )
    assert json.loads(stats_record.message.split("stats=", 1)[1]) == runner.stats()


def test_qwen_code2wav_stream_done_tail_replays_the_graph_of_its_shape(
    monkeypatch,
) -> None:
    model = FakeCode2WavModel(total_upsample=2)
    runner = FakeCudaGraphRunner(model)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=6,
        left_context_size=5,
        enable_cuda_graph=True,
        cuda_graph_runner=runner,
    )
    seed_stream_state(scheduler)
    events = activate_event_capture(monkeypatch)

    for chunk_id in range(11):
        deliver_code2wav_chunk(
            scheduler,
            "req-1",
            StreamItem(
                chunk_id,
                torch.tensor([chunk_id + 1, 10]),
                "talker",
                metadata={"stream": False},
            ),
        )
    scheduler.handle_stream_done("req-1")

    assert runner.calls == [((1, 2, 6), True), ((1, 2, 10), True)]
    decode_ends = [
        event for event in events if event["event_name"] == "code2wav_decode_end"
    ]
    tail_metadata = decode_ends[-1]["metadata"]
    assert tail_metadata["trigger"] == "stream_done"
    assert tail_metadata["window_frames"] == 10
    assert tail_metadata["execution_mode"] == "cuda_graph"
    assert tail_metadata["graph_key"] == {
        "batch_size": 1,
        "frames": 10,
        "decode": "window",
    }
    assert tail_metadata["fallback_reason"] is None


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
    seed_stream_state(scheduler)
    events = activate_event_capture(monkeypatch)

    for chunk_id, codes in enumerate(([1, 10], [2, 20], [3, 30])):
        deliver_code2wav_chunk(
            scheduler,
            "req-1",
            StreamItem(
                chunk_id,
                torch.tensor(codes),
                "talker",
                metadata={"stream": False},
            ),
        )
    scheduler.handle_stream_done("req-1")

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
        "rows": 1,
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
    runner = FakeCudaGraphRunner(model)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=6,
        left_context_size=0,
        enable_cuda_graph=True,
        cuda_graph_runner=runner,
    )
    seed_stream_state(scheduler)
    events = activate_event_capture(monkeypatch)

    for chunk_id in range(6):
        deliver_code2wav_chunk(
            scheduler,
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
        "decode": "window",
    }
    assert decode_end["metadata"]["fallback_reason"] == "key_miss"


def run_code2wav_stream(*, cuda_graph: bool) -> tuple[list[tuple], object]:
    model = FakeCode2WavModel(total_upsample=2)
    runner = FakeCudaGraphRunner(model) if cuda_graph else None
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=1,
        enable_cuda_graph=cuda_graph,
        cuda_graph_runner=runner,
    )
    seed_stream_state(scheduler)
    for chunk_id in range(11):
        deliver_code2wav_chunk(
            scheduler,
            "req-1",
            StreamItem(
                chunk_id,
                torch.tensor([chunk_id + 1, 10]),
                "talker",
                metadata={"stream": True},
            ),
        )
    scheduler.handle_stream_done("req-1")

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
    eager_snapshot, _ = run_code2wav_stream(cuda_graph=False)
    graph_snapshot, runner = run_code2wav_stream(cuda_graph=True)

    assert graph_snapshot == eager_snapshot
    assert [item[0] for item in graph_snapshot] == ["stream", "stream", "result"]
    assert [
        len(np.frombuffer(item[1], dtype=np.float32))
        for item in graph_snapshot
        if item[0] == "stream"
    ] == [20, 2]
    assert runner.calls == [((1, 2, 10), True), ((1, 2, 2), True)]


def test_qwen_code2wav_consumes_borrowed_output_under_state_lock() -> None:
    class BorrowedOutputRunner:
        def __init__(self) -> None:
            self.scheduler = None
            self.static_output = torch.zeros((1, 1, 2), dtype=torch.float32)
            self.lock_was_held: list[bool] = []
            self.replays = 0

        def available_batch_sizes(self, frames: int, decode: str) -> tuple[int, ...]:
            return (1,)

        def run(self, codes: torch.Tensor, *, eligible: bool) -> Code2WavRunResult:
            assert eligible
            self.lock_was_held.append(
                self.scheduler.state_lock._is_owned()
            )  # noqa: leading-underscore  # upstream name
            self.replays += 1
            self.static_output.fill_(float(self.replays))
            return Code2WavRunResult(
                self.static_output,
                "cuda_graph",
                GraphKey(1, int(codes.shape[-1])),
                None,
            )

    model = FakeCode2WavModel(total_upsample=2)
    runner = BorrowedOutputRunner()
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=1,
        left_context_size=0,
        enable_cuda_graph=True,
        cuda_graph_runner=runner,
    )
    runner.scheduler = scheduler
    seed_stream_state(scheduler)

    for chunk_id in range(2):
        deliver_code2wav_chunk(
            scheduler,
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
        chunk.tolist() for chunk in scheduler.stream_states["req-1"].audio_parts
    ] == [
        [1.0, 1.0],
        [2.0, 2.0],
    ]


def test_qwen_code2wav_replay_error_reaches_base_abort_without_eager_retry() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    replay_error = RuntimeError("replay failed")
    runner = FakeCudaGraphRunner(model, replay_error=replay_error)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=1,
        left_context_size=0,
        enable_cuda_graph=True,
        cuda_graph_runner=runner,
    )
    seed_stream_state(scheduler)
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
    assert scheduler.is_aborted("req-1")
    assert "req-1" not in scheduler.stream_states


def make_scheduler(model: FakeCode2WavModel) -> Code2WavScheduler:
    return Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )


def feed(
    scheduler: Code2WavScheduler,
    request_id: str,
    codes: tuple[int, ...],
    *,
    stream: bool,
) -> None:
    meta: dict[str, object] = {"stream": stream}
    for i, code in enumerate(codes):
        deliver_code2wav_chunk(
            scheduler,
            request_id,
            StreamItem(i, torch.tensor([code, code * 10]), "talker", metadata=meta),
        )


def test_qwen_code2wav_streams_incrementally_and_abort_clears_state() -> None:
    """Preserves incremental waveform windows and request-state cleanup on abort."""
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = make_scheduler(model)
    scheduler.stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    feed(scheduler, "req-1", (1, 2, 3), stream=False)
    scheduler.handle_stream_done("req-1")

    message = scheduler.outbox.get_nowait()
    audio = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 2, 2), (1, 2, 2)]
    assert audio.shape == (6,)

    scheduler.stream_payloads["req-2"] = make_qwen_payload(request_id="req-2")
    scheduler.get_or_create_stream_state("req-2")
    scheduler.abort("req-2")
    assert "req-2" not in scheduler.stream_states


def test_streaming_client_gets_stream_chunks_and_metadata_final() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = make_scheduler(model)
    scheduler.stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    feed(scheduler, "req-1", (1, 2, 3), stream=True)
    scheduler.handle_stream_done("req-1")

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
    scheduler = make_scheduler(model)
    scheduler.stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    feed(scheduler, "req-1", (1, 2), stream=False)
    assert model.calls == [(1, 2, 2)]

    deliver_code2wav_chunk(
        scheduler,
        "req-1",
        StreamItem(2, torch.tensor([2150, 0]), "talker", metadata={"stream": False}),
    )
    assert model.calls == [(1, 2, 2)]

    scheduler.handle_stream_done("req-1")
    message = scheduler.outbox.get_nowait()
    audio = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 2, 2)]
    assert audio.shape == (4,)


class RecordingDecodeStream:
    def __init__(self, log: list[object]) -> None:
        self.log = log

    def wait_stream(self, stream: object) -> None:
        self.log.append(("wait_stream", stream))

    def wait_event(self, event: object) -> None:
        self.log.append(("wait_event", event))


class LoggingCode2WavModel(FakeCode2WavModel):
    def __init__(self, log: list[object]) -> None:
        super().__init__(total_upsample=2)
        self.log = log

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        self.log.append("forward")
        return super().__call__(codes)


def test_serving_thread_decodes_after_the_receiving_stream(monkeypatch) -> None:
    log: list[object] = []
    selected: list[object] = []
    receiving_stream = object()
    monkeypatch.setattr(
        code2wav_scheduler.torch,
        "get_device_module",
        lambda device: SimpleNamespace(
            set_stream=selected.append, default_stream=lambda device: receiving_stream
        ),
    )
    decode_stream = RecordingDecodeStream(log)
    scheduler = Code2WavScheduler(
        LoggingCode2WavModel(log),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        decode_stream=decode_stream,
    )

    scheduler.on_serving_start()
    scheduler.stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    feed(scheduler, "req-1", (1, 2), stream=True)

    assert selected == [decode_stream]
    assert log.count("forward") == 1
    assert log[log.index("forward") - 1] == ("wait_stream", receiving_stream)


@pytest.mark.parametrize("enable_output_overlap", [False, True])
def test_decode_stream_waits_on_the_newest_codes_event_before_the_forward(
    enable_output_overlap: bool,
) -> None:
    log: list[object] = []
    scheduler = Code2WavScheduler(
        LoggingCode2WavModel(log),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        enable_output_overlap=enable_output_overlap,
        decode_stream=RecordingDecodeStream(log),
    )
    scheduler.stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    first_event, second_event = object(), object()

    for chunk_id, event in enumerate((first_event, second_event)):
        deliver_code2wav_chunk(
            scheduler,
            "req-1",
            StreamItem(
                chunk_id,
                torch.tensor([chunk_id + 1, 10]),
                "talker",
                metadata={"stream": True, "codes_ready_event": event},
            ),
        )

    assert log.count("forward") == 1
    assert log[log.index("forward") - 1] == ("wait_event", second_event)
    assert all(entry[0] == "wait_event" for entry in log if entry != "forward")


def test_qwen_code2wav_emits_full_chunk_despite_model_output_deficit() -> None:
    model = FakeCode2WavModel(total_upsample=2, output_deficit=1)
    scheduler = make_scheduler(model)
    scheduler.stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    feed(scheduler, "req-1", (1, 2, 3, 4), stream=True)
    scheduler.handle_stream_done("req-1")

    first = scheduler.outbox.get_nowait()
    first_audio = np.frombuffer(first.data["audio_waveform"], dtype=np.float32)
    assert first_audio.shape == (3,)

    second = scheduler.outbox.get_nowait()
    second_audio = np.frombuffer(second.data["audio_waveform"], dtype=np.float32)
    assert second_audio.shape == (4,)

    assert first_audio.shape[0] + second_audio.shape[0] == 4 * 2 - 1


def make_tiny_code2wav(
    device: str, dtype: torch.dtype, seed: int = 0, max_position_embeddings: int = 8000
) -> Qwen3OmniCode2Wav:
    config = Qwen3OmniMoeCode2WavConfig(
        codebook_size=16,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_quantizers=2,
        upsample_rates=[2, 3],
        upsampling_ratios=[2],
        decoder_dim=32,
        sliding_window=4,
        max_position_embeddings=max_position_embeddings,
    )
    torch.manual_seed(seed)
    model = Qwen3OmniCode2Wav(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.05)
        for parameter in model.pre_transformer.parameters():
            if parameter.dim() == 1:
                parameter.uniform_(0.5, 1.5)
            else:
                parameter.normal_(0.0, parameter.shape[1] ** -0.5)
    return model.to(device=device, dtype=dtype)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="channels-last convs run on NVIDIA CUDA only"
)
@pytest.mark.parametrize(
    ("batch_size", "frames"),
    [(1, 7), (3, 10)],
)
def test_channels_last_code2wav_matches_the_hf_forward(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, frames: int
) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    model = make_tiny_code2wav("cuda", torch.float32)
    codes = torch.randint(0, 16, (batch_size, 2, frames), device="cuda")

    with torch.inference_mode():
        expected = model(codes)
        model.use_channels_last()
        actual = model(codes)

    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_channels_last_code2wav_runs_every_snake_on_the_fused_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_tiny_code2wav("cuda", torch.bfloat16)
    model.use_channels_last()
    replaced = snake_beta.fuse_vocoder_decoder(model.decoder)
    assert replaced > 0
    launches: list[tuple[int, ...]] = []
    original_launch = snake_beta.launch

    def counted_launch(
        x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, eps: float
    ) -> torch.Tensor:
        launches.append(x.stride())
        return original_launch(x, alpha, beta, eps)

    monkeypatch.setattr(snake_beta, "launch", counted_launch)
    codes = torch.randint(0, 16, (2, 2, 9), device="cuda")

    with torch.inference_mode():
        model(codes)

    assert len(launches) == replaced
    assert all(stride[1] == 1 for stride in launches)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_cuda(),
    reason="requires NVIDIA CUDA",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("frames", [3, 9])
def test_fused_code2wav_transformer_error_to_fp32_is_below_the_hf_error(
    dtype: torch.dtype, frames: int
) -> None:
    hf_error = 0.0
    fused_error = 0.0
    for seed in range(8):
        model = make_tiny_code2wav("cuda", torch.float32, seed)
        hidden_states = torch.randn(2, frames, 128, device="cuda")
        with torch.inference_mode():
            reference = model.pre_transformer(
                inputs_embeds=hidden_states
            ).last_hidden_state
            model.to(dtype)
            expected = model.pre_transformer(
                inputs_embeds=hidden_states.to(dtype)
            ).last_hidden_state
            model.use_fused_transformer(
                current_platform.get_joint_rope_inplace_kernel(),
                max_positions=model.config.max_position_embeddings,
            )
            actual = model.pre_transformer(
                inputs_embeds=hidden_states.to(dtype)
            ).last_hidden_state
        hf_error += float((expected.float() - reference).norm() / reference.norm())
        fused_error += float((actual.float() - reference).norm() / reference.norm())

    assert fused_error <= 0.95 * hf_error


SLOT_DECODES = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_cuda(),
    reason="slot decodes run on the fused channels-last code2wav, NVIDIA CUDA only",
)


def make_fused_tiny_code2wav(seed: int = 0) -> Qwen3OmniCode2Wav:
    model = make_tiny_code2wav("cuda", torch.bfloat16, seed)
    model.use_channels_last()
    model.use_fused_transformer(
        current_platform.get_joint_rope_inplace_kernel(),
        max_positions=model.config.max_position_embeddings,
    )
    snake_beta.fuse_vocoder_decoder(model.decoder)
    return model


def make_pool(
    incremental_codec: Qwen3OmniIncrementalCodec, slots: int, largest_part: int
) -> CodecStatePool:
    return CodecStatePool(
        incremental_codec.spec,
        num_slots=slots,
        capture_slots=0,
        ring_frames=incremental_codec.spec.retained_context + largest_part,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )


def decode_in_parts(
    incremental_codec: Qwen3OmniIncrementalCodec,
    codes: torch.Tensor,
    parts: tuple[int, ...],
    pool: CodecStatePool | None = None,
    slot_index: torch.Tensor | None = None,
) -> torch.Tensor:
    rows = int(codes.shape[0])
    if pool is None:
        pool = make_pool(incremental_codec, rows, max(parts))
        slot_index = torch.arange(rows, device="cuda")
    else:
        pass
    waveforms = []
    start = 0
    for frames in parts:
        waveforms.append(
            incremental_codec.decode_slots(
                codes[..., start : start + frames],
                pool,
                slot_index,
                first_decode=start == 0,
            )
        )
        start += frames
    return torch.cat(waveforms, dim=-1)


@pytest.mark.accelerator
@SLOT_DECODES
@pytest.mark.parametrize(
    "is_paged_attention",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or not _is_fa3_supported(),
                reason="FlashAttention 3 runs on SM 8.x and 9.x with CUDA 12.3 or newer",
            ),
        ),
        False,
    ],
)
@pytest.mark.parametrize("parts", [(4, 6, 7), (1,) * 17, (10, 7)])
def test_slot_decodes_error_to_fp32_is_the_fused_forward_error(
    monkeypatch: pytest.MonkeyPatch, parts: tuple[int, ...], is_paged_attention: bool
) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    whole_error = 0.0
    incremental_error = 0.0
    for seed in range(4):
        model = make_tiny_code2wav("cuda", torch.float32, seed)
        codes = torch.randint(0, 16, (2, 2, sum(parts)), device="cuda")
        with torch.inference_mode():
            reference = model(codes)
            model.to(torch.bfloat16)
            model.use_channels_last()
            model.use_fused_transformer(
                current_platform.get_joint_rope_inplace_kernel(),
                max_positions=model.config.max_position_embeddings,
            )
            snake_beta.fuse_vocoder_decoder(model.decoder)
            whole = model(codes)
            incremental_codec = Qwen3OmniIncrementalCodec(model)
            incremental_codec.is_paged_attention = is_paged_attention
            incremental = decode_in_parts(incremental_codec, codes, parts)
        whole_error += float((whole.float() - reference).norm() / reference.norm())
        incremental_error += float(
            (incremental.float() - reference).norm() / reference.norm()
        )

    assert incremental_error <= 1.25 * whole_error


@pytest.mark.accelerator
@SLOT_DECODES
def test_slot_decodes_rotate_frames_past_the_code2wav_config_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    hf_error = 0.0
    whole_error = 0.0
    incremental_error = 0.0
    for seed in range(4):
        model = make_tiny_code2wav(
            "cuda", torch.float32, seed, max_position_embeddings=8
        )
        codes = torch.randint(0, 16, (2, 2, 30), device="cuda")
        with torch.inference_mode():
            reference = model(codes)
            model.to(torch.bfloat16)
            expected = model(codes)
            model.use_channels_last()
            model.use_fused_transformer(
                current_platform.get_joint_rope_inplace_kernel(), max_positions=30
            )
            snake_beta.fuse_vocoder_decoder(model.decoder)
            whole = model(codes)
            incremental = decode_in_parts(
                Qwen3OmniIncrementalCodec(model), codes, (10, 10, 10)
            )
        hf_error += float((expected.float() - reference).norm() / reference.norm())
        whole_error += float((whole.float() - reference).norm() / reference.norm())
        incremental_error += float(
            (incremental.float() - reference).norm() / reference.norm()
        )

    assert whole_error <= hf_error
    assert incremental_error <= 1.25 * whole_error


@pytest.mark.accelerator
@SLOT_DECODES
def test_a_reused_slot_decodes_a_new_stream_as_a_fresh_pool() -> None:
    model = make_fused_tiny_code2wav()
    incremental_codec = Qwen3OmniIncrementalCodec(model)
    first_codes = torch.randint(0, 16, (1, 2, 13), device="cuda")
    second_codes = torch.randint(0, 16, (1, 2, 9), device="cuda")
    pool = make_pool(incremental_codec, 1, 4)
    slot_index = torch.zeros(1, dtype=torch.long, device="cuda")

    with torch.inference_mode():
        decode_in_parts(incremental_codec, first_codes, (4, 4, 4, 1), pool, slot_index)
        reused = decode_in_parts(
            incremental_codec, second_codes, (3, 4, 2), pool, slot_index
        )
        fresh = decode_in_parts(incremental_codec, second_codes, (3, 4, 2))

    assert torch.equal(reused, fresh)


@pytest.mark.accelerator
@SLOT_DECODES
def test_slot_decode_rows_at_different_positions_keep_their_own_streams() -> None:
    model = make_fused_tiny_code2wav()
    incremental_codec = Qwen3OmniIncrementalCodec(model)
    first_frames = (5, 9)
    codes = [
        torch.randint(0, 16, (1, 2, frames + 4), device="cuda")
        for frames in first_frames
    ]
    pool = make_pool(incremental_codec, 2, 9)
    slots = [torch.tensor([row], device="cuda") for row in range(2)]

    with torch.inference_mode():
        heads = [
            incremental_codec.decode_slots(
                stream_codes[..., :frames], pool, slot, first_decode=True
            )
            for stream_codes, frames, slot in zip(codes, first_frames, slots)
        ]
        tails = incremental_codec.decode_slots(
            torch.cat([stream_codes[..., -4:] for stream_codes in codes]),
            pool,
            torch.cat(slots),
            first_decode=False,
        )
        alone = [
            decode_in_parts(incremental_codec, stream_codes, (frames, 4))
            for stream_codes, frames in zip(codes, first_frames)
        ]

    for row in range(2):
        torch.testing.assert_close(
            torch.cat((heads[row], tails[row : row + 1]), dim=-1),
            alone[row],
            rtol=2e-2,
            atol=2e-3,
        )


def make_incremental_scheduler(
    model: Qwen3OmniCode2Wav, codec_state_slots: int, *, cuda_graph: bool = False
) -> Code2WavScheduler:
    incremental_codec = Qwen3OmniIncrementalCodec(model)
    device = torch.device("cuda", torch.cuda.current_device())
    pool = CodecStatePool(
        incremental_codec.spec,
        num_slots=codec_state_slots,
        capture_slots=2,
        ring_frames=incremental_codec.spec.retained_context + 3,
        device=device,
        dtype=torch.bfloat16,
    )
    runner = None
    if cuda_graph:
        runner = Code2WavCudaGraphRunner.build(
            model,
            device=device,
            num_quantizers=2,
            total_gpu_memory_fraction=1.0,
            graph_keys=code2wav_scheduler.window_graph_keys(3, 2, 2)
            + code2wav_scheduler.incremental_graph_keys(3, 2),
            best_effort_keys=tuple(
                GraphKey(batch_size=1, frames=frames, decode="next")
                for frames in (1, 2)
            ),
            model_footprint_bytes=0,
            decode_stream=None,
            incremental_codec=incremental_codec,
            codec_pool=pool,
        )
    else:
        pass
    return Code2WavScheduler(
        model,
        device=str(device),
        stream_chunk_size=3,
        left_context_size=2,
        sample_rate=24000,
        enable_cuda_graph=cuda_graph,
        cuda_graph_runner=runner,
        incremental_codec=incremental_codec,
        codec_pool=pool,
    )


def run_streams(
    scheduler: Code2WavScheduler, codes: torch.Tensor, *, first_request: int = 0
) -> list[np.ndarray]:
    request_ids = [
        f"req-{first_request + index}" for index in range(int(codes.shape[0]))
    ]
    for request_id in request_ids:
        scheduler.stream_payloads[request_id] = make_qwen_payload(request_id=request_id)
    for frame in range(int(codes.shape[-1])):
        for request_id, request_codes in zip(request_ids, codes):
            deliver_code2wav_chunk(
                scheduler,
                request_id,
                StreamItem(
                    frame,
                    request_codes[:, frame].clone(),
                    "talker",
                    metadata={"stream": False},
                ),
            )
    waveforms = []
    for request_id in request_ids:
        scheduler.handle_stream_done(request_id)
        result = scheduler.outbox.get_nowait()
        waveforms.append(
            np.frombuffer(result.data.data["audio_waveform"], dtype=np.float32)
        )
    return waveforms


@pytest.mark.accelerator
@SLOT_DECODES
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_scheduler_decodes_streams_from_their_slots(
    cuda_graph: bool, caplog: pytest.LogCaptureFixture
) -> None:
    model = make_fused_tiny_code2wav()
    scheduler = make_incremental_scheduler(
        model, codec_state_slots=2, cuda_graph=cuda_graph
    )
    codes = torch.randint(0, 16, (2, 2, 11), device="cuda")

    with caplog.at_level(logging.WARNING):
        waveforms = run_streams(scheduler, codes)

    with torch.inference_mode():
        expected = decode_in_parts(
            Qwen3OmniIncrementalCodec(model), codes, (3, 3, 3, 2)
        )
    for waveform, wanted in zip(waveforms, expected.reshape(2, -1).float().cpu()):
        np.testing.assert_allclose(waveform, wanted.numpy(), rtol=2e-2, atol=2e-3)
    assert "codec state pool is full" not in caplog.text
    assert scheduler.codec_pool.available_size() == scheduler.codec_pool.num_slots
    if cuda_graph:
        runtime = scheduler.cuda_graph_runner.stats()["runtime"]
        assert runtime["fallback_counts"] == {}
        assert runtime["graph_replays"] > 0
    else:
        pass


@pytest.mark.accelerator
@SLOT_DECODES
def test_a_stream_without_a_free_slot_decodes_with_left_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = make_fused_tiny_code2wav()
    scheduler = make_incremental_scheduler(model, codec_state_slots=1)
    codes = torch.randint(0, 16, (2, 2, 7), device="cuda")

    with caplog.at_level(logging.WARNING):
        waveforms = run_streams(scheduler, codes)

    with torch.inference_mode():
        expected = decode_in_parts(
            Qwen3OmniIncrementalCodec(model), codes[:1], (3, 3, 1)
        )
    wanted = expected.reshape(-1).float().cpu().numpy()
    np.testing.assert_allclose(waveforms[0], wanted, rtol=2e-2, atol=2e-3)
    assert waveforms[1].shape == wanted.shape
    assert caplog.text.count("req-1 decodes with left context") == 1
    assert scheduler.codec_pool.available_size() == scheduler.codec_pool.num_slots


@pytest.mark.accelerator
@SLOT_DECODES
def test_a_stream_takes_its_slot_with_its_first_codes_and_abort_returns_it() -> None:
    scheduler = make_incremental_scheduler(
        make_fused_tiny_code2wav(), codec_state_slots=1
    )
    scheduler.stream_payloads["req-0"] = make_qwen_payload(request_id="req-0")
    deliver_code2wav_chunk(
        scheduler,
        "req-0",
        StreamItem(0, torch.randint(0, 16, (2,)), "talker", metadata={"stream": False}),
    )
    assert scheduler.codec_pool.available_size() == 0

    scheduler.abort("req-0")

    assert scheduler.codec_pool.available_size() == scheduler.codec_pool.num_slots


@pytest.mark.accelerator
@SLOT_DECODES
def test_a_stream_ending_before_its_first_window_decodes_without_slot_state() -> None:
    model = make_fused_tiny_code2wav()
    scheduler = make_incremental_scheduler(model, codec_state_slots=1)
    run_streams(scheduler, torch.randint(0, 16, (1, 2, 7), device="cuda"))
    short_codes = torch.randint(0, 16, (1, 2, 2), device="cuda")

    waveforms = run_streams(scheduler, short_codes, first_request=1)

    with torch.inference_mode():
        expected = model(short_codes).reshape(-1).float().cpu().numpy()
    np.testing.assert_allclose(waveforms[0], expected, rtol=2e-2, atol=2e-3)
    assert scheduler.codec_pool.available_size() == scheduler.codec_pool.num_slots


@pytest.mark.accelerator
@SLOT_DECODES
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_streams_decoded_in_one_replay_keep_their_own_slots(cuda_graph: bool) -> None:
    model = make_fused_tiny_code2wav()
    scheduler = make_incremental_scheduler(
        model, codec_state_slots=2, cuda_graph=cuda_graph
    )
    codes = torch.randint(0, 16, (2, 2, 11), device="cuda")
    request_ids = ["req-0", "req-1"]
    for request_id in request_ids:
        scheduler.stream_payloads[request_id] = make_qwen_payload(request_id=request_id)

    for start in (0, 3, 6):
        for request_id, request_codes in zip(request_ids, codes):
            scheduler.handle_stream_chunk(
                request_id,
                StreamItem(
                    start,
                    request_codes[:, start : start + 3].T,
                    "talker",
                    metadata={"stream": False},
                ),
            )
        assert len(scheduler.ready_window_group()) == 2
        scheduler.run_ready_step()
    waveforms = []
    for request_id, request_codes in zip(request_ids, codes):
        scheduler.handle_stream_chunk(
            request_id,
            StreamItem(9, request_codes[:, 9:].T, "talker", metadata={"stream": False}),
        )
        scheduler.handle_stream_done(request_id)
        result = scheduler.outbox.get_nowait()
        waveforms.append(
            np.frombuffer(result.data.data["audio_waveform"], dtype=np.float32)
        )

    with torch.inference_mode():
        expected = decode_in_parts(
            Qwen3OmniIncrementalCodec(model), codes, (3, 3, 3, 2)
        )
    for waveform, wanted in zip(waveforms, expected.reshape(2, -1).float().cpu()):
        np.testing.assert_allclose(waveform, wanted.numpy(), rtol=2e-2, atol=2e-3)
    assert scheduler.codec_pool.available_size() == scheduler.codec_pool.num_slots


@pytest.mark.accelerator
@SLOT_DECODES
@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_slot_graphs_and_capture_slots_stop_at_the_codec_slot_count(
    monkeypatch: pytest.MonkeyPatch, enable_cuda_graph: bool
) -> None:
    build_call: dict[str, object] = {}

    def build(model: Qwen3OmniCode2Wav, **kwargs: object) -> SimpleNamespace:
        build_call.update(kwargs)
        return SimpleNamespace(stats=lambda: {})

    monkeypatch.setattr(
        code2wav_scheduler,
        "load_code2wav_model",
        lambda *args, **kwargs: make_fused_tiny_code2wav(),
    )
    monkeypatch.setattr(
        code2wav_scheduler.Code2WavCudaGraphRunner, "build", staticmethod(build)
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler(
        "dummy",
        device="cuda",
        enable_cuda_graph=enable_cuda_graph,
        total_gpu_memory_fraction=0.5,
        stream_chunk_size=3,
        left_context_size=2,
        max_replay_rows=8,
        fused_snake_activation=False,
        codec_state_slots=2,
    )

    slot_graph_rows = {
        key.batch_size
        for key in build_call.get("graph_keys", ())
        if key.decode != "window"
    }
    assert (len(scheduler.codec_pool.scratch_slots), slot_graph_rows) == (
        (2, {1, 2}) if enable_cuda_graph else (0, set())
    )
