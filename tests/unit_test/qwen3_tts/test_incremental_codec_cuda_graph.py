# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalCodecState
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    IncrementalCodecGraphKey,
    IncrementalCodecGraphResult,
    Qwen3TTSIncrementalCodecCudaGraphRunner,
)
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    Qwen3TTSStreamingVocoderScheduler,
    _IncrementalDecodeBatch,
    _IncrementalDecodePlan,
)


class _FakeGraph:
    def __init__(self) -> None:
        self.replays = 0
        self.resets = 0

    def replay(self) -> None:
        self.replays += 1

    def reset(self) -> None:
        self.resets += 1


class _FailingGraph:
    def __init__(self) -> None:
        self.resets = 0

    def replay(self) -> None:
        raise RuntimeError("injected replay failure")

    def reset(self) -> None:
        self.resets += 1


class _DeviceContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


def _state(
    rows: int,
    *,
    offset: int = 0,
    device: torch.device | str = "cpu",
) -> Qwen3TTSIncrementalCodecState:
    positions = torch.arange(offset, offset + rows, dtype=torch.long, device=device)
    return Qwen3TTSIncrementalCodecState(
        transformer_context_length=2,
        frame_positions=positions,
        transformer_keys={
            0: torch.arange(rows * 2, dtype=torch.float32, device=device).view(
                rows, 1, 2, 1
            )
            + offset
        },
        transformer_values={
            0: torch.arange(rows * 2, dtype=torch.float32, device=device).view(
                rows, 1, 2, 1
            )
            + offset
            + 10
        },
        conv_histories={
            "conv": torch.arange(rows * 2, dtype=torch.float32, device=device).view(
                rows, 1, 2
            )
            + offset
            + 20
        },
        transconv_overlaps={
            "up": torch.arange(rows, dtype=torch.float32, device=device).view(
                rows, 1, 1
            )
            + offset
            + 30
        },
    )


def _async_incremental_scheduler(
    device: torch.device,
) -> Qwen3TTSStreamingVocoderScheduler:
    scheduler = Qwen3TTSStreamingVocoderScheduler.__new__(
        Qwen3TTSStreamingVocoderScheduler
    )
    scheduler._device = device
    scheduler._cuda_decode_failed = False
    scheduler._deterministic_inference = False
    scheduler._samples_per_frame = 1
    scheduler._pinned_staging_disabled = True
    scheduler._decode_staging = threading.local()
    scheduler._decode_stream = torch.cuda.Stream(device=device)
    scheduler._followup_decode_stream = torch.cuda.Stream(device=device)
    return scheduler


def test_incremental_codec_graph_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be 'cold' or 'warm'"):
        Qwen3TTSIncrementalCodecCudaGraphRunner(
            SimpleNamespace(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            num_quantizers=2,
            mode="unknown",
            fresh_frames=(8,),
            enabled=False,
        )


def test_incremental_codec_graph_stages_padding_and_returns_borrowed_views() -> None:
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_quantizers=2,
        mode="warm",
        fresh_frames=(8,),
        batch_sizes=(1, 4),
        enabled=False,
    )
    runner._enabled = True
    key = IncrementalCodecGraphKey(fresh_frames=8, batch_bucket=4)
    graph = _FakeGraph()
    static_codes = torch.full((4, 2, 8), -1, dtype=torch.long)
    input_state = _state(4, offset=100)
    output_state = _state(4, offset=200)
    waveform = torch.arange(4 * 8 * 4, dtype=torch.float32).view(4, 1, 32)
    runner._graphs[key] = SimpleNamespace(
        graph=graph,
        static_codes=static_codes,
        input_state=input_state,
        output_state=output_state,
        waveform=waveform,
    )

    codes = torch.arange(3 * 2 * 8, dtype=torch.long).view(3, 2, 8)
    source_state = _state(3, offset=7)
    result = runner.decode(codes, source_state)

    assert result is not None
    assert graph.replays == 1
    assert torch.equal(static_codes[:3], codes)
    assert torch.equal(static_codes[3], torch.zeros((2, 8), dtype=torch.long))
    assert torch.equal(input_state.frame_positions[:3], source_state.frame_positions)
    assert input_state.frame_positions[3].item() == 0
    assert torch.equal(
        input_state.transformer_keys[0][:3], source_state.transformer_keys[0]
    )
    assert torch.count_nonzero(input_state.transformer_keys[0][3]).item() == 0
    assert result.waveform.shape == (3, 1, 32)
    assert result.state.frame_positions is not None
    assert result.state.frame_positions.tolist() == [200, 201, 202]
    assert result.state.transformer_keys[0].shape == (3, 1, 2, 1)
    assert (
        result.waveform.untyped_storage().data_ptr()
        == waveform.untyped_storage().data_ptr()
    )


def test_incremental_codec_graph_uses_smallest_available_bucket() -> None:
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_quantizers=2,
        mode="warm",
        fresh_frames=(8,),
        batch_sizes=(1, 2, 4, 8),
        enabled=False,
    )
    runner._enabled = True
    graphs = {}
    for bucket in (2, 4, 8):
        key = IncrementalCodecGraphKey(8, bucket)
        graphs[key] = SimpleNamespace(
            graph=_FakeGraph(),
            static_codes=torch.zeros(bucket, 2, 8, dtype=torch.long),
            input_state=_state(bucket),
            output_state=_state(bucket, offset=10),
            waveform=torch.zeros(bucket, 1, 32),
        )
    runner._graphs = graphs

    result = runner.decode(torch.zeros(3, 2, 8, dtype=torch.long), _state(3))

    assert result is not None
    assert graphs[IncrementalCodecGraphKey(8, 2)].graph.replays == 0
    assert graphs[IncrementalCodecGraphKey(8, 4)].graph.replays == 1
    assert graphs[IncrementalCodecGraphKey(8, 8)].graph.replays == 0


def test_incremental_codec_graph_misses_uncaptured_frame_count() -> None:
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_quantizers=2,
        mode="warm",
        fresh_frames=(8,),
        batch_sizes=(1, 2, 4),
        enabled=False,
    )
    runner._enabled = True
    key = IncrementalCodecGraphKey(8, 1)
    runner._graphs[key] = SimpleNamespace(
        graph=_FakeGraph(),
        static_codes=torch.zeros(1, 2, 8, dtype=torch.long),
        input_state=_state(1),
        output_state=_state(1, offset=10),
        waveform=torch.zeros(1, 1, 32),
    )

    assert runner.decode(torch.zeros(1, 2, 3, dtype=torch.long), _state(1)) is None
    assert runner.stats()["runtime"]["fallback_counts"] == {
        "uncaptured_fresh_frames": 1
    }


def test_incremental_codec_graph_replay_failure_disables_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_quantizers=2,
        mode="warm",
        fresh_frames=(8,),
        batch_sizes=(1,),
        enabled=False,
    )
    runner._enabled = True
    key = IncrementalCodecGraphKey(8, 1)
    graph = _FailingGraph()
    runner._graphs[key] = SimpleNamespace(
        graph=graph,
        static_codes=torch.zeros(1, 2, 8, dtype=torch.long),
        input_state=_state(1),
        output_state=_state(1, offset=10),
        waveform=torch.zeros(1, 1, 32),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _device: _DeviceContext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="injected replay failure"):
        runner.decode(torch.zeros(1, 2, 8, dtype=torch.long), _state(1))

    stats = runner.stats()
    assert stats["enabled"] is False
    assert stats["runtime"]["replay_failures"] == 1
    assert stats["disable_reason"].startswith("runtime_replay_failed")
    assert runner.available_batch_sizes(8) == ()
    assert graph.resets == 1
    assert key not in runner._graphs


def test_incremental_codec_capture_rollback_retains_unsynchronized_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_quantizers=2,
        mode="warm",
        fresh_frames=(8,),
        enabled=False,
    )
    key = IncrementalCodecGraphKey(8, 1)
    temporary = {key: SimpleNamespace()}
    pool = object()
    capture_stream = object()
    monkeypatch.setattr(torch.cuda, "device", lambda _device: _DeviceContext())

    def fail_synchronize(_device) -> None:
        raise RuntimeError("injected synchronize failure")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_synchronize)

    runner._rollback_capture(
        temporary,
        pool=pool,
        capture_stream=capture_stream,
        reason="injected failure",
    )

    assert temporary
    assert runner.stats()["retained_capture_resource_sets"] == 1
    retained = runner._retained_capture_resources[0]
    assert retained.keepalives == [temporary]
    assert retained.pool is pool
    assert retained.stream is capture_stream


def test_incremental_codec_capture_rollback_resets_temporary_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_quantizers=2,
        mode="warm",
        fresh_frames=(8,),
        enabled=False,
    )
    graph = _FakeGraph()
    key = IncrementalCodecGraphKey(8, 1)
    temporary = {key: SimpleNamespace(graph=graph)}

    monkeypatch.setattr(torch.cuda, "device", lambda _device: _DeviceContext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    runner._rollback_capture(
        temporary,
        pool=object(),
        capture_stream=object(),
        reason="injected failure",
    )

    assert graph.resets == 1
    assert temporary == {}


def test_incremental_codec_graphs_capture_during_vocoder_warmup() -> None:
    captures: list[str] = []

    def graph_holder(name: str):
        return SimpleNamespace(capture=lambda: captures.append(name))

    scheduler = Qwen3TTSStreamingVocoderScheduler.__new__(
        Qwen3TTSStreamingVocoderScheduler
    )
    scheduler._async_decode = True
    scheduler._initial_decode_graphs = graph_holder("whole-sequence-initial")
    scheduler._followup_decode_graphs = graph_holder("whole-sequence-followup")
    scheduler._initial_incremental_decode_graphs = graph_holder("cold")
    scheduler._followup_incremental_decode_graphs = graph_holder("warm")

    scheduler.warmup_now()

    assert captures == [
        "whole-sequence-initial",
        "whole-sequence-followup",
        "warm",
        "cold",
    ]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_incremental_codec_launch_uses_graph_state_and_waveform() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    scheduler = _async_incremental_scheduler(device)

    graph_state = _state(1, offset=20, device=device)
    graph_waveform = torch.tensor([[[10.0, 11.0]]], device=device)

    class GraphRunner:
        calls = 0

        def decode(self, codes, state):
            self.calls += 1
            assert codes.shape == (1, 2, 2)
            assert state.frame_positions is not None
            return IncrementalCodecGraphResult(
                waveform=graph_waveform,
                state=graph_state,
            )

    class EagerDecoder:
        calls = 0

        def decode(self, codes, state):
            self.calls += 1
            raise AssertionError("eager decoder must not run on a graph hit")

    graph_runner = GraphRunner()
    eager_decoder = EagerDecoder()
    scatters = []
    arena = SimpleNamespace(
        scatter=lambda slots, state: scatters.append((slots, state))
    )
    scheduler._initial_incremental_decode_graphs = None
    scheduler._followup_incremental_decode_graphs = graph_runner
    plan = _IncrementalDecodePlan(
        decoder_input=torch.tensor([[[1, 2], [3, 4]]], device=device),
        slot=0,
        fresh_frames=2,
        reference_trim_frames=0,
        generated_frames=2,
        emitted_generated_frames=0,
    )
    batch = _IncrementalDecodeBatch(
        decoder=eager_decoder,
        arena=arena,
        slots=[0],
        cohort_state=_state(1, device=device),
    )

    handle = scheduler._launch_decode_plans(
        [plan],
        stream=scheduler._followup_decode_stream,
        incremental=batch,
    )
    deltas = handle.resolve()

    assert graph_runner.calls == 1
    assert eager_decoder.calls == 0
    assert batch.cohort_state is graph_state
    assert scatters == [([0], graph_state)]
    assert len(deltas) == 1
    assert deltas[0].tolist() == [10.0, 11.0]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_incremental_codec_launch_falls_back_to_eager_on_graph_miss() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    scheduler = _async_incremental_scheduler(device)

    cohort_state = _state(1, offset=5, device=device)
    eager_waveform = torch.tensor([[[20.0, 21.0]]], device=device)

    class GraphRunner:
        calls = 0

        def decode(self, codes, state):
            self.calls += 1
            assert codes.shape == (1, 2, 2)
            assert state is cohort_state
            return None

    class EagerDecoder:
        calls = 0

        def decode(self, codes, state):
            self.calls += 1
            assert codes.shape == (1, 2, 2)
            assert state is cohort_state
            state.frame_positions = state.frame_positions + 2
            return eager_waveform

    graph_runner = GraphRunner()
    eager_decoder = EagerDecoder()
    scatters = []
    arena = SimpleNamespace(
        scatter=lambda slots, state: scatters.append((slots, state))
    )
    scheduler._initial_incremental_decode_graphs = None
    scheduler._followup_incremental_decode_graphs = graph_runner
    plan = _IncrementalDecodePlan(
        decoder_input=torch.tensor([[[1, 2], [3, 4]]], device=device),
        slot=0,
        fresh_frames=2,
        reference_trim_frames=0,
        generated_frames=2,
        emitted_generated_frames=0,
    )
    batch = _IncrementalDecodeBatch(
        decoder=eager_decoder,
        arena=arena,
        slots=[0],
        cohort_state=cohort_state,
    )

    handle = scheduler._launch_decode_plans(
        [plan],
        stream=scheduler._followup_decode_stream,
        incremental=batch,
    )
    deltas = handle.resolve()

    assert graph_runner.calls == 1
    assert eager_decoder.calls == 1
    assert batch.cohort_state is cohort_state
    assert cohort_state.frame_positions.tolist() == [7]
    assert scatters == [([0], cohort_state)]
    assert len(deltas) == 1
    assert deltas[0].tolist() == [20.0, 21.0]


def test_incremental_codec_graph_cohort_splits_at_largest_bucket() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler.__new__(
        Qwen3TTSStreamingVocoderScheduler
    )
    scheduler._followup_incremental_decode_graphs = SimpleNamespace(
        available_batch_sizes=lambda fresh_frames: (
            (4, 2, 1) if fresh_frames == 8 else ()
        )
    )

    def plan(slot: int, fresh_frames: int = 8) -> _IncrementalDecodePlan:
        return _IncrementalDecodePlan(
            decoder_input=torch.zeros(1, 2, fresh_frames, dtype=torch.long),
            slot=slot,
            fresh_frames=fresh_frames,
            reference_trim_frames=0,
            generated_frames=fresh_frames,
            emitted_generated_frames=0,
        )

    group = [(str(index), None, plan(index)) for index in range(10)]
    split = scheduler._split_incremental_group_for_graph(group)

    assert [len(item) for item in split] == [4, 4, 2]
    assert [entry[0] for subgroup in split for entry in subgroup] == [
        str(index) for index in range(10)
    ]

    terminal = [("terminal", None, plan(11, fresh_frames=3))]
    assert scheduler._split_incremental_group_for_graph(terminal) == [terminal]


def test_incremental_codec_warm_graph_uses_standard_batch_bucket_prefix() -> None:
    select = (
        Qwen3TTSStreamingVocoderScheduler._resolve_incremental_warm_graph_batch_sizes
    )

    assert select(max_batch_size=1) == (1,)
    assert select(max_batch_size=3) == (1, 2, 4)
    assert select(max_batch_size=5) == (1, 2, 4, 8)
    assert select(max_batch_size=8) == (1, 2, 4, 8)
    assert select(max_batch_size=16) == (1, 2, 4, 8)
