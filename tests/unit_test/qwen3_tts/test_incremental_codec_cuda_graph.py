# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalCodecState
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    CaptureResourceSet,
    IncrementalCodecGraphKey,
    Qwen3TTSIncrementalCodecCudaGraphRunner,
    split_frames_by_width,
)
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    IncrementalDecodeBatch,
    IncrementalDecodePlan,
    Qwen3TTSStreamingVocoderScheduler,
)


class FakeGraph:
    def __init__(self) -> None:
        self.replays = 0
        self.resets = 0

    def replay(self) -> None:
        self.replays += 1

    def reset(self) -> None:
        self.resets += 1


class FailingGraph:
    def __init__(self) -> None:
        self.resets = 0

    def replay(self) -> None:
        raise RuntimeError("injected replay failure")

    def reset(self) -> None:
        self.resets += 1


class DeviceContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


def make_state(
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


def async_incremental_scheduler(
    device: torch.device,
) -> Qwen3TTSStreamingVocoderScheduler:
    scheduler = Qwen3TTSStreamingVocoderScheduler.__new__(
        Qwen3TTSStreamingVocoderScheduler
    )
    scheduler.device = device
    scheduler.cuda_decode_failed = False
    scheduler.deterministic_inference = False
    scheduler.samples_per_frame = 1
    scheduler.pinned_staging_disabled = True
    scheduler.decode_staging = threading.local()
    scheduler.decode_stream = torch.cuda.Stream(device=device)
    scheduler.followup_decode_stream = torch.cuda.Stream(device=device)
    scheduler.followup_decode_streams = (scheduler.followup_decode_stream,)
    scheduler.initial_window_decode_graphs = None
    scheduler.worker_ctx = SimpleNamespace(graphs=None)
    return scheduler


def test_incremental_codec_graph_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be 'cold', 'warm' or 'window'"):
        Qwen3TTSIncrementalCodecCudaGraphRunner(
            SimpleNamespace(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            num_quantizers=2,
            mode="unknown",
            fresh_frames=(8,),
            enabled=False,
            arena=SimpleNamespace(scratch_slot=0),
        )


class FakeArena:
    scratch_slot = 64

    def stage_index(self, slots):
        return torch.tensor(list(slots), dtype=torch.long)


def make_runner(**kwargs) -> Qwen3TTSIncrementalCodecCudaGraphRunner:
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_quantizers=2,
        mode="warm",
        fresh_frames=(8,),
        enabled=False,
        arena=FakeArena(),
        **kwargs,
    )
    runner.enabled = True
    return runner


def make_entry(bucket: int, graph=None) -> SimpleNamespace:
    return SimpleNamespace(
        graph=graph or FakeGraph(),
        static_codes=torch.full((bucket, 2, 8), -1, dtype=torch.long),
        static_index=torch.full((bucket,), -1, dtype=torch.long),
        waveform=torch.arange(bucket * 32, dtype=torch.float32).view(bucket, 1, 32),
    )


def test_incremental_codec_graph_stages_padding_and_returns_borrowed_views() -> None:
    runner = make_runner(batch_sizes=(1, 4))
    entry = make_entry(4)
    runner.graphs[IncrementalCodecGraphKey(fresh_frames=8, batch_bucket=4)] = entry
    codes = torch.arange(3 * 2 * 8, dtype=torch.long).view(3, 2, 8)

    waveform = runner.decode_slots(codes, [5, 6, 7])

    assert waveform is not None
    assert entry.graph.replays == 1
    assert torch.equal(entry.static_codes[:3], codes)
    assert torch.equal(entry.static_codes[3], torch.zeros((2, 8), dtype=torch.long))
    # note (luojiaxuan): padded rows read and write the arena's scratch row.
    assert entry.static_index.tolist() == [5, 6, 7, FakeArena.scratch_slot]
    assert waveform.shape == (3, 1, 32)
    assert (
        waveform.untyped_storage().data_ptr()
        == entry.waveform.untyped_storage().data_ptr()
    )


def test_incremental_codec_graph_uses_smallest_available_bucket() -> None:
    runner = make_runner(batch_sizes=(1, 2, 4, 8))
    graphs = {
        IncrementalCodecGraphKey(8, bucket): make_entry(bucket) for bucket in (2, 4, 8)
    }
    runner.graphs = graphs

    assert (
        runner.decode_slots(torch.zeros(3, 2, 8, dtype=torch.long), [0, 1, 2])
        is not None
    )
    assert graphs[IncrementalCodecGraphKey(8, 2)].graph.replays == 0
    assert graphs[IncrementalCodecGraphKey(8, 4)].graph.replays == 1
    assert graphs[IncrementalCodecGraphKey(8, 8)].graph.replays == 0


def test_incremental_codec_graph_misses_uncaptured_frame_count() -> None:
    runner = make_runner(batch_sizes=(1, 2, 4))
    runner.graphs[IncrementalCodecGraphKey(8, 1)] = make_entry(1)

    assert runner.decode_slots(torch.zeros(1, 2, 3, dtype=torch.long), [0]) is None
    assert runner.stats()["runtime"]["fallback_counts"] == {
        "uncaptured_fresh_frames": 1
    }


@pytest.mark.parametrize(
    ("total", "widths", "expected"),
    [
        (7, (1, 2, 4), (4, 2, 1)),
        (8, (1, 2, 4), (4, 4)),
        (4, (1, 2, 4), (4,)),
        (37, (1, 2, 4, 8, 16, 32), (32, 4, 1)),
        (5, (2, 4), None),
        (3, (4,), None),
        (0, (1, 2), ()),
    ],
)
def test_split_frames_by_width_takes_the_largest_width_first(
    total: int, widths: tuple[int, ...], expected: tuple[int, ...] | None
) -> None:
    assert split_frames_by_width(total, widths) == expected


def test_incremental_codec_graph_accepts_the_window_mode() -> None:
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        SimpleNamespace(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_quantizers=2,
        mode="window",
        fresh_frames=(8, 2, 4),
        enabled=False,
        arena=FakeArena(),
    )

    assert runner.stats()["binding"]["mode"] == "window"
    assert runner.stats()["graph_contract"]["fresh_frames"] == [2, 4, 8]


def test_incremental_codec_graph_splits_frames_by_captured_widths_only() -> None:
    runner = make_runner(batch_sizes=(1, 4))
    runner.graphs = {
        IncrementalCodecGraphKey(8, 1): make_entry(1),
        IncrementalCodecGraphKey(8, 4): make_entry(4),
        IncrementalCodecGraphKey(4, 1): make_entry(1),
        IncrementalCodecGraphKey(4, 4): make_entry(4),
    }

    assert runner.split_frames(20) == (8, 8, 4)
    assert runner.split_frames(8) == (8,)
    assert runner.split_frames(6) is None
    assert runner.largest_batch_bucket() == 4

    runner.enabled = False
    assert runner.split_frames(8) is None
    assert runner.largest_batch_bucket() == 0


def test_incremental_codec_graph_replay_failure_disables_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_runner(batch_sizes=(1,))
    key = IncrementalCodecGraphKey(8, 1)
    graph = FailingGraph()
    runner.graphs[key] = make_entry(1, graph=graph)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: DeviceContext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="injected replay failure"):
        runner.decode_slots(torch.zeros(1, 2, 8, dtype=torch.long), [0])

    stats = runner.stats()
    assert stats["enabled"] is False
    assert stats["runtime"]["replay_failures"] == 1
    assert stats["disable_reason"].startswith("runtime_replay_failed")
    assert runner.available_batch_sizes(8) == ()
    assert graph.resets == 1
    assert key not in runner.graphs


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
        arena=FakeArena(),
    )
    key = IncrementalCodecGraphKey(8, 1)
    temporary = {key: SimpleNamespace()}
    pool = object()
    capture_stream = object()
    monkeypatch.setattr(torch.cuda, "device", lambda _device: DeviceContext())

    def fail_synchronize(_device) -> None:
        raise RuntimeError("injected synchronize failure")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_synchronize)

    runner.rollback_capture(
        temporary,
        pool=pool,
        capture_stream=capture_stream,
        reason="injected failure",
    )

    assert temporary
    assert runner.stats()["retained_capture_resource_sets"] == 1
    retained = runner.retained_capture_resources[0]
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
        arena=FakeArena(),
    )
    graph = FakeGraph()
    key = IncrementalCodecGraphKey(8, 1)
    temporary = {key: SimpleNamespace(graph=graph)}

    monkeypatch.setattr(torch.cuda, "device", lambda _device: DeviceContext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    runner.rollback_capture(
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
    scheduler.async_decode = True
    scheduler.initial_decode_graphs = graph_holder("whole-sequence-initial")
    scheduler.followup_graph_holders = (graph_holder("whole-sequence-followup"),)
    scheduler.initial_incremental_decode_graphs = graph_holder("cold")
    scheduler.initial_window_decode_graphs = graph_holder("window")
    scheduler.followup_incremental_graph_holders = (graph_holder("warm"),)

    scheduler.warmup_now()

    assert captures == [
        "whole-sequence-initial",
        "whole-sequence-followup",
        "warm",
        "cold",
        "window",
    ]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_incremental_codec_warmup_traces_a_compiled_shape_on_its_own_tensors() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    traces: list[tuple] = []
    decodes: list[tuple] = []

    class Decoder:
        def precompile(self, codes, state):
            traces.append(
                (
                    codes,
                    state,
                    torch.is_inference_mode_enabled(),
                    torch.is_grad_enabled(),
                )
            )

        def decode(self, codes, state, *, compiled=False):
            decodes.append((codes, state, compiled))
            return torch.zeros(1, 1, 4, device=device)

    class Arena:
        scratch_slot = 3

        def gather_by_index(self, index):
            return SimpleNamespace(
                index=index,
                frame_positions=torch.zeros(int(index.shape[0]), device=device),
            )

    def runner(compile_fresh_frames):
        return Qwen3TTSIncrementalCodecCudaGraphRunner(
            Decoder(),
            device=device,
            dtype=torch.float32,
            num_quantizers=2,
            mode="cold",
            fresh_frames=(4,),
            batch_sizes=(1,),
            compile_fresh_frames=compile_fresh_frames,
            arena=Arena(),
            enabled=False,
        )

    static_codes = torch.zeros(1, 2, 4, dtype=torch.long, device=device)
    key = IncrementalCodecGraphKey(fresh_frames=4, batch_bucket=1)
    resources = CaptureResourceSet(
        pool=None, stream=torch.cuda.Stream(device=device), keepalives=[static_codes]
    )

    runner((4,)).warmup_capture_shape(key, static_codes, resources)

    assert len(traces) == 1
    codes, state, inference, grad = traces[0]
    assert codes is static_codes
    assert state.frame_positions.is_inference()
    assert state.index.tolist() == [Arena.scratch_slot]
    assert inference is True and grad is False
    assert [entry[2] for entry in decodes] == [True, True, True]
    assert all(entry[0] is static_codes for entry in decodes)
    assert resources.keepalives == [static_codes]

    traces.clear()
    decodes.clear()
    runner(()).warmup_capture_shape(key, static_codes, resources)

    assert traces == []
    assert [entry[2] for entry in decodes] == [False, False, False]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_incremental_codec_launch_uses_graph_state_and_waveform() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    scheduler = async_incremental_scheduler(device)

    graph_waveform = torch.tensor([[[10.0, 11.0]]], device=device)

    class GraphRunner:
        calls = 0

        def available_batch_sizes(self, fresh_frames):
            return (1,) if fresh_frames == 2 else ()

        def decode_slots(self, codes, slots):
            self.calls += 1
            assert codes.shape == (1, 2, 2)
            assert list(slots) == [0]
            return graph_waveform

    class EagerDecoder:
        calls = 0

        def decode(self, codes, state):
            self.calls += 1
            raise AssertionError("eager decoder must not run on a graph hit")

    graph_runner = GraphRunner()
    eager_decoder = EagerDecoder()
    scatters = []
    arena = SimpleNamespace(
        scatter=lambda slots, state: scatters.append((slots, state)),
        gather=lambda slots: (_ for _ in ()).throw(
            AssertionError("no gather on a hit")
        ),
    )
    scheduler.initial_incremental_decode_graphs = None
    scheduler.followup_incremental_graph_holders = (graph_runner,)
    scheduler.worker_ctx.incremental_graphs = graph_runner
    plan = IncrementalDecodePlan(
        decoder_input=torch.tensor([[[1, 2], [3, 4]]], device=device),
        slot=0,
        fresh_frames=2,
        reference_trim_frames=0,
        generated_frames=2,
        emitted_generated_frames=0,
    )
    batch = IncrementalDecodeBatch(decoder=eager_decoder, arena=arena, slots=[0])

    handle = scheduler.launch_decode_plans(
        [plan],
        stream=scheduler.followup_decode_stream,
        incremental=batch,
    )
    deltas = handle.resolve()

    assert graph_runner.calls == 1
    assert eager_decoder.calls == 0
    assert batch.cohort_state is None, "the graph reads and writes the arena itself"
    assert scatters == []
    assert len(deltas) == 1
    assert deltas[0].tolist() == [10.0, 11.0]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_incremental_codec_launch_falls_back_to_eager_on_graph_miss() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    scheduler = async_incremental_scheduler(device)

    cohort_state = make_state(1, offset=5, device=device)
    eager_waveform = torch.tensor([[[20.0, 21.0]]], device=device)

    class GraphRunner:
        calls = 0

        def available_batch_sizes(self, fresh_frames):
            return ()

        def decode_slots(self, codes, slots):
            self.calls += 1
            assert codes.shape == (1, 2, 2)
            assert list(slots) == [0]
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
        scatter=lambda slots, state: scatters.append((slots, state)),
        gather=lambda slots: cohort_state,
    )
    scheduler.initial_incremental_decode_graphs = None
    scheduler.followup_incremental_graph_holders = (graph_runner,)
    scheduler.worker_ctx.incremental_graphs = graph_runner
    plan = IncrementalDecodePlan(
        decoder_input=torch.tensor([[[1, 2], [3, 4]]], device=device),
        slot=0,
        fresh_frames=2,
        reference_trim_frames=0,
        generated_frames=2,
        emitted_generated_frames=0,
    )
    batch = IncrementalDecodeBatch(decoder=eager_decoder, arena=arena, slots=[0])

    handle = scheduler.launch_decode_plans(
        [plan],
        stream=scheduler.followup_decode_stream,
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
    scheduler.followup_incremental_graph_holders = (
        SimpleNamespace(
            available_batch_sizes=lambda fresh_frames: {8: (4, 2, 1)}.get(
                fresh_frames, ()
            )
        ),
    )

    def plan(slot: int, fresh_frames: int = 8) -> IncrementalDecodePlan:
        return IncrementalDecodePlan(
            decoder_input=torch.zeros(1, 2, fresh_frames, dtype=torch.long),
            slot=slot,
            fresh_frames=fresh_frames,
            reference_trim_frames=0,
            generated_frames=fresh_frames,
            emitted_generated_frames=0,
        )

    group = [(str(index), None, plan(index)) for index in range(10)]
    runner = scheduler.followup_incremental_graph_holders[0]
    split = scheduler.split_incremental_group_for_graph(group, runner=runner)

    assert [len(item) for item in split] == [4, 4, 2]
    assert [entry[0] for subgroup in split for entry in subgroup] == [
        str(index) for index in range(10)
    ]

    terminal = [("terminal", None, plan(11, fresh_frames=3))]
    assert scheduler.split_incremental_group_for_graph(terminal, runner=runner) == [
        terminal
    ]


def test_incremental_codec_warm_graph_uses_standard_batch_bucket_prefix() -> None:
    select = (
        Qwen3TTSStreamingVocoderScheduler.resolve_incremental_warm_graph_batch_sizes
    )

    assert select(max_batch_size=1) == (1,)
    assert select(max_batch_size=3) == (1, 2, 4)
    assert select(max_batch_size=5) == (1, 2, 4, 8)
    assert select(max_batch_size=8) == (1, 2, 4, 8)
    assert select(max_batch_size=16) == (1, 2, 4, 8)
