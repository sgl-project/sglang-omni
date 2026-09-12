# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import sglang_omni.models.fun_asr.encoder_cuda_graph as encoder_cuda_graph
from sglang_omni.models.fun_asr.encoder_cuda_graph import (
    FunASREncoderCudaGraphRunner,
    _bucket_batch,
    _bucket_t,
)
from sglang_omni.models.fun_asr.sglang_model import FunAsrNanoForConditionalGeneration
from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)


def test_bucket_batch_rounds_up_within_max() -> None:
    assert _bucket_batch(1, 8) == 1
    assert _bucket_batch(2, 8) == 2
    assert _bucket_batch(3, 8) == 4
    assert _bucket_batch(5, 8) == 8
    assert _bucket_batch(8, 8) == 8
    # max_batch not a power of two: fall through to max itself
    assert _bucket_batch(5, 6) == 6
    # over the max -> no bucket
    assert _bucket_batch(9, 8) is None


def test_bucket_t_rounds_up_to_step() -> None:
    assert _bucket_t(1) == 64
    assert _bucket_t(64) == 64
    assert _bucket_t(65) == 128
    assert _bucket_t(500) == 512
    # beyond the 30s ceiling -> no bucket
    assert _bucket_t(513) is None


class _EagerTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.calls: list[tuple] = []

    def forward(self, xs, mask):
        self.calls.append((xs.shape, None if mask is None else mask.shape))
        return xs


class _EagerProjector(nn.Module):
    def __init__(self, llm_dim: int = 4) -> None:
        super().__init__()
        self.llm_dim = llm_dim

    def forward(self, enc_out, mask):
        b, t, _ = enc_out.shape
        t_out = int(fun_asr_low_frame_rate_length(t))
        return torch.arange(b * t_out * self.llm_dim, dtype=torch.float32).reshape(
            b, t_out, self.llm_dim
        )


def _model_with(runner) -> FunAsrNanoForConditionalGeneration:
    model = object.__new__(FunAsrNanoForConditionalGeneration)
    nn.Module.__init__(model)
    model.audio_tower = _EagerTower()
    model.multi_modal_projector = _EagerProjector()
    if runner is not None:
        model.encoder_cuda_graph_runner = runner
    return model


def _item(num_frames: int) -> SimpleNamespace:
    return SimpleNamespace(
        feature=torch.randn(1, 560, num_frames),
        feature_attention_mask=torch.ones(1, num_frames, dtype=torch.long),
    )


def test_get_audio_feature_routes_through_graph_runner() -> None:
    observed = {}

    class _Runner:
        def run(self, xs, lengths):
            observed["xs_shape"] = tuple(xs.shape)
            observed["lengths"] = list(lengths)
            b = xs.shape[0]
            t_out = int(fun_asr_low_frame_rate_length(xs.shape[1]))
            return torch.ones(b, t_out, 4)

    model = _model_with(_Runner())
    out = model.get_audio_feature([_item(17), _item(9)])

    assert observed["xs_shape"] == (2, 17, 560)
    assert observed["lengths"] == [17, 9]
    expected_rows = int(fun_asr_low_frame_rate_length(17)) + int(
        fun_asr_low_frame_rate_length(9)
    )
    assert out.shape == (expected_rows, 4)
    # eager tower must not have run
    assert model.audio_tower.calls == []


def test_get_audio_feature_falls_back_to_eager_when_runner_declines() -> None:
    class _DecliningRunner:
        def run(self, xs, lengths):
            return None

    model = _model_with(_DecliningRunner())
    out = model.get_audio_feature([_item(17), _item(9)])

    # eager path ran, with a mask (batched input)
    assert len(model.audio_tower.calls) == 1
    xs_shape, mask_shape = model.audio_tower.calls[0]
    assert tuple(xs_shape) == (2, 17, 560)
    assert tuple(mask_shape) == (2, 1, 17)
    expected_rows = int(fun_asr_low_frame_rate_length(17)) + int(
        fun_asr_low_frame_rate_length(9)
    )
    assert out.shape == (expected_rows, 4)


def test_get_audio_feature_without_runner_matches_previous_behavior() -> None:
    model = _model_with(None)
    out = model.get_audio_feature([_item(12)])

    # single unpadded item keeps the maskless fast path
    assert model.audio_tower.calls == [((1, 12, 560), None)]
    assert out.shape == (int(fun_asr_low_frame_rate_length(12)), 4)


class _FakeGraph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


class _ParamOn:
    """A parameter stand-in reporting a device type the host may not register.

    torch.device("musa") raises without torch_musa, and the constructor only
    reads the type and index, so a stand-in covers the device-surface policy.
    """

    def __init__(self, device_type: str) -> None:
        self.device = SimpleNamespace(type=device_type, index=0)
        self.dtype = torch.float32


class _InertStream:
    """Streams and events: the runner calls them, no test reads them back."""

    def wait_stream(self, other: "_InertStream") -> None:
        pass

    def record(self, stream: "_InertStream") -> None:
        pass

    def wait(self, stream: "_InertStream") -> None:
        pass


class _FakeGraphBackend:
    """The platform's DeviceGraphBackend, recording how the runner captures."""

    def __init__(self, log: list[str], capture_kwargs: list[dict]) -> None:
        self.log = log
        self.capture_kwargs = capture_kwargs

    @contextlib.contextmanager
    def capture(self, **kwargs):
        self.capture_kwargs.append(kwargs)
        self.log.append("capture:enter")
        yield _FakeGraph()
        self.log.append("capture:exit")


class _FakeDeviceModule:
    """Stands in for torch.cuda / torch.xpu.

    The runner reaches the device only through this object, so a torch.cuda call
    creeping back in surfaces here as an AttributeError instead of as a crash on
    an accelerator no test host has.
    """

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.capture_kwargs: list[dict] = []

    def Event(self) -> _InertStream:  # noqa: N802 - mirrors the torch spelling
        return _InertStream()

    def Stream(self, device=None) -> _InertStream:  # noqa: N802 - ditto
        return _InertStream()

    def current_stream(self, device=None) -> _InertStream:
        return _InertStream()

    @contextlib.contextmanager
    def stream(self, stream: _InertStream):
        self.log.append("warmup-stream:enter")
        yield

    def synchronize(self, device=None) -> None:
        pass

    def graph_pool_handle(self) -> str:
        return "pool-token"

    def device(self, device):
        return contextlib.nullcontext()


def _runner_on(
    module: _FakeDeviceModule, monkeypatch, free_gb: float = 40.0
) -> FunASREncoderCudaGraphRunner:
    """A runner whose device and graph backend are the fakes above."""
    backend = _FakeGraphBackend(module.log, module.capture_kwargs)
    monkeypatch.setattr(
        encoder_cuda_graph.current_platform,
        "get_device_graph_backend",
        lambda device: backend,
        raising=False,
    )
    monkeypatch.setattr(
        encoder_cuda_graph,
        "get_available_gpu_memory",
        lambda device_type, gpu_id, **kwargs: free_gb,
    )
    return FunASREncoderCudaGraphRunner(
        _EagerTower(),
        _EagerProjector(),
        max_batch_size=4,
        device_module=module,
        # These fakes allocate on the host, so the device they claim is cpu.
        graph_device_types=frozenset({"cpu"}),
    )


@contextlib.contextmanager
def _recording_sdpa_context(log: list[str]):
    log.append("sdpa:enter")
    yield
    log.append("sdpa:exit")


def _record_sdpa_pin(monkeypatch, log: list[str]) -> None:
    monkeypatch.setattr(
        encoder_cuda_graph.current_platform,
        "graph_capture_attention",
        lambda: _recording_sdpa_context(log),
        raising=False,
    )


def test_runner_resolves_its_device_module_from_the_model() -> None:
    runner = FunASREncoderCudaGraphRunner(_EagerTower(), _EagerProjector())

    # A CPU-resident tower must not reach for torch.cuda, which is how this
    # constructor used to fail on an accelerator without a CUDA runtime.
    assert runner._device_module is torch.get_device_module(torch.device("cpu"))


def test_runner_declines_every_bucket_on_a_device_without_graphs() -> None:
    runner = FunASREncoderCudaGraphRunner(_EagerTower(), _EagerProjector())

    # torch.cpu cannot capture; the caller then runs the encoder eager rather
    # than raising once per request from inside the capture path.
    assert runner.run(torch.zeros(1, 17, 560), [17]) is None


def test_a_device_without_graphs_is_never_asked_for_an_event() -> None:
    """torch.mps.Event() raises without the MPS backend, so a runner that will
    never replay must not build one at construction."""

    class _NoEventModule:
        def device(self, device):
            return contextlib.nullcontext()

    runner = FunASREncoderCudaGraphRunner(
        _EagerTower(), _EagerProjector(), device_module=_NoEventModule()
    )

    assert runner._done_event is None
    assert runner.run(torch.zeros(1, 17, 560), [17]) is None


def test_a_bucket_is_declined_when_the_card_is_below_the_headroom(monkeypatch) -> None:
    runner = _runner_on(_FakeDeviceModule([]), monkeypatch, free_gb=1.0)

    assert runner._enough_free_gb() == (False, 1.0)


def test_a_bucket_is_captured_when_the_card_has_headroom(monkeypatch) -> None:
    runner = _runner_on(_FakeDeviceModule([]), monkeypatch, free_gb=5.0)

    assert runner._enough_free_gb() == (True, 5.0)


def test_the_memory_query_is_local_and_leaves_the_allocator_cache(monkeypatch) -> None:
    calls: list[dict] = []
    module = _FakeDeviceModule([])
    monkeypatch.setattr(
        encoder_cuda_graph.current_platform,
        "get_device_graph_backend",
        lambda device: _FakeGraphBackend(module.log, module.capture_kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        encoder_cuda_graph,
        "get_available_gpu_memory",
        lambda device_type, gpu_id, **kwargs: calls.append(kwargs) or 40.0,
    )
    runner = FunASREncoderCudaGraphRunner(
        _EagerTower(),
        _EagerProjector(),
        device_module=module,
        graph_device_types=frozenset({"cpu"}),
    )

    runner._enough_free_gb()

    # A capture is triggered lazily by one worker, so a min-reduction across
    # ranks would deadlock; emptying the cache would stall and drop the blocks
    # the graph pool reuses.
    assert calls == [{"distributed": False, "empty_cache": False}]


def test_capture_warms_up_and_records_under_the_platform_sdpa_context(
    monkeypatch,
) -> None:
    log: list[str] = []
    module = _FakeDeviceModule(log)
    runner = _runner_on(module, monkeypatch)
    _record_sdpa_pin(monkeypatch, log)

    runner.run(torch.zeros(1, 17, 560), [17])

    assert log.index("sdpa:enter") < log.index("warmup-stream:enter")
    assert log.index("capture:exit") < log.index("sdpa:exit")


def test_capture_asks_the_platform_backend_for_a_thread_local_capture(
    monkeypatch,
) -> None:
    log: list[str] = []
    module = _FakeDeviceModule(log)
    runner = _runner_on(module, monkeypatch)
    _record_sdpa_pin(monkeypatch, log)

    runner.run(torch.zeros(1, 17, 560), [17])

    # The scheduler thread keeps launching kernels while this captures, so a
    # capture failure must stay on this thread. No pool is named: a capture runs
    # while other buckets replay, and a shared pool is safe only in capture order.
    assert module.capture_kwargs == [{"thread_local_errors": True}]


def test_replay_pads_the_bucket_and_captures_once(monkeypatch) -> None:
    log: list[str] = []
    module = _FakeDeviceModule(log)
    runner = _runner_on(module, monkeypatch)
    _record_sdpa_pin(monkeypatch, log)
    xs = torch.zeros(1, 17, 560)

    first = runner.run(xs, [17])
    runner.run(xs, [17])

    assert first is not None
    assert first.shape == (1, int(fun_asr_low_frame_rate_length(64)), 4)
    graph, _, static_ilens, _ = runner._graphs[(1, 64)]
    assert graph.replays == 2
    assert static_ilens.tolist() == [17]
    assert len(module.capture_kwargs) == 1


def test_capture_does_not_hold_the_replay_lock(monkeypatch) -> None:
    module = _FakeDeviceModule([])
    runner = _runner_on(module, monkeypatch)
    _record_sdpa_pin(monkeypatch, module.log)
    observed: dict[str, bool] = {}
    capture = runner._capture

    def _watching_capture(*args, **kwargs):
        acquired = runner._lock.acquire(blocking=False)
        observed["replay_lock_free"] = acquired
        if acquired:
            runner._lock.release()
        return capture(*args, **kwargs)

    monkeypatch.setattr(runner, "_capture", _watching_capture)

    runner.run(torch.zeros(1, 17, 560), [17])

    # A capture costs seconds on XPU; holding the replay lock through it would
    # stall every encode for buckets that are already recorded.
    assert observed["replay_lock_free"] is True


@pytest.mark.parametrize(
    ("device_type", "captures"),
    [("cuda", True), ("xpu", True), ("musa", False)],
)
def test_a_backend_on_another_device_surface_stays_eager(
    monkeypatch, device_type: str, captures: bool
) -> None:
    module = _FakeDeviceModule([])
    backend = _FakeGraphBackend(module.log, module.capture_kwargs)
    monkeypatch.setattr(
        encoder_cuda_graph.current_platform,
        "get_device_graph_backend",
        lambda device: backend,
        raising=False,
    )
    tower = _EagerTower()
    monkeypatch.setattr(tower, "parameters", lambda: iter([_ParamOn(device_type)]))

    runner = FunASREncoderCudaGraphRunner(
        tower, _EagerProjector(), device_module=module
    )

    # MUSA's platform hands out the CUDA backend, which captures through
    # torch.cuda, while a musa tensor resolves to torch.musa: recording a warmup
    # on one surface and the capture on the other is untested, so it stays eager.
    assert (runner._graph_backend is not None) is captures
    assert (runner._done_event is not None) is captures
