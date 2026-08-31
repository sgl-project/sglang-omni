# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import sglang_omni.models.fun_asr.encoder_cuda_graph as encoder_cuda_graph
from sglang_omni.models.fun_asr.encoder_cuda_graph import (
    FunASREncoderCudaGraphRunner,
    _bucket_batch,
    _bucket_t,
    _graph_api,
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


class _FakeStream:
    def __init__(self, label: str) -> None:
        self.label = label
        self.joined: list[str] = []

    def wait_stream(self, other: "_FakeStream") -> None:
        self.joined.append(other.label)


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded: list[str] = []
        self.awaited: list[str] = []

    def record(self, stream: _FakeStream) -> None:
        self.recorded.append(stream.label)

    def wait(self, stream: _FakeStream) -> None:
        self.awaited.append(stream.label)


class _FakeDeviceModule:
    """Stands in for torch.cuda / torch.xpu.

    The runner may only reach the device through this object, so anything it
    needs that is missing here shows up as an AttributeError in these tests
    rather than as a CUDA-only runtime on someone else's accelerator.
    """

    def __init__(
        self,
        log: list[str],
        *,
        free_bytes: int = 40 * 1024**3,
        total_bytes: int = 40 * 1024**3,
        reserved_bytes: int = 0,
    ) -> None:
        self.__name__ = "fake_device_module"
        self.log = log
        self.free_bytes = free_bytes
        self.total_bytes = total_bytes
        self.reserved_bytes = reserved_bytes
        self.capture_kwargs: list[dict] = []
        self.event = _FakeEvent()

    def Event(self) -> _FakeEvent:  # noqa: N802 - mirrors the torch spelling
        return self.event

    def Stream(self, device=None) -> _FakeStream:  # noqa: N802 - ditto
        return _FakeStream("side")

    def current_stream(self, device=None) -> _FakeStream:
        return _FakeStream("current")

    @contextlib.contextmanager
    def stream(self, stream: _FakeStream):
        self.log.append("warmup-stream:enter")
        yield
        self.log.append("warmup-stream:exit")

    def synchronize(self, device=None) -> None:
        self.log.append("synchronize")

    def graph_pool_handle(self) -> str:
        return "pool-token"

    @contextlib.contextmanager
    def device(self, device):
        self.log.append("device:enter")
        yield
        self.log.append("device:exit")

    def mem_get_info(self, device=None) -> tuple[int, int]:
        return self.free_bytes, self.total_bytes

    def memory_reserved(self, device=None) -> int:
        return self.reserved_bytes

    @contextlib.contextmanager
    def graph(self, graph, **kwargs):
        self.capture_kwargs.append(kwargs)
        self.log.append("capture:enter")
        yield
        self.log.append("capture:exit")


def _runner_on(module: _FakeDeviceModule) -> FunASREncoderCudaGraphRunner:
    """A runner whose device is the fake module, capturing fake graphs."""
    runner = FunASREncoderCudaGraphRunner(
        _EagerTower(), _EagerProjector(), max_batch_size=4
    )
    runner._device_module = module
    runner._done_event = module.Event()
    runner._graph_cls = _FakeGraph
    runner._capture_kwargs = {}
    return runner


def _sdpa_flags() -> tuple[bool, bool, bool]:
    return (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )


@contextlib.contextmanager
def _recording_sdpa_context(log: list[str]):
    log.append("sdpa:enter")
    yield
    log.append("sdpa:exit")


def _record_sdpa_pin(monkeypatch, log: list[str]) -> None:
    """Stand in for the platform hook plus the context it feeds."""
    monkeypatch.setattr(
        encoder_cuda_graph,
        "_sdpa_capture_context",
        lambda: _recording_sdpa_context(log),
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


@pytest.mark.parametrize(
    ("device_type", "expected_graph_cls", "expected_kwargs"),
    [
        ("cuda", torch.cuda.CUDAGraph, {"capture_error_mode": "thread_local"}),
        ("xpu", torch.xpu.XPUGraph, {}),
        ("cpu", None, {}),
    ],
)
def test_graph_api_is_named_per_device(
    device_type: str, expected_graph_cls, expected_kwargs: dict
) -> None:
    graph_cls, kwargs = _graph_api(torch.device(device_type))

    assert graph_cls is expected_graph_cls
    assert kwargs == expected_kwargs


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_capture_keywords_are_ones_the_device_context_accepts(device_type: str) -> None:
    # torch.xpu.graph takes no capture_error_mode and torch.cuda.graph does;
    # this fails if torch changes either signature under us, which would
    # otherwise only surface as a TypeError mid-capture on that hardware.
    _, kwargs = _graph_api(torch.device(device_type))
    context = torch.get_device_module(torch.device(device_type)).graph

    accepted = inspect.signature(context).parameters
    assert set(kwargs) <= set(accepted)
    assert "pool" in accepted


def test_free_vram_check_clamps_a_runtime_that_reports_the_whole_card_free() -> None:
    over_reporting = _FakeDeviceModule(
        [],
        free_bytes=24 * 1024**3,
        total_bytes=24 * 1024**3,
        reserved_bytes=23 * 1024**3,
    )

    assert _runner_on(over_reporting)._enough_free_vram() == (False, 1024**3)


def test_free_vram_check_keeps_mem_get_info_when_it_is_the_tighter_bound() -> None:
    accounting = _FakeDeviceModule(
        [],
        free_bytes=5 * 1024**3,
        total_bytes=24 * 1024**3,
        reserved_bytes=1024**3,
    )

    assert _runner_on(accounting)._enough_free_vram() == (True, 5 * 1024**3)


def test_capture_warms_up_and_records_under_the_platform_sdpa_context(
    monkeypatch,
) -> None:
    log: list[str] = []
    module = _FakeDeviceModule(log)
    runner = _runner_on(module)
    _record_sdpa_pin(monkeypatch, log)

    runner.run(torch.zeros(1, 17, 560), [17])

    # The warmup is inside the pinned dispatch, so what was captured is what was
    # exercised, and the pin is released once the capture closes.
    assert log.index("sdpa:enter") < log.index("warmup-stream:enter")
    assert log.index("capture:exit") < log.index("sdpa:exit")


def test_capture_asks_the_device_for_its_own_pool_and_keywords(monkeypatch) -> None:
    log: list[str] = []
    module = _FakeDeviceModule(log)
    runner = _runner_on(module)
    runner._capture_kwargs = {"capture_error_mode": "thread_local"}
    _record_sdpa_pin(monkeypatch, log)

    runner.run(torch.zeros(1, 17, 560), [17])

    assert module.capture_kwargs == [
        {"pool": "pool-token", "capture_error_mode": "thread_local"}
    ]


def test_replay_pads_the_bucket_and_captures_once(monkeypatch) -> None:
    log: list[str] = []
    module = _FakeDeviceModule(log)
    runner = _runner_on(module)
    _record_sdpa_pin(monkeypatch, log)
    xs = torch.zeros(1, 17, 560)

    first = runner.run(xs, [17])
    runner.run(xs, [17])

    # 17 frames replay through the 64-frame bucket, and the second call reuses
    # the capture instead of paying for another one.
    assert first is not None
    assert first.shape == (1, int(fun_asr_low_frame_rate_length(64)), 4)
    graph, _, static_ilens, _ = runner._graphs[(1, 64)]
    assert graph.replays == 2
    assert static_ilens.tolist() == [17]
    assert len(module.capture_kwargs) == 1


def test_sdpa_capture_context_pins_nothing_when_the_platform_names_nothing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        encoder_cuda_graph.current_platform,
        "get_graph_capture_sdpa_backends",
        lambda: (),
        raising=False,
    )
    before = _sdpa_flags()

    with encoder_cuda_graph._sdpa_capture_context():
        # A platform whose default selection already captures keeps every kernel
        # it had; narrowing the choice would only cost it.
        assert _sdpa_flags() == before


def test_sdpa_capture_context_pins_what_the_platform_names(monkeypatch) -> None:
    from torch.nn.attention import SDPBackend

    monkeypatch.setattr(
        encoder_cuda_graph.current_platform,
        "get_graph_capture_sdpa_backends",
        lambda: (SDPBackend.MATH,),
        raising=False,
    )

    with encoder_cuda_graph._sdpa_capture_context():
        assert _sdpa_flags() == (False, True, False)

    assert _sdpa_flags() != (False, True, False)
