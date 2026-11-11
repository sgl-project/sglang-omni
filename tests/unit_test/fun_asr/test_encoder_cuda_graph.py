# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch
import torch.nn as nn

import sglang_omni.models.fun_asr.encoder_cuda_graph as encoder_cuda_graph
from sglang_omni.models.fun_asr.encoder_cuda_graph import (
    FunASREncoderCudaGraphRunner,
    bucket_batch,
    bucket_t,
)
from sglang_omni.models.fun_asr.sglang_model import FunAsrNanoForConditionalGeneration


def test_bucket_batch_rounds_up_within_max() -> None:
    assert bucket_batch(1, 8) == 1
    assert bucket_batch(2, 8) == 2
    assert bucket_batch(3, 8) == 4
    assert bucket_batch(5, 8) == 8
    assert bucket_batch(8, 8) == 8
    # max_batch not a power of two: fall through to max itself
    assert bucket_batch(5, 6) == 6
    # over the max -> no bucket
    assert bucket_batch(9, 8) is None


def test_bucket_t_rounds_up_to_step() -> None:
    assert bucket_t(1) == 64
    assert bucket_t(64) == 64
    assert bucket_t(65) == 128
    assert bucket_t(500) == 512
    # beyond the 30s ceiling -> no bucket
    assert bucket_t(513) is None


class EagerTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))
        self.calls: list[tuple] = []

    def forward(self, xs, mask):
        self.calls.append((xs.shape, None if mask is None else mask.shape))
        return xs


class EagerProjector(nn.Module):
    def __init__(self, llm_dim: int = 4) -> None:
        super().__init__()
        self.llm_dim = llm_dim

    def forward(self, enc_out, mask):
        b, t, _ = enc_out.shape
        t_out = t
        return torch.arange(b * t_out * self.llm_dim, dtype=torch.float32).reshape(
            b, t_out, self.llm_dim
        )


def model_with(runner) -> FunAsrNanoForConditionalGeneration:
    model = object.__new__(FunAsrNanoForConditionalGeneration)
    nn.Module.__init__(model)
    model.audio_tower = EagerTower()
    model.multi_modal_projector = EagerProjector()
    if runner is not None:
        model.encoder_cuda_graph_runner = runner
    return model


def item(num_frames: int) -> SimpleNamespace:
    return SimpleNamespace(
        feature=torch.randn(1, 560, num_frames),
        feature_attention_mask=torch.ones(1, num_frames, dtype=torch.long),
    )


def test_get_audio_feature_routes_through_graph_runner() -> None:
    observed = {}

    class Runner:
        def run(self, xs, lengths):
            observed["xs_shape"] = tuple(xs.shape)
            observed["lengths"] = list(lengths)
            b = xs.shape[0]
            t_out = xs.shape[1]
            return torch.ones(b, t_out, 4)

    model = model_with(Runner())
    out = model.get_audio_feature([item(17), item(9)])

    assert observed["xs_shape"] == (2, 17, 560)
    assert observed["lengths"] == [17, 9]
    expected_rows = 3 + 2  # ceil(17 / 8) + ceil(9 / 8)
    assert out.shape == (expected_rows, 4)
    # eager tower must not have run
    assert model.audio_tower.calls == []


def test_get_audio_feature_falls_back_to_eager_when_runner_declines() -> None:
    class DecliningRunner:
        def run(self, xs, lengths):
            return None

    model = model_with(DecliningRunner())
    out = model.get_audio_feature([item(17), item(9)])

    # eager path ran, with a mask (batched input)
    assert len(model.audio_tower.calls) == 1
    xs_shape, mask_shape = model.audio_tower.calls[0]
    assert tuple(xs_shape) == (2, 17, 560)
    assert tuple(mask_shape) == (2, 1, 17)
    expected_rows = 3 + 2  # ceil(17 / 8) + ceil(9 / 8)
    assert out.shape == (expected_rows, 4)


def test_get_audio_feature_without_runner_truncates_embeddings() -> None:
    model = model_with(None)
    out = model.get_audio_feature([item(12)])

    # single unpadded item keeps the maskless fast path
    assert model.audio_tower.calls == [((1, 12, 560), None)]
    assert out.shape == (2, 4)  # ceil(12 / 8)


class FakeGraph:
    def __init__(self, log: list[str]) -> None:
        self.replays = 0
        self.log = log

    def replay(self) -> None:
        self.replays += 1
        self.log.append("replay")


class InertStream:
    def wait_stream(self, other: "InertStream") -> None:
        pass

    def record(self, stream: "InertStream") -> None:
        pass

    def wait(self, stream: "InertStream") -> None:
        pass


class FakeGraphBackend:
    def __init__(
        self,
        log: list[str],
        capture_kwargs: list[dict[str, str | bool]],
        fail: bool = False,
    ) -> None:
        self.log = log
        self.capture_kwargs = capture_kwargs
        self.fail = fail

    @contextlib.contextmanager
    def capture(self, **kwargs):
        self.capture_kwargs.append(kwargs)
        self.log.append("capture:enter")
        yield FakeGraph(self.log)
        if self.fail:
            raise RuntimeError("capture_end exploded")
        self.log.append("capture:exit")


class FakeDeviceModule:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.capture_kwargs: list[dict[str, str | bool]] = []
        self.entry_stream = InertStream()

    def Event(self) -> InertStream:  # noqa: N802 - mirrors the torch spelling
        return InertStream()

    def graph_pool_handle(self) -> str:
        return "pool-token"

    def Stream(self, device=None) -> InertStream:  # noqa: N802 - ditto
        return InertStream()

    def current_stream(self, device=None) -> InertStream:
        # One stable object per module: the entry stream a caller records has to
        # be distinguishable from the warmup stream, or restoring either passes.
        return self.entry_stream

    def set_stream(self, stream: InertStream) -> None:
        self.log.append(("set_stream", stream))

    @contextlib.contextmanager
    def stream(self, stream: InertStream):
        self.log.append("warmup-stream:enter")
        yield

    def synchronize(self, device=None) -> None:
        pass

    @contextlib.contextmanager
    def device(self, device):
        self.log.append("device:enter")
        yield
        self.log.append("device:exit")


def runner_on(
    module: FakeDeviceModule,
    monkeypatch,
    free_gb: float = 40.0,
    backend: FakeGraphBackend | None = None,
) -> FunASREncoderCudaGraphRunner:
    backend = backend or FakeGraphBackend(module.log, module.capture_kwargs)
    monkeypatch.setattr(
        encoder_cuda_graph,
        "get_available_gpu_memory",
        lambda device_type, gpu_id, **kwargs: free_gb,
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: module)
    return FunASREncoderCudaGraphRunner(
        EagerTower(), EagerProjector(), graph_backend=backend, max_batch_size=4
    )


def record_sdpa_pin(monkeypatch, log: list[str]) -> None:
    @contextlib.contextmanager
    def pin():
        log.append("sdpa:enter")
        yield
        log.append("sdpa:exit")

    monkeypatch.setattr(
        encoder_cuda_graph.current_platform, "graph_capture_attention", pin
    )


def test_a_bucket_is_declined_when_the_card_is_below_the_headroom(monkeypatch) -> None:
    runner = runner_on(FakeDeviceModule([]), monkeypatch, free_gb=1.0)

    assert runner.run(torch.zeros(1, 17, 560), [17]) is None


def test_capture_warms_up_and_records_under_the_platform_sdpa_context(
    monkeypatch,
) -> None:
    log: list[str] = []
    module = FakeDeviceModule(log)
    runner = runner_on(module, monkeypatch)
    record_sdpa_pin(monkeypatch, log)

    runner.run(torch.zeros(1, 17, 560), [17])

    assert log.index("sdpa:enter") < log.index("warmup-stream:enter")
    assert log.index("capture:exit") < log.index("sdpa:exit")
    assert module.capture_kwargs == [
        {"pool": "pool-token", "thread_local_errors": True}
    ]


def test_replay_pads_the_bucket_and_captures_once(monkeypatch) -> None:
    log: list[str] = []
    module = FakeDeviceModule(log)
    runner = runner_on(module, monkeypatch)
    record_sdpa_pin(monkeypatch, log)
    xs = torch.zeros(1, 17, 560)

    first = runner.run(xs, [17])
    runner.run(xs, [17])

    assert first is not None
    assert first.shape == (1, 64, 4)
    graph, _, static_ilens, _ = runner.graphs[(1, 64)]
    assert graph.replays == 2
    assert static_ilens.tolist() == [17]
    assert len(module.capture_kwargs) == 1


def test_a_failed_capture_restores_the_stream_it_was_entered_on(monkeypatch) -> None:
    log: list[str] = []
    module = FakeDeviceModule(log)
    runner = runner_on(
        module, monkeypatch, backend=FakeGraphBackend(log, [], fail=True)
    )
    record_sdpa_pin(monkeypatch, log)

    assert runner.run(torch.zeros(1, 17, 560), [17]) is None

    restored = log.index(("set_stream", module.entry_stream))
    assert log.index("capture:enter") < restored
    assert runner.graphs == {}
