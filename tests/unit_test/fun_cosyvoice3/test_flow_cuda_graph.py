# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import torch

import sglang_omni.models.fun_cosyvoice3.stages as stages
import sglang_omni.models.fun_cosyvoice3.streaming_vocoder as streaming_vocoder


@pytest.fixture(autouse=True)
def _cpu_autocast(monkeypatch) -> None:
    monkeypatch.setattr(
        torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext()
    )


def _flow(*, channels: int = 4, max_frames: int = 512) -> SimpleNamespace:
    parameter = torch.nn.Parameter(torch.zeros(1))
    return SimpleNamespace(
        parameters=lambda: iter((parameter,)),
        decoder=SimpleNamespace(
            t_scheduler="linear",
            inference_cfg_rate=0.0,
            rand_noise=torch.zeros(1, channels, max_frames),
            estimator=torch.nn.Identity(),
        ),
        output_size=channels,
        token_mel_ratio=1,
        spk_embed_affine_layer=torch.nn.Linear(3, 5),
        input_embedding=lambda token: torch.ones(*token.shape, channels),
        pre_lookahead_layer=lambda x, context=None: x,
        pre_lookahead_len=3,
        cuda_graph_runner=None,
    )


class _ReplayGraph:
    def __init__(
        self,
        static_inputs: tuple[torch.Tensor, ...],
        static_output: torch.Tensor,
    ) -> None:
        self._static_inputs = static_inputs
        self._static_output = static_output

    def replay(self) -> None:
        self._static_output.copy_(
            self._static_inputs[0] + self._static_inputs[2] + self._static_inputs[5]
        )


def _runner() -> stages.FlowCudaGraphRunner:
    return stages.FlowCudaGraphRunner(
        _flow(), device=torch.device("cpu"), autocast_dtype=None
    )


def _install(runner: stages.FlowCudaGraphRunner, key: tuple[int, int]) -> None:
    static_inputs = runner.capture_inputs(*key)
    static_output = torch.empty_like(static_inputs[0])
    runner.graphs[key] = stages.CapturedFlowCudaGraph(
        _ReplayGraph(static_inputs, static_output),
        static_inputs,
        static_output,
    )


def _solver_inputs(
    batch_size: int, mel_frame: int, channels: int = 4
) -> tuple[torch.Tensor, ...]:
    noisy_mel = torch.ones(batch_size, channels, mel_frame)
    return (
        noisy_mel,
        torch.linspace(0, 1, 11),
        torch.full_like(noisy_mel, 2),
        torch.ones(batch_size, 1, mel_frame),
        torch.zeros(batch_size, 5),
        torch.full_like(noisy_mel, 3),
    )


def _packed_tokens(flow: SimpleNamespace, length: int = 17) -> stages.PackedFlowBatch:
    return stages.pack_flow_inputs(
        flow,
        [
            stages.FlowBatchInput(
                token=torch.ones(1, length, dtype=torch.int32),
                prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                prompt_feat=torch.zeros(1, 0, 4),
                embedding=torch.ones(1, 3),
            )
        ],
    )


class _FakeStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waited: list[str] = []

    def wait_stream(self, other: "_FakeStream") -> None:
        self.waited.append(other.name)


class _FakeDeviceModule:
    def __init__(self) -> None:
        self.default_stream = _FakeStream("default")
        self.side_stream = _FakeStream("side")
        self.pool_handle = (0, 1)
        self.scoped: list[torch.device] = []
        self.streamed: list[str] = []
        self.empty_cache_calls = 0

    def current_stream(self, device: torch.device) -> _FakeStream:
        del device
        return self.default_stream

    def Stream(self, device: torch.device) -> _FakeStream:
        del device
        return self.side_stream

    @contextlib.contextmanager
    def stream(self, stream: _FakeStream) -> Iterator[None]:
        self.streamed.append(stream.name)
        yield

    @contextlib.contextmanager
    def device(self, device: torch.device) -> Iterator[None]:
        self.scoped.append(torch.device(device))
        yield

    def graph_pool_handle(self) -> tuple[int, int]:
        return self.pool_handle

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _RecordingBackend:
    def __init__(self) -> None:
        self.captures: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def capture(
        self,
        *,
        pool: Any | None = None,
        stream: Any | None = None,
        thread_local_errors: bool = False,
    ) -> Iterator[Any]:
        graph = object()
        self.captures.append(
            {
                "pool": pool,
                "stream": stream,
                "thread_local_errors": thread_local_errors,
                "graph": graph,
            }
        )
        yield graph


class _FakePlatform:
    device_type = "fake"

    def __init__(self, backend: _RecordingBackend | None) -> None:
        self.backend = backend
        self.attention_scopes = 0

    def get_device_graph_backend(self, device: torch.device) -> Any:
        del device
        return self.backend

    @contextlib.contextmanager
    def graph_capture_attention(self) -> Iterator[None]:
        self.attention_scopes += 1
        yield


def _install_fake_device(
    monkeypatch, *, backend: _RecordingBackend | None
) -> tuple[_FakeDeviceModule, _FakePlatform]:
    module = _FakeDeviceModule()
    platform = _FakePlatform(backend)
    monkeypatch.setattr(torch, "get_device_module", lambda device=None: module)
    monkeypatch.setattr(stages, "current_platform", platform)
    return module, platform


def test_verify_capture_shapes_rejects_unaligned_frames() -> None:
    with pytest.raises(ValueError, match="multiples"):
        stages.verify_flow_cuda_graph_capture_shapes(((1, 495),))


def test_resident_replay_crops_to_actual_frames() -> None:
    runner = _runner()
    _install(runner, (2, 496))
    noisy_mel, time_span, token_condition, mel_mask, speaker_embedding, prompt_mel = (
        _solver_inputs(2, 489)
    )
    output = runner.run(
        noisy_mel,
        time_span,
        token_condition,
        mel_mask,
        speaker_embedding,
        prompt_mel,
    )

    assert output is not None
    assert output.shape == (2, 4, 489)
    assert torch.equal(output, noisy_mel + token_condition + prompt_mel)


def test_nonresident_shape_returns_none() -> None:
    runner = _runner()
    _install(runner, (2, 496))
    assert runner.run(*_solver_inputs(2, 1)) is None


def test_generate_flow_does_not_retry_eager_after_replay_failure(monkeypatch) -> None:
    eager_calls: list[object] = []
    monkeypatch.setattr(
        stages, "solve_flow_euler", lambda *args, **kwargs: eager_calls.append(args)
    )

    class _FailingRunner:
        def run(self, *args, **kwargs):
            raise RuntimeError("replay failed")

    flow = _flow(max_frames=64)
    flow.cuda_graph_runner = _FailingRunner()
    with pytest.raises(RuntimeError, match="replay failed"):
        stages.generate_flow(flow, _packed_tokens(flow))
    assert eager_calls == []


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_capture_records_each_shape_through_the_platform_backend(
    monkeypatch, device_type: str
) -> None:
    backend = _RecordingBackend()
    module, platform = _install_fake_device(monkeypatch, backend=backend)
    solved: list[tuple[int, int]] = []

    def _solve(decoder, noisy_mel, *rest):
        del decoder, rest
        solved.append((int(noisy_mel.shape[0]), int(noisy_mel.shape[2])))
        return torch.zeros_like(noisy_mel)

    monkeypatch.setattr(stages, "solve_flow_euler", _solve)
    device = torch.device(device_type, 0)
    runner = stages.FlowCudaGraphRunner(_flow(), device=device, autocast_dtype=None)

    runner.capture(((2, 496), (1, 512)))

    assert sorted(runner.graphs) == [(1, 512), (2, 496)]
    assert [entry["graph"] for entry in backend.captures] == [
        runner.graphs[(2, 496)].graph,
        runner.graphs[(1, 512)].graph,
    ]
    assert runner.pool == module.pool_handle
    assert [entry["pool"] for entry in backend.captures] == [module.pool_handle] * 2
    assert [entry["stream"] for entry in backend.captures] == [module.side_stream] * 2
    assert all(entry["thread_local_errors"] for entry in backend.captures)
    assert module.streamed == ["side"]
    assert module.side_stream.waited == ["default"]
    assert module.default_stream.waited == ["side"]
    assert platform.attention_scopes == 1
    assert module.scoped == [device]
    assert module.empty_cache_calls == 1
    assert solved == [(2, 496), (2, 496), (1, 512), (1, 512)]


def test_capture_without_a_platform_graph_backend_raises(monkeypatch) -> None:
    _install_fake_device(monkeypatch, backend=None)
    runner = stages.FlowCudaGraphRunner(
        _flow(), device=torch.device("xpu", 0), autocast_dtype=None
    )

    with pytest.raises(RuntimeError, match="device-graph backend"):
        runner.capture(((1, 512),))


class _RecordingRunner:
    instances: list["_RecordingRunner"] = []

    def __init__(self, flow, *, device: torch.device, autocast_dtype) -> None:
        self.flow = flow
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.captured: tuple[tuple[int, int], ...] | None = None
        self.graphs: dict[tuple[int, int], Any] = {}
        _RecordingRunner.instances.append(self)

    def capture(self, capture_shapes: tuple[tuple[int, int], ...]) -> None:
        self.captured = capture_shapes
        self.graphs = {shape: object() for shape in capture_shapes}


def _stub_vocoder_build(monkeypatch, flow: SimpleNamespace) -> list[_RecordingRunner]:
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device(device)
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(stages, "patch_chunk_mask", lambda: None)
    monkeypatch.setattr(
        stages,
        "load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16, **kwargs: (flow, torch.nn.Identity()),
    )
    monkeypatch.setattr(
        stages,
        "CosyVoice3Vocoder",
        lambda *args, **kwargs: SimpleNamespace(flow_scheduler_cost=lambda payload: 0),
    )
    monkeypatch.setattr(
        streaming_vocoder,
        "FunCosyVoice3StreamingVocoderScheduler",
        lambda *args, **kwargs: SimpleNamespace(warmup_now=lambda: None),
    )
    instances: list[_RecordingRunner] = []
    _RecordingRunner.instances = instances
    monkeypatch.setattr(stages, "FlowCudaGraphRunner", _RecordingRunner)
    return instances


def _gate_flow() -> SimpleNamespace:
    flow = _flow()
    flow.attach_cuda_graph_runner = lambda runner: setattr(
        flow, "cuda_graph_runner", runner
    )
    return flow


def test_vocoder_skips_flow_graphs_when_the_platform_records_none(monkeypatch) -> None:
    flow = _gate_flow()
    instances = _stub_vocoder_build(monkeypatch, flow)
    stages.create_vocoder_executor(
        "model", device="cpu", flow_cuda_graph_capture_shapes=((1, 512),)
    )

    assert instances == []
    assert flow.cuda_graph_runner is None


def test_vocoder_captures_flow_graphs_whenever_the_platform_records_them(
    monkeypatch,
) -> None:
    flow = _gate_flow()
    instances = _stub_vocoder_build(monkeypatch, flow)
    monkeypatch.setattr(stages, "current_platform", _FakePlatform(_RecordingBackend()))

    stages.create_vocoder_executor(
        "model", device="cpu", flow_cuda_graph_capture_shapes=((1, 512), (2, 496))
    )

    assert len(instances) == 1
    runner = instances[0]
    assert runner.captured == ((1, 512), (2, 496))
    assert flow.cuda_graph_runner is runner
