# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import inspect
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages


@pytest.fixture(autouse=True)
def _mock_cuda_contexts(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda *_: contextlib.nullcontext())
    monkeypatch.setattr(torch, "autocast", lambda **_: contextlib.nullcontext())


def _flow(channels: int = 4, max_frames: int = 512) -> SimpleNamespace:
    parameter = torch.nn.Parameter(torch.zeros(1))
    decoder = SimpleNamespace(
        t_scheduler="linear",
        inference_cfg_rate=0.5,
        rand_noise=torch.arange(channels * max_frames, dtype=torch.float32).reshape(
            1, channels, max_frames
        ),
        estimator=torch.nn.Identity(),
    )
    return SimpleNamespace(
        parameters=lambda: iter((parameter,)),
        output_size=channels,
        token_mel_ratio=1,
        decoder=decoder,
        spk_embed_affine_layer=torch.nn.Linear(3, 5, bias=False),
        input_embedding=lambda token: token.new_zeros(
            *token.shape, channels, dtype=torch.float32
        ),
        pre_lookahead_layer=lambda token_embedding: token_embedding,
    )


def _runner(flow=None) -> stages._FlowCudaGraphRunner:
    return stages._FlowCudaGraphRunner(
        flow or _flow(), device=torch.device("cpu"), compute_dtype=torch.bfloat16
    )


def _inputs(batch_size: int, frames: int, channels: int = 4):
    x = torch.arange(batch_size * channels * frames, dtype=torch.float32).reshape(
        batch_size, channels, frames
    )
    return (
        x,
        torch.linspace(0, 1, 11),
        torch.full_like(x, 2),
        torch.ones(batch_size, 1, frames),
        torch.arange(batch_size * 5, dtype=torch.bfloat16).reshape(batch_size, 5),
        torch.full_like(x, 3),
    )


def _install_graph(runner, key: tuple[int, int], *, fail: bool = False):
    static_inputs = runner._capture_inputs(*key)
    graph = SimpleNamespace(
        static_inputs=static_inputs,
        static_output=torch.zeros_like(static_inputs[0]),
        fail=fail,
        replay_calls=0,
    )

    def replay() -> None:
        graph.replay_calls += 1
        if graph.fail:
            raise RuntimeError("synthetic replay failure")
        graph.static_output.copy_(
            graph.static_inputs[0] + graph.static_inputs[2] + graph.static_inputs[5]
        )

    graph.replay = replay
    runner._graphs[key] = stages._CapturedFlowCudaGraph(
        graph, static_inputs, graph.static_output
    )
    return graph


def test_defaults_and_q16_policy() -> None:
    assert (
        inspect.signature(stages.create_vocoder_executor)
        .parameters["enable_flow_cuda_graph"]
        .default
        is False
    )
    shapes = stages._DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES
    assert len(shapes) == len(set(shapes)) == 32
    assert shapes == tuple(sorted(shapes))
    assert all(frames % 16 == 0 for _, frames in shapes)
    assert {(1, 288), (2, 368), (3, 448)} <= set(shapes)
    assert (1, 624) not in shapes


@pytest.mark.parametrize("frames, expected", [(1, 16), (3073, 3088), (0, 0)])
def test_q16_alignment(frames: int, expected: int) -> None:
    assert stages._align_flow_cuda_graph_frames(frames) == expected


def test_capture_inputs_follow_flow_and_runtime_hit_contract() -> None:
    flow = _flow(channels=6)
    runner = _runner(flow)
    graph = _install_graph(runner, (2, 496))
    inputs = _inputs(2, 489, channels=6)
    output = runner.run(*inputs)
    assert output is not None and output.shape == (2, 6, 489)
    torch.testing.assert_close(
        output, inputs[0] + inputs[2] + inputs[5], rtol=0, atol=0
    )
    for index in (0, 2, 3, 5):
        static = runner._graphs[(2, 496)].static_inputs[index]
        torch.testing.assert_close(static[..., :489], inputs[index])
        assert torch.count_nonzero(static[..., 489:]) == 0
    for index in (1, 4):
        static = runner._graphs[(2, 496)].static_inputs[index]
        torch.testing.assert_close(static, inputs[index])
        assert static.shape == inputs[index].shape
    assert graph.replay_calls == 1


def test_misses_do_not_nearest_lookup_capture_or_replay(monkeypatch) -> None:
    runner = _runner()
    graph = _install_graph(runner, (1, 464))
    monkeypatch.setattr(
        runner, "capture", lambda *_: pytest.fail("request-time capture is forbidden")
    )
    assert runner.run(*_inputs(1, 1)) is None
    assert graph.replay_calls == 0

    inputs = _inputs(1, 449)
    assert all(
        runner.run(*bad) is None
        for bad in (
            (inputs[0].bfloat16(), *inputs[1:]),
            (torch.empty_like(inputs[0], device="meta"), *inputs[1:]),
            (torch.empty(1, 3, 449), *inputs[1:]),
        )
    )
    assert graph.replay_calls == 0

    flow, packed = _generation_case()
    eager_inputs = []
    monkeypatch.setattr(
        stages,
        "_solve_flow_euler",
        lambda _decoder, *inputs: eager_inputs.append(inputs) or inputs[0],
    )
    output = stages._generate_flow(flow, packed, runner)
    assert output.shape[-1] == eager_inputs[0][0].shape[-1] == 17


def test_capture_uses_independent_default_pools(monkeypatch) -> None:
    runner = _runner()
    monkeypatch.setattr(
        stages, "_solve_flow_euler", lambda _decoder, *inputs: inputs[0]
    )
    stream = SimpleNamespace(wait_stream=lambda _: None, synchronize=lambda: None)
    graph_kwargs = []

    @contextlib.contextmanager
    def fake_graph(graph, **kwargs):
        del graph
        graph_kwargs.append(kwargs)
        yield

    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: object())
    monkeypatch.setattr(torch.cuda, "graph", fake_graph)
    runner._capture_one(1, 16)
    assert graph_kwargs == [{"stream": stream, "capture_error_mode": "thread_local"}]


def test_startup_capture_is_atomic(monkeypatch) -> None:
    runner = _runner()
    runner._device = torch.device("cuda")

    def capture_one(batch_size: int, _frames: int):
        if batch_size == 2:
            raise RuntimeError("synthetic capture failure")
        return object()

    monkeypatch.setattr(runner, "_capture_one", capture_one)
    with pytest.raises(RuntimeError, match="synthetic capture failure"):
        runner.capture([(1, 16), (2, 16)])
    assert runner._graphs == {}


def _fake_dit_module(monkeypatch) -> ModuleType:
    names = [
        "cosyvoice",
        "cosyvoice.flow",
        "cosyvoice.flow.DiT",
        "cosyvoice.flow.DiT.dit",
    ]
    modules = {name: ModuleType(name) for name in names}
    modules["cosyvoice"].flow = modules["cosyvoice.flow"]
    modules["cosyvoice.flow"].DiT = modules["cosyvoice.flow.DiT"]
    modules["cosyvoice.flow.DiT"].dit = modules["cosyvoice.flow.DiT.dit"]
    modules["cosyvoice.flow.DiT.dit"].add_optional_chunk_mask = object()
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    return modules["cosyvoice.flow.DiT.dit"]


def test_dit_compat_is_in_place_and_persistent(monkeypatch) -> None:
    monkeypatch.setattr(torch._dynamo, "graph_break", lambda: None)
    masks = torch.tensor([[[True, False]], [[False, False]]])
    assert (
        stages._graph_safe_nonstreaming_chunk_mask(
            torch.zeros(2, 2, 1), masks, False, False, 0, 0, -1
        )
        is masks
    )
    assert masks.tolist() == [[[True, False]], [[True, True]]]
    dit = _fake_dit_module(monkeypatch)
    assert stages._enable_flow_cuda_graph_dit_compat() is None
    assert dit.add_optional_chunk_mask is stages._graph_safe_nonstreaming_chunk_mask


def _patch_factory_loader(monkeypatch, flow, device) -> None:
    monkeypatch.setattr(stages, "resolve_device_spec", lambda *_: device)
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda _: "/checkpoint")
    monkeypatch.setattr(
        stages, "_load_cosyvoice3_flow_hift", lambda _checkpoint, **_: (flow, object())
    )


def test_factory_orders_compat_compile_capture(monkeypatch) -> None:
    events = []
    _patch_factory_loader(monkeypatch, _flow(), "cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        stages, "_enable_flow_cuda_graph_dit_compat", lambda: events.append("compat")
    )
    monkeypatch.setattr(
        stages, "_compile_dit_backbone", lambda *_a, **_k: events.append("compile")
    )

    def fake_runner(*_args, **_kwargs):
        events.append("runner")

        def capture(shapes):
            events.append("capture")
            assert shapes is stages._DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES

        return SimpleNamespace(capture=capture)

    monkeypatch.setattr(stages, "_FlowCudaGraphRunner", fake_runner)
    monkeypatch.setattr(
        stages.FunCosyVoice3Flow,
        "attach_cuda_graph_runner",
        lambda *_args: events.append("attach"),
    )
    stages.create_vocoder_executor(
        "model",
        dtype="bfloat16",
        enable_dit_torch_compile=True,
        enable_flow_cuda_graph=True,
    )
    assert events == ["compat", "compile", "runner", "capture", "attach"]


def test_unsupported_graphs_do_not_disable_compile(monkeypatch, caplog) -> None:
    events = []
    attached = []
    _patch_factory_loader(monkeypatch, _flow(), "cpu")
    monkeypatch.setattr(
        stages, "_compile_dit_backbone", lambda *_a, **_k: events.append("compile")
    )
    monkeypatch.setattr(
        stages.FunCosyVoice3Flow,
        "attach_cuda_graph_runner",
        lambda *_args: attached.append(True),
    )
    caplog.set_level(logging.WARNING, logger=stages.__name__)
    stages.create_vocoder_executor(
        "model",
        dtype="float16",
        enable_dit_torch_compile=True,
        enable_flow_cuda_graph=True,
    )
    assert events == ["compile"]
    assert attached == []
    assert "CUDA graphs are disabled" in caplog.text


def test_startup_failure_leaves_normal_serving_available(monkeypatch, caplog) -> None:
    attached = []
    _patch_factory_loader(monkeypatch, _flow(), "cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(stages, "_enable_flow_cuda_graph_dit_compat", lambda: None)

    def fail_capture(_shapes):
        raise RuntimeError("synthetic startup capture failure")

    monkeypatch.setattr(
        stages,
        "_FlowCudaGraphRunner",
        lambda *_args, **_kwargs: SimpleNamespace(capture=fail_capture),
    )
    monkeypatch.setattr(
        stages.FunCosyVoice3Flow,
        "attach_cuda_graph_runner",
        lambda *_args: attached.append(True),
    )
    caplog.set_level(logging.WARNING, logger=stages.__name__)
    stages.create_vocoder_executor("model", enable_flow_cuda_graph=True)
    assert attached == []
    assert "using the normal solver" in caplog.text


def _generation_case():
    return _flow(max_frames=64), SimpleNamespace(
        embedding=torch.ones(1, 3),
        token=torch.ones(1, 17, dtype=torch.int32),
        token_mask=torch.ones(1, 17, 1, dtype=torch.bool),
        total_mel_lengths_tensor=torch.tensor([17]),
        prompt_mel_lengths=(0,),
        prompt_feat=torch.zeros(1, 0, 4),
    )


def test_replay_failure_raises_disables_all_graphs_and_does_not_retry(
    monkeypatch,
) -> None:
    runner = _runner()
    failing = _install_graph(runner, (1, 464), fail=True)
    surviving = _install_graph(runner, (1, 480))
    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        runner.run(*_inputs(1, 449))
    assert failing.replay_calls == 1 and runner._graphs == {}
    assert runner.run(*_inputs(1, 465)) is None and surviving.replay_calls == 0

    flow, packed = _generation_case()
    eager_calls = []

    class ReplayFailure:
        def run(self, *_inputs):
            raise RuntimeError("synthetic replay failure")

    monkeypatch.setattr(
        stages, "_solve_flow_euler", lambda *args: eager_calls.append(args)
    )
    with pytest.raises(RuntimeError, match="synthetic replay failure"):
        stages._generate_flow(flow, packed, ReplayFailure())
    assert eager_calls == []
