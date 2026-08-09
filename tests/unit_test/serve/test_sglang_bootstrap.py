# SPDX-License-Identifier: Apache-2.0
"""SGLang bootstrap helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.model_runner import model_worker as model_worker_module
from sglang_omni.scheduling import bootstrap, sglang_backend
from tests.unit_test.fakes import FakeServerArgs


def test_runtime_configuration_reports_global_backend_for_each_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "get_visible_gpu_sm_version",
        lambda _gpu_id: 89,
        raising=False,
    )
    server_args = SimpleNamespace(
        attention_backend="flashinfer",
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend="pytorch",
        get_attention_backends=lambda: ("flashinfer", "flashinfer"),
    )

    description = bootstrap._describe_sglang_runtime_configuration(
        server_args,
        gpu_id=0,
    )

    assert description == (
        "SGLang runtime configuration: gpu_id=0, sm=89, architecture=ada, "
        "attention_backend=flashinfer, decode_attention_backend=flashinfer, "
        "prefill_attention_backend=flashinfer, sampling_backend=pytorch"
    )


def test_runtime_configuration_reports_explicit_phase_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "get_visible_gpu_sm_version",
        lambda _gpu_id: 90,
        raising=False,
    )
    server_args = SimpleNamespace(
        attention_backend="flashinfer",
        decode_attention_backend="triton",
        prefill_attention_backend="fa3",
        sampling_backend="pytorch",
        get_attention_backends=lambda: ("fa3", "triton"),
    )

    description = bootstrap._describe_sglang_runtime_configuration(
        server_args,
        gpu_id=1,
    )

    assert description == (
        "SGLang runtime configuration: gpu_id=1, sm=90, architecture=hopper, "
        "attention_backend=flashinfer, decode_attention_backend=triton, "
        "prefill_attention_backend=fa3, sampling_backend=pytorch"
    )


def test_create_sglang_infrastructure_runs_0515_initialization_phases(
    monkeypatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda _server_args, _gpu_id: events.append("runtime_configuration")
        or "runtime configuration",
    )

    class FakeRunner:
        model = object()

        def alloc_memory_pool(self) -> None:
            events.append("alloc_memory_pool")

        def init_attention_backends(self) -> None:
            events.append("init_attention_backends")

        def init_cuda_graphs(self) -> None:
            events.append("init_cuda_graphs")

    class FakeWorker:
        model_config = SimpleNamespace(is_multimodal=False)
        enable_prefill_input_embeds = False

        def __init__(self, **kwargs) -> None:
            del kwargs
            events.append("model_worker")
            self.model_runner = FakeRunner()

        def get_memory_pool(self):
            events.append("get_memory_pool")
            return "req_pool", "kv_pool"

    class FakePrefillManager:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def add_one_request(self, req) -> None:
            del req

    class FakeDecodeManager:
        def __init__(self, **kwargs) -> None:
            del kwargs

    monkeypatch.setattr(model_worker_module, "ModelWorker", FakeWorker)
    monkeypatch.setattr(sglang_backend, "PrefillManager", FakePrefillManager)
    monkeypatch.setattr(sglang_backend, "DecodeManager", FakeDecodeManager)
    monkeypatch.setattr(
        sglang_backend,
        "create_tree_cache",
        lambda *args: ("tree_cache", args),
    )

    server_args = SimpleNamespace(
        attention_backend=None,
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend=None,
        page_size=1,
        disable_overlap_schedule=False,
        chunked_prefill_size=8,
        max_prefill_tokens=16,
    )
    infrastructure = bootstrap.create_sglang_infrastructure(server_args, 0)

    assert events == [
        "runtime_configuration",
        "model_worker",
        "alloc_memory_pool",
        "init_attention_backends",
        "init_cuda_graphs",
        "get_memory_pool",
    ]
    assert infrastructure[0].model_runner.model is FakeRunner.model


def test_cuda_graph_init_scopes_prefill_embedding_capture_flag() -> None:
    model_config = SimpleNamespace(is_multimodal=False)
    capture_values: list[bool] = []
    model_worker = SimpleNamespace(
        model_config=model_config,
        enable_prefill_input_embeds=True,
        model_runner=SimpleNamespace(
            init_cuda_graphs=lambda: capture_values.append(model_config.is_multimodal)
        ),
    )

    bootstrap.init_sglang_cuda_graphs(model_worker)

    assert capture_values == [True]
    assert model_config.is_multimodal is False


def test_defer_cuda_graph_restores_requested_graph_capture(monkeypatch) -> None:
    server_args = FakeServerArgs(disable_cuda_graph=False)
    seen: list[bool] = []

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        seen.append(bool(server_args.disable_cuda_graph))
        return ("infra", gpu_id, kwargs)

    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )

    want_cuda_graph, infrastructure = (
        bootstrap.create_sglang_infrastructure_defer_cuda_graph(
            server_args,
            3,
            model_arch_override="TestModel",
        )
    )

    assert want_cuda_graph is True
    assert seen == [True]
    assert server_args.disable_cuda_graph is False
    assert infrastructure == (
        "infra",
        3,
        {
            "defer_cuda_graph_capture": True,
            "model_arch_override": "TestModel",
        },
    )


def test_defer_cuda_graph_leaves_disabled_graph_capture_disabled(monkeypatch) -> None:
    server_args = FakeServerArgs(disable_cuda_graph=True)
    seen: list[bool] = []

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        del gpu_id, kwargs
        seen.append(bool(server_args.disable_cuda_graph))
        return object()

    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure",
        fake_create_sglang_infrastructure,
    )

    want_cuda_graph, _ = bootstrap.create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        0,
    )

    assert want_cuda_graph is False
    assert seen == [True]
    assert server_args.disable_cuda_graph is True
