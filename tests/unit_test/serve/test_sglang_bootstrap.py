# SPDX-License-Identifier: Apache-2.0
"""SGLang bootstrap helpers."""

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.scheduling import bootstrap


def test_defer_cuda_graph_restores_requested_graph_capture(monkeypatch) -> None:
    server_args = SimpleNamespace(disable_cuda_graph=False)
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
    assert infrastructure == ("infra", 3, {"model_arch_override": "TestModel"})


def test_defer_cuda_graph_leaves_disabled_graph_capture_disabled(monkeypatch) -> None:
    server_args = SimpleNamespace(disable_cuda_graph=True)
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
