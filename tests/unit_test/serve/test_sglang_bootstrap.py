# SPDX-License-Identifier: Apache-2.0
"""SGLang bootstrap helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.model_runner import _hidden_capture as hidden_capture_module
from sglang_omni.model_runner import model_worker as model_worker_module
from sglang_omni.scheduling import bootstrap, sglang_backend


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

    monkeypatch.setattr(model_worker_module, "ModelWorker", FakeWorker)
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


def test_an_engine_is_refused_in_a_process_with_a_published_context(
    monkeypatch,
) -> None:
    """ModelRunner publishes the process-wide runtime context, so a process
    that already holds one cannot host a second engine.
    """
    from sglang.srt.runtime_context import get_context

    def constructed(**kwargs):
        raise AssertionError("engine constructed in a published process")

    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda _server_args, _gpu_id: "runtime configuration",
    )
    monkeypatch.setattr(model_worker_module, "ModelWorker", constructed)
    server_args = SimpleNamespace(
        attention_backend=None,
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend=None,
    )

    with get_context().override_server_args():
        with pytest.raises(RuntimeError, match="published SGLang runtime context"):
            bootstrap.create_sglang_infrastructure(server_args, 0)


def test_a_construction_that_failed_after_publishing_is_not_retried(
    monkeypatch,
) -> None:
    from sglang.srt.runtime_context import get_context

    published = get_context().override_server_args()

    def publish_then_fail(**kwargs):
        published.install()
        raise RuntimeError("weights missing")

    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda _server_args, _gpu_id: "runtime configuration",
    )
    monkeypatch.setattr(model_worker_module, "ModelWorker", publish_then_fail)
    server_args = SimpleNamespace(
        attention_backend=None,
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend=None,
    )

    try:
        with pytest.raises(RuntimeError, match="weights missing"):
            bootstrap.create_sglang_infrastructure(server_args, 0)
        with pytest.raises(RuntimeError, match="published SGLang runtime context"):
            bootstrap.create_sglang_infrastructure(server_args, 0)
    finally:
        published.restore()


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


@pytest.mark.parametrize(
    ("server_args", "expected"),
    [
        (
            SimpleNamespace(
                chunked_prefill_size=8192,
                max_prefill_tokens=16384,
                context_length=32768,
                max_running_requests=64,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(max_bs=128, bs=[1, 128]),
                    prefill=SimpleNamespace(max_bs=256, bs=[4, 256]),
                ),
            ),
            8192,
        ),
        (
            SimpleNamespace(
                chunked_prefill_size=-1,
                max_prefill_tokens=16384,
                context_length=8192,
                max_running_requests=64,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(max_bs=None),
                    prefill=SimpleNamespace(max_bs=None),
                ),
            ),
            16384,
        ),
        (
            SimpleNamespace(
                chunked_prefill_size=512,
                max_prefill_tokens=16384,
                context_length=8192,
                max_running_requests=64,
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(max_bs=128, bs=[1, 128]),
                    prefill=SimpleNamespace(max_bs=1024, bs=[4, 1024]),
                ),
            ),
            1024,
        ),
    ],
)
def test_hidden_capture_max_tokens_covers_eager_and_graph_forwards(
    server_args: SimpleNamespace,
    expected: int,
) -> None:
    assert bootstrap._hidden_capture_max_tokens(server_args) == expected


def test_hidden_capture_max_tokens_covers_non_chunked_context_length() -> None:
    server_args = SimpleNamespace(
        chunked_prefill_size=-1,
        max_prefill_tokens=8192,
        context_length=32768,
        max_running_requests=64,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(max_bs=None),
            prefill=SimpleNamespace(max_bs=None),
        ),
    )

    assert bootstrap._hidden_capture_max_tokens(server_args) == 32768


def test_hidden_capture_max_tokens_rejects_missing_capacity_sources() -> None:
    server_args = SimpleNamespace(
        chunked_prefill_size=-1,
        max_prefill_tokens=None,
        context_length=None,
        max_running_requests=0,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(max_bs=None),
            prefill=SimpleNamespace(max_bs=None),
        ),
    )

    with pytest.raises(ValueError, match="hidden capture capacity"):
        bootstrap._hidden_capture_max_tokens(server_args)


def test_hidden_capture_is_installed_before_graph_initialization(monkeypatch) -> None:
    events: list[object] = []

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
            self.model_runner = FakeRunner()
            events.append("model_worker")

        def get_memory_pool(self):
            events.append("get_memory_pool")
            return "req_pool", "kv_pool"

    def fake_install(model, layers, *, max_tokens) -> None:
        events.append(("install_hidden_capture", model, layers, max_tokens))

    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda *_args: "runtime configuration",
    )
    monkeypatch.setattr(model_worker_module, "ModelWorker", FakeWorker)
    monkeypatch.setattr(
        hidden_capture_module,
        "install_hidden_capture_hooks",
        fake_install,
    )
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
        chunked_prefill_size=8192,
        max_prefill_tokens=16384,
        context_length=32768,
        max_running_requests=64,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(max_bs=128, bs=[1, 128]),
            prefill=SimpleNamespace(max_bs=256, bs=[4, 256]),
        ),
    )

    bootstrap.create_sglang_infrastructure(
        server_args,
        0,
        capture_hidden_layers=[0, 24],
    )

    assert events == [
        "model_worker",
        ("install_hidden_capture", FakeRunner.model, [0, 24], 8192),
        "alloc_memory_pool",
        "init_attention_backends",
        "init_cuda_graphs",
        "get_memory_pool",
    ]


def test_defer_cuda_graph_requests_deferred_capture_without_touching_args(
    monkeypatch,
) -> None:
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
    assert seen == [False]
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
    server_args = SimpleNamespace(disable_cuda_graph=True)
    seen: list[tuple[bool, bool]] = []

    def fake_create_sglang_infrastructure(server_args, gpu_id, **kwargs):
        del gpu_id
        seen.append(
            (bool(server_args.disable_cuda_graph), kwargs["defer_cuda_graph_capture"])
        )
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
    assert seen == [(True, False)]
    assert server_args.disable_cuda_graph is True


def test_create_sglang_infrastructure_consumes_scoped_kv_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.scheduling.stage_kv_budget import stage_kv_cache_budget

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda _server_args, _gpu_id: "runtime configuration",
    )

    class FakeRunner:
        model = object()

        def alloc_memory_pool(self) -> None:
            pass

        def init_attention_backends(self) -> None:
            pass

        def init_cuda_graphs(self) -> None:
            pass

    class FakeWorker:
        model_config = SimpleNamespace(is_multimodal=False)
        enable_prefill_input_embeds = False

        def __init__(self, *, config, **kwargs) -> None:
            del kwargs
            captured["kv_cache_bytes"] = config.kv_cache_bytes
            self.model_runner = FakeRunner()

        def get_memory_pool(self):
            return "req_pool", "kv_pool"

    monkeypatch.setattr(model_worker_module, "ModelWorker", FakeWorker)
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

    with stage_kv_cache_budget("thinker", 2 * 1024**3):
        bootstrap.create_sglang_infrastructure(server_args, 0)
    assert captured["kv_cache_bytes"] == 2 * 1024**3

    captured.clear()
    bootstrap.create_sglang_infrastructure(server_args, 0)
    assert captured["kv_cache_bytes"] is None
