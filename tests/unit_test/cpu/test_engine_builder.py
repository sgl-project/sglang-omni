# SPDX-License-Identifier: Apache-2.0
"""CPU behaviour of the shared SGLang engine builder.

Kept in the CPU folder rather than parametrised into the shared engine-factory
test so CI can select CPU coverage by directory, and so the CUDA cases stay
exactly as they are (RFC #1310: "Preserve existing CUDA behavior").
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sglang_omni import platforms


def _build_on(monkeypatch, device: str) -> dict[str, Any]:
    """Run the shared builder against fakes and return the server-args kwargs."""
    from sglang_omni.scheduling import bootstrap, sglang_backend
    from sglang_omni.scheduling.engine_factory import TtsEngineBuilder

    monkeypatch.setattr(platforms.current_platform, "is_cpu", lambda: True)
    monkeypatch.setattr("sglang.srt.utils.get_device", lambda device_id=None: "cpu")

    build_kwargs: dict[str, Any] = {}
    events: list[str] = []

    class FakeModel:
        pass

    class FakeRunner:
        def __init__(self, server_args: Any) -> None:
            self.server_args = server_args
            self.model = FakeModel()

        def init_cuda_graphs(self) -> None:
            events.append("init_graphs")

    class FakeWorker:
        def __init__(self, server_args: Any) -> None:
            self.model_runner = FakeRunner(server_args)
            self.model_config = SimpleNamespace(is_multimodal=False)
            self.enable_prefill_input_embeds = False

    def fake_build_server_args(
        checkpoint_dir: str, *, context_length: int, **kwargs: Any
    ) -> Any:
        build_kwargs.update(kwargs)
        return SimpleNamespace(
            checkpoint_dir=checkpoint_dir,
            context_length=context_length,
            cuda_graph_bs=kwargs["cuda_graph_bs"],
            cuda_graph_max_bs=kwargs["cuda_graph_max_bs"],
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(
                    max_bs=kwargs["cuda_graph_max_bs"], bs=kwargs["cuda_graph_bs"]
                ),
                prefill=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
            ),
            disable_cuda_graph=kwargs["disable_cuda_graph"],
            enable_torch_compile=kwargs["enable_torch_compile"],
            max_running_requests=kwargs["max_running_requests"],
            mem_fraction_static=kwargs["mem_fraction_static"],
            torch_compile_max_bs=kwargs["torch_compile_max_bs"],
        )

    def fake_infrastructure(server_args: Any, gpu_id: int, **kwargs: Any):
        events.append("infrastructure")
        build_kwargs["_gpu_id"] = gpu_id
        build_kwargs["_defer_capture"] = kwargs["defer_cuda_graph_capture"]
        return (FakeWorker(server_args), "tree", "req", "kv", "cfg")

    monkeypatch.setattr(
        sglang_backend, "build_sglang_server_args", fake_build_server_args
    )
    monkeypatch.setattr(bootstrap, "create_sglang_infrastructure", fake_infrastructure)
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    class Builder(TtsEngineBuilder):
        model_name = "CPU probe"
        context_length = 123

        def resolve_checkpoint(self, model_path: str) -> str:
            return model_path

        def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
            # A stage default that wants graphs: the CPU decision must beat it.
            return {
                "max_running_requests": 4,
                "cuda_graph_max_bs": 4,
                "torch_compile_max_bs": 4,
                "dtype": dtype,
                "disable_cuda_graph": False,
                "enable_torch_compile": True,
                "mem_fraction_static": 0.5,
            }

        def setup_model(self, **kwargs: Any) -> None:
            build_kwargs["_device"] = kwargs["device"]

        def get_model_buffer_bs(self, model: Any) -> int | None:
            # Must cover max_running_requests above or the policy check rejects it.
            return 4

        def compile_model(self, model: Any, server_args: Any) -> None:
            events.append("compile_model")

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            return SimpleNamespace(model_worker=model_worker)

        def make_adapters(self, model: Any) -> tuple[Any, Any]:
            return ("req_adapter", "res_adapter")

        def make_scheduler(self, **kwargs: Any) -> Any:
            return SimpleNamespace(kwargs=kwargs)

    Builder().build("model", device=device, gpu_id=2)
    build_kwargs["_events"] = events
    return build_kwargs


def test_cpu_placement_forces_graph_capture_off(monkeypatch):
    """generation_defaults() asks for disable_cuda_graph=False; on CPU the
    builder's decision has to win over that stage default, not merely fill a gap.
    """
    build_kwargs = _build_on(monkeypatch, device="cpu")

    assert build_kwargs["disable_cuda_graph"] is True
    assert build_kwargs["device"] == "cpu"


def test_cpu_placement_skips_the_capture_phases(monkeypatch):
    """Skipping capture entirely, rather than running it against a disabled
    config, is what keeps the failure at configuration time.
    """
    build_kwargs = _build_on(monkeypatch, device="cpu")

    assert build_kwargs["_defer_capture"] is False
    assert "init_graphs" not in build_kwargs["_events"]
    assert "compile_model" in build_kwargs["_events"]


def test_graph_disabled_infrastructure_still_initializes_the_eager_runner(
    monkeypatch,
):
    """SGLang installs its eager runner from init_cuda_graphs(), even when graph
    capture is disabled. CPU must initialize that eager path during infrastructure
    construction without requesting a deferred capture from the builder.
    """
    from sglang_omni.model_runner import model_worker as model_worker_mod
    from sglang_omni.scheduling import bootstrap, sglang_backend

    events: list[str] = []

    class FakeContext:
        def is_config_namespace_published(self, namespace: str) -> bool:
            assert namespace == "model"
            return False

    class FakeRunner:
        def alloc_memory_pool(self) -> None:
            events.append("alloc_memory_pool")

        def init_attention_backends(self) -> None:
            events.append("init_attention_backends")

        def init_cuda_graphs(self) -> None:
            events.append("init_eager_runner")

    class FakeWorker:
        enable_prefill_input_embeds = False
        model_config = SimpleNamespace(is_multimodal=False)

        def __init__(self) -> None:
            self.model_runner = FakeRunner()

        def get_memory_pool(self):
            return "req", "kv"

    monkeypatch.setattr("sglang.srt.runtime_context.get_context", lambda: FakeContext())
    monkeypatch.setattr(
        bootstrap,
        "_describe_sglang_runtime_configuration",
        lambda server_args, gpu_id: "CPU test runtime",
    )
    monkeypatch.setattr(model_worker_mod, "ModelWorker", lambda **kwargs: FakeWorker())
    monkeypatch.setattr(model_worker_mod, "ModelWorkerConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(sglang_backend, "create_tree_cache", lambda *args: "tree")

    want_cuda_graph, infrastructure = (
        bootstrap.create_sglang_infrastructure_defer_cuda_graph(
            SimpleNamespace(disable_cuda_graph=True, page_size=1),
            gpu_id=0,
        )
    )

    assert want_cuda_graph is False
    assert events == [
        "alloc_memory_pool",
        "init_attention_backends",
        "init_eager_runner",
    ]
    assert infrastructure[1:4] == ("tree", "req", "kv")


def test_a_cpu_stage_drops_its_placement_index(monkeypatch):
    """gpu_id=2 is handed in, but a CPU device carries no index, so the builder
    must fall back to 0 rather than build 'cpu:2'.
    """
    build_kwargs = _build_on(monkeypatch, device="cpu")

    assert build_kwargs["_device"] == "cpu"
    assert build_kwargs["_gpu_id"] == 0
