# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sglang_omni.compilation import CompilePlan
from sglang_omni.scheduling import bootstrap, engine_factory, sglang_backend
from sglang_omni.scheduling.engine_factory import TtsEngineBuilder


def test_tts_engine_builder_runs_compile_plan_around_cuda_graph(
    monkeypatch,
) -> None:
    events: list[str] = []
    server_args = SimpleNamespace(
        disable_cuda_graph=False,
        enable_torch_compile=True,
    )
    model = object()
    model_runner = SimpleNamespace(
        model=model,
        init_device_graphs=lambda: events.append("primary_cuda_graph"),
    )
    model_worker = SimpleNamespace(model_runner=model_runner)

    def create_infrastructure(*args: Any, **kwargs: Any):
        del args, kwargs
        return True, (
            model_worker,
            "tree_cache",
            "req_pool",
            "kv_pool",
            "prefill",
            "decode",
            "model_config",
        )

    monkeypatch.setattr(
        bootstrap,
        "create_sglang_infrastructure_defer_cuda_graph",
        create_infrastructure,
    )
    monkeypatch.setattr(
        sglang_backend,
        "build_sglang_server_args",
        lambda *args, **kwargs: server_args,
    )
    monkeypatch.setattr(
        sglang_backend,
        "SGLangOutputProcessor",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        engine_factory,
        "validate_generation_batch_policy",
        lambda **kwargs: None,
    )

    class FakeCompileManager:
        def __init__(self, plan: CompilePlan, **kwargs: Any) -> None:
            assert (
                kwargs["configure_fn"] is engine_factory.configure_sglang_torch_compile
            )
            events.append("compile_manager")
            self.plan = plan

        def apply(self, phase: Any) -> None:
            events.append(phase.value)

        def finish_startup(self) -> None:
            events.append("compile_finish")

    monkeypatch.setattr(engine_factory, "StageCompileManager", FakeCompileManager)

    def create_plan(model: Any) -> CompilePlan:
        del model
        events.append("create_compile_plan")
        return CompilePlan(name="test.plan", targets=())

    class Builder(TtsEngineBuilder):
        model_name = "test"
        context_length = 1
        compile_plan_factory = create_plan

        def resolve_checkpoint(self, model_path: str) -> str:
            return model_path

        def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
            del dtype
            return {"max_running_requests": 1}

        def setup_model(self, **kwargs: Any) -> None:
            del kwargs
            events.append("setup_model")

        def compile_model(self, model: Any, server_args: Any) -> None:
            del model, server_args
            events.append("legacy_compile_model")

        def post_cuda_graph_setup(self, model: Any, server_args: Any) -> None:
            del model, server_args
            events.append("aux_cuda_graph")

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            del model_worker, output_proc
            return "runner"

        def make_adapters(self, model: Any) -> tuple[Any, Any]:
            del model
            return "request_builder", "result_adapter"

        def make_scheduler(self, **kwargs: Any) -> Any:
            assert server_args.enable_torch_compile is False
            del kwargs
            return SimpleNamespace()

    scheduler = Builder().build("model")

    assert events == [
        "setup_model",
        "legacy_compile_model",
        "create_compile_plan",
        "compile_manager",
        "before_primary_cuda_graph",
        "primary_cuda_graph",
        "after_primary_cuda_graph",
        "aux_cuda_graph",
        "before_stage_ready",
        "compile_finish",
    ]
    assert scheduler._stage_compile_manager.plan.name == "test.plan"


def test_compile_plan_only_consumes_enabled_custom_plan() -> None:
    class BuilderWithoutPlan:
        compile_plan_factory = None

    class BuilderWithPlan:
        def compile_plan_factory(model: Any) -> CompilePlan:
            del model
            return CompilePlan(name="test.plan", targets=())

    upstream_args = SimpleNamespace(enable_torch_compile=True)
    disabled_args = SimpleNamespace(enable_torch_compile=False)
    enabled_args = SimpleNamespace(enable_torch_compile=True)

    assert (
        TtsEngineBuilder._create_compile_plan(
            BuilderWithoutPlan(), object(), upstream_args
        )
        is None
    )
    assert (
        TtsEngineBuilder._create_compile_plan(
            BuilderWithPlan(), object(), disabled_args
        )
        is None
    )
    plan = TtsEngineBuilder._create_compile_plan(
        BuilderWithPlan(), object(), enabled_args
    )

    assert plan is not None and plan.name == "test.plan"
    assert upstream_args.enable_torch_compile is True
    assert disabled_args.enable_torch_compile is False
    assert enabled_args.enable_torch_compile is False
