# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import sglang_omni.engines.omni.factory as factory
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)


class _DummyModelWorkerConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyModelWorker:
    def __init__(self, config, server_args, gpu_id, tp_rank=0):
        self.config = config
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.model_runner = SimpleNamespace(model=object())
        self.model_config = object()

    def get_memory_pool(self):
        return object(), object()


class _DummyPrefillManager:
    instances: list["_DummyPrefillManager"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).instances.append(self)

    def add_one_request(self, req):
        del req


class _DummyDecodeManager:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyBatchPlanner:
    def __init__(self, prefill_mgr, decode_mgr, server_args):
        self.prefill_mgr = prefill_mgr
        self.decode_mgr = decode_mgr
        self.server_args = server_args
        self._abort_callback = None


class _DummyResourceManager:
    def __init__(self, *args, **kwargs):
        del args, kwargs


class _DummyIterationController:
    def __init__(self, tree_cache, feedback_enabled=False):
        self.tree_cache = tree_cache
        self.feedback_enabled = feedback_enabled


class _DummyOutputProcessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyModelRunner:
    def __init__(self, model_worker, output_proc, batch_planner=None):
        self.model_worker = model_worker
        self.output_proc = output_proc
        self.batch_planner = batch_planner


class _DummyScheduler:
    def __init__(
        self,
        batch_planner,
        resource_manager,
        iteration_controller,
        stream_adapter=None,
    ):
        self.batch_planner = batch_planner
        self.resource_manager = resource_manager
        self.iteration_controller = iteration_controller
        self.stream_adapter = stream_adapter

    def abort_request(self, request_id):
        return request_id


class _DummyEngine:
    def __init__(self, scheduler, model_runner, enable_overlap, feedback_mailbox=None):
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.enable_overlap = enable_overlap
        self.feedback_mailbox = feedback_mailbox


def _install_sglang_stubs(monkeypatch):
    _DummyPrefillManager.instances.clear()

    model_worker_mod = ModuleType("sglang_omni.engines.ar.sglang_backend.model_worker")
    model_worker_mod.ModelWorker = _DummyModelWorker
    model_worker_mod.ModelWorkerConfig = _DummyModelWorkerConfig
    monkeypatch.setitem(sys.modules, model_worker_mod.__name__, model_worker_mod)

    cache_mod = ModuleType("sglang_omni.engines.ar.sglang_backend.scheduler.cache")
    cache_mod.create_tree_cache = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, cache_mod.__name__, cache_mod)

    decode_mod = ModuleType("sglang_omni.engines.ar.sglang_backend.scheduler.decode")
    decode_mod.DecodeManager = _DummyDecodeManager
    monkeypatch.setitem(sys.modules, decode_mod.__name__, decode_mod)

    prefill_mod = ModuleType("sglang_omni.engines.ar.sglang_backend.scheduler.prefill")
    prefill_mod.PrefillManager = _DummyPrefillManager
    monkeypatch.setitem(sys.modules, prefill_mod.__name__, prefill_mod)

    runtime_mod = ModuleType("sglang_omni.engines.omni.runtime.sglang_ar")
    runtime_mod.SGLangBatchPlanner = _DummyBatchPlanner
    runtime_mod.SGLangIterationController = _DummyIterationController
    runtime_mod.SGLangModelRunner = _DummyModelRunner
    runtime_mod.SGLangOutputProcessor = _DummyOutputProcessor
    runtime_mod.SGLangResourceManager = _DummyResourceManager
    monkeypatch.setitem(sys.modules, runtime_mod.__name__, runtime_mod)

    monkeypatch.setattr(factory, "Scheduler", _DummyScheduler)
    monkeypatch.setattr(factory, "OmniEngine", _DummyEngine)


def _import_qwen_stages(monkeypatch):
    server_args_mod = ModuleType("sglang.srt.server_args")

    class _DummyServerArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    server_args_mod.ServerArgs = _DummyServerArgs
    server_args_mod.get_global_server_args = lambda: None

    srt_mod = ModuleType("sglang.srt")
    srt_mod.server_args = server_args_mod

    sglang_mod = ModuleType("sglang")
    sglang_mod.srt = srt_mod

    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_mod)

    sys.modules.pop("sglang_omni.engines.ar.sglang_backend.server_args_builder", None)
    sys.modules.pop("sglang_omni.models.qwen3_omni.pipeline.stages", None)
    return importlib.import_module("sglang_omni.models.qwen3_omni.pipeline.stages")


def test_create_sglang_ar_engine_disables_overlap_for_feedback(monkeypatch) -> None:
    _install_sglang_stubs(monkeypatch)
    server_args = SimpleNamespace(
        disable_overlap_schedule=False,
        page_size=16,
        chunked_prefill_size=32,
        max_prefill_tokens=64,
    )

    engine = factory.create_sglang_ar_engine(
        server_args=server_args,
        enable_overlap=True,
        feedback_enabled=True,
    )

    assert engine.enable_overlap is False
    assert _DummyPrefillManager.instances[-1].kwargs["enable_overlap"] is False
    assert engine.scheduler.iteration_controller.feedback_enabled is True


def test_create_sglang_ar_engine_keeps_overlap_without_feedback(monkeypatch) -> None:
    _install_sglang_stubs(monkeypatch)
    server_args = SimpleNamespace(
        disable_overlap_schedule=False,
        page_size=16,
        chunked_prefill_size=32,
        max_prefill_tokens=64,
    )

    engine = factory.create_sglang_ar_engine(
        server_args=server_args,
        enable_overlap=True,
        feedback_enabled=False,
    )

    assert engine.enable_overlap is True
    assert _DummyPrefillManager.instances[-1].kwargs["enable_overlap"] is True


def test_create_sglang_ar_engine_forwards_tp_metadata(monkeypatch) -> None:
    _install_sglang_stubs(monkeypatch)
    server_args = SimpleNamespace(
        disable_overlap_schedule=False,
        page_size=16,
        chunked_prefill_size=32,
        max_prefill_tokens=64,
    )

    engine = factory.create_sglang_ar_engine(
        server_args=server_args,
        gpu_id=3,
        tp_rank=1,
        nccl_port=23456,
    )

    model_worker = engine.model_runner.model_worker
    assert model_worker.gpu_id == 3
    assert model_worker.tp_rank == 1
    assert model_worker.config.kwargs["nccl_port"] == 23456


def test_qwen3_speech_pipeline_enables_talker_feedback() -> None:
    cfg = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    talker_stage = next(stage for stage in cfg.stages if stage.name == "talker_ar")

    assert talker_stage.executor.args["feedback_enabled"] is True


def test_qwen3_text_pipeline_injects_server_args_overrides() -> None:
    cfg = Qwen3OmniPipelineConfig(
        model_path="dummy",
        server_args_overrides={"tp_size": 2, "mem_fraction_static": 0.75},
    )
    thinker_stage = next(stage for stage in cfg.stages if stage.name == "thinker")

    assert thinker_stage.executor.args["server_args_overrides"] == {
        "tp_size": 2,
        "mem_fraction_static": 0.75,
    }


def test_qwen3_speech_pipeline_injects_server_args_overrides() -> None:
    cfg = Qwen3OmniSpeechPipelineConfig(
        model_path="dummy",
        server_args_overrides={"tp_size": 2, "quantization": "fp8"},
    )
    thinker_stage = next(stage for stage in cfg.stages if stage.name == "thinker")

    assert thinker_stage.executor.args["server_args_overrides"] == {
        "tp_size": 2,
        "quantization": "fp8",
    }


def test_qwen3_thinker_from_config_uses_tp_wrapper(monkeypatch) -> None:
    qwen_stages = _import_qwen_stages(monkeypatch)
    captured = {}

    class _DummyTensorParallelExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        qwen_stages,
        "build_sglang_server_args",
        lambda *args, **kwargs: SimpleNamespace(tp_size=kwargs["tp_size"]),
    )
    monkeypatch.setattr(
        qwen_stages,
        "TensorParallelExecutor",
        _DummyTensorParallelExecutor,
    )

    executor = qwen_stages.create_sglang_thinker_executor_from_config(
        model_path="dummy",
        gpu_id=4,
        thinker_max_seq_len=4096,
        server_args_overrides={"tp_size": 2, "quantization": "fp8"},
        speech_enabled=True,
    )

    assert isinstance(executor, _DummyTensorParallelExecutor)
    assert captured["factory_path"].endswith(
        "create_sglang_thinker_executor_rank_local"
    )
    assert captured["factory_kwargs"] == {
        "model_path": "dummy",
        "thinker_max_seq_len": 4096,
        "server_args_overrides": {"tp_size": 2, "quantization": "fp8"},
        "speech_enabled": True,
    }
    assert captured["tp_size"] == 2
    assert captured["base_gpu_id"] == 4


def test_qwen3_thinker_executor_forwards_tp_rank(monkeypatch) -> None:
    qwen_stages = _import_qwen_stages(monkeypatch)
    captured = {}

    class _DummyTokenizer:
        eos_token_id = 7
        vocab_size = 11

    monkeypatch.setattr(
        qwen_stages.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )
    def _capture_engine_kwargs(**kwargs):
        captured["engine_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(qwen_stages, "create_sglang_ar_engine", _capture_engine_kwargs)
    common_mod = ModuleType("sglang_omni.models.qwen3_omni.components.common")
    common_mod.load_thinker_config = lambda model_path: object()
    monkeypatch.setitem(sys.modules, common_mod.__name__, common_mod)

    qwen_stages.create_sglang_thinker_executor(
        server_args=SimpleNamespace(tp_size=2),
        model_path="dummy",
        gpu_id=5,
        tp_rank=1,
        nccl_port=34567,
        speech_enabled=False,
    )

    assert captured["engine_kwargs"]["gpu_id"] == 5
    assert captured["engine_kwargs"]["tp_rank"] == 1
    assert captured["engine_kwargs"]["nccl_port"] == 34567
