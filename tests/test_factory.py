# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import sglang_omni.engines.omni.factory as factory
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig


class _DummyModelWorkerConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyModelWorker:
    def __init__(self, config, server_args, gpu_id):
        self.config = config
        self.server_args = server_args
        self.gpu_id = gpu_id
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


def test_qwen3_speech_pipeline_enables_talker_feedback() -> None:
    cfg = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    talker_stage = next(stage for stage in cfg.stages if stage.name == "talker_ar")

    assert talker_stage.executor.args["feedback_enabled"] is True
