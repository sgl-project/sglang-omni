# SPDX-License-Identifier: Apache-2.0
"""Import/config/version-dispatch tests."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_ming_text_config_imports_and_uses_current_stage_schema() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig

    config = MingOmniPipelineConfig(model_path="dummy")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "audio_encoder",
        "image_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
    ]
    assert config.terminal_stages == ["decode"]
    assert all(
        stage.factory.startswith("sglang_omni.models.ming_omni.stages.create_")
        for stage in config.stages
    )
    assert all("executor" not in stage.model_dump() for stage in config.stages)
    assert all("input_handler" not in stage.model_dump() for stage in config.stages)
    assert all("get_next" not in stage.model_dump() for stage in config.stages)


def test_ming_speech_config_routes_decode_and_talker() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

    config = MingOmniSpeechPipelineConfig(model_path="dummy")
    stages = {stage.name: stage for stage in config.stages}

    assert list(stages) == [
        "preprocessing",
        "audio_encoder",
        "image_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
        "talker",
    ]
    assert stages["preprocessing"].next == [
        "audio_encoder",
        "image_encoder",
        "mm_aggregate",
    ]
    assert set(stages["preprocessing"].project_payload) == {
        "audio_encoder",
        "image_encoder",
        "mm_aggregate",
    }
    assert stages["mm_aggregate"].wait_for == [
        "preprocessing",
        "audio_encoder",
        "image_encoder",
    ]
    assert (
        stages["mm_aggregate"].merge_fn
        == "sglang_omni.models.ming_omni.pipeline.merge.merge_for_thinker"
    )
    assert stages["thinker"].next == ["decode", "talker"]
    assert stages["decode"].terminal is True
    assert stages["talker"].terminal is True
    assert config.terminal_stages == ["decode", "talker"]


def test_ming_speech_launcher_exposes_tp_size_arg(monkeypatch) -> None:
    from examples.run_ming_omni_speech_server import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ming_omni_speech_server.py",
            "--tp-size",
            "4",
        ],
    )

    args = parse_args()

    assert args.tp_size == 4


def test_ming_speech_launcher_places_thinker_tp_and_talker(monkeypatch) -> None:
    from examples.run_ming_omni_speech_server import _launch_speech_server

    captured: dict[str, object] = {}
    serve_module = ModuleType("sglang_omni.serve")

    def fake_launch_server(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs

    serve_module.launch_server = fake_launch_server
    monkeypatch.setitem(sys.modules, "sglang_omni.serve", serve_module)

    args = SimpleNamespace(
        model_path="dummy",
        relay_backend="shm",
        tp_size=4,
        gpu_thinker=0,
        gpu_talker=4,
        voice="DB30",
        mem_fraction_static=0.8,
        host="127.0.0.1",
        port=8000,
        model_name="ming-omni",
    )

    _launch_speech_server(args)

    config = captured["config"]
    stages = {stage.name: stage for stage in config.stages}
    thinker = stages["thinker"]
    talker = stages["talker"]
    overrides = thinker.factory_args["server_args_overrides"]

    assert thinker.tp_size == 4
    assert thinker.gpu == [0, 1, 2, 3]
    assert talker.gpu == 4
    assert overrides["disable_custom_all_reduce"] is True
    assert overrides["mem_fraction_static"] == 0.8


def test_ming_stages_import_light_and_accept_mp_injection_args() -> None:
    stages = importlib.import_module("sglang_omni.models.ming_omni.stages")

    sig = inspect.signature(stages.create_sglang_thinker_executor_from_config)

    assert "tp_rank" in sig.parameters
    assert "tp_size" in sig.parameters
    assert "nccl_port" in sig.parameters


def test_ming_talker_factory_returns_scheduler_contract(monkeypatch) -> None:
    talker_module = ModuleType(
        "sglang_omni.models.ming_omni.components.talker_executor"
    )

    class FakeMingTalkerExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self):
            pass

        async def add_request(self, payload):
            self.payload = payload

        async def get_result(self):
            return getattr(self, "payload", None)

    talker_module.MingTalkerExecutor = FakeMingTalkerExecutor
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_omni.components.talker_executor",
        talker_module,
    )

    weight_loader_module = ModuleType("sglang_omni.models.weight_loader")
    weight_loader_module.resolve_model_path = (
        lambda model_path: f"/resolved/{model_path}"
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.weight_loader",
        weight_loader_module,
    )

    from sglang_omni.models.ming_omni.stages import create_talker_executor

    scheduler = create_talker_executor(
        model_path="dummy",
        talker_model_path="talker",
        device="cuda:1",
        voice="DB30",
    )

    assert hasattr(scheduler, "inbox")
    assert hasattr(scheduler, "outbox")
    assert callable(scheduler.start)
    assert callable(scheduler.stop)
    assert callable(scheduler.abort)
    assert not isinstance(scheduler, FakeMingTalkerExecutor)


def test_ming_audio_encoder_moves_inputs_to_component_device() -> None:
    source = Path("sglang_omni/models/ming_omni/components/audio_encoder.py").read_text(
        encoding="utf-8"
    )

    assert "audio_feats = audio_feats.to(device=self._device)" in source
    assert "audio_feats_lengths = audio_feats_lengths.to(device=self._device)" in source


def test_ming_text_launcher_places_tp_ranks_on_distinct_gpus(monkeypatch) -> None:
    from examples.run_ming_omni_server import _launch_text_server

    captured: dict[str, object] = {}
    serve_module = ModuleType("sglang_omni.serve")

    def fake_launch_server(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs

    serve_module.launch_server = fake_launch_server
    monkeypatch.setitem(sys.modules, "sglang_omni.serve", serve_module)

    args = SimpleNamespace(
        model_path="dummy",
        relay_backend="shm",
        tp_size=3,
        quantization=None,
        cpu_offload_gb=0,
        gpu_audio_encoder=None,
        gpu_image_encoder=None,
        thinker_only=False,
        mem_fraction_static=None,
        thinker_max_seq_len=8192,
        host="127.0.0.1",
        port=8000,
        model_name="ming-omni",
    )

    _launch_text_server(args)

    config = captured["config"]
    thinker = next(stage for stage in config.stages if stage.name == "thinker")
    assert thinker.tp_size == 3
    assert thinker.gpu == [0, 1, 2]


def test_ming_text_launcher_allows_encoder_gpu_overrides(monkeypatch) -> None:
    from examples.run_ming_omni_server import _launch_text_server

    captured: dict[str, object] = {}
    serve_module = ModuleType("sglang_omni.serve")

    def fake_launch_server(config, **kwargs):
        del kwargs
        captured["config"] = config

    serve_module.launch_server = fake_launch_server
    monkeypatch.setitem(sys.modules, "sglang_omni.serve", serve_module)

    args = SimpleNamespace(
        model_path="dummy",
        relay_backend="shm",
        tp_size=4,
        quantization=None,
        cpu_offload_gb=0,
        gpu_audio_encoder=4,
        gpu_image_encoder=4,
        thinker_only=False,
        mem_fraction_static=None,
        thinker_max_seq_len=8192,
        host="127.0.0.1",
        port=8000,
        model_name="ming-omni",
    )

    _launch_text_server(args)

    config = captured["config"]
    stages = {stage.name: stage for stage in config.stages}
    assert stages["thinker"].gpu == [0, 1, 2, 3]
    assert stages["audio_encoder"].gpu == 4
    assert stages["image_encoder"].gpu == 4


def test_ming_text_launcher_can_build_thinker_only_smoke_pipeline(
    monkeypatch,
) -> None:
    from examples.run_ming_omni_server import _launch_text_server

    captured: dict[str, object] = {}
    serve_module = ModuleType("sglang_omni.serve")

    def fake_launch_server(config, **kwargs):
        del kwargs
        captured["config"] = config

    serve_module.launch_server = fake_launch_server
    monkeypatch.setitem(sys.modules, "sglang_omni.serve", serve_module)

    args = SimpleNamespace(
        model_path="dummy",
        relay_backend="shm",
        tp_size=4,
        quantization=None,
        cpu_offload_gb=0,
        gpu_audio_encoder=None,
        gpu_image_encoder=None,
        thinker_only=True,
        mem_fraction_static=None,
        thinker_max_seq_len=8192,
        host="127.0.0.1",
        port=8000,
        model_name="ming-omni",
    )

    _launch_text_server(args)

    config = captured["config"]
    stages = {stage.name: stage for stage in config.stages}
    assert list(stages) == ["preprocessing", "mm_aggregate", "thinker", "decode"]
    assert stages["preprocessing"].next == "mm_aggregate"
    assert set(stages["preprocessing"].project_payload) == {"mm_aggregate"}
    assert stages["mm_aggregate"].wait_for == ["preprocessing"]
    assert stages["thinker"].gpu == [0, 1, 2, 3]
    assert stages["thinker"].tp_size == 4


def test_ming_thinker_factory_registers_hf_config_before_server_args(
    monkeypatch,
) -> None:
    from sglang_omni.models.ming_omni import stages

    call_order: list[str] = []
    captured_server_args_kwargs: dict[str, object] = {}

    registration_module = ModuleType("sglang_omni.models.ming_omni.registration")

    def register_ming_hf_config() -> None:
        call_order.append("register")

    registration_module.register_ming_hf_config = register_ming_hf_config
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_omni.registration",
        registration_module,
    )

    backend_module = ModuleType("sglang_omni.scheduling.sglang_backend")

    def build_sglang_server_args(*args, **kwargs):
        del args
        assert call_order == ["register"]
        call_order.append("build_server_args")
        captured_server_args_kwargs.update(kwargs)
        return SimpleNamespace(tp_size=1)

    backend_module.build_sglang_server_args = build_sglang_server_args
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.scheduling.sglang_backend",
        backend_module,
    )

    bootstrap_module = ModuleType("sglang_omni.models.ming_omni.bootstrap")

    def create_thinker_scheduler(*args, **kwargs):
        del args, kwargs
        call_order.append("create_scheduler")
        return object()

    bootstrap_module.create_thinker_scheduler = create_thinker_scheduler
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_omni.bootstrap",
        bootstrap_module,
    )

    stages.create_sglang_thinker_executor_from_config(model_path="dummy")

    assert call_order == ["register", "build_server_args", "create_scheduler"]
    assert captured_server_args_kwargs["trust_remote_code"] is False


def test_ming_arch_override_uses_composite_llm_config() -> None:
    from sglang_omni.model_runner.model_worker import ModelWorker

    llm_config = SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=4,
        hidden_size=4096,
        num_hidden_layers=32,
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[], llm_config=llm_config),
        hf_text_config=None,
        num_attention_heads=None,
        num_key_value_heads=None,
        hidden_size=None,
        num_hidden_layers=None,
    )

    ModelWorker._apply_arch_override(model_config, "BailingMoeV2ForCausalLM")

    assert model_config.hf_config.architectures == ["BailingMoeV2ForCausalLM"]
    assert model_config.hf_text_config is llm_config
    assert model_config.num_attention_heads == 32
    assert model_config.num_key_value_heads == 4
    assert model_config.hidden_size == 4096
    assert model_config.num_hidden_layers == 32


def test_ming_init_model_config_registers_auto_config_before_loading(
    monkeypatch,
) -> None:
    from sglang_omni.model_runner.model_worker import ModelWorker

    call_order: list[str] = []

    registration_module = ModuleType("sglang_omni.models.ming_omni.registration")

    def register_ming_hf_config() -> None:
        call_order.append("register")

    registration_module.register_ming_hf_config = register_ming_hf_config
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_omni.registration",
        registration_module,
    )

    model_config_module = ModuleType("sglang.srt.configs.model_config")

    class FakeModelConfig:
        @classmethod
        def from_server_args(cls, **kwargs):
            del kwargs
            call_order.append("from_server_args")
            return SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=[],
                    llm_config=SimpleNamespace(
                        num_attention_heads=32,
                        num_key_value_heads=4,
                        hidden_size=4096,
                        num_hidden_layers=32,
                    ),
                )
            )

    model_config_module.ModelConfig = FakeModelConfig
    monkeypatch.setitem(sys.modules, "sglang", ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", ModuleType("sglang.srt"))
    monkeypatch.setitem(
        sys.modules, "sglang.srt.configs", ModuleType("sglang.srt.configs")
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.configs.model_config", model_config_module
    )

    worker = object.__new__(ModelWorker)
    worker.server_args = SimpleNamespace(model_path="dummy", revision=None)
    worker.model_arch_override = "BailingMoeV2ForCausalLM"

    worker._init_model_config()

    assert call_order == ["register", "from_server_args"]


def test_ming_decode_metadata_includes_usage_and_finish_reason() -> None:
    from sglang_omni.models.ming_omni.io import PipelineState
    from sglang_omni.models.ming_omni.stages import _attach_decode_final_metadata

    class TensorLike:
        def numel(self) -> int:
            return 5

    state = PipelineState(prompt={"input_ids": TensorLike()})
    thinker_out = {
        "output_ids": [10, 11, 12],
        "finish_reason": "length",
    }
    result: dict[str, object] = {}

    _attach_decode_final_metadata(result, state, thinker_out)

    assert result["finish_reason"] == "length"
    assert result["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "total_tokens": 8,
    }
