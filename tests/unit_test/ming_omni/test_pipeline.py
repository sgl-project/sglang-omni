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
    stages = {stage.name: stage for stage in config.stages}
    assert stages["thinker"].stream_to == ["decode"]
    assert stages["decode"].can_accept_stream_before_payload is True
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
    assert stages["thinker"].stream_to == ["decode"]
    assert stages["decode"].terminal is True
    assert stages["decode"].can_accept_stream_before_payload is True
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


def test_ming_image_encoder_factory_accepts_tp_injection_args() -> None:
    stages = importlib.import_module("sglang_omni.models.ming_omni.stages")

    sig = inspect.signature(stages.create_image_encoder_executor)

    assert "tp_rank" in sig.parameters
    assert "tp_size" in sig.parameters
    assert "nccl_port" in sig.parameters
    assert sig.parameters["tp_rank"].default == 0
    assert sig.parameters["tp_size"].default == 1
    assert sig.parameters["nccl_port"].default is None


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
    weight_loader_module.resolve_model_path = lambda model_path: (
        f"/resolved/{model_path}"
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
        image_encoder_tp=1,
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
        gpu_image_encoder=[4],
        image_encoder_tp=1,
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
        image_encoder_tp=1,
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


def test_ming_text_launcher_configures_image_encoder_tp(monkeypatch) -> None:
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
        tp_size=1,
        quantization=None,
        cpu_offload_gb=0,
        gpu_audio_encoder=None,
        gpu_image_encoder=[2, 3],
        image_encoder_tp=2,
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
    assert stages["image_encoder"].tp_size == 2
    assert stages["image_encoder"].gpu == [2, 3]


def test_ming_text_launcher_rejects_image_encoder_tp_zero(monkeypatch) -> None:
    import pytest

    from examples.run_ming_omni_server import _launch_text_server

    serve_module = ModuleType("sglang_omni.serve")
    serve_module.launch_server = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "sglang_omni.serve", serve_module)

    args = SimpleNamespace(
        model_path="dummy",
        relay_backend="shm",
        tp_size=1,
        quantization=None,
        cpu_offload_gb=0,
        gpu_audio_encoder=None,
        gpu_image_encoder=None,
        image_encoder_tp=0,
        thinker_only=False,
        mem_fraction_static=None,
        thinker_max_seq_len=8192,
        host="127.0.0.1",
        port=8000,
        model_name="ming-omni",
    )

    with pytest.raises(ValueError, match="--image-encoder-tp must be >= 1"):
        _launch_text_server(args)


def test_ming_text_launcher_rejects_thinker_only_with_image_encoder_tp(
    monkeypatch,
) -> None:
    import pytest

    from examples.run_ming_omni_server import _launch_text_server

    serve_module = ModuleType("sglang_omni.serve")
    serve_module.launch_server = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "sglang_omni.serve", serve_module)

    args = SimpleNamespace(
        model_path="dummy",
        relay_backend="shm",
        tp_size=1,
        quantization=None,
        cpu_offload_gb=0,
        gpu_audio_encoder=None,
        gpu_image_encoder=None,
        image_encoder_tp=2,
        thinker_only=True,
        mem_fraction_static=None,
        thinker_max_seq_len=8192,
        host="127.0.0.1",
        port=8000,
        model_name="ming-omni",
    )

    with pytest.raises(ValueError, match="--thinker-only cannot be used"):
        _launch_text_server(args)


def test_ming_text_launcher_requires_gpu_ids_for_image_encoder_tp(
    monkeypatch,
) -> None:
    import pytest

    from examples.run_ming_omni_server import _launch_text_server

    serve_module = ModuleType("sglang_omni.serve")
    serve_module.launch_server = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "sglang_omni.serve", serve_module)

    args = SimpleNamespace(
        model_path="dummy",
        relay_backend="shm",
        tp_size=1,
        quantization=None,
        cpu_offload_gb=0,
        gpu_audio_encoder=None,
        gpu_image_encoder=None,
        image_encoder_tp=2,
        thinker_only=False,
        mem_fraction_static=None,
        thinker_max_seq_len=8192,
        host="127.0.0.1",
        port=8000,
        model_name="ming-omni",
    )

    with pytest.raises(ValueError, match="--gpu-image-encoder must be specified"):
        _launch_text_server(args)


def test_ming_text_launcher_rejects_mismatched_gpu_count(monkeypatch) -> None:
    import pytest

    from examples.run_ming_omni_server import _launch_text_server

    serve_module = ModuleType("sglang_omni.serve")
    serve_module.launch_server = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "sglang_omni.serve", serve_module)

    args = SimpleNamespace(
        model_path="dummy",
        relay_backend="shm",
        tp_size=1,
        quantization=None,
        cpu_offload_gb=0,
        gpu_audio_encoder=None,
        gpu_image_encoder=[2],
        image_encoder_tp=2,
        thinker_only=False,
        mem_fraction_static=None,
        thinker_max_seq_len=8192,
        host="127.0.0.1",
        port=8000,
        model_name="ming-omni",
    )

    with pytest.raises(ValueError, match="requires exactly 2 GPU ids"):
        _launch_text_server(args)


def test_ming_text_launcher_rejects_duplicate_gpu_ids(monkeypatch) -> None:
    import pytest

    from examples.run_ming_omni_server import _launch_text_server

    serve_module = ModuleType("sglang_omni.serve")
    serve_module.launch_server = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "sglang_omni.serve", serve_module)

    args = SimpleNamespace(
        model_path="dummy",
        relay_backend="shm",
        tp_size=1,
        quantization=None,
        cpu_offload_gb=0,
        gpu_audio_encoder=None,
        gpu_image_encoder=[3, 3],
        image_encoder_tp=2,
        thinker_only=False,
        mem_fraction_static=None,
        thinker_max_seq_len=8192,
        host="127.0.0.1",
        port=8000,
        model_name="ming-omni",
    )

    with pytest.raises(ValueError, match="GPU ids must be unique"):
        _launch_text_server(args)


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
    assert captured_server_args_kwargs["sampling_backend"] == "pytorch"


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
    from sglang_omni.models.ming_omni.components.streaming_detokenizer import (
        _attach_decode_final_metadata,
    )
    from sglang_omni.models.ming_omni.io import MingOmniPipelineState

    class TensorLike:
        def numel(self) -> int:
            return 5

    state = MingOmniPipelineState(prompt={"input_ids": TensorLike()})
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


def test_ming_preprocessor_injects_top_level_videos_as_inline_content() -> None:
    """``videos=[...]`` at the top level becomes an inline ``video_url`` item.

    Mirrors the existing ``_inject_top_level_images`` contract so the rest of
    the preprocessor handles top-level and inline video requests identically.
    """
    from sglang_omni.models.ming_omni.components.preprocessor import (
        _inject_top_level_videos,
    )

    messages = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "What is happening?"},
    ]
    out = _inject_top_level_videos(messages, ["/tmp/clip.mp4"])

    # System message untouched, only first user message extended.
    assert out[0] == {"role": "system", "content": "你是助手"}
    assert out[1]["role"] == "user"
    # Ming-Omni was trained with text BEFORE the media item in user turns
    # (see _inject_top_level_audios docstring), so videos go after text too.
    assert out[1]["content"] == [
        {"type": "text", "text": "What is happening?"},
        {"type": "video_url", "video_url": {"url": "/tmp/clip.mp4"}},
    ]
    # Original list unchanged (helper does a shallow copy).
    assert messages[1]["content"] == "What is happening?"


def test_ming_image_encoder_forward_accepts_video_inputs() -> None:
    """MingImageEncoder.forward should accept optional video kwargs.

    Videos reuse the image encoder stage (Qwen3-Omni pattern); the signature
    must expose ``pixel_values_videos`` + ``video_grid_thw`` so the
    preprocessing → image_encoder stage path can flow both modalities.
    """
    from sglang_omni.models.ming_omni.components.image_encoder import MingImageEncoder

    params = inspect.signature(MingImageEncoder.forward).parameters
    for name in (
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ):
        assert name in params, f"missing forward kwarg: {name}"
        assert params[name].default is None, (
            f"{name} must default to None so the image stage can be invoked "
            "with images only, videos only, or both."
        )


def test_ming_merge_extracts_video_embeds_into_thinker_inputs() -> None:
    """build_thinker_inputs must route ``video_embeds`` from the image stage.

    The thinker model_runner injects multimodal embeddings by token id; this
    test verifies ``merge.build_thinker_inputs`` exposes video_embeds the same
    way it already exposes image_embeds/audio_embeds.
    """
    import torch

    from sglang_omni.models.ming_omni.io import MingOmniPipelineState
    from sglang_omni.models.ming_omni.pipeline.merge import build_thinker_inputs
    from sglang_omni.models.ming_omni.pipeline.next_stage import (
        AUDIO_STAGE,
        IMAGE_STAGE,
    )

    state = MingOmniPipelineState(
        raw_inputs={},
        prompt={
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
            "prompt_text": "",
        },
        encoder_inputs={
            IMAGE_STAGE: {"cache_key": "img:abc|vid:def"},
        },
    )

    encoder_outs = {
        AUDIO_STAGE: {},
        IMAGE_STAGE: {
            "image_embeds": torch.randn(4, 8),
            "video_embeds": torch.randn(12, 8),
        },
    }

    result = build_thinker_inputs(state, encoder_outs)

    model_inputs = result.get("model_inputs", {})
    assert "image_embeds" in model_inputs
    assert "video_embeds" in model_inputs
    assert tuple(model_inputs["video_embeds"].shape) == (12, 8)
    assert result["media_cache_keys"]["image"] == "image:img:abc|vid:def"
    # Video must have its own modality-keyed cache entry; the SGLang adapter
    # looks up media_cache_keys.get("video") separately and without this
    # entry video patch tokens would alias in the radix prefix cache.
    assert result["media_cache_keys"]["video"] == "video:img:abc|vid:def"


def test_compute_video_cache_key_changes_with_decode_params() -> None:
    """Different fps/max_frames/pixel limits must produce distinct cache keys.

    Without this, the encoder cache could return ``video_embeds`` whose
    length doesn't match the prompt placeholders for the new request.
    """
    from sglang_omni.preprocessing.video import compute_video_cache_key

    videos = ["/tmp/clip.mp4"]
    base = compute_video_cache_key(videos)
    k_fps_1 = compute_video_cache_key(videos, fps=1.0)
    k_fps_8 = compute_video_cache_key(videos, fps=8.0)
    k_frames_16 = compute_video_cache_key(videos, max_frames=16)
    k_min_px = compute_video_cache_key(videos, min_pixels=128)
    k_max_px = compute_video_cache_key(videos, max_pixels=4096)
    k_total_px = compute_video_cache_key(videos, total_pixels=65536)
    k_all = compute_video_cache_key(
        videos,
        fps=8.0,
        max_frames=16,
        min_pixels=128,
        max_pixels=4096,
        total_pixels=65536,
    )

    # Every param shift produces a distinct key.
    distinct = {
        base,
        k_fps_1,
        k_fps_8,
        k_frames_16,
        k_min_px,
        k_max_px,
        k_total_px,
        k_all,
    }
    assert len(distinct) == 8

    # Same params -> same key (deterministic).
    assert k_fps_8 == compute_video_cache_key(videos, fps=8.0)

    # Empty / None input still returns None (no cache).
    assert compute_video_cache_key(None, fps=8.0) is None
    assert compute_video_cache_key([], fps=8.0) is None


def _make_fake_ming_image_encoder(spatial_merge_size: int = 2):
    """Build a MingImageEncoder shell whose ``_encode`` returns synthetic
    tensors with the real shape contract (embeds rows == sum(token_counts)).

    Bypasses nn.Module.__init__ so we don't need vision-encoder weights, and
    only exercises ``forward``'s dispatch + the embeds/token_counts invariant.
    """
    import types

    import torch

    from sglang_omni.models.ming_omni.components.image_encoder import MingImageEncoder

    enc = object.__new__(MingImageEncoder)
    enc.__dict__["_spatial_merge_size"] = spatial_merge_size
    enc.__dict__["visual"] = types.SimpleNamespace(device=torch.device("cpu"))

    def fake_encode(pixel_values, grid_thw):
        merge_sq = spatial_merge_size**2
        token_counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]) // merge_sq
        total = int(token_counts.sum().item())
        embeds = torch.zeros(total, 8)  # hidden_dim doesn't matter for shape test
        return embeds, token_counts

    enc.__dict__["_encode"] = fake_encode
    return enc


def test_ming_image_encoder_forward_video_embeds_match_token_counts() -> None:
    """video_embeds.shape[0] MUST equal sum(video_token_counts).

    This is the placeholder-count contract: the encoder's output length has
    to match the number of <video_patch> slots the prompt reserved for it.
    Signature tests alone don't catch a regression where forward silently
    returns wrong-length embeds.
    """
    import torch

    from sglang_omni.models.ming_omni.components.image_encoder import MingImageEncoder

    enc = _make_fake_ming_image_encoder()
    # Two videos: (t=2, h=4, w=4) and (t=1, h=6, w=6).
    # With merge_sq=4: tokens = 8 and 9, total = 17.
    video_grid_thw = torch.tensor([[2, 4, 4], [1, 6, 6]], dtype=torch.long)
    pixel_values_videos = torch.zeros(100, 16)

    out = MingImageEncoder.forward(
        enc,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
    )

    assert {"video_embeds", "video_grid_thw", "video_token_counts"} <= set(out)
    assert "image_embeds" not in out
    expected_total = int(out["video_token_counts"].sum().item())
    assert expected_total == 17
    assert out["video_embeds"].shape[0] == expected_total


def test_ming_image_encoder_forward_handles_image_and_video_together() -> None:
    """Mixed image+video request must produce both modalities with the
    embeds/token_counts invariant holding independently for each."""
    import torch

    from sglang_omni.models.ming_omni.components.image_encoder import MingImageEncoder

    enc = _make_fake_ming_image_encoder()
    out = MingImageEncoder.forward(
        enc,
        pixel_values=torch.zeros(50, 16),
        image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.long),  # 4 tokens
        pixel_values_videos=torch.zeros(100, 16),
        video_grid_thw=torch.tensor([[2, 4, 4]], dtype=torch.long),  # 8 tokens
    )

    assert {
        "image_embeds",
        "image_grid_thw",
        "image_token_counts",
        "video_embeds",
        "video_grid_thw",
        "video_token_counts",
    } <= set(out)
    assert (
        out["image_embeds"].shape[0] == int(out["image_token_counts"].sum().item()) == 4
    )
    assert (
        out["video_embeds"].shape[0] == int(out["video_token_counts"].sum().item()) == 8
    )


def test_ming_image_encoder_forward_skips_video_when_grid_thw_missing() -> None:
    """If only one of (pixel_values_videos, video_grid_thw) is provided,
    the encoder must silently skip the video path rather than crash.

    This locks the current defensive behavior: upstream stages that fail to
    produce a usable video pair (e.g. partial decode failure) won't take
    down the encoder, the request just produces no video_embeds.
    """
    import torch

    from sglang_omni.models.ming_omni.components.image_encoder import MingImageEncoder

    enc = _make_fake_ming_image_encoder()

    # pixel_values_videos without video_grid_thw -> skipped.
    out = MingImageEncoder.forward(
        enc,
        pixel_values_videos=torch.zeros(100, 16),
        video_grid_thw=None,
    )
    assert "video_embeds" not in out
    assert "video_grid_thw" not in out

    # video_grid_thw without pixel_values_videos -> also skipped.
    out = MingImageEncoder.forward(
        enc,
        pixel_values_videos=None,
        video_grid_thw=torch.tensor([[2, 4, 4]], dtype=torch.long),
    )
    assert "video_embeds" not in out
    assert "video_grid_thw" not in out
