"""Exercise native model discovery and T2I request conversion."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("sglang")

import torch
from accelerate import dispatch_model
from safetensors.torch import save_file
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import base as stage_base
from sglang.multimodal_gen.runtime.platforms import current_platform
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.sensenova_u1.config import SenseNovaU1PipelineConfig
from sglang_omni.models.sensenova_u1.native import cache_dit as sensenova_cache_dit
from sglang_omni.models.sensenova_u1.native import loader
from sglang_omni.models.sensenova_u1.native import options as loader_options
from sglang_omni.models.sensenova_u1.native import pipeline
from sglang_omni.models.sensenova_u1.native import sampling as sampling_module
from sglang_omni.models.sensenova_u1.native.generation import SenseNovaU1GenerationStage
from sglang_omni.models.sensenova_u1.native.pipeline import SenseNovaU1Pipeline
from sglang_omni.models.sensenova_u1.native.sampling import SenseNovaU1SamplingParams
from sglang_omni.vendor.sensenova_u1 import modeling_neo_chat
from sglang_omni.vendor.sensenova_u1.configuration_neo_chat import (
    NEOChatConfig,
    NEOMoELLMConfig,
)
from sglang_omni.vendor.sensenova_u1.modeling_neo_chat import NEOChatModel

CACHE_DIT_STEPS = 4
CACHE_DIT_REUSE_PARAMS = {
    "Fn_compute_blocks": 1,
    "Bn_compute_blocks": 0,
    "max_warmup_steps": 0,
    "residual_diff_threshold": 1,
    "max_continuous_cached_steps": -1,
}


def test_external_package_in_fresh_process(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "neo_chat", "architectures": ["NEOChatModel"]})
    )
    script = textwrap.dedent(
        """
        import sys
        import importlib.abc
        from importlib.machinery import ModuleSpec
        from types import ModuleType

        class BlockBuiltin(importlib.abc.MetaPathFinder):
            def find_spec(
                self, fullname: str, path: list[str] | None,
                target: ModuleType | None = None,
            ) -> ModuleSpec | None:
                if fullname.startswith((
                    "sglang.multimodal_gen.runtime.models.sensenova_u1",
                    "sglang.multimodal_gen.runtime.pipelines.sensenova_u1",
                )):
                    raise ModuleNotFoundError(fullname, name=fullname)
                else:
                    return None

        sys.meta_path.insert(0, BlockBuiltin())
        from sglang.multimodal_gen.registry import get_model_info
        from transformers import AutoConfig

        model_info = get_model_info(sys.argv[1], backend="sglang")
        prefix = "sglang_omni.models.sensenova_u1.native."
        assert model_info.pipeline_cls.__module__.startswith(prefix)
        assert model_info.pipeline_config_cls.__module__.startswith(prefix)
        assert model_info.sampling_param_cls.__module__.startswith(prefix)
        config = AutoConfig.for_model("neo_chat")
        assert type(config).__module__.startswith("sglang_omni.vendor.sensenova_u1.")
    """
    )
    config = SenseNovaU1PipelineConfig(model_path=str(checkpoint))
    environment = os.environ | config.resolved_stage_env_defaults("generation")
    result = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_generation_preserves_request_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    sampling_params = SenseNovaU1SamplingParams(
        prompt="a red apple",
        width=64,
        height=64,
        num_inference_steps=2,
        think_mode=True,
        seed=42,
        cfg_norm="channel",
        timestep_shift=2.0,
        enable_timestep_shift=False,
        cfg_interval=(0.25, 0.75),
        t_eps=0.05,
    )
    pixels = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    model = Mock()
    model.t2i_generate.return_value = (pixels, "reasoning")
    tokenizer = Mock()
    stage = SenseNovaU1GenerationStage(model, tokenizer)
    request = Req(sampling_params=sampling_params)
    sampling_params.apply_request_extra(request)
    output = stage.forward(request, None)
    generation_arguments = model.t2i_generate.call_args.kwargs
    assert generation_arguments["seed"] == 42
    assert generation_arguments["image_size"] == (64, 64)
    assert generation_arguments["think_mode"] is True
    assert generation_arguments["num_steps"] == 2
    assert generation_arguments["cfg_norm"] == "channel"
    assert generation_arguments["timestep_shift"] == 2.0
    assert generation_arguments["enable_timestep_shift"] is False
    assert generation_arguments["cfg_interval"] == (0.25, 0.75)
    assert generation_arguments["t_eps"] == 0.05
    torch.testing.assert_close(output.output[0], torch.tensor([[[0.0, 0.5, 1.0]]]))
    assert output.usage == {"think_text": "reasoning"}


def test_think_mode_attention_keeps_the_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Think Mode image attention covers the reasoning appended to the prompt cache."""
    prompt_length, cache_length = 5, 12

    class Cache:
        layers = [
            SimpleNamespace(
                keys=torch.zeros(1, 1, cache_length, 2),
                values=torch.zeros(1, 1, cache_length, 2),
            )
        ]

        def get_seq_length(self) -> int:
            return cache_length

    class Model:
        concat_time_token_num = 0
        downsample_ratio = 0.5
        patch_size = 16
        config = SimpleNamespace()
        _build_t2i_image_indexes = (  # noqa: leading-underscore -- the model method t2i_generate calls.
            modeling_neo_chat.NEOChatModel._build_t2i_image_indexes
        )

        def _build_t2i_query(self, *args: object, **kwargs: object) -> str:
            return "query"

        def _build_t2i_text_inputs(self, *args: object) -> tuple:
            return (
                torch.zeros(1, prompt_length, dtype=torch.long),
                torch.zeros(1, 3, prompt_length, dtype=torch.long),
                None,
                None,
                torch.tensor([prompt_length]),
            )

        def _think_prefix_forward(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(past_key_values=Cache(), logits=torch.zeros(1, 1, 2))

        def _generate_think(self, tokenizer, outputs, cache, t_index, image_start):
            return cache, t_index + cache_length - prompt_length, "reasoning"

    class Prepared(Exception):
        pass

    def prepare(cache, current_len, batch_size, prefix_lengths) -> None:
        raise Prepared(prefix_lengths.tolist())

    monkeypatch.setattr(modeling_neo_chat, "prepare_flash_kv_cache", prepare)
    with pytest.raises(Prepared) as prepared:
        modeling_neo_chat.NEOChatModel.t2i_generate(
            Model(), None, "a cat", image_size=(32, 32), think_mode=True
        )
    assert prepared.value.args == ([cache_length],)


def test_npu_startup_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    config = get_model_info(
        "sensenova/SenseNova-U1.5-8B-MoT", backend="sglang"
    ).pipeline_config_cls()
    monkeypatch.setattr(current_platform, "is_npu", lambda: True)
    with pytest.raises(ValueError, match="does not support NPU devices"):
        config.validate_server_args(SimpleNamespace(num_gpus=1))


def test_parallel_request_fails_before_loading_weights() -> None:
    model_info = get_model_info("sensenova/SenseNova-U1.5-8B-MoT", backend="sglang")
    config = model_info.pipeline_config_cls()
    with pytest.raises(ValueError, match="num_gpus=1"):
        config.validate_server_args(SimpleNamespace(num_gpus=2))


@pytest.mark.parametrize("explicit_offload", [False, True])
def test_offload_default_is_reset_but_explicit_request_is_rejected(
    explicit_offload: bool,
) -> None:
    config = get_model_info(
        "sensenova/SenseNova-U1.5-8B-MoT", backend="sglang"
    ).pipeline_config_cls()
    server_args = SimpleNamespace(
        num_gpus=1,
        dit_cpu_offload=True,
        enable_torch_compile=False,
        lora_path=None,
        is_arg_explicitly_set=lambda option: explicit_offload
        and option == "dit_cpu_offload",
    )
    if explicit_offload:
        with pytest.raises(ValueError, match="DiT CPU offload"):
            config.validate_server_args(server_args)
        assert server_args.dit_cpu_offload is True
    else:
        config.validate_server_args(server_args)
        assert server_args.dit_cpu_offload is False


@pytest.mark.parametrize(
    ("configured", "is_xpu_device", "expected"),
    [
        ({}, False, ["expandable_segments:True"]),
        ({}, True, []),
        ({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}, False, []),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:False"}, False, []),
    ],
)
def test_worker_enables_expandable_segments_unless_configured(
    monkeypatch: pytest.MonkeyPatch,
    configured: dict[str, str],
    is_xpu_device: bool,
    expected: list[str],
) -> None:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
    } | configured
    monkeypatch.setattr(os, "environ", environment)
    monkeypatch.setattr(pipeline, "is_xpu", lambda: is_xpu_device)
    set_allocator_settings = Mock()
    monkeypatch.setattr(
        torch._C,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        "_accelerator_setAllocatorSettings",
        set_allocator_settings,
    )
    launcher_environment = dict(environment)
    config = get_model_info(
        "sensenova/SenseNova-U1.5-8B-MoT", backend="sglang"
    ).pipeline_config_cls()
    config.validate_server_args(
        SimpleNamespace(
            num_gpus=1,
            enable_torch_compile=False,
            lora_path=None,
            is_arg_explicitly_set=lambda option: False,
        )
    )
    assert environment == launcher_environment
    modules = {"model": Mock(), "tokenizer": Mock()}
    assert SenseNovaU1Pipeline.load_modules(Mock(), Mock(), modules) is modules
    assert [call.args[0] for call in set_allocator_settings.call_args_list] == expected


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"width": 65, "height": 64}, "divisible"),
        ({"timestep_shift": 0.0}, "timestep_shift"),
        ({"timestep_shift": float("inf")}, "timestep_shift"),
        ({"t_eps": float("nan")}, "t_eps"),
    ],
)
def test_invalid_sampling_values_are_rejected(
    overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SenseNovaU1SamplingParams(**overrides)


@pytest.mark.parametrize("think_mode", [False, True])
def test_dynamic_batching_excludes_think_requests(think_mode: bool) -> None:
    config = get_model_info(
        "sensenova/SenseNova-U1.5-8B-MoT", backend="sglang"
    ).pipeline_config_cls()
    sampling_params = SenseNovaU1SamplingParams(
        width=64, height=64, think_mode=think_mode
    )
    request = Req(sampling_params=sampling_params)
    sampling_params.apply_request_extra(request)
    assert config.supports_dynamic_batching_for_request(request) is (not think_mode)


def test_warmup_generation_uses_native_extra_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing model extras use native defaults rather than sampling_params overrides."""
    monkeypatch.setattr(
        stage_base,
        "get_global_server_args",
        lambda: SimpleNamespace(comfyui_mode=False),
    )
    sampling_params = SenseNovaU1SamplingParams(
        prompt="a red apple",
        width=64,
        height=64,
        num_inference_steps=2,
        seed=42,
        cfg_norm="channel",
        timestep_shift=2.0,
        enable_timestep_shift=False,
        cfg_interval=(0.25, 0.75),
        t_eps=0.05,
        think_mode=True,
    )
    request = Req(sampling_params=sampling_params)
    request.set_as_warmup(warmup_steps=1)
    pixels = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    model = Mock()
    model.t2i_generate.return_value = pixels
    stage = SenseNovaU1GenerationStage(model, Mock())
    output = stage.forward(request, None)
    generation_arguments = model.t2i_generate.call_args.kwargs
    assert generation_arguments["cfg_norm"] == "none"
    assert generation_arguments["num_steps"] == 1
    assert generation_arguments["timestep_shift"] == 3.0
    assert generation_arguments["enable_timestep_shift"] is True
    assert generation_arguments["cfg_interval"] == (0.0, 1.0)
    assert generation_arguments["t_eps"] == 0.02
    assert generation_arguments["think_mode"] is False
    assert output.usage is None


@pytest.mark.parametrize(
    ("free_gib_by_device", "has_index", "expected_device_indexes"),
    [
        ((96, 96, 96, 96, 96), True, None),
        ((24, 24, 24, 24, 24), True, (1, 2)),
        ((12, 12, 12, 12, 12), True, (1, 2, 3, 4)),
        ((24, 24, 24, 24, 24), False, None),
        ((0, 40, 1, 24, 24), True, None),
        ((0, 1, 24, 24, 24), True, (2, 3)),
        ((0, 24, 1, 24, 24), True, (1, 3)),
        ((0, 36, 1, 1, 1), True, None),
    ],
)
def test_backbone_spans_only_as_many_devices_as_the_weights_need(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    free_gib_by_device: tuple[int, ...],
    has_index: bool,
    expected_device_indexes: tuple[int, ...] | None,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    if has_index:
        (checkpoint / loader.CHECKPOINT_INDEX_FILE_NAME).write_text(
            json.dumps({"metadata": {"total_size": 33 * 2**30}, "weight_map": {}})
        )
    else:
        pass
    monkeypatch.setattr(
        loader.torch,
        "get_device_module",
        lambda device: SimpleNamespace(
            device_count=lambda: len(free_gib_by_device),
            mem_get_info=lambda device_index: (
                free_gib_by_device[device_index] * 2**30,
                0,
            ),
        ),
    )

    device_map = loader.build_spanning_device_map(
        str(checkpoint),
        SimpleNamespace(num_hidden_layers=42, vocab_size=151936, hidden_size=4096),
        torch.bfloat16,
        torch.bfloat16,
        torch.device("xpu", 1),
        None,
    )

    if expected_device_indexes is None:
        assert device_map is None
    else:
        assert {device.index for device in device_map.values()} == set(
            expected_device_indexes
        )
        last_device = torch.device("xpu", expected_device_indexes[-1])
        assert device_map["fm_modules"] == last_device
        assert device_map["language_model.model.norm"] == last_device
        assert device_map["language_model.model.embed_tokens"] == torch.device(
            "xpu", expected_device_indexes[0]
        )
        placed_layers = [
            device_map[f"language_model.model.layers.{layer_index}"].index
            for layer_index in range(42)
        ]
        assert placed_layers == sorted(placed_layers)
        assert set(placed_layers) == set(expected_device_indexes)


def test_spanning_refuses_when_no_device_set_holds_the_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / loader.CHECKPOINT_INDEX_FILE_NAME).write_text(
        json.dumps({"metadata": {"total_size": 33 * 2**30}, "weight_map": {}})
    )
    monkeypatch.setattr(
        loader.torch,
        "get_device_module",
        lambda device: SimpleNamespace(
            device_count=lambda: 1,
            mem_get_info=lambda device_index: (24 * 2**30, 0),
        ),
    )

    with pytest.raises(RuntimeError, match="hold only"):
        loader.build_spanning_device_map(
            str(checkpoint),
            SimpleNamespace(num_hidden_layers=42, vocab_size=151936, hidden_size=4096),
            torch.bfloat16,
            torch.bfloat16,
            torch.device("xpu", 0),
            None,
        )


def test_spanning_is_refused_for_the_moe_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / loader.CHECKPOINT_INDEX_FILE_NAME).write_text(
        json.dumps({"metadata": {"total_size": 33 * 2**30}, "weight_map": {}})
    )
    monkeypatch.setattr(
        loader.torch,
        "get_device_module",
        lambda device: SimpleNamespace(
            device_count=lambda: 4,
            mem_get_info=lambda device_index: (24 * 2**30, 0),
        ),
    )
    moe_config = NEOMoELLMConfig(
        num_hidden_layers=42, vocab_size=151936, hidden_size=4096, num_experts=128
    )

    with pytest.raises(RuntimeError, match="mixes dense and sparse"):
        loader.build_spanning_device_map(
            str(checkpoint),
            moe_config,
            torch.bfloat16,
            torch.bfloat16,
            torch.device("xpu", 0),
            None,
        )


def build_small_dense_model() -> tuple[NEOChatModel, PreTrainedTokenizerFast]:
    torch.manual_seed(0)
    config = NEOChatConfig(
        vision_config={
            "hidden_size": 32,
            "llm_hidden_size": 32,
            "patch_size": 16,
            "max_position_embeddings_vision": 64,
        },
        llm_config={
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 16,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 2048,
            "max_position_embeddings_hw": 64,
        },
        template="neo1_0",
        use_pixel_head=False,
        fm_head_layers=2,
        concat_time_token_num=0,
        noise_scale=1.0,
        noise_scale_mode="constant",
        noise_scale_base_image_seq_len=1,
        add_noise_scale_embedding=True,
        noise_scale_max_value=10.0,
    )
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "<img>": 2, "</img>": 3, "<IMG_CONTEXT>": 4}
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<img>", "</img>", "<IMG_CONTEXT>"],
    )
    return NEOChatModel(config).eval(), tokenizer


@pytest.mark.accelerator
def test_dispatched_denoising_spans_two_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device_type = current_platform.device_type
    device_module = torch.get_device_module(device_type)
    if device_module.device_count() < 2:
        pytest.skip(f"needs two {device_type} devices")
    else:
        pass

    model, tokenizer = build_small_dense_model()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / loader.CHECKPOINT_INDEX_FILE_NAME).write_text(
        json.dumps({"metadata": {"total_size": 4 * 2**30}, "weight_map": {}})
    )
    monkeypatch.setattr(
        loader.torch,
        "get_device_module",
        lambda device: SimpleNamespace(
            device_count=lambda: 2,
            mem_get_info=lambda device_index: (5 * 2**29, 0),
        ),
    )
    device_map = loader.build_spanning_device_map(
        str(checkpoint),
        model.config.llm_config,
        torch.bfloat16,
        torch.bfloat16,
        torch.device(device_type, 0),
        None,
    )
    assert {device.index for device in device_map.values()} == {0, 1}

    model = dispatch_model(
        model,
        device_map={
            name: torch.device(device_type, device.index)
            for name, device in device_map.items()
        },
    )
    head_device = next(model.fm_modules["fm_head"].parameters()).device
    assert head_device == model.language_model.model.norm.weight.device

    cache_devices: list[torch.device] = []
    original_prepare = modeling_neo_chat.prepare_flash_kv_cache

    def recording_prepare(
        past_key_values, current_len, batch_size, prefix_lengths=None
    ):
        cache_devices.extend(
            layer_cache.keys.device for layer_cache in past_key_values.layers
        )
        return original_prepare(
            past_key_values, current_len, batch_size, prefix_lengths
        )

    monkeypatch.setattr(modeling_neo_chat, "prepare_flash_kv_cache", recording_prepare)

    decoder = model.language_model.model
    assert sensenova_cache_dit.spans_multiple_devices(decoder)
    assert (
        sensenova_cache_dit.resolve_cache_dit_switch(
            None, branch_count=2, cfg_interval=(0.0, 1.0), decoder=decoder
        )
        is False
    )
    with pytest.raises(ValueError, match="spanned across devices"):
        sensenova_cache_dit.resolve_cache_dit_switch(
            True, branch_count=2, cfg_interval=(0.0, 1.0), decoder=decoder
        )

    layer_calls = [0]

    def count_last_layer(module, inputs, output) -> None:
        layer_calls[0] += 1

    last_layer = decoder.layers[-1]
    assert next(last_layer.parameters()).device == torch.device(device_type, 1)
    handle = last_layer.register_forward_hook(count_last_layer)

    def generate() -> torch.Tensor:
        cache_devices.clear()
        layer_calls[0] = 0
        images = model.t2i_generate(
            tokenizer,
            "a red apple",
            image_size=(32, 32),
            cfg_scale=4.0,
            num_steps=CACHE_DIT_STEPS,
            seed=0,
        )
        device_module.synchronize()
        return images

    def assert_layer_local_caches() -> None:
        assert cache_devices[:2] == [
            device_map["language_model.model.layers.0"],
            device_map["language_model.model.layers.1"],
        ]

    uncached = generate()
    uncached_calls = layer_calls[0]
    assert_layer_local_caches()
    assert uncached.shape == (1, 3, 32, 32)
    assert uncached.device == head_device
    assert torch.isfinite(uncached).all().item()

    cache_dit = sensenova_cache_dit.SenseNovaCacheDit()
    config = sensenova_cache_dit.resolve_cache_dit_config(
        CACHE_DIT_REUSE_PARAMS, CACHE_DIT_STEPS
    )
    try:
        cache_dit.mount(
            decoder,
            config,
            sensenova_cache_dit.CacheDitMountKey(
                overrides=sensenova_cache_dit.cache_dit_overrides_key(
                    CACHE_DIT_REUSE_PARAMS
                ),
                separate_cfg=False,
                max_warmup_steps=config.max_warmup_steps,
            ),
        )
        with pytest.raises(RuntimeError, match="same device"):
            generate()
    finally:
        cache_dit.unmount(decoder)

    unmounted = generate()
    handle.remove()
    assert layer_calls[0] == uncached_calls
    assert_layer_local_caches()
    assert torch.isfinite(unmounted).all().item()
    assert torch.equal(unmounted, uncached)


def test_checkpoint_dtype_falls_back_to_the_shard_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(
        {
            "language_model.model.layers.0.mlp.gate_proj.weight": torch.zeros(
                4, 4, dtype=torch.bfloat16
            )
        },
        checkpoint / "model-00001-of-00001.safetensors",
    )
    (checkpoint / loader.CHECKPOINT_INDEX_FILE_NAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": 33 * 2**30},
                "weight_map": {
                    "language_model.model.layers.0.mlp.gate_proj.weight": (
                        "model-00001-of-00001.safetensors"
                    )
                },
            }
        )
    )
    monkeypatch.setattr(
        loader.torch,
        "get_device_module",
        lambda device: SimpleNamespace(
            device_count=lambda: 4,
            mem_get_info=lambda device_index: (24 * 2**30, 0),
        ),
    )

    device_map = loader.build_spanning_device_map(
        str(checkpoint),
        SimpleNamespace(num_hidden_layers=42, vocab_size=151936, hidden_size=4096),
        None,
        torch.float32,
        torch.device("xpu", 0),
        None,
    )

    assert {device.index for device in device_map.values()} == {0, 1, 2, 3}


@pytest.mark.parametrize(
    ("request_switch", "expected_enabled", "expected_enabled_by_request"),
    [(None, True, False), (True, True, True), (False, False, False)],
)
def test_server_switch_records_whether_the_request_asked_for_cache_dit(
    monkeypatch: pytest.MonkeyPatch,
    request_switch: bool | None,
    expected_enabled: bool,
    expected_enabled_by_request: bool,
) -> None:
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
    sampling = SenseNovaU1SamplingParams(
        prompt="a red apple",
        width=64,
        height=64,
        num_inference_steps=2,
        guidance_scale=4.0,
        enable_cache_dit=request_switch,
    )
    monkeypatch.setattr(
        sampling_module,
        "checkpoint_decoder_layer_count",
        lambda model_path, revision: 2,
    )

    sampling.resolve_cache_dit("checkpoint", None)

    assert sampling.enable_cache_dit is expected_enabled
    assert sampling.cache_dit_enabled_by_request is expected_enabled_by_request
    extra = sampling.build_request_extra()[
        loader_options.SENSENOVA_U1_REQUEST_EXTRA_KEY
    ]
    assert extra["cache_dit_enabled_by_request"] is expected_enabled_by_request


@pytest.mark.parametrize(
    ("enabled_by_request", "expected"),
    [(False, False), (True, "rejected")],
)
def test_spanned_model_rejects_only_an_explicit_cache_dit_request(
    enabled_by_request: bool, expected: bool | str
) -> None:
    decoder = SimpleNamespace(
        layers=[
            SimpleNamespace(parameters=lambda: iter([torch.zeros(1, device="cpu")])),
            SimpleNamespace(parameters=lambda: iter([torch.zeros(1, device="meta")])),
        ]
    )
    assert sensenova_cache_dit.spans_multiple_devices(decoder)

    if expected == "rejected":
        with pytest.raises(ValueError, match="spanned across devices"):
            sensenova_cache_dit.resolve_cache_dit_switch(
                True,
                branch_count=2,
                cfg_interval=(0.0, 1.0),
                decoder=decoder,
                enabled_by_request=enabled_by_request,
            )
    else:
        assert (
            sensenova_cache_dit.resolve_cache_dit_switch(
                True,
                branch_count=2,
                cfg_interval=(0.0, 1.0),
                decoder=decoder,
                enabled_by_request=enabled_by_request,
            )
            is expected
        )


def test_devices_drawing_on_host_memory_keep_single_device_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / loader.CHECKPOINT_INDEX_FILE_NAME).write_text(
        json.dumps({"metadata": {"total_size": 33 * 2**30}, "weight_map": {}})
    )
    monkeypatch.setattr(current_platform, "device_shares_host_memory", lambda: True)
    monkeypatch.setattr(
        loader.torch,
        "get_device_module",
        lambda device: SimpleNamespace(
            device_count=lambda: 4,
            mem_get_info=lambda device_index: (1 * 2**30, 0),
        ),
    )

    assert (
        loader.build_spanning_device_map(
            str(checkpoint),
            SimpleNamespace(num_hidden_layers=42, vocab_size=151936, hidden_size=4096),
            torch.bfloat16,
            torch.bfloat16,
            torch.device("xpu", 0),
            None,
        )
        is None
    )
