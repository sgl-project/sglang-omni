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
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import base as stage_base

from sglang_omni.models.sensenova_u1.native.generation import SenseNovaU1GenerationStage
from sglang_omni.models.sensenova_u1.native.sampling import SenseNovaU1SamplingParams


@pytest.mark.parametrize(
    "local_checkpoint,block_builtin", [(False, False), (True, False), (True, True)]
)
def test_external_package_in_fresh_process(
    tmp_path: Path, local_checkpoint: bool, block_builtin: bool
) -> None:
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

        if sys.argv[2] == "True":
            sys.meta_path.insert(0, BlockBuiltin())
        else:
            pass
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
    environment = os.environ | {
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "sglang_omni.models.sensenova_u1.native",
    }
    model_path = (
        str(checkpoint) if local_checkpoint else "sensenova/SenseNova-U1.5-8B-MoT"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, model_path, str(block_builtin)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("think_mode", [False, True])
def test_native_generation_preserves_request_and_output(
    think_mode: bool,
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
        think_mode=think_mode,
        seed=42,
        cfg_norm="channel",
        timestep_shift=2.0,
        enable_timestep_shift=False,
        cfg_interval=(0.25, 0.75),
        t_eps=0.05,
    )
    pixels = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    model = Mock()
    model.t2i_generate.return_value = (pixels, "reasoning") if think_mode else pixels
    tokenizer = Mock()
    stage = SenseNovaU1GenerationStage(model, tokenizer)
    request = Req(sampling_params=sampling_params)
    sampling_params.apply_request_extra(request)
    output = stage.forward(request, None)
    generation_arguments = model.t2i_generate.call_args.kwargs
    assert generation_arguments["seed"] == 42
    assert generation_arguments["image_size"] == (64, 64)
    assert generation_arguments["think_mode"] == think_mode
    assert generation_arguments["num_steps"] == 2
    assert generation_arguments["cfg_norm"] == "channel"
    assert generation_arguments["timestep_shift"] == 2.0
    assert generation_arguments["enable_timestep_shift"] is False
    assert generation_arguments["cfg_interval"] == (0.25, 0.75)
    assert generation_arguments["t_eps"] == 0.05
    torch.testing.assert_close(output.output[0], torch.tensor([[[0.0, 0.5, 1.0]]]))
    assert output.usage == ({"think_text": "reasoning"} if think_mode else None)


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


def test_invalid_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SenseNovaU1SamplingParams(width=65, height=64)


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


def test_generation_preserves_native_extra_defaults_and_conversion(
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
    torch.testing.assert_close(output.output[0], torch.tensor([[[0.0, 0.5, 1.0]]]))
