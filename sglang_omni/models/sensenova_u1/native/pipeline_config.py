# SPDX-License-Identifier: Apache-2.0
"""Validate deployment options for native SenseNova generation."""
from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from sglang_omni.models.sensenova_u1.native.options import (
    DEFAULT_THINK_MODE,
    RESOLUTION_ALIGNMENT,
)


def is_runtime_option_requested(
    option_value: (
        str
        | bool
        | int
        | dict[str, str]
        | list[str]
        | tuple[str, ...]
        | set[str]
        | None
    ),
) -> bool:
    if option_value is None:
        return False
    elif isinstance(option_value, bool):
        return option_value
    elif isinstance(option_value, (dict, list, tuple, set)):
        return bool(option_value)
    else:
        return True


def component_residency_requests_offload(
    component_residency: str | dict[str, str] | list[str] | None,
) -> bool:
    if not is_runtime_option_requested(component_residency):
        return False
    else:
        pass
    if isinstance(component_residency, dict):
        residency_modes = component_residency.values()
    elif isinstance(component_residency, str):
        residency_modes = component_residency.split(",")
    else:
        residency_modes = component_residency

    for residency_entry in residency_modes:
        residency_mode = (
            str(residency_entry).split("=", 1)[-1].strip().replace("_", "-").lower()
        )
        if residency_mode in ("component-offload", "layerwise-offload"):
            return True
        else:
            pass
    return False


def set_compatible_runtime_defaults(server_args: ServerArgs) -> None:
    compatible_defaults = {
        "component_residency": None,
        "cpu_offload_components": None,
        "dit_cpu_offload": False,
        "text_encoder_cpu_offload": False,
        "image_encoder_cpu_offload": False,
        "vae_cpu_offload": False,
        "dit_layerwise_offload": False,
        "layerwise_offload_components": None,
        "quantization": None,
        "quantization_ignored_layers": None,
        "transformer_weights_path": None,
        "component_paths": {},
        "component_weights_paths": {},
        "component_quantizations": {},
        "component_quantization_ignored_layers": {},
        "component_precisions": {},
        "attention_backend": None,
        "component_attention_backends": {},
        "attention_backend_config": None,
    }
    for option_name, option_value in compatible_defaults.items():
        if not server_args.is_arg_explicitly_set(option_name):
            setattr(server_args, option_name, option_value)
        else:
            pass


@dataclass
class SenseNovaU1PipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I
    model_precision: str = "bf16"
    should_use_guidance: bool = True
    supports_cfg_parallel: bool = False

    def supports_dynamic_batching(self) -> bool:
        return True

    def supports_dynamic_batching_for_request(self, batch: Req) -> bool:
        sampling_params = batch.sampling_params
        return not bool(getattr(sampling_params, "think_mode", DEFAULT_THINK_MODE))

    def estimate_request_cost(self, batch: Req) -> float:
        image_token_count = (int(batch.width) // RESOLUTION_ALIGNMENT) * (
            int(batch.height) // RESOLUTION_ALIGNMENT
        )
        guidance_branch_count = 2 if float(batch.guidance_scale) > 1 else 1
        return float(
            image_token_count
            * int(batch.num_inference_steps)
            * guidance_branch_count
            * int(batch.num_outputs_per_prompt)
        )

    def supports_disaggregation(self) -> bool:
        return False

    def supports_sequential_multi_output_inference(self) -> bool:
        return True

    def validate_server_args(self, server_args: ServerArgs) -> None:
        if current_platform.is_npu():
            raise ValueError("SenseNovaU1Pipeline does not support NPU devices.")
        else:
            pass
        if server_args.num_gpus != 1:
            raise ValueError(
                "SenseNovaU1Pipeline currently supports num_gpus=1. "
                "Native tensor/pipeline parallelism is not implemented yet."
            )
        else:
            pass
        if server_args.enable_torch_compile:
            raise ValueError(
                "SenseNovaU1Pipeline does not support torch.compile yet. "
                "Please omit --enable-torch-compile."
            )
        else:
            pass
        if server_args.lora_path:
            raise ValueError(
                "SenseNovaU1Pipeline does not support LoRA adapters yet. "
                "Please omit --lora-path."
            )
        else:
            pass
        set_compatible_runtime_defaults(server_args)
        if server_args.is_arg_explicitly_set(
            "component_residency"
        ) and component_residency_requests_offload(server_args.component_residency):
            raise ValueError(
                "SenseNovaU1Pipeline does not support component residency "
                "offload modes yet. Please omit --component-residency."
            )
        else:
            pass
        unsupported_runtime_options = {
            "cpu_offload_components": "CPU offload",
            "dit_cpu_offload": "DiT CPU offload",
            "text_encoder_cpu_offload": "text encoder CPU offload",
            "image_encoder_cpu_offload": "image encoder CPU offload",
            "vae_cpu_offload": "VAE CPU offload",
            "dit_layerwise_offload": "DiT layerwise offload",
            "layerwise_offload_components": "layerwise offload",
            "quantization": "quantization",
            "quantization_ignored_layers": "quantization ignored layers",
            "transformer_weights_path": "pre-quantized transformer weights",
            "component_paths": "component path overrides",
            "component_weights_paths": "component weight path overrides",
            "component_quantizations": "component quantization",
            "component_quantization_ignored_layers": (
                "component quantization ignored layers"
            ),
            "component_precisions": "component precision overrides",
        }
        for option_name, option_description in unsupported_runtime_options.items():
            if server_args.is_arg_explicitly_set(
                option_name
            ) and is_runtime_option_requested(getattr(server_args, option_name)):
                raise ValueError(
                    f"SenseNovaU1Pipeline does not support {option_description} yet. "
                    f"Please omit --{option_name.replace('_', '-')}."
                )
            else:
                pass
        if server_args.is_arg_explicitly_set(
            "attention_backend"
        ) and is_runtime_option_requested(server_args.attention_backend):
            raise ValueError(
                "SenseNovaU1Pipeline does not support custom attention backends yet. "
                "Please omit --attention-backend."
            )
        else:
            pass
        if server_args.is_arg_explicitly_set(
            "component_attention_backends"
        ) and is_runtime_option_requested(server_args.component_attention_backends):
            raise ValueError(
                "SenseNovaU1Pipeline does not support component attention backends yet. "
                "Please omit --component-attention-backends."
            )
        else:
            pass
        if server_args.is_arg_explicitly_set(
            "attention_backend_config"
        ) and is_runtime_option_requested(server_args.attention_backend_config):
            raise ValueError(
                "SenseNovaU1Pipeline does not support attention backend config yet. "
                "Please omit --attention-backend-config."
            )
        else:
            pass

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            speed_mode_enable_torch_compile_by_default=False,
            keep_resident_min_available_gb=80,
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
        )
