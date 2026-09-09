# SPDX-License-Identifier: Apache-2.0
"""Strict Apple Silicon runtime policy for Qwen3-Omni."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sglang_omni.platforms import current_platform

_ARCHITECTURE = "Qwen3OmniMoeForConditionalGeneration"
MPS_QUANTIZATION_ENV = "SGLANG_QWEN3_OMNI_MPS_QUANTIZATION"
_TORCH_MPS_CHECKPOINT_ERROR = (
    "Qwen3-Omni Torch MPS requires a dense Hugging Face checkpoint, such as "
    "Qwen/Qwen3-Omni-30B-A3B-Instruct, unless native MPS quantization is enabled. "
    f"For MLX affine weights, set {MPS_QUANTIZATION_ENV}=int4 or int8, "
    "or select the native MLX backend with SGLANG_USE_MLX=1."
)


def qwen3_omni_uses_mlx_backend() -> bool:
    from sglang.srt.utils.tensor_bridge import use_mlx

    return bool(use_mlx())


def qwen3_omni_uses_apple_backend() -> bool:
    return bool(current_platform.is_mps())


def get_qwen3_omni_mps_quantization(*, use_mlx: bool | None = None) -> int | None:
    value = os.environ.get(MPS_QUANTIZATION_ENV)
    if value is None:
        return None
    if value not in ("int4", "int8"):
        raise ValueError(f"{MPS_QUANTIZATION_ENV} must be int4 or int8; got {value!r}")
    if not qwen3_omni_uses_apple_backend():
        raise ValueError(f"{MPS_QUANTIZATION_ENV} requires the Apple MPS backend")
    if use_mlx is None:
        use_mlx = qwen3_omni_uses_mlx_backend()
    if use_mlx:
        raise ValueError(
            f"{MPS_QUANTIZATION_ENV} selects Torch MPS; unset SGLANG_USE_MLX"
        )
    return 4 if value == "int4" else 8


def _reject_explicit_override(
    explicit_overrides: Mapping[str, Any],
    *,
    key: str,
    supported: tuple[Any, ...],
    stage_name: str,
) -> None:
    if key not in explicit_overrides:
        return
    value = explicit_overrides[key]
    if value not in supported:
        raise ValueError(
            f"Apple Qwen3-Omni {stage_name} does not support {key}={value!r}"
        )


def apply_qwen3_omni_apple_profile(
    overrides: dict[str, Any],
    *,
    explicit_overrides: Mapping[str, Any],
    stage_name: str,
) -> dict[str, Any]:
    if not qwen3_omni_uses_apple_backend():
        return dict(overrides)
    if int(overrides.get("tp_size", 1)) != 1:
        raise ValueError(f"Apple Qwen3-Omni {stage_name} requires tp_size=1")

    supported_explicit = {
        "max_running_requests": (None, 1),
        "disable_cuda_graph": (None, True),
        "disable_decode_cuda_graph": (None, True),
        "disable_overlap_schedule": (None, True),
        "disable_radix_cache": (None, True),
        "enable_torch_compile": (None, False),
        "enable_mixed_chunk": (None, False),
        "chunked_prefill_size": (None, -1),
        "sampling_backend": (None, "pytorch"),
    }
    for key, supported in supported_explicit.items():
        _reject_explicit_override(
            explicit_overrides,
            key=key,
            supported=supported,
            stage_name=stage_name,
        )

    result = dict(overrides)
    result.update(
        max_running_requests=1,
        disable_cuda_graph=True,
        disable_decode_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        enable_torch_compile=False,
        enable_mixed_chunk=False,
        chunked_prefill_size=-1,
        sampling_backend="pytorch",
    )
    return result


def _validate_option(
    params: Mapping[str, Any],
    *,
    key: str,
    allowed: tuple[Any, ...],
    stage_name: str,
) -> None:
    value = params.get(key)
    if value not in allowed:
        raise ValueError(
            f"Apple Qwen3-Omni {stage_name} requires greedy generation; "
            f"unsupported {key}={value!r}"
        )


def validate_qwen3_omni_apple_request(
    params: Mapping[str, Any],
    *,
    stage_name: str,
) -> None:
    if not qwen3_omni_uses_apple_backend():
        return

    if stage_name == "thinker":
        options = {
            "temperature": (None, 0, 0.0),
            "top_p": (None, 1, 1.0),
            "top_k": (None, -1),
            "min_p": (None, 0, 0.0),
            "repetition_penalty": (None, 1, 1.0),
        }
        for key, allowed in options.items():
            _validate_option(
                params,
                key=key,
                allowed=allowed,
                stage_name=stage_name,
            )
        if params.get("return_logprob"):
            raise ValueError(
                "Apple Qwen3-Omni thinker does not support return_logprob=True"
            )
        return

    if stage_name == "talker_ar":
        options = {
            "talker_temperature": (None, 0, 0.0),
            "talker_top_p": (None, 1, 1.0),
            "talker_top_k": (None, -1),
            "talker_repetition_penalty": (None, 1, 1.0),
        }
        for key, allowed in options.items():
            _validate_option(
                params,
                key=key,
                allowed=allowed,
                stage_name=stage_name,
            )
        return

    raise ValueError(f"Unknown Qwen3-Omni Apple stage {stage_name!r}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read Qwen3-Omni checkpoint metadata: {path}"
        ) from exc


def _resolve_metadata_paths(model_path: str) -> tuple[Path, list[Path]]:
    local_path = Path(model_path)
    if local_path.exists():
        config_path = local_path / "config.json"
        indexes = sorted(local_path.rglob("*.safetensors.index.json"))
        return config_path, indexes

    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.errors import HfHubHTTPError

    try:
        config_path = Path(hf_hub_download(model_path, "config.json"))
        index_names = sorted(
            file_name
            for file_name in list_repo_files(model_path)
            if file_name.endswith(".safetensors.index.json")
        )
        index_paths = [
            Path(hf_hub_download(model_path, index_name)) for index_name in index_names
        ]
    except (HfHubHTTPError, OSError) as exc:
        raise ValueError(
            f"Unable to read Qwen3-Omni checkpoint metadata for {model_path!r}"
        ) from exc
    if not index_paths:
        raise ValueError(
            f"Qwen3-Omni checkpoint {model_path!r} must expose a safetensors index"
        )
    return config_path, index_paths


def _components_for_key(
    key: str,
    *,
    component_dir: str | None,
    use_mlx: bool,
    allow_packed: bool = False,
) -> set[str]:
    if not (use_mlx or allow_packed) and key.endswith((".scales", ".biases")):
        raise ValueError(
            f"{_TORCH_MPS_CHECKPOINT_ERROR} Found quantized tensor metadata {key!r}."
        )
    root_prefixes = {
        "thinker": (
            "thinker.model.",
            "thinker.lm_head.",
            "thinker.language_model.model.",
            "thinker.language_model.lm_head.",
        ),
        "talker": (
            "talker.model.",
            "talker.codec_head.",
            "talker.text_projection.",
            "talker.hidden_projection.",
        ),
        "predictor": ("talker.code_predictor.",),
        "code2wav": ("code2wav.",),
    }
    root_components = {
        component
        for component, prefixes in root_prefixes.items()
        if key.startswith(prefixes)
    }
    if component_dir is None:
        return root_components

    local_prefixes = {
        "thinker": ("model.", "lm_head."),
        "talker": (
            "model.",
            "codec_head.",
            "text_projection.",
            "hidden_projection.",
        ),
        "predictor": ("code_predictor.",),
        "code2wav": (
            "pre_transformer.",
            "code_embedding.",
            "upsample.",
            "decoder.",
        ),
    }
    if component_dir == "code2wav":
        if key.startswith(local_prefixes["code2wav"]):
            return {"code2wav"}
        return set()
    if not use_mlx:
        return set()
    if component_dir == "talker":
        return {
            component
            for component in ("talker", "predictor")
            if key.startswith(local_prefixes[component])
        }
    if component_dir == "thinker" and key.startswith(local_prefixes["thinker"]):
        return {component_dir}
    return set()


def _components_from_indexes(
    root: Path,
    index_paths: list[Path],
    *,
    use_mlx: bool,
    allow_packed: bool = False,
) -> set[str]:
    components: set[str] = set()
    for index_path in index_paths:
        weight_map = _read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Missing weight_map in checkpoint index {index_path}")
        relative_parts: tuple[str, ...] = ()
        try:
            relative_parts = index_path.relative_to(root).parts
        except ValueError:
            pass
        component_dir = relative_parts[0] if len(relative_parts) > 1 else None
        for key in weight_map:
            components.update(
                _components_for_key(
                    str(key),
                    component_dir=component_dir,
                    use_mlx=use_mlx,
                    allow_packed=allow_packed,
                )
            )
    return components


def _components_from_single_files(
    root: Path,
    *,
    use_mlx: bool,
    index_paths: Sequence[Path] = (),
    allow_packed: bool = False,
) -> set[str]:
    from safetensors import safe_open

    components: set[str] = set()
    indexed_directories = {index_path.parent.resolve() for index_path in index_paths}
    for shard_path in sorted(root.rglob("*.safetensors")):
        if not shard_path.is_file():
            continue
        if shard_path.parent.resolve() in indexed_directories:
            continue
        relative_parts = shard_path.relative_to(root).parts
        component_dir = relative_parts[0] if len(relative_parts) > 1 else None
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                components.update(
                    _components_for_key(
                        key,
                        component_dir=component_dir,
                        use_mlx=use_mlx,
                        allow_packed=allow_packed,
                    )
                )
    return components


def validate_qwen3_omni_apple_checkpoint(
    model_path: str,
    *,
    speech_enabled: bool,
    use_mlx: bool,
) -> None:
    bits = get_qwen3_omni_mps_quantization(use_mlx=use_mlx)
    if not qwen3_omni_uses_apple_backend():
        return

    config_path, index_paths = _resolve_metadata_paths(model_path)
    config = _read_json(config_path)
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or _ARCHITECTURE not in architectures:
        raise ValueError(
            f"Qwen3-Omni checkpoint architecture must include {_ARCHITECTURE!r}; "
            f"got {architectures!r}"
        )
    quantizations = [
        config[key]
        for key in ("quantization", "quantization_config")
        if config.get(key) is not None
    ]
    quantization = quantizations[0] if quantizations else None
    if not use_mlx and bits is None and quantization:
        raise ValueError(f"{_TORCH_MPS_CHECKPOINT_ERROR} Checkpoint: {model_path!r}.")
    if bits is not None and quantizations:
        if (
            not isinstance(quantization, dict)
            or any(item != quantization for item in quantizations)
            or set(quantization) - {"bits", "group_size", "mode"}
            or quantization.get("bits") not in (4, 8)
            or quantization.get("mode", "affine") != "affine"
            or quantization.get("group_size") not in (32, 64, 128, 256)
            or quantization.get("quant_method") is not None
        ):
            raise ValueError(
                "Torch MPS quantization supports dense HF or MLX affine 4/8-bit "
                "checkpoints; AWQ/compressed-tensors and GPTQ formats are not supported"
            )

    root = config_path.parent
    components = _components_from_indexes(
        root, index_paths, use_mlx=use_mlx, allow_packed=bits is not None
    )
    if Path(model_path).exists():
        components.update(
            _components_from_single_files(
                root,
                use_mlx=use_mlx,
                index_paths=index_paths,
                allow_packed=bits is not None,
            )
        )

    required = {"thinker"}
    if speech_enabled:
        required.update(("talker", "predictor", "code2wav"))
    missing = sorted(required - components)
    if missing:
        raise ValueError(
            f"Qwen3-Omni checkpoint {model_path!r} is missing required "
            f"{', '.join(missing)} weights"
        )
