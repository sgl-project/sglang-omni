# SPDX-License-Identifier: Apache-2.0
"""Load split MiniMax Music 3 components from an MLX artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3 import ModelArgs as Qwen3Args

from .config import ModelConfig
from .depth import RVQDepthDecoder
from .dit import FlowMatchingTransformer
from .fusion import ConditionEncoder
from .vocoder import Vocoder


def _make_qwen3(config: ModelConfig) -> Qwen3Model:
    return Qwen3Model(
        Qwen3Args(
            model_type="qwen3",
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            rms_norm_eps=config.rms_norm_eps,
            vocab_size=config.vocab_size,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            head_dim=config.head_dim,
            tie_word_embeddings=config.tie_word_embeddings,
        )
    )


class MiniMaxMusic3MlxARModel(nn.Module):
    """Global Qwen3 generator plus the local RVQ depth decoder."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.language_model = _make_qwen3(config)
        self.rvq_depth_decoder = RVQDepthDecoder(config)

    @staticmethod
    def model_quant_predicate(path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if path.endswith("lm_head") or ".audio_heads." in path:
            return False
        return path.startswith(
            (
                "language_model.model.layers.",
                "rvq_depth_decoder.layers.",
            )
        )


class MiniMaxMusic3MlxAcousticModel(nn.Module):
    """Condition encoder, Flow/DiT, and stereo DAV decoder."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.condition_encoder = ConditionEncoder(config)
        self.transformer = FlowMatchingTransformer(config)
        self.vocoder = Vocoder(config)

    @staticmethod
    def model_quant_predicate(path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        return path.startswith(
            (
                "transformer.proj_in",
                "transformer.proj_out",
                "transformer.transformer_blocks.",
            )
        )


def _resolve_model_directory(model_path: str, revision: str | None) -> Path:
    local_path = Path(model_path).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    from sglang.srt.hardware_backend.mlx.remote_code_gate import (
        ensure_remote_code_allowed,
        resolve_model_directory,
    )

    model_dir = Path(resolve_model_directory(model_path, revision=revision))
    ensure_remote_code_allowed(model_dir, trust_remote_code=False)
    return model_dir


def _read_config(model_dir: Path) -> tuple[dict[str, Any], ModelConfig]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"MiniMax Music 3 MLX artifact is missing {config_path}"
        )
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = set(raw.get("architectures") or ())
    if architectures and "MiniMaxMusic3ForConditionalGeneration" not in architectures:
        raise ValueError(
            "MiniMax Music 3 MLX artifact has an incompatible architecture: "
            f"{sorted(architectures)}"
        )
    config = ModelConfig.from_dict(raw)
    config.model_path = str(model_dir)
    return raw, config


def _load_component_weights(
    model_dir: Path,
    prefixes: tuple[str, ...],
) -> dict[str, mx.array]:
    weight_files = sorted(model_dir.glob("*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_dir.glob("*.npz"))
    if not weight_files:
        raise FileNotFoundError(
            "MiniMax Music 3 native MLX requires a converted artifact with "
            f"safetensors or npz weights: {model_dir}"
        )

    selected: dict[str, mx.array] = {}
    for weight_file in weight_files:
        shard = mx.load(str(weight_file))
        for name, value in shard.items():
            if not name.startswith(prefixes):
                continue
            if name in selected:
                raise ValueError(f"duplicate MiniMax Music 3 MLX tensor {name!r}")
            selected[name] = value
    if not selected:
        raise ValueError(
            "MiniMax Music 3 MLX artifact has no weights matching "
            + ", ".join(prefixes)
        )
    return selected


def _apply_quantization(
    model: nn.Module,
    raw_config: dict[str, Any],
    weights: dict[str, mx.array],
) -> None:
    quantization = raw_config.get("quantization") or raw_config.get(
        "quantization_config"
    )
    if not isinstance(quantization, dict):
        return
    group_size = int(quantization.get("group_size", 64))
    predicate = getattr(model, "model_quant_predicate", None)

    def class_predicate(path: str, module: nn.Module):
        if not hasattr(module, "to_quantized"):
            return False
        if hasattr(module, "weight") and module.weight.shape[-1] % group_size:
            return False
        if predicate is not None and not predicate(path, module):
            return False
        if path in quantization:
            return quantization[path]
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=group_size,
        bits=int(quantization["bits"]),
        mode=str(quantization.get("mode", "affine")),
        class_predicate=class_predicate,
    )


def _load_split_model(
    model_path: str,
    revision: str | None,
    model_cls: type[nn.Module],
    prefixes: tuple[str, ...],
) -> nn.Module:
    model_dir = _resolve_model_directory(model_path, revision)
    raw_config, config = _read_config(model_dir)
    model = model_cls(config)
    weights = _load_component_weights(model_dir, prefixes)
    _apply_quantization(model, raw_config, weights)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def load_mlx_ar_model(
    model_path: str,
    revision: str | None = None,
) -> MiniMaxMusic3MlxARModel:
    return _load_split_model(
        model_path,
        revision,
        MiniMaxMusic3MlxARModel,
        ("language_model.", "rvq_depth_decoder."),
    )


def load_mlx_acoustic_model(
    model_path: str,
    revision: str | None = None,
) -> MiniMaxMusic3MlxAcousticModel:
    return _load_split_model(
        model_path,
        revision,
        MiniMaxMusic3MlxAcousticModel,
        ("condition_encoder.", "transformer.", "vocoder."),
    )


__all__ = [
    "MiniMaxMusic3MlxARModel",
    "MiniMaxMusic3MlxAcousticModel",
    "load_mlx_acoustic_model",
    "load_mlx_ar_model",
]
