# SPDX-License-Identifier: Apache-2.0
"""Strict loading of official Ming weights into native MLX components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


def checkpoint_files(model_path: str | Path) -> list[Path]:
    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if index.exists():
        filenames = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in filenames):
            raise ValueError("Checkpoint shard paths must remain within the model directory")
        # HF snapshots use symlinks to sibling blobs; keep those paths intact.
        return [root / name for name in filenames]
    path = root / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"No safetensors checkpoint found in {root}")
    return [path]


def read_config(model_path: str | Path) -> dict[str, Any]:
    return json.loads((Path(model_path) / "config.json").read_text())


def load_component_weights(
    model_path: str | Path, *, component: Literal["ar", "audio"]
) -> dict[str, Any]:
    import mlx.core as mx

    weights: dict[str, Any] = {}
    for path in checkpoint_files(model_path):
        for key, value in mx.load(str(path)).items():
            if key.startswith("audio.") != (component == "audio"):
                continue
            target = key.removeprefix("audio.") if component == "audio" else key
            if target in weights:
                raise ValueError(f"Duplicate checkpoint tensor: {target}")
            weights[target] = value
    return weights


def load_ming_tts_model(
    model_path: str | Path, *, quantization: str | None = None
) -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    from .config import ModelConfig
    from .model import MingTTSModel

    raw = read_config(model_path)
    if raw.get("quantization") or raw.get("quantization_config"):
        raise ValueError(
            "Ming MLX currently loads official unquantized checkpoints; use "
            "quantization=mlx_q4 or mlx_q8 for backbone-only on-load quantization"
        )
    if quantization not in (None, "mlx_q4", "mlx_q8"):
        raise ValueError("Ming MLX quantization must be mlx_q4 or mlx_q8")
    model = MingTTSModel(ModelConfig.from_dict(raw))
    weights = model.sanitize(load_component_weights(model_path, component="ar"))
    model.load_weights(list(weights.items()), strict=True)
    del weights
    if quantization is not None:
        # Router scores and all acoustic modules retain checkpoint precision.
        nn.quantize(
            model.model,
            bits=4 if quantization == "mlx_q4" else 8,
            group_size=64,
            class_predicate=lambda path, module: hasattr(module, "to_quantized")
            and path.rsplit(".", 1)[-1] not in {
                "gate", "image_gate", "audio_gate", "word_embeddings"
            },
        )
    model.eval()
    mx.eval(model.parameters())
    return model


def load_ming_audio_vae(
    model_path: str | Path, *, component: Literal["encoder", "decoder"]
) -> Any:
    import mlx.core as mx

    from .audio_vae import AudioVAE

    raw = read_config(model_path)
    if raw.get("quantization") or raw.get("quantization_config"):
        raise ValueError("Ming AudioVAE requires an official unquantized checkpoint")
    model = AudioVAE(raw["audio_tokenizer_config"], component=component)
    weights = load_component_weights(model_path, component="audio")
    weights = {
        key: value for key, value in weights.items()
        if key.startswith(component + ".")
    }
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
