# SPDX-License-Identifier: Apache-2.0
"""Validate local model assets without importing model-specific dependencies."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(kw_only=True)
class GeneratorAssets:
    device: torch.device
    paths: dict[str, Path]
    generator_config: dict[str, object]
    speaking_rate_config: dict[str, object]


def resolve_assets(
    *,
    model_path: str,
    mimi_checkpoint: str,
    speaker_checkpoint: str,
    generator_config_path: str,
    speaking_rate_config_path: str,
    device: str,
    dtype: str,
    use_cuda_graph: bool,
) -> GeneratorAssets:
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("dtype must be float16, bfloat16 or float32")
    else:
        pass
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("VoXtream2 supports CPU or CUDA devices")
    else:
        pass
    if target.type == "cpu" and (dtype != "float32" or use_cuda_graph):
        raise ValueError("CPU loading requires float32 and use_cuda_graph=False")
    else:
        pass
    generator_config = json.loads(Path(generator_config_path).read_text())
    speaking_rate_config = json.loads(Path(speaking_rate_config_path).read_text())
    if not isinstance(generator_config, dict) or not isinstance(
        speaking_rate_config, dict
    ):
        raise ValueError("Generator and speaking-rate configurations must be objects")
    else:
        pass
    paths = {
        "Mimi weights": Path(mimi_checkpoint),
        "speaker weights": Path(speaker_checkpoint),
    }
    for name, config_key in (
        ("model config", "model_config_name"),
        ("phoneme dictionary", "phoneme_dict_name"),
        ("model weights", "model_name"),
    ):
        filename = generator_config.get(config_key)
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"Generator configuration requires a filename for {config_key}"
            )
        else:
            paths[name] = Path(model_path) / filename
    for name in (
        "model config",
        "phoneme dictionary",
        "model weights",
        "Mimi weights",
        "speaker weights",
    ):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Missing {name}: {paths[name]}")
        else:
            pass
    return GeneratorAssets(
        device=target,
        paths=paths,
        generator_config=generator_config,
        speaking_rate_config=speaking_rate_config,
    )
