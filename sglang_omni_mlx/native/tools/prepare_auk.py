# SPDX-License-Identifier: MIT AND Apache-2.0
# Copyright (C) 2026 Tencent. All rights reserved.
# Copyright (c) 2022 NVIDIA CORPORATION.
"""Prepare an unquantized AuK native artifact from local official checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from omegaconf import DictConfig, OmegaConf

NATIVE_DIRECTORY = Path(__file__).resolve().parents[1]
WEIGHT_PREFIXES = ("ema_model.", "module.", "model.", "net.")
CONDITIONER_PREFIXES = ("thinker.model.", "thinker.audio_tower.")
FLASH_TIME_GRID = [
    0.0,
    0.07612049579620361,
    0.2928932309150696,
    0.6173166036605835,
    1.0,
]

# Copied into each artifact under the names the runtime installs them as.
ARTIFACT_LICENSES = {
    "NOTICE": NATIVE_DIRECTORY / "licenses/NOTICE",
    "qwen-research.txt": NATIVE_DIRECTORY / "licenses/qwen-research.txt",
    "auk.txt": NATIVE_DIRECTORY.parents[1] / "sglang_omni/models/auk/LICENSE",
    "Apache-2.0.txt": NATIVE_DIRECTORY.parents[1] / "LICENSE",
}


def pinned_mlx_version() -> str:
    lock = NATIVE_DIRECTORY / "scripts/build-tools.lock"
    pins = [
        line.removeprefix("mlx==")
        for line in lock.read_text().splitlines()
        if line.startswith("mlx==")
    ]
    if len(pins) != 1 or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", pins[0]):
        raise ValueError(f"Expected one exact MLX pin in {lock}")
    else:
        return pins[0]


def write_json(path: Path, value: DictConfig) -> None:
    path.write_text(
        json.dumps(OmegaConf.to_container(value, resolve=True), indent=2) + "\n",
        encoding="utf-8",
    )


def read_config(path: Path) -> DictConfig:
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected a configuration mapping in {path}")
    elif "quantization" in config or "quantization_config" in config:
        raise ValueError(f"Quantized checkpoints are unsupported: {path}")
    else:
        return config


def find_file(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
        else:
            pass
    raise FileNotFoundError(
        f"Expected one of {names} under {directory}; PyTorch .pt/.bin checkpoints are unsupported"
    )


def normalized_weights(path: Path) -> dict[str, mx.array]:
    weights = mx.load(str(path))
    normalized: dict[str, mx.array] = {}
    for name, value in weights.items():
        for prefix in WEIGHT_PREFIXES:
            if name.startswith(prefix):
                name = name.removeprefix(prefix)
                break
            else:
                pass
        if name in normalized:
            raise ValueError(f"Duplicate normalized tensor {name} in {path}")
        elif name.endswith((".scales", ".biases")) or not mx.issubdtype(
            value.dtype, mx.floating
        ):
            raise ValueError(f"Non-floating or quantized tensor {name} in {path}")
        else:
            normalized[name] = value
    return normalized


def prepare_diffusion(
    source: Path, directory: Path, architecture: DictConfig, latent_dimension: int
) -> None:
    weights = normalized_weights(source)
    for name, value in weights.items():
        if name.endswith(
            ("conv_pos_embed.conv1d.0.weight", "conv_pos_embed.conv1d.2.weight")
        ):
            weights[name] = value.transpose(0, 2, 1)
        else:
            pass
    architecture.setdefault("dim_head", 64)
    architecture.setdefault("attn_mask_enabled", True)
    write_json(
        directory / "config.json",
        OmegaConf.create({"arch": architecture, "latent_dim": latent_dimension}),
    )
    mx.save_safetensors(str(directory / "model.safetensors"), weights)


def prepare_vae(source: Path, directory: Path, config: DictConfig) -> None:
    weights = normalized_weights(source)
    converted: dict[str, mx.array] = {}
    for name, value in weights.items():
        if name.startswith("flow.") or name.endswith(".weight_g"):
            continue
        elif name.endswith(".weight_v"):
            name = name.removesuffix("_v")
            value = value.astype(mx.float32)
            magnitude = weights[f"{name}_g"].astype(mx.float32)
            value = value * (
                magnitude / mx.sqrt(mx.sum(value * value, axis=(1, 2), keepdims=True))
            )
        else:
            pass
        if name.endswith(".weight") and value.ndim == 3:
            value = value.transpose((1, 2, 0) if name.startswith("ups.") else (0, 2, 1))
        elif name.endswith(".filter"):
            value = value.transpose(0, 2, 1)
        else:
            pass
        converted[name] = value.astype(mx.float32)
    write_json(directory / "config.json", config)
    mx.save_safetensors(str(directory / "model.safetensors"), converted)


def conditioner_sources(directory: Path) -> tuple[list[Path], list[Path]]:
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_config(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, DictConfig):
            raise ValueError(f"Missing weight_map in {index_path}")
        else:
            filenames = sorted(
                {
                    filename
                    for name, filename in weight_map.items()
                    if name.startswith(CONDITIONER_PREFIXES)
                }
            )
        weight_files = []
        for filename in filenames:
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not filename.endswith(".safetensors")
            ):
                raise ValueError(f"Invalid checkpoint shard name in {index_path}")
            elif not (directory / filename).is_file():
                raise FileNotFoundError(directory / filename)
            else:
                weight_files.append(directory / filename)
    else:
        weight_files = [find_file(directory, ("model.safetensors",))]
    if not weight_files:
        raise ValueError(f"No Qwen text/audio tensors in {directory}")
    else:
        pass
    required_metadata_filenames = (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    for name in required_metadata_filenames:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
        else:
            pass
    metadata = sorted(
        directory / name
        for name in (
            *required_metadata_filenames,
            "added_tokens.json",
            "chat_template.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
        )
        if (directory / name).is_file()
    )
    return weight_files, metadata


def prepare_conditioner(
    weight_files: list[Path], metadata: list[Path], directory: Path
) -> None:
    weights: dict[str, mx.array] = {}
    for path in weight_files:
        for name, value in mx.load(str(path)).items():
            if not name.startswith(CONDITIONER_PREFIXES):
                continue
            elif name in weights:
                raise ValueError(f"Duplicate conditioner tensor {name}")
            elif name.endswith((".scales", ".biases")) or not mx.issubdtype(
                value.dtype, mx.floating
            ):
                raise ValueError(f"Non-floating or quantized conditioner tensor {name}")
            else:
                weights[name] = value
    if not all(
        any(name.startswith(prefix) for name in weights)
        for prefix in CONDITIONER_PREFIXES
    ):
        raise ValueError(
            "Conditioner requires both thinker.model and thinker.audio_tower tensors"
        )
    else:
        pass
    for path in metadata:
        shutil.copyfile(path, directory / path.name)
    # Note (XXXXRT666): the Qwen tensors keep their stored values and Torch
    # convolution layout; the runtime transposes the two convolutions itself.
    mx.save_safetensors(
        str(directory / "model.safetensors"),
        weights,
        metadata={
            "modification": "Qwen2.5-Omni-3B Thinker text/audio tensors extracted and repackaged for native serving.",
            "license": "Qwen Research License; see ../licenses/qwen-research.txt and ../licenses/NOTICE.",
        },
    )
    mx.save_safetensors(str(directory / "frontend.safetensors"), frontend_constants())


def frontend_constants() -> dict[str, mx.array]:
    """The 400-point periodic Hann and 128-bin Slaney filter bank used by AuK."""
    window = read_config(Path(__file__).with_name("auk_hann_window.json"))
    if window.n_fft != 400 or len(window.samples) != 400:
        raise ValueError("AuK requires its 400-point periodic Hann window")
    else:
        pass
    maximum_mel = 15.0 + math.log(8.0) * (27.0 / math.log(6.4))
    step_mel = maximum_mel / 129
    mel_edges = [index * step_mel for index in range(130)]
    mel_edges[-1] = maximum_mel
    frequencies_hz = [
        (
            200.0 * mel / 3.0
            if mel < 15.0
            else 1000.0 * math.exp(math.log(6.4) / 27.0 * (mel - 15.0))
        )
        for mel in mel_edges
    ]
    filters = []
    for frequency_hz in range(0, 8001, 40):
        row = []
        for band in range(128):
            low_hz, center_hz, high_hz = frequencies_hz[band : band + 3]
            row.append(
                max(
                    0.0,
                    min(
                        (frequency_hz - low_hz) / (center_hz - low_hz),
                        (high_hz - frequency_hz) / (high_hz - center_hz),
                    ),
                )
                * (2.0 / (high_hz - low_hz))
            )
        filters.append(row)
    return {
        "frontend.hann_window": mx.array(list(window.samples), dtype=mx.float32),
        "frontend.mel_filters": mx.array(filters, dtype=mx.float32),
    }


def prepare(auk_directory: Path, conditioner_directory: Path, output: Path) -> None:
    required_mlx_version = pinned_mlx_version()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output {output}")
    elif version("mlx") != required_mlx_version:
        raise RuntimeError(
            f"AuK artifact preparation requires mlx=={required_mlx_version}; found {version('mlx')}"
        )
    else:
        pass
    config_path = find_file(auk_directory, ("config.yaml", "config.yml", "config.json"))
    config = read_config(config_path)
    model = config.get("model", config)
    if not isinstance(model, DictConfig) or model.get("name") not in {
        "AuK",
        "AuK-Flash",
    }:
        raise ValueError("Expected an official AuK or AuK-Flash configuration")
    else:
        name = str(model.name)
    architecture, vae = model.get("arch"), model.get("vae")
    if (
        not isinstance(architecture, DictConfig)
        or not isinstance(vae, DictConfig)
        or not isinstance(vae.get("model_init_kwargs"), DictConfig)
    ):
        raise ValueError("AuK configuration requires arch and vae.model_init_kwargs")
    else:
        OmegaConf.resolve(config)
    read_config(conditioner_directory / "config.json")
    weight_source = find_file(
        auk_directory,
        (
            "auk_flash.safetensors" if name == "AuK-Flash" else "auk_base.safetensors",
            "auk.safetensors",
            "model.safetensors",
        ),
    )
    vae_source = find_file(
        auk_directory,
        ("vae.safetensors", "vae/vae.safetensors", "audio_vae.safetensors"),
    )
    conditioner_weights, conditioner_metadata = conditioner_sources(
        conditioner_directory
    )
    is_flash = name == "AuK-Flash"
    # Note (XXXXRT666): released recipes, matching sglang_omni/models/auk/constants.py.
    runtime_config = OmegaConf.create(
        {
            "model_type": "auk",
            "name": name,
            "sample_rate": int(vae.target_sample_rate),
            "downsample_rate": int(vae.downsample_rate),
            "latent_dim": int(vae.latent_dim),
            "nfe": 4 if is_flash else 32,
            "cfg_strength": 0.0 if is_flash else 2.0,
            "sway_sampling_coef": None if is_flash else -1.0,
            "time_grid": FLASH_TIME_GRID if is_flash else [],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent)
    )
    try:
        for component in ("dit", "vae", "conditioner", "licenses"):
            (directory / component).mkdir()
        write_json(directory / "config.json", runtime_config)
        prepare_diffusion(
            weight_source, directory / "dit", architecture, int(vae.latent_dim)
        )
        prepare_vae(vae_source, directory / "vae", vae.model_init_kwargs)
        prepare_conditioner(
            conditioner_weights, conditioner_metadata, directory / "conditioner"
        )
        for license_name, license_path in ARTIFACT_LICENSES.items():
            shutil.copyfile(license_path, directory / "licenses" / license_name)
    except BaseException:
        shutil.rmtree(directory)
        raise
    directory.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auk-directory", type=Path, required=True)
    parser.add_argument("--conditioner-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    mx.set_default_device(mx.cpu)
    prepare(arguments.auk_directory, arguments.conditioner_directory, arguments.output)
    print(f"Artifact ready: {arguments.output}")


if __name__ == "__main__":
    main()
