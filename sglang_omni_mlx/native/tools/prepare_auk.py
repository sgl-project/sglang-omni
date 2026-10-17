# SPDX-License-Identifier: MIT AND Apache-2.0
# Copyright (C) 2026 Tencent. All rights reserved.
# Copyright (c) 2022 NVIDIA CORPORATION.
"""Prepare an unquantized AuK native artifact from local official checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

import mlx.core as mx
from omegaconf import DictConfig, OmegaConf

WEIGHT_PREFIXES = ("ema_model.", "module.", "model.", "net.")
CONDITIONER_PREFIXES = ("thinker.model.", "thinker.audio_tower.")
FLASH_TIME_GRID = [
    0.0,
    0.07612049579620361,
    0.2928932309150696,
    0.6173166036605835,
    1.0,
]


class FileDigest(TypedDict):
    path: str
    size: int
    sha256: str


class SourceManifest(TypedDict):
    revision: str | None
    files: list[FileDigest]


class ArtifactManifest(TypedDict):
    format_version: int
    mlx_version: str
    sources: dict[str, SourceManifest]
    outputs: list[FileDigest]


def file_digest(path: Path, directory: Path) -> FileDigest:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.relative_to(directory).as_posix(),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def write_json(path: Path, value: DictConfig | ArtifactManifest) -> None:
    content = (
        OmegaConf.to_container(value, resolve=True)
        if isinstance(value, DictConfig)
        else value
    )
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


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
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
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


def verify_reuse(output: Path, sources: dict[str, SourceManifest]) -> None:
    manifest_path = output / "manifest.json"
    manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_version") != 2
        or manifest.get("mlx_version") != version("mlx")
    ):
        raise ValueError(f"Incompatible artifact manifest in {output}")
    else:
        recorded_sources = manifest.get("sources")
    if (
        not isinstance(recorded_sources, dict)
        or recorded_sources.keys() != sources.keys()
    ):
        raise ValueError(f"Artifact source roles differ in {output}")
    else:
        pass
    for role, source in sources.items():
        recorded = recorded_sources[role]
        if not isinstance(recorded, dict) or recorded.get("files") != source["files"]:
            raise ValueError(
                f"Artifact {role} source files changed; refusing reuse of {output}"
            )
        elif (
            source["revision"] is not None
            and recorded.get("revision") != source["revision"]
        ):
            raise ValueError(
                f"Artifact {role} revision changed; refusing reuse of {output}"
            )
        else:
            pass
    actual_outputs = [
        file_digest(path, output)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    if manifest.get("outputs") != actual_outputs:
        raise ValueError(f"Artifact output files changed; refusing reuse of {output}")
    else:
        pass


def prepare(
    auk_directory: Path,
    conditioner_directory: Path,
    output: Path,
    auk_revision: str | None,
    conditioner_revision: str | None,
    *,
    reuse: bool = False,
) -> None:
    if (output.exists() or output.is_symlink()) and not reuse:
        raise FileExistsError(f"Refusing to overwrite existing output {output}")
    elif version("mlx") != "0.32.3":
        raise RuntimeError("AuK artifact preparation requires mlx==0.32.3")
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
    conditioner_files = conditioner_weights + conditioner_metadata
    conditioner_index = conditioner_directory / "model.safetensors.index.json"
    if conditioner_index.is_file():
        conditioner_files.append(conditioner_index)
    else:
        pass
    sources: dict[str, SourceManifest] = {
        "auk": {
            "revision": auk_revision,
            "files": [
                file_digest(path, auk_directory)
                for path in [config_path, weight_source, vae_source]
            ],
        },
        "conditioner": {
            "revision": conditioner_revision,
            "files": [
                file_digest(path, conditioner_directory)
                for path in sorted(conditioner_files)
            ],
        },
    }
    if output.exists() or output.is_symlink():
        verify_reuse(output, sources)
        return
    else:
        pass
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary:
        directory = Path(temporary)
        for component in ("dit", "vae", "conditioner"):
            (directory / component).mkdir()
        write_json(directory / "config.json", runtime_config)
        prepare_diffusion(
            weight_source, directory / "dit", architecture, int(vae.latent_dim)
        )
        prepare_vae(vae_source, directory / "vae", vae.model_init_kwargs)
        prepare_conditioner(
            conditioner_weights, conditioner_metadata, directory / "conditioner"
        )
        manifest: ArtifactManifest = {
            "format_version": 2,
            "mlx_version": version("mlx"),
            "sources": sources,
            "outputs": [
                file_digest(path, directory)
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            ],
        }
        write_json(directory / "manifest.json", manifest)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output was created during preparation: {output}")
        else:
            directory.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auk-directory", type=Path, required=True)
    parser.add_argument("--conditioner-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse an existing artifact only after source and output hash verification",
    )
    parser.add_argument(
        "--auk-revision", help="Source revision, if known; recorded verbatim"
    )
    parser.add_argument(
        "--conditioner-revision", help="Source revision, if known; recorded verbatim"
    )
    arguments = parser.parse_args()
    mx.set_default_device(mx.cpu)
    prepare(
        arguments.auk_directory,
        arguments.conditioner_directory,
        arguments.output,
        arguments.auk_revision,
        arguments.conditioner_revision,
        reuse=arguments.reuse,
    )
    print(f"Artifact ready: {arguments.output}")


if __name__ == "__main__":
    main()
