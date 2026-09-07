# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for checkpoints produced by MLX-VLM."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from safetensors import safe_open

from sglang_omni.models.weight_loader import resolve_model_path

_MLX_VLM_THINKER_PREFIX = "thinker.language_model."
_SIDECAR_PROVENANCE_CACHE: set[tuple[tuple[str, int, int], ...]] = set()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_has_root_code2wav_weights(root: Path) -> bool:
    """Return whether the root checkpoint owns namespaced code2wav weights."""

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))[
                "weight_map"
            ]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Unable to read checkpoint index {index_path}") from exc
        return any(str(key).startswith("code2wav.") for key in weight_map)

    single = root / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return any(key.startswith("code2wav.") for key in handle.keys())
    return False


def _provenance_signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths
    )


def _resolve_source_shard(root: Path, shard_name: str) -> Path:
    source = (root / shard_name).resolve()
    try:
        source.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Code2wav manifest references a shard outside {root}: {shard_name!r}"
        ) from exc
    if not source.is_file():
        raise ValueError(f"Code2wav source shard does not exist: {source}")
    return source


def validate_code2wav_sidecar_provenance(root: Path) -> None:
    """Require a generated sidecar to identify its exact source tensors."""

    sidecar = root / "code2wav"
    manifest_path = sidecar / "conversion.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"Code2wav sidecar {sidecar} has no conversion.json; rerun the "
            "checkpoint preparation command"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read code2wav manifest {manifest_path}") from exc

    root_index = root / "model.safetensors.index.json"
    root_model = root / "model.safetensors"
    raw_shard_hashes: dict[str, str] = {}
    if root_index.is_file():
        source_path = root_index
        manifest_key = "source_index_sha256"
        try:
            weight_map = json.loads(root_index.read_text(encoding="utf-8"))[
                "weight_map"
            ]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Unable to read checkpoint index {root_index}") from exc
        expected_shards = {
            str(shard_name)
            for key, shard_name in weight_map.items()
            if str(key).startswith("code2wav.")
        }
        raw_hashes = manifest.get("source_shards_sha256")
        if not isinstance(raw_hashes, dict) or set(raw_hashes) != expected_shards:
            raise ValueError(
                f"Code2wav sidecar {sidecar} does not identify every source "
                "code2wav shard; rerun the checkpoint preparation command"
            )
        raw_shard_hashes = {
            str(shard_name): str(shard_hash)
            for shard_name, shard_hash in raw_hashes.items()
        }
        source_shards = {
            shard_name: _resolve_source_shard(root, shard_name)
            for shard_name in sorted(expected_shards)
        }
    elif root_model.is_file():
        source_path = root_model
        manifest_key = "source_model_sha256"
        source_shards = {}
    else:
        raise ValueError(
            f"Code2wav sidecar {sidecar} has no source checkpoint file to verify"
        )

    signature = _provenance_signature(
        [manifest_path, source_path, *source_shards.values()]
    )
    if signature in _SIDECAR_PROVENANCE_CACHE:
        return

    expected_hash = _sha256_file(source_path)
    if manifest.get(manifest_key) != expected_hash:
        raise ValueError(
            f"Code2wav sidecar {sidecar} does not match {source_path}; rerun "
            "the checkpoint preparation command"
        )
    for shard_name, shard_path in source_shards.items():
        if raw_shard_hashes.get(shard_name) != _sha256_file(shard_path):
            raise ValueError(
                f"Code2wav sidecar {sidecar} does not match {shard_path}; rerun "
                "the checkpoint preparation command"
            )
    _SIDECAR_PROVENANCE_CACHE.add(signature)


@lru_cache(maxsize=4)
def checkpoint_uses_mlx_vlm_layout(model_path: str) -> bool:
    """Return whether a checkpoint uses MLX-VLM's Qwen3-Omni namespaces."""

    root = resolve_model_path(model_path)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Unable to read checkpoint index {index_path}") from exc
        return any(str(key).startswith(_MLX_VLM_THINKER_PREFIX) for key in weight_map)

    single = root / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return any(key.startswith(_MLX_VLM_THINKER_PREFIX) for key in handle.keys())
    return False


def restore_torch_convolution_layout(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Move MLX convolution input channels back to Torch's second dimension."""

    expected = module.state_dict()
    restored: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        expected_tensor = expected.get(key)
        if (
            expected_tensor is None
            or tensor.ndim not in (3, 4, 5)
            or tuple(tensor.shape) == tuple(expected_tensor.shape)
        ):
            restored[key] = tensor
            continue

        permutation = (0, tensor.ndim - 1, *range(1, tensor.ndim - 1))
        candidate = tensor.permute(permutation)
        if tuple(candidate.shape) == tuple(expected_tensor.shape):
            restored[key] = candidate.contiguous()
        else:
            restored[key] = tensor
    return restored


def restore_torch_mlx_vlm_layout(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Restore MLX-VLM vision names and convolution layouts for Torch."""

    renamed: dict[str, torch.Tensor] = {}
    for source_key, tensor in state_dict.items():
        key = source_key
        if key.startswith("deepstack_merger_list."):
            key = f"merger_list.{key[len('deepstack_merger_list.') :]}"
        if key.startswith(("merger.", "merger_list.")):
            key = key.replace(".norm.", ".ln_q.")
            key = key.replace(".linear_fc1.", ".mlp.0.")
            key = key.replace(".linear_fc2.", ".mlp.2.")
        if key in renamed:
            raise ValueError(
                f"MLX-VLM vision weights {source_key!r} and another source "
                f"both map to {key!r}"
            )
        renamed[key] = tensor
    return restore_torch_convolution_layout(module, renamed)
