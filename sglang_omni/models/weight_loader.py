# SPDX-License-Identifier: Apache-2.0
"""Direct weight loading utilities for split model components."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers.utils.hub import cached_file


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        # Default to BF16 to avoid unintentionally loading FP32 weights.
        return torch.bfloat16
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype string: {dtype}")
    return mapping[key]


@lru_cache(maxsize=4)
def resolve_model_path(model_path: str, *, local_files_only: bool = False) -> Path:
    """Resolve a model_path to a local path, downloading if needed."""
    path = Path(model_path)
    if path.exists():
        return path
    if local_files_only:
        config_path = cached_file(model_path, "config.json", local_files_only=True)
        return Path(config_path).parent
    return Path(snapshot_download(model_path, local_files_only=False))


def _load_bin_shard(path: str) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def _read_safetensors_keys(path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    state_dict: dict[str, torch.Tensor] = {}
    if not keys:
        return state_dict
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in keys:
            state_dict[key] = f.get_tensor(key)
    return state_dict


def _load_safetensors_sharded(model_path: Path, prefix: str) -> dict[str, torch.Tensor]:
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        return {}

    with index_file.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shards.setdefault(shard, []).append(key)

    state_dict: dict[str, torch.Tensor] = {}
    for shard, keys in shards.items():
        shard_path = model_path / shard
        shard_weights = _read_safetensors_keys(shard_path, keys)
        for key, tensor in shard_weights.items():
            new_key = key[len(prefix) :]
            state_dict[new_key] = tensor
    return state_dict


def _load_safetensors_single(model_path: Path, prefix: str) -> dict[str, torch.Tensor]:
    single = model_path / "model.safetensors"
    if not single.exists():
        return {}

    from safetensors import safe_open

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(single), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state_dict[key[len(prefix) :]] = f.get_tensor(key)
    return state_dict


def _load_bin_sharded(model_path: Path, prefix: str) -> dict[str, torch.Tensor]:
    index_file = model_path / "pytorch_model.bin.index.json"
    if not index_file.exists():
        return {}

    with index_file.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix):
            shards.setdefault(shard, []).append(key)

    state_dict: dict[str, torch.Tensor] = {}
    for shard, keys in shards.items():
        shard_weights = _load_bin_shard(str(model_path / shard))
        for key in keys:
            new_key = key[len(prefix) :]
            state_dict[new_key] = shard_weights[key]
    return state_dict


def _load_bin_single(model_path: Path, prefix: str) -> dict[str, torch.Tensor]:
    single = model_path / "pytorch_model.bin"
    if not single.exists():
        return {}

    all_weights = _load_bin_shard(str(single))
    return {k[len(prefix) :]: v for k, v in all_weights.items() if k.startswith(prefix)}


def _normalize_prefixes(prefixes: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(prefixes, str):
        return (prefixes,)
    return tuple(prefixes)


def _load_weights_from_resolved_path(
    model_path: Path, prefixes: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Return the first matching state_dict, or {} if no prefix matches.

    An empty dict here means "no weights found for the requested prefixes" for
    this resolved path. Exceptions from missing/corrupted shard reads are
    allowed to propagate so the caller can decide whether to retry with a
    refreshed remote snapshot.
    """
    for prefix_item in prefixes:
        state_dict = _load_safetensors_sharded(model_path, prefix_item)
        if state_dict:
            return state_dict
        state_dict = _load_safetensors_single(model_path, prefix_item)
        if state_dict:
            return state_dict
        state_dict = _load_bin_sharded(model_path, prefix_item)
        if state_dict:
            return state_dict
        state_dict = _load_bin_single(model_path, prefix_item)
        if state_dict:
            return state_dict
    return {}


def _should_retry_remote_weight_load(
    *,
    model_path: str,
    local_files_only: bool,
) -> bool:
    return not local_files_only and not Path(model_path).exists()


def load_weights_by_prefix(
    model_path: str,
    *,
    prefix: str | tuple[str, ...] | list[str],
    local_files_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Load weights matching one of the prefixes, stripping the matched prefix."""
    resolved_model_path = resolve_model_path(
        model_path, local_files_only=local_files_only
    )
    prefixes = _normalize_prefixes(prefix)
    should_retry_remote_load = _should_retry_remote_weight_load(
        model_path=model_path,
        local_files_only=local_files_only,
    )

    try:
        state_dict = _load_weights_from_resolved_path(resolved_model_path, prefixes)
    except Exception:
        if not should_retry_remote_load:
            raise
        state_dict = {}

    if state_dict:
        return state_dict

    # A poisoned/partial HF cache can still yield a snapshot path that is missing
    # weight index files or shards. Refresh once before failing for remote models.
    if should_retry_remote_load:
        resolve_model_path.cache_clear()
        resolved_model_path = Path(
            snapshot_download(model_path, local_files_only=False, force_download=True)
        )
        state_dict = _load_weights_from_resolved_path(resolved_model_path, prefixes)
        if state_dict:
            return state_dict

    raise FileNotFoundError(
        f"No weights found for prefixes {list(prefixes)!r} under {resolved_model_path}"
    )


def load_module(
    module: nn.Module,
    model_path: str,
    *,
    prefix: str | tuple[str, ...] | list[str],
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
    local_files_only: bool = False,
) -> nn.Module:
    """Load weights into module by prefix, optionally move to device."""
    state_dict = load_weights_by_prefix(
        model_path,
        prefix=prefix,
        local_files_only=local_files_only,
    )
    # Prefer assign=True to avoid expensive in-place tensor copies during load.
    try:
        module.load_state_dict(state_dict, strict=strict, assign=True)
    except (TypeError, RuntimeError):
        module.load_state_dict(state_dict, strict=strict)
    module.eval()
    if device is not None or dtype is not None:
        if device is not None and dtype is not None:
            module = module.to(device=device, dtype=dtype)
        elif device is not None:
            module = module.to(device=device)
        else:
            module = module.to(dtype=dtype)
    return module


# picked from sglang.srt.model_loader.weight_utils.py
def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise
