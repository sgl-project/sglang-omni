# SPDX-License-Identifier: Apache-2.0
"""Direct weight loading utilities for split model components."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers.utils.hub import cached_file

_WEIGHT_CACHE: dict[tuple[str, str, str, tuple[str, ...]], dict[str, torch.Tensor]] = {}


def _maybe_fuse_qwen3_moe_experts(
    state_dict: dict[str, torch.Tensor],
    module: nn.Module,
) -> dict[str, torch.Tensor]:
    """Fuse per-expert gate/up/down weights into gate_up_proj/down_proj if needed.

    Newer Qwen3-Omni HF implementations expect fused expert tensors:
      - experts.gate_up_proj (num_experts, 2*intermediate, hidden)
      - experts.down_proj (num_experts, hidden, intermediate)

    Older checkpoints store per-expert Linear weights:
      - experts.{i}.gate_proj.weight
      - experts.{i}.up_proj.weight
      - experts.{i}.down_proj.weight
    """
    targets = [
        name
        for name, _ in module.named_parameters()
        if name.endswith("mlp.experts.gate_up_proj")
    ]
    if not targets:
        return state_dict

    for gate_up_key in targets:
        if gate_up_key in state_dict:
            continue
        prefix = gate_up_key[: -len(".gate_up_proj")]
        down_key = f"{prefix}.down_proj"
        if down_key in state_dict:
            continue

        prefix_dot = prefix + "."
        expert_indices: list[int] = []
        for key in state_dict.keys():
            if not key.startswith(prefix_dot):
                continue
            if not key.endswith(".gate_proj.weight"):
                continue
            idx_str = key[len(prefix_dot) :].split(".", 1)[0]
            if idx_str.isdigit():
                expert_indices.append(int(idx_str))

        if not expert_indices:
            continue
        expert_indices = sorted(set(expert_indices))

        gate_up_list: list[torch.Tensor] = []
        down_list: list[torch.Tensor] = []
        missing = False
        for idx in expert_indices:
            gate_key = f"{prefix}.{idx}.gate_proj.weight"
            up_key = f"{prefix}.{idx}.up_proj.weight"
            down_key_per = f"{prefix}.{idx}.down_proj.weight"
            if gate_key not in state_dict or up_key not in state_dict or down_key_per not in state_dict:
                missing = True
                break
            gate = state_dict[gate_key]
            up = state_dict[up_key]
            down = state_dict[down_key_per]
            gate_up_list.append(torch.cat([gate, up], dim=0))
            down_list.append(down)

        if missing:
            continue

        state_dict[gate_up_key] = torch.stack(gate_up_list, dim=0)
        state_dict[down_key] = torch.stack(down_list, dim=0)

        for idx in expert_indices:
            state_dict.pop(f"{prefix}.{idx}.gate_proj.weight", None)
            state_dict.pop(f"{prefix}.{idx}.up_proj.weight", None)
            state_dict.pop(f"{prefix}.{idx}.down_proj.weight", None)

    return state_dict

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
def resolve_model_path(model_id: str, *, local_files_only: bool = False) -> Path:
    """Resolve a model_id to a local path, downloading if needed."""
    path = Path(model_id)
    if path.exists():
        return path
    try:
        config_path = cached_file(model_id, "config.json", local_files_only=True)
        return Path(config_path).parent
    except Exception:
        if local_files_only:
            raise
    return Path(snapshot_download(model_id, local_files_only=local_files_only))


def _read_safetensors_keys(
    path: Path,
    keys: list[str],
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    state_dict: dict[str, torch.Tensor] = {}
    if not keys:
        return state_dict
    with safe_open(str(path), framework="pt", device=device) as f:
        for key in keys:
            tensor = f.get_tensor(key)
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            state_dict[key] = tensor
    return state_dict


def _load_safetensors_sharded(
    model_path: Path,
    prefix: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
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
        shard_weights = _read_safetensors_keys(
            shard_path, keys, device=device, dtype=dtype
        )
        for key, tensor in shard_weights.items():
            new_key = key[len(prefix) :]
            state_dict[new_key] = tensor
    return state_dict


def _load_safetensors_single(
    model_path: Path,
    prefix: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    single = model_path / "model.safetensors"
    if not single.exists():
        return {}

    from safetensors import safe_open

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(single), framework="pt", device=device) as f:
        for key in f.keys():
            if key.startswith(prefix):
                tensor = f.get_tensor(key)
                if dtype is not None and tensor.dtype != dtype:
                    tensor = tensor.to(dtype)
                state_dict[key[len(prefix) :]] = tensor
    return state_dict


def _normalize_prefixes(prefixes: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(prefixes, str):
        return (prefixes,)
    return tuple(prefixes)


def preload_weights(
    model_path: str | Path,
    *,
    prefix_groups: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    max_workers: int | None = None,
) -> None:
    """Preload safetensors for multiple prefix groups in parallel into a cache."""
    from safetensors import safe_open

    model_path = Path(model_path)
    groups = [tuple(g) for g in prefix_groups]
    index_file = model_path / "model.safetensors.index.json"

    if index_file.exists():
        with index_file.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        shards: dict[str, list[tuple[int, str, str]]] = {}
        for key, shard in weight_map.items():
            for idx, prefixes in enumerate(groups):
                for prefix in prefixes:
                    if key.startswith(prefix):
                        new_key = key[len(prefix) :]
                        shards.setdefault(shard, []).append((idx, key, new_key))
                        break
                else:
                    continue
                break

        if not shards:
            raise FileNotFoundError(
                f"No safetensors weights found for prefixes {groups!r} under {model_path}"
            )

        group_dicts: list[dict[str, torch.Tensor]] = [
            {} for _ in range(len(groups))
        ]

        def _load_shard(shard: str, entries: list[tuple[int, str, str]]):
            shard_path = model_path / shard
            out: dict[int, dict[str, torch.Tensor]] = {}
            with safe_open(str(shard_path), framework="pt", device=device) as f:
                for idx, original_key, new_key in entries:
                    tensor = f.get_tensor(original_key)
                    if dtype is not None and tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    out.setdefault(idx, {})[new_key] = tensor
            return out

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_load_shard, shard, entries)
                for shard, entries in shards.items()
            ]
            for future in as_completed(futures):
                shard_out = future.result()
                for idx, weights in shard_out.items():
                    group_dicts[idx].update(weights)

        # Each thread loaded tensors to GPU on its own CUDA stream.
        # Synchronize to ensure all H2D transfers are complete before
        # the main thread (on a different stream) uses these tensors.
        if device != "cpu":
            torch.cuda.synchronize()

        for idx, prefixes in enumerate(groups):
            if not group_dicts[idx]:
                raise FileNotFoundError(
                    f"No safetensors weights found for prefixes {prefixes!r} under {model_path}"
                )
            cache_key = (str(model_path), device, str(dtype), prefixes)
            _WEIGHT_CACHE[cache_key] = group_dicts[idx]
        return

    single = model_path / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(
            f"No safetensors weights found for prefixes {groups!r} under {model_path}"
        )

    group_dicts: list[dict[str, torch.Tensor]] = [{} for _ in range(len(groups))]
    with safe_open(str(single), framework="pt", device=device) as f:
        for key in f.keys():
            for idx, prefixes in enumerate(groups):
                for prefix in prefixes:
                    if key.startswith(prefix):
                        tensor = f.get_tensor(key)
                        if dtype is not None and tensor.dtype != dtype:
                            tensor = tensor.to(dtype)
                        group_dicts[idx][key[len(prefix) :]] = tensor
                        break
                else:
                    continue
                break

    for idx, prefixes in enumerate(groups):
        if not group_dicts[idx]:
            raise FileNotFoundError(
                f"No safetensors weights found for prefixes {prefixes!r} under {model_path}"
            )
        cache_key = (str(model_path), device, str(dtype), prefixes)
        _WEIGHT_CACHE[cache_key] = group_dicts[idx]

def load_weights_by_prefixes(
    model_path: str | Path,
    *,
    prefixes: str | tuple[str, ...] | list[str],
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Load safetensors weights matching any prefix, stripping the matched prefix."""
    from safetensors import safe_open

    model_path = Path(model_path)
    prefixes = _normalize_prefixes(prefixes)
    cache_key = (str(model_path), device, str(dtype), prefixes)
    cached = _WEIGHT_CACHE.pop(cache_key, None)
    if cached is not None:
        return cached
    index_file = model_path / "model.safetensors.index.json"

    if index_file.exists():
        with index_file.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        shards: dict[str, list[tuple[str, str]]] = {}
        for key, shard in weight_map.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    new_key = key[len(prefix) :]
                    shards.setdefault(shard, []).append((key, new_key))
                    break

        if not shards:
            raise FileNotFoundError(
                f"No safetensors weights found for prefixes {list(prefixes)!r} under {model_path}"
            )

        state_dict: dict[str, torch.Tensor] = {}
        for shard, entries in shards.items():
            shard_path = model_path / shard
            with safe_open(str(shard_path), framework="pt", device=device) as f:
                for original_key, new_key in entries:
                    tensor = f.get_tensor(original_key)
                    if dtype is not None and tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    state_dict[new_key] = tensor
        return state_dict

    single = model_path / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(
            f"No safetensors weights found for prefixes {list(prefixes)!r} under {model_path}"
        )

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(single), framework="pt", device=device) as f:
        for key in f.keys():
            for prefix in prefixes:
                if key.startswith(prefix):
                    tensor = f.get_tensor(key)
                    if dtype is not None and tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    state_dict[key[len(prefix) :]] = tensor
                    break

    if not state_dict:
        raise FileNotFoundError(
            f"No safetensors weights found for prefixes {list(prefixes)!r} under {model_path}"
        )
    return state_dict


def load_weights_by_prefix(
    model_path: str | Path,
    *,
    prefix: str | tuple[str, ...] | list[str],
) -> dict[str, torch.Tensor]:
    """Load safetensors weights matching one of the prefixes, stripping the matched prefix."""
    model_path = Path(model_path)
    prefixes = _normalize_prefixes(prefix)

    for prefix_item in prefixes:
        state_dict = _load_safetensors_sharded(model_path, prefix_item)
        if state_dict:
            return state_dict
        state_dict = _load_safetensors_single(model_path, prefix_item)
        if state_dict:
            return state_dict

    raise FileNotFoundError(
        f"No safetensors weights found for prefixes {list(prefixes)!r} under {model_path}"
    )


def load_module(
    module: nn.Module,
    model_path: str | Path,
    *,
    prefix: str | tuple[str, ...] | list[str],
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> nn.Module:
    """Load weights into module by prefix, optionally move to device."""
    state_dict = load_weights_by_prefix(
        model_path,
        prefix=prefix,
    )
    state_dict = _maybe_fuse_qwen3_moe_experts(state_dict, module)
    module.load_state_dict(state_dict, strict=strict, assign=True)
    module.eval()
    if device is not None or dtype is not None:
        if device is not None and dtype is not None:
            module = module.to(device=device, dtype=dtype)
        elif device is not None:
            module = module.to(device=device)
        else:
            module = module.to(dtype=dtype)
    return module
