# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointFilterConfig:
    """Checkpoint tensor-name filter for one model-loading stage."""

    name: str
    accept_prefixes: tuple[str, ...]
    strip_prefixes: tuple[str, ...] = ()
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("checkpoint filter name must be non-empty")
        if not self.accept_prefixes:
            raise ValueError("checkpoint filter accept_prefixes must be non-empty")


def filter_safetensor_files(
    *,
    model_dir: str | Path,
    files: Sequence[str],
    profile: CheckpointFilterConfig,
    index_name: str = "model.safetensors.index.json",
) -> list[str]:
    """Select safetensor files that contain tensors accepted by profile."""

    model_path = Path(model_dir)
    index_path = model_path / index_name
    if not index_path.is_file():
        return _filter_safetensor_files_by_header(files, profile)

    with index_path.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    selected_shards = {
        model_path / shard
        for name, shard in weight_map.items()
        if _is_accepted(name, profile)
    }
    if not selected_shards:
        if profile.required:
            raise ValueError(
                f"Checkpoint filter {profile.name!r} matched no tensors in "
                f"{index_path}"
            )
        return list(files)

    selected = {_normalize_path(path) for path in selected_shards}
    filtered_files = [file for file in files if _normalize_path(Path(file)) in selected]
    if not filtered_files and profile.required:
        raise ValueError(
            f"Checkpoint filter {profile.name!r} matched tensors in "
            f"{index_path}, but none of their shard files are available"
        )
    return filtered_files


def filter_and_remap_weight_iterator(
    weights: Iterable[tuple[str, Any]],
    profile: CheckpointFilterConfig,
) -> Iterator[tuple[str, Any]]:
    """Yield only accepted checkpoint tensors, applying stage-local name remaps."""

    for name, tensor in weights:
        if not _is_accepted(name, profile):
            continue
        yield _strip_prefix(name, profile.strip_prefixes), tensor


def install_checkpoint_filter(
    loader: Any,
    profile: CheckpointFilterConfig,
    *,
    log: Any | None = None,
) -> None:
    """Install shard and tensor-name filtering on one SGLang loader instance."""

    original_prepare_weights = loader._prepare_weights
    original_get_weights_iterator = loader._get_weights_iterator

    def _prepare_weights(model_name_or_path, revision, fall_back_to_pt):
        hf_folder, hf_weights_files, use_safetensors = original_prepare_weights(
            model_name_or_path,
            revision,
            fall_back_to_pt,
        )
        if not use_safetensors:
            return hf_folder, hf_weights_files, use_safetensors

        filtered_files = filter_safetensor_files(
            model_dir=hf_folder,
            files=hf_weights_files,
            profile=profile,
        )
        if log is not None and len(filtered_files) != len(hf_weights_files):
            log.info(
                "Checkpoint filter %s selected %d/%d safetensor shard(s)",
                profile.name,
                len(filtered_files),
                len(hf_weights_files),
            )
        return hf_folder, filtered_files, use_safetensors

    def _get_weights_iterator(source):
        weights = original_get_weights_iterator(source)
        return filter_and_remap_weight_iterator(weights, profile)

    loader._prepare_weights = _prepare_weights
    loader._get_weights_iterator = _get_weights_iterator


def _is_accepted(name: str, profile: CheckpointFilterConfig) -> bool:
    return name.startswith(profile.accept_prefixes)


def _filter_safetensor_files_by_header(
    files: Sequence[str],
    profile: CheckpointFilterConfig,
) -> list[str]:
    selected_files = [
        file
        for file in files
        if any(_is_accepted(name, profile) for name in _read_safetensor_keys(file))
    ]
    if not selected_files and profile.required:
        raise ValueError(
            f"Checkpoint filter {profile.name!r} matched no tensors in "
            "safetensors headers"
        )
    if not selected_files:
        return list(files)
    return selected_files


def _read_safetensor_keys(path: str) -> list[str]:
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        return list(handle.keys())


def _strip_prefix(name: str, strip_prefixes: Sequence[str]) -> str:
    for prefix in sorted(strip_prefixes, key=len, reverse=True):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _normalize_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)
