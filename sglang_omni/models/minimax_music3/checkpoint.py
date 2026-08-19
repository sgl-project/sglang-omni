# SPDX-License-Identifier: Apache-2.0
"""MiniMax Music 3 checkpoint path and weight loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_source


@dataclass(frozen=True)
class TTMCheckpointPaths:
    root: Path
    qwen_dir: Path
    tokenizer_dir: Path
    config_path: Path
    qwen_index_path: Path
    dit_path: Path
    dav_path: Path


@lru_cache(maxsize=None)
def _download_once(model_path: str) -> str:
    return _resolve_source(model_path)


def resolve_checkpoint(model_path: str | Path) -> TTMCheckpointPaths:
    root = Path(_download_once(str(Path(model_path).expanduser()))).expanduser()
    direct_qwen = root / "qwen_7B" / "qwen_7B"
    direct_dit = root / "flowmatching_vae.pth"
    if not (direct_qwen.is_dir() and direct_dit.exists()) and root.is_dir():
        candidates = [
            child
            for child in root.iterdir()
            if child.is_dir()
            and (child / "qwen_7B" / "qwen_7B").is_dir()
            and (child / "flowmatching_vae.pth").exists()
        ]
        if len(candidates) == 1:
            root = candidates[0]
    qwen_dir = root / "qwen_7B" / "qwen_7B"
    tokenizer_dir = root / "qwen_7B" / "qwen3-8B-tokenizer-music"
    paths = TTMCheckpointPaths(
        root=root,
        qwen_dir=qwen_dir,
        tokenizer_dir=tokenizer_dir,
        config_path=qwen_dir / "config.json",
        qwen_index_path=qwen_dir / "model.safetensors.index.json",
        dit_path=root / "flowmatching_vae.pth",
        dav_path=root / "dav.pth",
    )
    return paths


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def load_torch_state(path: str | Path, *, device: torch.device) -> dict[str, Any]:
    """Load a .pth state dict without accepting arbitrary object payloads."""

    state = torch.load(
        str(path),
        map_location=device,
        weights_only=True,
    )
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError(f"expected tensor state dict in {path}")
    return state


def load_audio_state(paths: TTMCheckpointPaths) -> dict[str, torch.Tensor]:
    """Load Qwen's audio embedding and RVQ decoder weights."""

    from safetensors.torch import load_file

    prefixes = ("model.audio_extra_embedding", "model.audio_decoder.")
    index = load_json(paths.qwen_index_path)
    selected_files = {
        str(filename)
        for key, filename in index["weight_map"].items()
        if any(str(key).startswith(prefix) for prefix in prefixes)
    }
    result: dict[str, torch.Tensor] = {}
    for filename in sorted(selected_files):
        shard = load_file(str(paths.qwen_dir / filename), device="cpu")
        for key, value in shard.items():
            if any(key.startswith(prefix) for prefix in prefixes):
                result[key] = value
    return result


__all__ = [
    "TTMCheckpointPaths",
    "load_audio_state",
    "load_json",
    "load_torch_state",
    "resolve_checkpoint",
]
