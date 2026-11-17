# SPDX-License-Identifier: Apache-2.0
"""MiniMax Music 3 checkpoint path and weight loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_source
from sglang_omni.utils.json import JsonValue


@dataclass(frozen=True)
class TTMCheckpointPaths:
    root: Path
    language_model_dir: Path
    language_model_config_path: Path
    qwen_dir: Path
    tokenizer_dir: Path
    audio_config_path: Path
    qwen_index_path: Path
    dit_path: Path
    dav_path: Path


@lru_cache(maxsize=None)
def download_once(model_path: str) -> str:
    return _resolve_source(model_path)


def resolve_checkpoint(model_path: str | Path) -> TTMCheckpointPaths:
    root = Path(download_once(str(Path(model_path).expanduser()))).expanduser()
    direct_qwen = root / "qwen_7B" / "qwen_7B"
    direct_language_model = root / "language_model"
    direct_dit = root / "flowmatching_vae.pth"
    if (
        not (
            direct_qwen.is_dir()
            and direct_language_model.is_dir()
            and direct_dit.exists()
        )
        and root.is_dir()
    ):
        candidates = [
            child
            for child in root.iterdir()
            if child.is_dir()
            and (child / "qwen_7B" / "qwen_7B").is_dir()
            and (child / "language_model").is_dir()
            and (child / "flowmatching_vae.pth").exists()
        ]
        if len(candidates) == 1:
            root = candidates[0]
        else:
            pass
    else:
        pass
    qwen_dir = root / "qwen_7B" / "qwen_7B"
    tokenizer_dir = root / "qwen_7B" / "qwen3-8B-tokenizer-music"
    language_model_dir = root / "language_model"
    language_model_config_path = language_model_dir / "config.json"
    audio_config_path = qwen_dir / "config.json"
    paths = TTMCheckpointPaths(
        root=root,
        language_model_dir=language_model_dir,
        language_model_config_path=language_model_config_path,
        qwen_dir=qwen_dir,
        tokenizer_dir=tokenizer_dir,
        audio_config_path=audio_config_path,
        qwen_index_path=qwen_dir / "model.safetensors.index.json",
        dit_path=root / "flowmatching_vae.pth",
        dav_path=root / "dav.pth",
    )
    validate_checkpoint_configs(paths)
    return paths


def load_json(path: Path) -> dict[str, JsonValue]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    else:
        pass
    return value


def validate_checkpoint_configs(paths: TTMCheckpointPaths) -> None:
    if not paths.language_model_config_path.is_file():
        raise ValueError(
            "MiniMax Music 3 checkpoint is missing the native Qwen3 backbone "
            f"config: {paths.language_model_config_path}"
        )
    else:
        pass
    if not paths.audio_config_path.is_file():
        raise ValueError(
            "MiniMax Music 3 checkpoint is missing the audio module config: "
            f"{paths.audio_config_path}"
        )
    else:
        pass
    language_model_config = load_json(paths.language_model_config_path)
    audio_config = load_json(paths.audio_config_path)
    if language_model_config.get("model_type") != "qwen3":
        raise ValueError(
            "MiniMax Music 3 native backbone must declare model_type='qwen3'; "
            f"got {language_model_config.get('model_type')!r}"
        )
    else:
        pass
    language_hidden_size = int(language_model_config["hidden_size"])
    audio_hidden_size = int(audio_config["hidden_size"])
    if language_hidden_size != audio_hidden_size:
        raise ValueError(
            "MiniMax Music 3 backbone/audio hidden_size mismatch: "
            f"{language_hidden_size} != {audio_hidden_size}"
        )
    else:
        pass


def load_torch_state(path: str | Path, *, device: torch.device) -> dict[str, object]:
    """Load a .pth state dict without accepting arbitrary object payloads."""

    state = torch.load(
        str(path),
        map_location=device,
        weights_only=True,
        mmap=True,
    )
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    else:
        pass
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError(f"expected tensor state dict in {path}")
    else:
        pass
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
            else:
                pass
    return result


__all__ = [
    "TTMCheckpointPaths",
    "load_audio_state",
    "load_json",
    "load_torch_state",
    "resolve_checkpoint",
    "validate_checkpoint_configs",
]
