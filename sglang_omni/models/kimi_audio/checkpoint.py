# SPDX-License-Identifier: Apache-2.0
"""Resolve the text-output subset of a Kimi-Audio checkpoint."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

_INDEX_NAME = "model.safetensors.index.json"
_DOWNLOAD_SOURCE_ENV = "SGLANG_OMNI_KIMI_AUDIO_DOWNLOAD_SOURCE"


def _is_text_output_weight(name: str) -> bool:
    if name in (
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ):
        return True
    if name.startswith("model.vq_adaptor."):
        return True
    return name.startswith("model.layers.")


def _make_filtered_view(snapshot: Path, index: dict) -> str:
    view = Path(tempfile.mkdtemp(prefix="sglang-omni-kimi-audio-"))
    for source in snapshot.rglob("*"):
        relative = source.relative_to(snapshot)
        if (
            not source.is_file()
            or source.name == _INDEX_NAME
            or relative == Path("config.json")
        ):
            continue
        destination = view / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source.resolve(), destination)

    with (snapshot / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    config.pop("auto_map", None)
    config["model_type"] = "moonshot_kimia"
    with (view / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    filtered = dict(index)
    filtered["weight_map"] = {
        name: shard
        for name, shard in index["weight_map"].items()
        if _is_text_output_weight(name)
    }
    with (view / _INDEX_NAME).open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, indent=2, sort_keys=True)
    return str(view)


def resolve_kimi_audio_text_checkpoint(model_path: str) -> str:
    """Download only files needed by the audio-to-text serving path."""
    local_path = Path(model_path)
    if local_path.is_dir():
        with (local_path / _INDEX_NAME).open(encoding="utf-8") as handle:
            index = json.load(handle)
        return _make_filtered_view(local_path, index)

    source = os.getenv(_DOWNLOAD_SOURCE_ENV, "huggingface").strip().lower()
    if source not in ("huggingface", "modelscope"):
        raise ValueError(
            f"{_DOWNLOAD_SOURCE_ENV} must be 'huggingface' or 'modelscope', "
            f"got {source!r}"
        )

    def download(patterns: list[str]) -> Path:
        if source == "modelscope":
            from modelscope import snapshot_download as modelscope_download

            return Path(
                modelscope_download(
                    model_path,
                    allow_patterns=patterns,
                    cache_dir=os.getenv("MODELSCOPE_CACHE") or None,
                    max_workers=16,
                )
            )
        return Path(snapshot_download(model_path, allow_patterns=patterns))

    metadata_snapshot = download(
        [
            "*.json",
            "*.py",
            "*.model",
            _INDEX_NAME,
        ]
    )
    with (metadata_snapshot / _INDEX_NAME).open(encoding="utf-8") as handle:
        index = json.load(handle)
    shards = sorted(
        {
            shard
            for name, shard in index["weight_map"].items()
            if _is_text_output_weight(name)
        }
    )
    snapshot = download(
        [
            "*.json",
            "*.py",
            "*.model",
            *shards,
            "whisper-large-v3/*",
        ]
    )
    return _make_filtered_view(snapshot, index)


def resolve_glm4_audio_tokenizer(checkpoint: str) -> str:
    if Path(checkpoint).is_dir():
        return checkpoint
    source = os.getenv(_DOWNLOAD_SOURCE_ENV, "huggingface").strip().lower()
    if source == "modelscope":
        from modelscope import snapshot_download as modelscope_download

        model_id = (
            "ZhipuAI/glm-4-voice-tokenizer"
            if checkpoint == "THUDM/glm-4-voice-tokenizer"
            else checkpoint
        )
        return modelscope_download(
            model_id,
            cache_dir=os.getenv("MODELSCOPE_CACHE") or None,
            max_workers=16,
        )
    if source != "huggingface":
        raise ValueError(
            f"{_DOWNLOAD_SOURCE_ENV} must be 'huggingface' or 'modelscope', "
            f"got {source!r}"
        )
    return snapshot_download(checkpoint)


__all__ = ["resolve_glm4_audio_tokenizer", "resolve_kimi_audio_text_checkpoint"]
