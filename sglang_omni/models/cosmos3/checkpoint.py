# SPDX-License-Identifier: Apache-2.0
"""Resolve a Cosmos checkpoint without replacing native model metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sglang_omni.utils.checkpoint import resolve_checkpoint


def resolve_native_checkpoint(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Give native loaders a local snapshot and retain the public model name.

    Omni accepts ``repo@revision``; native loaders need the resolved directory.
    Keep the checkpoint's geometry, processor, and component configurations
    intact so the native Cosmos family implementation can select them.
    """
    resolved = dict(kwargs)
    model_path = resolved["model_path"]
    is_local = Path(model_path).is_dir()
    resolved["model_path"] = str(Path(resolve_checkpoint(model_path)).resolve())
    if resolved.get("served_model_name") is None:
        resolved["served_model_name"] = (
            model_path if is_local else model_path.partition("@")[0]
        )
    return resolved
