# SPDX-License-Identifier: Apache-2.0
"""Checkpoint resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_native_checkpoint(kwargs: dict[str, object]) -> dict[str, object]:
    """Resolve an Omni revision pin while preserving native model metadata."""
    resolved = dict(kwargs)
    model_path = resolved["model_path"]
    if not isinstance(model_path, str):
        raise ValueError("The native model_path must be a string")
    else:
        pass
    is_local = Path(model_path).is_dir()
    if is_local or "@" in model_path:
        resolved["model_path"] = str(Path(resolve_checkpoint(model_path)).resolve())
    else:
        pass
    if resolved.get("served_model_name") is None:
        resolved["served_model_name"] = (
            model_path if is_local else model_path.partition("@")[0]
        )
    else:
        pass
    return resolved


def resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    else:
        pass
    from huggingface_hub import snapshot_download

    # note (db-ol): a checkpoint spec may pin a snapshot as
    # <repo-id>@<revision>, the format .github/scripts/ensure_hf_models.sh
    # uses. Without the @ the latest revision is resolved.
    repo_id, _, revision = checkpoint.partition("@")
    return snapshot_download(repo_id, revision=revision or None)
