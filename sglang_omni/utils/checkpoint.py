# SPDX-License-Identifier: Apache-2.0
"""Checkpoint resolution helpers."""

from __future__ import annotations

import os

import huggingface_hub
from huggingface_hub import constants


def resolve_checkpoint(
    checkpoint: str, *, allow_patterns: list[str] | None = None
) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    else:
        pass
    # note (db-ol): a checkpoint spec may pin a snapshot as
    # <repo-id>@<revision>, the format .github/scripts/ensure_hf_models.sh
    # uses. Without the @ the latest revision is resolved.
    repo_id, _, revision = checkpoint.partition("@")
    if allow_patterns is None:
        return huggingface_hub.snapshot_download(repo_id, revision=revision or None)
    else:
        return huggingface_hub.snapshot_download(
            repo_id,
            revision=revision or None,
            allow_patterns=allow_patterns,
            local_files_only=constants.HF_HUB_OFFLINE,
        )
