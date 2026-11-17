# SPDX-License-Identifier: Apache-2.0
"""Checkpoint resolution helpers."""

from __future__ import annotations

import os


def split_checkpoint_revision(
    model_path: str, revision: str | None
) -> tuple[str, str | None]:
    """Translate an Omni checkpoint pin into native model and revision arguments."""
    if revision is not None and not isinstance(revision, str):
        raise ValueError("Checkpoint revision must be a string")
    elif os.path.isdir(model_path):
        return model_path, revision
    else:
        repository_id, separator, pinned_revision = model_path.partition("@")
        if separator and not pinned_revision:
            raise ValueError("Checkpoint revision after @ must not be empty")
        elif separator and revision is not None and revision != pinned_revision:
            raise ValueError(
                "model_path and revision pin different checkpoint revisions"
            )
        else:
            return repository_id, pinned_revision or revision


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
