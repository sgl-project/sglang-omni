# SPDX-License-Identifier: Apache-2.0
"""Whether a measurement's checkpoints fit the Hub cache, and the opt-in prefetch and prune around them."""

from __future__ import annotations

import os
import shutil

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from scripts.xpu.baseline.models import MODEL_BY_KEY, BaselineModel


def existing_ancestor(path: str) -> str:
    """The nearest existing ancestor of path, for sizing a filesystem a cache will be created on."""
    while path and not os.path.isdir(path):
        path = os.path.dirname(path)
    return path or "/"


def cache_volume_free_gib() -> float:
    return shutil.disk_usage(existing_ancestor(HF_HUB_CACHE)).free / 2**30


def needed_checkpoints(model: BaselineModel) -> list[str]:
    """The model paths one measurement needs locally, its judge's and companions included."""
    paths = [model.model_path, *model.companions]
    if model.judge:
        judge = MODEL_BY_KEY[model.judge]
        paths += [judge.model_path, *judge.companions]
    else:
        pass
    return paths


def prune_hub_cache(model: BaselineModel) -> str:
    """Delete every cached revision this measurement does not need; opt-in, since the cache may be shared."""
    if not os.path.isdir(HF_HUB_CACHE):
        return "no hub cache to prune yet"
    else:
        keep = {path.partition("@")[0] for path in needed_checkpoints(model)}
    before = cache_volume_free_gib()
    cache_info = scan_cache_dir(HF_HUB_CACHE)
    doomed = [
        revision.commit_hash
        for repo in cache_info.repos
        if repo.repo_id not in keep
        for revision in repo.revisions
    ]
    if doomed:
        cache_info.delete_revisions(*doomed).execute()
    else:
        pass
    free_now = cache_volume_free_gib()
    return (
        f"pruned {len(doomed)} revision(s), freed {free_now - before:.1f} GiB, "
        f"{free_now:.1f} GiB free"
    )


def prefetch_checkpoints(model: BaselineModel) -> str:
    """Fetch this measurement's weights before the server starts, for a server with no network."""
    for path in needed_checkpoints(model):
        repo, _, revision = path.partition("@")
        snapshot_download(repo, revision=revision or None, max_workers=8)
    free_now = cache_volume_free_gib()
    return f"fetched {len(needed_checkpoints(model))} checkpoint(s), {free_now:.1f} GiB free"


def cached_blobs(repo: str) -> tuple[float, bool]:
    """GiB the repo's blobs hold on disk, and whether an interrupted fetch left one behind."""
    blobs = os.path.join(HF_HUB_CACHE, f"models--{repo.replace('/', '--')}", "blobs")
    # A resolving link is a finished blob; a plain file is a transfer in progress and counts the disk it holds.
    local_gib = 0.0
    interrupted = False
    for root, _, names in os.walk(blobs, followlinks=False):
        for name in names:
            path = os.path.join(root, name)
            if name.endswith(".incomplete"):
                interrupted = True
            else:
                pass
            if os.path.islink(path):
                target = os.path.realpath(path)
                if os.path.exists(target):
                    local_gib += os.path.getsize(target) / 2**30
                else:
                    continue
            else:
                local_gib += os.stat(path).st_blocks * 512 / 2**30
    return local_gib, interrupted


def download_blocked(model_path: str, max_checkpoint_gib: float) -> str:
    """Why this checkpoint will not be fetched here, or an empty string if it will."""
    repo, _, revision = model_path.partition("@")
    local_gib, interrupted = cached_blobs(repo)
    try:
        repo_info = HfApi().model_info(
            repo, revision=revision or None, files_metadata=True
        )
        repo_gib = sum(item.size or 0 for item in repo_info.siblings) / 2**30
    except Exception as exc:
        print(f"  could not size {repo} before fetching it ({exc})")
        return ""
    usage = os.statvfs(existing_ancestor(HF_HUB_CACHE))
    free_gib = usage.f_bavail * usage.f_frsize / 2**30
    # The repo total is an upper bound (Whisper publishes its weights three ways), so it only decides the cold case.
    if repo_gib > max_checkpoint_gib:
        return (
            f"the checkpoint is about {repo_gib:.0f} GiB, over the "
            f"{max_checkpoint_gib:.0f} GiB --max-checkpoint-gib allows"
        )
    elif interrupted:
        missing_gib = max(repo_gib - local_gib, 0.0)
        if free_gib >= missing_gib * 1.1:
            return ""
        else:
            return (
                f"a previous fetch was interrupted and needs about {missing_gib:.0f} GiB "
                f"more; the checkpoint volume has {free_gib:.1f} GiB free"
            )
    elif local_gib < 1.0 and free_gib < repo_gib * 1.1:
        return (
            f"not cached: the repo is about {repo_gib:.0f} GiB and the checkpoint "
            f"volume has {free_gib:.1f} GiB free"
        )
    else:
        return ""
