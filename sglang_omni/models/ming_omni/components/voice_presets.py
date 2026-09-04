# SPDX-License-Identifier: Apache-2.0
"""Prompt-wav path resolution for Ming talker voice manifests."""

from __future__ import annotations

import os

import sglang_omni

ASSET_ROOT_ENV = "SGLANG_OMNI_VOICE_ASSET_ROOT"


def _fallback_roots() -> list[str]:
    roots = []
    env_root = os.environ.get(ASSET_ROOT_ENV)
    if env_root:
        roots.append(env_root)
    # Checkpoint manifests may reference assets from the authoring checkout
    # (Ming-flash-omni-2.0's DB30 points at tests/data). A wheel install's
    # package parent is site-packages and ships no such assets, so only offer
    # the parent when it actually looks like a checkout.
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(sglang_omni.__file__)))
    if os.path.isdir(os.path.join(pkg_root, "tests")):
        roots.append(pkg_root)
    return roots


def _contained(candidate: str, root: str) -> bool:
    return os.path.realpath(candidate).startswith(os.path.realpath(root) + os.sep)


def resolve_prompt_wav_path(raw_path: str, talker_dir: str) -> str | None:
    """Return an existing absolute path for a manifest prompt-wav entry.

    Tries the checkpoint-relative join first, then re-resolves path suffixes
    (longest first) against the talker dir and the fallback roots
    (``SGLANG_OMNI_VOICE_ASSET_ROOT``, then the checkout root when running
    from a checkout), so absolute paths baked in on the authoring machine
    keep working. Parent traversal fails closed; bare filenames only match
    inside the talker dir; every fallback match must stay inside its root.
    Returns None when no candidate exists.
    """
    parts = [p for p in raw_path.replace("\\", "/").split("/") if p and p != "."]
    if ".." in parts:
        return None
    if os.path.isabs(raw_path):
        if os.path.isfile(raw_path):
            return raw_path
    else:
        primary = os.path.join(talker_dir, raw_path)
        if os.path.isfile(primary) and _contained(primary, talker_dir):
            return primary
    roots = [talker_dir, *_fallback_roots()]
    for start in range(len(parts)):
        tail_parts = parts[start:]
        tail = os.path.join(*tail_parts)
        for root in roots:
            if len(tail_parts) < 2 and root != talker_dir:
                continue
            candidate = os.path.join(root, tail)
            if os.path.isfile(candidate) and _contained(candidate, root):
                return candidate
    return None
