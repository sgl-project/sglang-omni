# SPDX-License-Identifier: Apache-2.0
"""Model hub backend selection.

ModelScope mirrors the Hugging Face API, so when ``SGLANG_OMNI_USE_MODELSCOPE``
is set we source the loaders and download helpers from ModelScope instead of
``transformers`` / ``huggingface_hub``. Remote ids are passed through unchanged;
the active backend's ``from_pretrained`` / download functions fetch them.
"""

from __future__ import annotations

from sglang_omni.environ import OMNIENV

_USE_MODELSCOPE: bool = OMNIENV.SGLANG_OMNI_USE_MODELSCOPE.get()


if _USE_MODELSCOPE:
    from modelscope import AutoConfig
    from modelscope import model_file_download as _model_file_download
    from modelscope import snapshot_download as _snapshot_download
else:
    from huggingface_hub import hf_hub_download as _hf_hub_download
    from huggingface_hub import snapshot_download as _snapshot_download
    from transformers import AutoConfig
    from transformers.utils.hub import cached_file as _cached_file

__all__ = ["AutoConfig", "snapshot_download", "hf_hub_download", "cached_file"]


def snapshot_download(repo_id: str, **kwargs) -> str:
    """Download a full model snapshot from the active hub backend."""
    if _USE_MODELSCOPE:
        # ModelScope re-fetches missing/corrupt files on its own and has no
        # force_download flag.
        kwargs.pop("force_download", None)
    return _snapshot_download(repo_id, **kwargs)


def hf_hub_download(repo_id: str, filename: str, **kwargs) -> str:
    """Download a single file from the active hub backend."""
    if _USE_MODELSCOPE:
        return _model_file_download(model_id=repo_id, file_path=filename, **kwargs)
    return _hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)


def cached_file(model_path: str, filename: str, **kwargs) -> str:
    """Locate a single file in the active hub backend cache, downloading if needed."""
    if _USE_MODELSCOPE:
        return _model_file_download(model_id=model_path, file_path=filename, **kwargs)
    return _cached_file(model_path, filename, **kwargs)
