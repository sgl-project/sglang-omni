# SPDX-License-Identifier: Apache-2.0
"""Model hub backend selection.

ModelScope mirrors the Hugging Face API, so when ``SGLANG_OMNI_USE_MODELSCOPE``
is set we source the loaders and download helpers from ModelScope instead of
``transformers`` / ``huggingface_hub``. Remote ids are passed through unchanged;
the active backend's ``from_pretrained`` / download functions fetch them.
"""

from __future__ import annotations

from sglang_omni.environ import OMNIENV


def _use_modelscope() -> bool:
    return OMNIENV.SGLANG_OMNI_USE_MODELSCOPE.get()


# AutoConfig is selected at import time, mirroring how the loaders are imported
# at the call sites. The process-wide flag is fixed before launch.
if _use_modelscope():
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

__all__ = ["AutoConfig", "snapshot_download", "hf_hub_download", "cached_file"]


def snapshot_download(repo_id: str, **kwargs) -> str:
    """Download a full model snapshot from the active hub backend."""
    if _use_modelscope():
        from modelscope import snapshot_download as _download

        # ModelScope re-fetches missing/corrupt files on its own and has no
        # force_download flag.
        kwargs.pop("force_download", None)
        return _download(repo_id, **kwargs)
    from huggingface_hub import snapshot_download as _download

    return _download(repo_id, **kwargs)


def hf_hub_download(repo_id: str, filename: str, **kwargs) -> str:
    """Download a single file from the active hub backend."""
    if _use_modelscope():
        from modelscope import model_file_download

        return model_file_download(model_id=repo_id, file_path=filename, **kwargs)
    from huggingface_hub import hf_hub_download as _download

    return _download(repo_id=repo_id, filename=filename, **kwargs)


def cached_file(model_path: str, filename: str, **kwargs) -> str:
    """Locate a single file in the active hub backend cache, downloading if needed."""
    if _use_modelscope():
        from modelscope import model_file_download

        return model_file_download(model_id=model_path, file_path=filename, **kwargs)
    from transformers.utils.hub import cached_file as _cached_file

    return _cached_file(model_path, filename, **kwargs)
