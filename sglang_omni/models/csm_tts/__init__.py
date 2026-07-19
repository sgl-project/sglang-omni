# SPDX-License-Identifier: Apache-2.0
"""CSM-1B TTS model support for sglang-omni (``sesame/csm-1b``).

Registers :class:`CsmTtsHfConfig` with ``transformers.AutoConfig`` under the
NATIVE model type ``"csm"`` (``exist_ok=True`` overrides the in-process
mapping; semantics + container check inR7), then imports
``config`` so the pipeline registry pkgutil scan finds ``EntryClass``.

This module (and ``config``) MUST stay sglang/CUDA-free — the registry scan
silently skips packages whose import fails (contract). The sglang model
class is registered in
:meth:`sglang_omni.model_runner.sglang_model_runner.SGLModelRunner._register_omni_model`.
"""

from __future__ import annotations

from transformers import AutoConfig

from .hf_config import CsmTtsHfConfig

AutoConfig.register("csm", CsmTtsHfConfig, exist_ok=True)

from . import config  # noqa: E402  (must follow the AutoConfig override)

__all__ = ["config", "CsmTtsHfConfig"]
