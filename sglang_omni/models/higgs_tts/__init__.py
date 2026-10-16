# SPDX-License-Identifier: Apache-2.0
"""HiggsMultimodalQwen3 TTS model support for sglang-omni.

:mod:`.hf_config` registers :class:`HiggsMultimodalQwen3Config` with
``transformers.AutoConfig`` when it is imported, which the stage modules do, so
``AutoConfig.from_pretrained()`` works before any Higgs stage factory runs.
Importing this package itself stays free of transformers. The model class is registered in
:meth:`sglang_omni.model_runner.sglang_model_runner.SGLModelRunner._register_omni_model`
alongside the other sglang-omni models.
"""

from __future__ import annotations

from sglang_omni.models.model_capabilities import ModelCapabilities

from . import config

CAPABILITIES = ModelCapabilities(
    supports_reference_audio=True,
    supports_batch_vocoder=True,
    supports_streaming_vocoder=True,
    supports_cuda_graph=True,
    supports_torch_compile=True,
    supports_breakable_prefill_cuda_graph=True,
    supports_full_prefill_cuda_graph=False,
)

__all__ = ["CAPABILITIES", "config", "HiggsMultimodalQwen3Config"]


def __getattr__(name: str):
    if name == "HiggsMultimodalQwen3Config":
        from .hf_config import HiggsMultimodalQwen3Config

        return HiggsMultimodalQwen3Config
    else:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
