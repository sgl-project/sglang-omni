# SPDX-License-Identifier: Apache-2.0
"""Machine-readable accelerator support declarations for model pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class AcceleratorSupportStatus(str, Enum):
    SUPPORTED = "supported"
    PREVIEW = "preview"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ModelAcceleratorSupport:
    """Validation state for one canonical architecture on one accelerator."""

    architecture: str
    accelerator: str
    status: AcceleratorSupportStatus
    gpu_architectures: tuple[str, ...]
    validated_features: frozenset[str] = frozenset()
    limitations: tuple[str, ...] = ()
    evidence: str | None = None

    def supports_gpu_architecture(self, gpu_architecture: str) -> bool:
        normalized = gpu_architecture.strip().lower().split(":", 1)[0]
        return (
            self.status is AcceleratorSupportStatus.SUPPORTED
            and normalized in self.gpu_architectures
        )


_ROCM_ARCHITECTURES = ("gfx942", "gfx950")
_PREVIEW_LIMITATION = (
    "Single-request functional E2E passed on gfx942 and gfx950; the complete "
    "CUDA feature contract and scheduled stability gates remain pending.",
)


def _rocm_preview(architecture: str) -> ModelAcceleratorSupport:
    return ModelAcceleratorSupport(
        architecture=architecture,
        accelerator="rocm",
        status=AcceleratorSupportStatus.PREVIEW,
        gpu_architectures=_ROCM_ARCHITECTURES,
        limitations=_PREVIEW_LIMITATION,
        evidence="Dual-architecture functional matrix: 18/18 on gfx942 and gfx950",
    )


_ROCM_SUPPORT = {
    architecture: _rocm_preview(architecture)
    for architecture in (
        "ArkasrForConditionalGeneration",
        "AudarTTSForConditionalGeneration",
        "DotsTTSForConditionalGeneration",
        "FishQwen3OmniForCausalLM",
        "FunAsrNanoForConditionalGeneration",
        "HiggsMultimodalQwen3ForConditionalGeneration",
        "LLaDA2MoeModelLM",
        "BailingMM2NativeForConditionalGeneration",
        "BailingMMNativeForConditionalGeneration",
        "MossTranscribeDiarizeForConditionalGeneration",
        "MossTTSDelayModel",
        "MossTTSLocalModel",
        "Qwen3ASRForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3TTSForConditionalGeneration",
        "VoxtralTTSForConditionalGeneration",
        "WhisperForConditionalGeneration",
        "Zonos2ForCausalLM",
    )
}
_ROCM_SUPPORT["MiniMaxMusic3ForConditionalGeneration"] = ModelAcceleratorSupport(
    architecture="MiniMaxMusic3ForConditionalGeneration",
    accelerator="rocm",
    status=AcceleratorSupportStatus.SUPPORTED,
    gpu_architectures=_ROCM_ARCHITECTURES,
    validated_features=frozenset(
        {
            "openai-audio-speech-api",
            "eager-generation",
            "compiled-acoustic-dit-dav",
            "aiter-autoregressive-stage",
            "torch-sdpa-acoustic-stage",
        }
    ),
    limitations=(
        "Generation and RVQ device graphs are disabled; DIT and DAV remain compiled.",
    ),
    evidence="PR #1534: ROCm 7.2 E2E generation on MI300X and MI355X",
)


def _canonical_architecture(architecture: str) -> str | None:
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    config_cls = PIPELINE_CONFIG_REGISTRY.configs.get(architecture)
    if config_cls is None:
        return None
    return str(config_cls.architecture)


def get_model_accelerator_support(
    architecture: str,
    accelerator: str,
) -> ModelAcceleratorSupport | None:
    """Return declared support for a registered architecture and accelerator."""

    canonical = _canonical_architecture(architecture)
    if canonical is None:
        return None
    normalized_accelerator = accelerator.strip().lower()
    if normalized_accelerator == "rocm":
        return _ROCM_SUPPORT.get(canonical)
    return None


def iter_model_accelerator_support(
    accelerator: str | None = None,
) -> Iterable[ModelAcceleratorSupport]:
    """Iterate declarations in stable architecture order."""

    normalized = accelerator.strip().lower() if accelerator is not None else None
    declarations = _ROCM_SUPPORT.values() if normalized in (None, "rocm") else ()
    return iter(sorted(declarations, key=lambda item: item.architecture))


__all__ = [
    "AcceleratorSupportStatus",
    "ModelAcceleratorSupport",
    "get_model_accelerator_support",
    "iter_model_accelerator_support",
]
