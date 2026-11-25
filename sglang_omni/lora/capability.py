# SPDX-License-Identifier: Apache-2.0
"""Capability interface and architecture audit for engine-side merged LoRA."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import torch.nn as nn

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

MergedLoRAStatus = Literal["supported", "runtime_only", "not_capable"]


class MergedLoRACapabilityError(RuntimeError):
    """Raised when a model or stage cannot participate in merged LoRA updates."""


@dataclass(frozen=True)
class MergedLoRASpec:
    """Declarative binding specification for engine-side merged LoRA updates.

    Attributes:
        target_module_root: Dotted path from the model root to the module tree
            targeted by LoRA adapters, or empty string for the model root.
        packed_modules_mapping: Mapping from packed physical module suffix
            (for example qkv_proj) to the ordered logical sublayer names packed
            along its output dimension.
        name_aliases: Mapping from adapter-side logical module names to the
            engine-side module suffix (for example to_out -> to_out.0).
        strip_prefixes: Ordered prefixes stripped from incoming adapter module
            names before matching (for example transformer. or model.).
        allow_plain_linear: Whether plain torch.nn.Linear modules are valid
            merge targets (True for single-device DiT/flow stages, False for
            TP-sharded AR stages that require sharded linear metadata).
    """

    target_module_root: str = ""
    packed_modules_mapping: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    name_aliases: Mapping[str, str] = field(default_factory=dict)
    strip_prefixes: tuple[str, ...] = ()
    allow_plain_linear: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.target_module_root, str):
            raise TypeError(
                "target_module_root must be str, "
                f"got {type(self.target_module_root).__name__}"
            )
        else:
            pass

        normalized_packed: dict[str, tuple[str, ...]] = {}
        seen_sublayers: dict[str, str] = {}
        for physical_name, logical_names in self.packed_modules_mapping.items():
            if not isinstance(physical_name, str) or not physical_name.strip():
                raise ValueError("packed_modules_mapping keys must be non-empty str")
            else:
                pass
            if isinstance(logical_names, str) or not logical_names:
                raise ValueError(
                    f"packed_modules_mapping[{physical_name!r}] must be a non-empty "
                    "sequence of sublayer names"
                )
            else:
                pass
            sublayer_tuple = tuple(logical_names)
            for sublayer in sublayer_tuple:
                if not isinstance(sublayer, str) or not sublayer.strip():
                    raise ValueError(
                        f"packed_modules_mapping[{physical_name!r}] contains invalid "
                        f"sublayer name {sublayer!r}"
                    )
                else:
                    pass
                prior = seen_sublayers.get(sublayer)
                if prior is not None:
                    raise ValueError(
                        f"logical sublayer {sublayer!r} is mapped to both "
                        f"{prior!r} and {physical_name!r}"
                    )
                else:
                    pass
                seen_sublayers[sublayer] = physical_name
            normalized_packed[physical_name] = sublayer_tuple
        object.__setattr__(self, "packed_modules_mapping", normalized_packed)

        normalized_aliases: dict[str, str] = {}
        for alias_key, alias_target in self.name_aliases.items():
            if (
                not isinstance(alias_key, str)
                or not alias_key.strip()
                or not isinstance(alias_target, str)
                or not alias_target.strip()
            ):
                raise ValueError(
                    f"name_aliases entries must be non-empty strings, "
                    f"got {alias_key!r}: {alias_target!r}"
                )
            else:
                pass
            normalized_aliases[alias_key] = alias_target
        object.__setattr__(self, "name_aliases", normalized_aliases)

        prefixes: list[str] = []
        for prefix in self.strip_prefixes:
            if not isinstance(prefix, str) or not prefix:
                raise ValueError(
                    f"strip_prefixes must contain non-empty strings, got {prefix!r}"
                )
            else:
                pass
            prefixes.append(prefix)
        object.__setattr__(self, "strip_prefixes", tuple(prefixes))


@runtime_checkable
class SupportsMergedLoRA(Protocol):
    """Protocol implemented by models that expose a merged LoRA specification."""

    def merged_lora_spec(self) -> MergedLoRASpec:
        """Return the validated merged-LoRA binding specification for this model."""


@dataclass(frozen=True)
class MergedLoRARegistryEntry:
    """Explicit merged-LoRA capability status for one registered architecture."""

    status: MergedLoRAStatus
    reason: str

    def __post_init__(self) -> None:
        if not self.reason or not self.reason.strip():
            raise ValueError("MergedLoRARegistryEntry.reason must be non-empty")
        else:
            pass


def supported_entry(reason: str) -> MergedLoRARegistryEntry:
    return MergedLoRARegistryEntry(status="supported", reason=reason)


def runtime_only_entry(reason: str) -> MergedLoRARegistryEntry:
    return MergedLoRARegistryEntry(status="runtime_only", reason=reason)


def not_capable_entry(reason: str) -> MergedLoRARegistryEntry:
    return MergedLoRARegistryEntry(status="not_capable", reason=reason)


PENDING_AR_MAPPING = "AR stage has not declared SupportsMergedLoRA yet"
PENDING_PIPELINE_MAPPING = (
    "Multi-stage pipeline has not declared SupportsMergedLoRA yet"
)
NON_MERGE_STAGE_ONLY = "Pipeline contains non-LLM encoder/codec or MLX-only stages"

MERGED_LORA_ARCHITECTURE_REGISTRY: dict[str, MergedLoRARegistryEntry] = {
    "ArkasrForConditionalGeneration": not_capable_entry(PENDING_AR_MAPPING),
    "AuKForConditionalGeneration": supported_entry(
        "AuKDit implements SupportsMergedLoRA for packed QKV, SwiGLU, and output projections"
    ),
    "AudarTTSForConditionalGeneration": not_capable_entry(PENDING_AR_MAPPING),
    "BailingMM2NativeForConditionalGeneration": not_capable_entry(
        PENDING_PIPELINE_MAPPING
    ),
    "BailingMMNativeForConditionalGeneration": not_capable_entry(
        PENDING_PIPELINE_MAPPING
    ),
    "BailingMoeV3AudioForConditionalGeneration": not_capable_entry(
        PENDING_PIPELINE_MAPPING
    ),
    "Cosmos3ForConditionalGeneration": not_capable_entry(PENDING_PIPELINE_MAPPING),
    "DotsTTSForConditionalGeneration": not_capable_entry(PENDING_PIPELINE_MAPPING),
    "FireRedTTS3ForConditionalGeneration": not_capable_entry(PENDING_AR_MAPPING),
    "FishOmniModel": not_capable_entry(PENDING_PIPELINE_MAPPING),
    "FishQwen3OmniForCausalLM": not_capable_entry(PENDING_AR_MAPPING),
    "FunAsrNanoForConditionalGeneration": not_capable_entry(PENDING_AR_MAPPING),
    "FunCosyVoice3SGLangModel": not_capable_entry(PENDING_PIPELINE_MAPPING),
    "HYV3SpeechForCausalLM": not_capable_entry(PENDING_AR_MAPPING),
    "HiggsMultimodalQwen3ForConditionalGeneration": not_capable_entry(
        PENDING_AR_MAPPING
    ),
    "LLaDA2MoeModelLM": not_capable_entry(PENDING_AR_MAPPING),
    "LingV3TTSForConditionalGeneration": not_capable_entry(PENDING_PIPELINE_MAPPING),
    "MingV3ASRForConditionalGeneration": not_capable_entry(PENDING_AR_MAPPING),
    "MiniCPMO": not_capable_entry(PENDING_PIPELINE_MAPPING),
    "MiniMaxMusic3ForConditionalGeneration": not_capable_entry(
        PENDING_PIPELINE_MAPPING
    ),
    "MossTTSDelayModel": runtime_only_entry(
        "MossTTSDelaySGLangModel declares packed_modules_mapping on the AR stage"
    ),
    "MossTTSLocalModel": runtime_only_entry(
        "MossTTSLocalSGLangModel declares packed_modules_mapping on the AR stage"
    ),
    "MossTranscribeDiarizeForConditionalGeneration": not_capable_entry(
        PENDING_AR_MAPPING
    ),
    "NEOChatModel": not_capable_entry(PENDING_PIPELINE_MAPPING),
    "Nemotron3DiarizationForAudioFrameClassification": not_capable_entry(
        NON_MERGE_STAGE_ONLY
    ),
    "Nemotron3_5AsrForRNNT": not_capable_entry(PENDING_AR_MAPPING),
    "Nemotron3_5TranscribeForConditionalGeneration": not_capable_entry(
        PENDING_AR_MAPPING
    ),
    "NemotronVoiceChatForCausalLM": not_capable_entry(PENDING_PIPELINE_MAPPING),
    "PersonaPlexForCausalLM": not_capable_entry(PENDING_PIPELINE_MAPPING),
    "Qwen3ASRForConditionalGeneration": not_capable_entry(PENDING_AR_MAPPING),
    "Qwen3OmniMoeForConditionalGeneration": supported_entry(
        "Qwen3OmniThinkerForCausalLM and Qwen3OmniTalker implement SupportsMergedLoRA"
    ),
    "Qwen3TTSForConditionalGeneration": not_capable_entry(PENDING_PIPELINE_MAPPING),
    "StreamingCascadePipeline": not_capable_entry(NON_MERGE_STAGE_ONLY),
    "VoxtralTTSForConditionalGeneration": not_capable_entry(PENDING_AR_MAPPING),
    "WhisperForConditionalGeneration": not_capable_entry(NON_MERGE_STAGE_ONLY),
    "YuE2ForCausalLM": not_capable_entry(PENDING_AR_MAPPING),
    "Zonos2ForCausalLM": not_capable_entry(PENDING_PIPELINE_MAPPING),
    "Zonos3ForCausalLM": not_capable_entry(PENDING_PIPELINE_MAPPING),
}


def resolve_submodule(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted submodule path from model, raising MergedLoRACapabilityError."""
    if not path:
        return model
    else:
        pass
    current: nn.Module = model
    for part in path.split("."):
        child = getattr(current, part, None)
        if not isinstance(child, nn.Module):
            raise MergedLoRACapabilityError(
                f"model {type(model).__name__} is missing target_module_root "
                f"{path!r} (segment {part!r} not found)"
            )
        else:
            pass
        current = child
    return current


def is_sharded_linear_module(module: nn.Module) -> bool:
    """Return whether module looks like a sharded parallel linear layer."""
    if isinstance(module, nn.Linear):
        return False
    else:
        pass
    weight = getattr(module, "weight", None)
    if weight is None or getattr(weight, "ndim", 0) != 2:
        return False
    else:
        pass
    return (
        hasattr(module, "tp_size")
        or hasattr(module, "output_sizes")
        or hasattr(module, "input_size_per_partition")
        or hasattr(module, "output_size_per_partition")
    )


def validate_spec_against_model(model: nn.Module, spec: MergedLoRASpec) -> None:
    """Validate that the resolved model tree actually exposes mergeable linears."""
    root = resolve_submodule(model, spec.target_module_root)
    has_sharded = False
    has_plain = False
    seen_suffixes: set[str] = set()
    for name, module in root.named_modules():
        if not name:
            continue
        else:
            pass
        suffix = name.rsplit(".", 1)[-1]
        if is_sharded_linear_module(module):
            has_sharded = True
            seen_suffixes.add(suffix)
        elif isinstance(module, nn.Linear):
            has_plain = True
            seen_suffixes.add(suffix)
        else:
            pass

    if not spec.allow_plain_linear and not has_sharded:
        if has_plain:
            raise MergedLoRACapabilityError(
                f"model {type(model).__name__} exposes only plain torch.nn.Linear "
                "projections while MergedLoRASpec.allow_plain_linear is False"
            )
        else:
            raise MergedLoRACapabilityError(
                f"model {type(model).__name__} exposes no mergeable linear modules "
                f"under {spec.target_module_root or '<root>'!r}"
            )
    else:
        pass

    if spec.allow_plain_linear and not has_plain and not has_sharded:
        raise MergedLoRACapabilityError(
            f"model {type(model).__name__} exposes no linear modules under "
            f"{spec.target_module_root or '<root>'!r}"
        )
    else:
        pass

    missing_packed = [
        packed_name
        for packed_name in spec.packed_modules_mapping
        if packed_name not in seen_suffixes
    ]
    if missing_packed:
        raise MergedLoRACapabilityError(
            f"model {type(model).__name__} declares packed_modules_mapping entries "
            f"not present under {spec.target_module_root or '<root>'!r}: "
            f"{', '.join(sorted(missing_packed))}"
        )
    else:
        pass


def resolve_merged_lora_spec(
    model: nn.Module,
    *,
    architecture: str | None = None,
) -> MergedLoRASpec:
    """Resolve and validate the MergedLoRASpec for a loaded model instance."""
    if architecture is not None:
        config_cls = PIPELINE_CONFIG_REGISTRY.configs.get(architecture)
        canonical_arch = (
            getattr(config_cls, "architecture", architecture)
            if config_cls is not None
            else architecture
        )
        entry = MERGED_LORA_ARCHITECTURE_REGISTRY.get(canonical_arch)
        if entry is not None and entry.status != "supported":
            raise MergedLoRACapabilityError(
                f"architecture {canonical_arch!r} is marked {entry.status} for "
                f"merged LoRA updates: {entry.reason}"
            )
        else:
            pass
    else:
        pass

    if isinstance(model, SupportsMergedLoRA):
        spec = model.merged_lora_spec()
        if not isinstance(spec, MergedLoRASpec):
            raise MergedLoRACapabilityError(
                f"{type(model).__name__}.merged_lora_spec() returned "
                f"{type(spec).__name__}, expected MergedLoRASpec"
            )
        else:
            pass
        validate_spec_against_model(model, spec)
        return spec
    else:
        pass

    raw_packed = getattr(model, "packed_modules_mapping", None)
    if isinstance(raw_packed, Mapping) and raw_packed:
        spec = MergedLoRASpec(
            packed_modules_mapping={str(k): tuple(v) for k, v in raw_packed.items()},
            allow_plain_linear=False,
        )
        validate_spec_against_model(model, spec)
        return spec
    else:
        pass

    raise MergedLoRACapabilityError(
        f"model {type(model).__name__} does not implement SupportsMergedLoRA "
        "and does not declare a non-empty packed_modules_mapping"
    )


def audit_merged_lora_registry(
    registered_architectures: Mapping[str, object] | None = None,
) -> list[str]:
    """Return missing or stale registry entries relative to PIPELINE_CONFIG_REGISTRY."""
    if registered_architectures is None:
        canonical_architectures: set[str] = set()
        for config_cls in PIPELINE_CONFIG_REGISTRY.configs.values():
            arch = getattr(config_cls, "architecture", None)
            if isinstance(arch, str) and arch:
                canonical_architectures.add(arch)
            else:
                pass
    else:
        canonical_architectures = set(registered_architectures.keys())

    violations: list[str] = []
    missing = sorted(
        canonical_architectures - set(MERGED_LORA_ARCHITECTURE_REGISTRY.keys())
    )
    stale = sorted(
        set(MERGED_LORA_ARCHITECTURE_REGISTRY.keys()) - canonical_architectures
    )
    for arch in missing:
        violations.append(
            f"missing merged-LoRA registry entry for {arch!r}; add an entry to "
            "MERGED_LORA_ARCHITECTURE_REGISTRY in sglang_omni/lora/capability.py"
        )
    for arch in stale:
        violations.append(
            f"stale merged-LoRA registry entry for {arch!r}; remove or update it in "
            "MERGED_LORA_ARCHITECTURE_REGISTRY in sglang_omni/lora/capability.py"
        )
    return violations
