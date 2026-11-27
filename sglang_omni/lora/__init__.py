# SPDX-License-Identifier: Apache-2.0
"""Engine-side merged LoRA weight management and synchronized IPC weight transfer."""

from __future__ import annotations

from sglang_omni.lora.capability import (
    MERGED_LORA_ARCHITECTURE_REGISTRY,
    MergedLoRACapabilityError,
    MergedLoRARegistryEntry,
    MergedLoRASpec,
    SupportsMergedLoRA,
    audit_merged_lora_registry,
    resolve_merged_lora_spec,
)
from sglang_omni.lora.ipc_wire import (
    decode_wire_tensor,
    encode_cuda_ipc_handle,
    encode_inline_tensor,
    release_unpublished_ipc_handle,
)
from sglang_omni.lora.merged_manager import (
    LoRAAdapterTensor,
    LoRARequestPayload,
    MergedLoRAManager,
    TargetModuleMatcher,
    build_target_module_matcher,
    normalize_target_modules,
)
from sglang_omni.lora.merged_weights import (
    DEFAULT_DEVICE_PARK_HEADROOM_BYTES,
    MergedLinearWeights,
    ParkingPolicy,
)
from sglang_omni.lora.weight_transfer import (
    LoRAAdapterWeightSource,
    MergedLoRATrainerSync,
    MergedLoRAWorkerSession,
    build_merged_lora_admin_handler,
)

__all__ = [
    "DEFAULT_DEVICE_PARK_HEADROOM_BYTES",
    "MERGED_LORA_ARCHITECTURE_REGISTRY",
    "LoRAAdapterTensor",
    "LoRAAdapterWeightSource",
    "LoRARequestPayload",
    "MergedLinearWeights",
    "MergedLoRACapabilityError",
    "MergedLoRAManager",
    "MergedLoRARegistryEntry",
    "MergedLoRASpec",
    "MergedLoRATrainerSync",
    "MergedLoRAWorkerSession",
    "ParkingPolicy",
    "SupportsMergedLoRA",
    "TargetModuleMatcher",
    "audit_merged_lora_registry",
    "build_merged_lora_admin_handler",
    "build_target_module_matcher",
    "decode_wire_tensor",
    "encode_cuda_ipc_handle",
    "encode_inline_tensor",
    "normalize_target_modules",
    "release_unpublished_ipc_handle",
    "resolve_merged_lora_spec",
]
