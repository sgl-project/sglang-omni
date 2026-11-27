"""Vendor wrapper for sglang.srt.hardware_backend.npu.attention.ascend_kda_backend.

Centralize third-party imports and apply monkey patches here.

Patch applied to fused_kda_gate_npu (named fused_kda_gate on newer upstream):
  - Pass the gate projection as a contiguous tensor. The sgl-kernel-npu KDA
    gate kernel requires contiguous input, while AscendKDAAttnBackend hands it
    the gate slice of the packed mixed_qkv buffer, which is a non-contiguous
    view.
This patch can be removed once upstream SGLang or sgl-kernel-npu accepts a
non-contiguous gate input.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 0.5.x exports fused_kda_gate_npu; newer upstream renamed it fused_kda_gate.
_GATE_NAMES = ("fused_kda_gate_npu", "fused_kda_gate")


def patch_gate(module: object, name: str) -> bool:
    original = getattr(module, name, None)
    if original is None or getattr(original, "_omni_contiguous_patch", False):
        return False

    def kda_gate(gate, *args, **kwargs):
        return original(gate.contiguous(), *args, **kwargs)

    kda_gate._omni_contiguous_patch = True
    setattr(module, name, kda_gate)
    logger.debug("Patched %s for contiguous gate input", name)
    return True


try:
    from sglang.srt.hardware_backend.npu.attention import ascend_kda_backend
except ImportError:
    # Non-Ascend hosts have no NPU KDA backend to patch.
    ascend_kda_backend = None

if ascend_kda_backend is not None:
    for _gate_name in _GATE_NAMES:
        if patch_gate(ascend_kda_backend, _gate_name):
            break
