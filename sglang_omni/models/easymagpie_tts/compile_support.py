# SPDX-License-Identifier: Apache-2.0
"""Whether the installed SGLang can torch.compile the EasyMagpie talker.

SGLang compiles the whole decode forward (Nemotron-H backbone plus the TTS
heads) for each decode CUDA graph batch size. Four SGLang fixes are needed
for that to be correct and pay off; without them the talker stays eager.
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)


def missing_compile_fixes() -> list[str]:
    """The SGLang fixes the talker compile needs that this install lacks."""
    from sglang.kernels.jit.utils import is_arch_support_pdl
    from sglang.kernels.ops.mamba.triton_ops import mamba_ssm
    from sglang.srt.layers.attention.mamba import mamba
    from sglang.srt.layers.moe import fused_moe_native
    from sglang.srt.models import nemotron_h

    missing = []
    # Without it, compiling the decode state updates fails on sm_80/86/89 with
    # "launcher() missing 1 required positional argument".
    passes_pdl_off = '"USE_GDC": False' in inspect.getsource(
        mamba_ssm.selective_state_update
    )
    if not is_arch_support_pdl() and not passes_pdl_off:
        missing.append("Triton launches pass PDL constexprs on GPUs without PDL")
    else:
        pass
    # Without it, Inductor clones each layer's whole Mamba state pool around
    # every decode state update, which makes the compiled step slower than eager.
    if not hasattr(mamba, "mamba2_decode_ssm_state_update"):
        missing.append("Mamba2 decode state updates stay in place under compile")
    else:
        pass
    # Without it, replaying the compiled MoE side-stream switch fails with
    # "User object is no longer alive".
    moe = nemotron_h.NemotronHMoE
    dispatch = moe._forward_core  # noqa: leading-underscore  # upstream spelling
    if "is_compiling" not in inspect.getsource(dispatch):
        missing.append("Nemotron-H MoE runs serially under compile")
    else:
        pass
    # Without it, compiled decode at batch size 1 drops the 2.5x routed expert
    # scale and the talker produces garbled speech.
    native_moe = inspect.getsource(fused_moe_native.fused_moe_forward_native)
    if "routed_scaling_factor" not in native_moe:
        missing.append("native MoE applies routed_scaling_factor")
    else:
        pass
    return missing


def resolve_torch_compile(requested: bool) -> bool:
    """``requested``, unless SGLang lacks a fix the talker compile needs."""
    if not requested:
        return False
    else:
        pass
    missing = missing_compile_fixes()
    if missing:
        logger.warning(
            "EasyMagpie TTS runs the talker without torch.compile; this SGLang "
            "lacks: %s. Upgrade SGLang to compile it.",
            "; ".join(missing),
        )
        return False
    else:
        return True


__all__ = ["missing_compile_fixes", "resolve_torch_compile"]
