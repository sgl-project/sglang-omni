# SPDX-License-Identifier: Apache-2.0
"""XPU corrections to SGLang runtime helpers that assume CUDA semantics."""

from __future__ import annotations

import logging
import threading

import torch

logger = logging.getLogger(__name__)

_PATCH_LOCK = threading.Lock()
_MEM_PATCHED = False


def patch_available_gpu_memory_for_xpu() -> bool:
    """Cap XPU free memory at ``total - memory_reserved``, so the allocator's
    cached-but-freed blocks are not offered to the KV pool (SGLang caps at
    ``total - memory_allocated``, which counts them free, and on Arc Pro B60
    ``mem_get_info`` cannot catch it: it always returns the full capacity --
    measured 17.91 GiB reported against 13.91 GiB real).

    No-op off XPU; idempotent; thread-safe.
    """
    global _MEM_PATCHED
    if _MEM_PATCHED:
        return True
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return False
    if not hasattr(torch.xpu, "mem_get_info"):
        return False
    with _PATCH_LOCK:
        if _MEM_PATCHED:  # re-check under lock
            return True
        try:
            from sglang.srt.utils import common as _common
        except Exception:  # noqa: BLE001 - SGLang layout drift must not crash startup
            return False

        _orig = _common.get_available_gpu_memory

        def get_available_gpu_memory(  # noqa: ANN001, ANN201
            device, gpu_id, distributed=False, empty_cache=True, cpu_group=None
        ):
            # Normalize torch.device / "xpu:0" to the backend type before matching.
            dev_type = getattr(device, "type", None) or str(device).split(":", 1)[0]
            if dev_type == "xpu":
                if empty_cache:
                    try:
                        torch.xpu.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                free_gpu_memory, total = torch.xpu.mem_get_info(gpu_id)
                # Some Intel GPUs (e.g. Arc Pro B60) report total capacity as
                # "free" regardless of live allocations, causing KV-pool OOM.
                # Cross-check against the allocator, which does track correctly.
                # Prefer memory_reserved (the caching allocator's pool); fall
                # back to memory_allocated on torch+xpu builds that lack it, and
                # skip the cross-check entirely if neither exists.
                reserved_fn = getattr(torch.xpu, "memory_reserved", None) or getattr(
                    torch.xpu, "memory_allocated", None
                )
                if reserved_fn is not None:
                    torch_free = total - reserved_fn(gpu_id)
                    free_gpu_memory = min(free_gpu_memory, torch_free)
                if distributed:
                    # Match CUDA path: reduce to min free across the group.
                    tensor = torch.tensor(float(free_gpu_memory), dtype=torch.float32)
                    torch.distributed.all_reduce(
                        tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
                    )
                    free_gpu_memory = tensor.item()
                # SGLang returns GB (callers multiply by 1<<30 to get bytes).
                return free_gpu_memory / (1 << 30)
            return _orig(
                device,
                gpu_id,
                distributed=distributed,
                empty_cache=empty_cache,
                cpu_group=cpu_group,
            )

        # Rebind the source module + every module that captured the function by
        # value, so no stale reference to the original survives.
        import sys

        _common.get_available_gpu_memory = get_available_gpu_memory
        for mod in list(sys.modules.values()):
            try:
                if getattr(mod, "get_available_gpu_memory", None) is _orig:
                    mod.get_available_gpu_memory = get_available_gpu_memory
            except (
                Exception
            ):  # noqa: BLE001 - lazy-import shims may raise on attr access
                continue

        _MEM_PATCHED = True
    logger.info(
        "Intel XPU: patched get_available_gpu_memory to use torch.xpu.mem_get_info "
        "(real free memory, matching the CUDA path)."
    )
    return True


__all__ = ["patch_available_gpu_memory_for_xpu"]
