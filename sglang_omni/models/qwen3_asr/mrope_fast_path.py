# SPDX-License-Identifier: Apache-2.0
"""Vectorized decode-step MRoPE positions for degenerate ASR inputs.

Qwen3-ASR inherits mrope_section from the Omni thinker config, so sglang treats
the model as mrope and rebuilds mrope_positions per request on every decode
step. ASR has no spatial axes, so that work is pure overhead.

Can set env SGLANG_OMNI_ASR_FAST_MROPE=0 to disable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Attribute stamped by request_builders on MultimodalInputs it constructs with
# degenerate (text-equivalent) mrope: identical rows, delta == 0.
DEGENERATE_MROPE_FLAG = "_asr_degenerate_mrope"

_orig_compute_mrope_positions: Any = None


def _all_degenerate(multimodal_inputs: list[Any]) -> bool:
    for mm_input in multimodal_inputs:
        if mm_input is not None and not getattr(mm_input, DEGENERATE_MROPE_FLAG, False):
            return False
    return True


def _fast_compute_mrope_positions(self: Any, model_runner: Any, batch: Any) -> None:
    """Drop-in replacement for ForwardBatch._compute_mrope_positions.

    Upstream rebuilds the positions with a per-request loop on every decode
    step -- one (3, 1) tensor per request, then a concatenate and a device copy, ref:
    https://github.com/sgl-project/sglang/blob/v0.5.16/python/sglang/srt/model_executor/forward_batch_info.py#L1095-L1109
    Every degenerate ASR request lands on seq_len - 1, so the loop collapses into one broadcast.
    """
    if not (self.forward_mode.is_decode() and _all_degenerate(batch.multimodal_inputs)):
        _orig_compute_mrope_positions(self, model_runner, batch)
        return

    row = self.seq_lens_cpu.to(torch.int64) - 1
    mrope_positions = (
        row.unsqueeze(0)
        .expand(3, -1)
        .contiguous()
        .to(device=model_runner.device, non_blocking=True)
    )

    self.mrope_positions = mrope_positions


def apply_asr_mrope_fast_path() -> None:
    """Patch ForwardBatch._compute_mrope_positions with the fast path."""
    global _orig_compute_mrope_positions

    if os.environ.get("SGLANG_OMNI_ASR_FAST_MROPE", "1") == "0":
        logger.info("[qwen3-asr] fast mrope decode path disabled via env")
        return
    if _orig_compute_mrope_positions is not None:
        return

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    _orig_compute_mrope_positions = (
        ForwardBatch._compute_mrope_positions
    )  # save the original function
    ForwardBatch._compute_mrope_positions = (
        _fast_compute_mrope_positions  # replace with the fast path
    )
    logger.info("[qwen3-asr] fast mrope decode path applied")
