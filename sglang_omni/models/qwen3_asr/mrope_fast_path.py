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
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import MultimodalInputs, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
else:
    pass

logger = logging.getLogger(__name__)

# Attribute stamped by request_builders on MultimodalInputs it constructs with
# degenerate (text-equivalent) mrope: identical rows, delta == 0.
DEGENERATE_MROPE_FLAG = "_asr_degenerate_mrope"

_orig_compute_mrope_positions: (
    Callable[[ForwardBatch, ModelRunner, ScheduleBatch], None] | None
) = None


def all_degenerate(multimodal_inputs: list[MultimodalInputs | None]) -> bool:
    for mm_input in multimodal_inputs:
        if mm_input is not None and not getattr(mm_input, DEGENERATE_MROPE_FLAG, False):
            return False
        else:
            pass
    return True


def fast_compute_mrope_positions(
    self: ForwardBatch, model_runner: ModelRunner, batch: ScheduleBatch
) -> None:
    """Drop-in replacement for ForwardBatch._compute_mrope_positions.

    For a batch with multimodal inputs, upstream still walks the requests on
    every decode step to read each mrope_position_delta before one broadcast.
    Every degenerate ASR request lands on seq_len - 1, so the loop collapses
    into one broadcast from seq_lens_cpu.
    """
    if not (self.forward_mode.is_decode() and all_degenerate(batch.multimodal_inputs)):
        _orig_compute_mrope_positions(self, model_runner, batch)
        return
    else:
        pass

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
    else:
        pass
    if _orig_compute_mrope_positions is not None:
        return
    else:
        pass

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    _orig_compute_mrope_positions = (
        ForwardBatch._compute_mrope_positions  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    )  # save the original function
    ForwardBatch._compute_mrope_positions = (  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        fast_compute_mrope_positions  # replace with the fast path
    )
    logger.info("[qwen3-asr] fast mrope decode path applied")
