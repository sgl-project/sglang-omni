# SPDX-License-Identifier: Apache-2.0
"""Decode and speculative MRoPE positions for degenerate ASR inputs.

Qwen3-ASR inherits mrope_section from the Omni thinker config, so sglang treats
the model as mrope and rebuilds mrope_positions per request on every decode
step, and copies every request's mrope delta to the GPU before every speculative
forward. ASR has no spatial axes, so that work is pure overhead.

Set SGLANG_OMNI_ASR_FAST_MROPE=0 to disable both shortcuts.
"""

from __future__ import annotations

import logging
import os

import torch
from sglang.srt.managers.schedule_batch import MultimodalInputs, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Attribute stamped by request_builders on MultimodalInputs it constructs with
# degenerate (text-equivalent) mrope: identical rows, delta == 0.
DEGENERATE_MROPE_FLAG = "_asr_degenerate_mrope"

ORIGINAL_COMPUTE_MROPE_POSITIONS = (
    ForwardBatch._compute_mrope_positions  # noqa: leading-underscore  # inherited method
)
ORIGINAL_COMPUTE_SPEC_MROPE_POSITIONS = ForwardBatch.compute_spec_mrope_positions


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
        ORIGINAL_COMPUTE_MROPE_POSITIONS(self, model_runner, batch)
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


def fast_compute_spec_mrope_positions(
    self: ForwardBatch,
    model_runner: ModelRunner,
    batch: ScheduleBatch,
    seq_positions: torch.Tensor | None = None,
) -> None:
    """Text-only positions for degenerate ASR batches, without the blocking delta copy."""
    if not all_degenerate(batch.multimodal_inputs):
        ORIGINAL_COMPUTE_SPEC_MROPE_POSITIONS(self, model_runner, batch, seq_positions)
    else:
        positions = (
            batch.spec_info.positions if seq_positions is None else seq_positions
        )
        self.mrope_positions = (
            positions.to(torch.int64).flatten().unsqueeze(0).repeat(3, 1)
        )


def apply_asr_mrope_fast_path() -> None:
    """Install the ASR decode and speculative MRoPE fast paths."""
    if os.environ.get("SGLANG_OMNI_ASR_FAST_MROPE", "1") == "0":
        logger.info(
            "[qwen3-asr] fast mrope decode and speculative paths disabled via env"
        )
    else:
        ForwardBatch._compute_mrope_positions = (  # noqa: leading-underscore  # inherited method
            fast_compute_mrope_positions
        )
        ForwardBatch.compute_spec_mrope_positions = fast_compute_spec_mrope_positions
        logger.info("[qwen3-asr] fast mrope decode and speculative paths applied")
