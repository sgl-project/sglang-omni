# SPDX-License-Identifier: Apache-2.0
"""Torch MPS runner for Whisper."""

from __future__ import annotations

import torch

from sglang_omni.model_runner.base import ModelRunner


class WhisperTorchMpsModelRunner(ModelRunner):
    """Whisper's Torch/MPS step, with grad held off across the whole step.

    Omni's scheduler loops do not carry SGLang's DynamicGradMode, so each
    request would otherwise retain its autograd graph.
    """

    model_name = "Whisper"

    @torch.no_grad()
    def prepare_and_forward(
        self,
        forward_batch,
        schedule_batch,
        requests,
        is_prefill,
        *,
        is_lookahead: bool = False,
    ):
        return super().prepare_and_forward(
            forward_batch,
            schedule_batch,
            requests,
            is_prefill,
            is_lookahead=is_lookahead,
        )


__all__ = ["WhisperTorchMpsModelRunner"]
