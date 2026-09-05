# SPDX-License-Identifier: Apache-2.0
"""Lazy decode state for the MLX Qwen3-TTS runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingDecode


@dataclass
class Qwen3TTSMlxPendingDecode(MlxPendingDecode):
    """One lazily launched codec frame.

    ``lazy_tokens`` stays the base contract's group-0 token per request, so
    SGLang's bookkeeping is unchanged. Two fields are added:

    ``lazy_codes``    the full ``[B, num_code_groups]`` frame for the vocoder.
    ``lazy_feedback`` the ``[B, 1, hidden]`` input for the next step, which is
                      what makes a chained step possible without evaluating
                      this one first.
    """

    lazy_codes: Any = None
    lazy_feedback: Any = None

    def feedback_row(self, index: int) -> mx.array:
        """The still-lazy next input for one request in the batch."""
        return self.lazy_feedback[index : index + 1]
