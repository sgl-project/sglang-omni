# SPDX-License-Identifier: Apache-2.0
"""End a Nemotron Transcribe decode once its output repeats itself.

A short sentence of SHORT_PERIOD_MIN to SHORT_PERIOD_MAX tokens repeated
SHORT_MIN_REPEATS times, or any pattern of up to PERIOD_MAX tokens repeated
MIN_REPEATS times, forces the end token at the next step. Shorter patterns keep
the stricter window, so read speech such as "percent sign, percent sign" stays.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
else:
    pass

PERIOD_MAX = 32
SPAN = 32
MIN_REPEATS = 4
SHORT_PERIOD_MIN = 4
SHORT_PERIOD_MAX = 12
SHORT_SPAN = 15
SHORT_MIN_REPEATS = 3


class LoopGuardParams(TypedDict):
    eos_token_id: int
    __req__: Req


def is_periodic(tokens: Sequence[int], start: int, end: int, period: int) -> bool:
    """True when tokens[start:end] repeats with this period."""
    for index in range(start, end - period):
        if tokens[index] != tokens[index + period]:
            return False
        else:
            pass
    return True


def loop_period(tokens: Sequence[int]) -> int:
    """The period of a repeating tail of tokens, or 0 when the tail does not repeat."""
    end = len(tokens)
    while end and tokens[end - 1] < 0:
        end -= 1
    if end < min(SPAN, SHORT_SPAN):
        return 0
    else:
        pass
    last = tokens[end - 1]
    for period in range(1, PERIOD_MAX + 1):
        short = SHORT_PERIOD_MIN <= period <= SHORT_PERIOD_MAX
        if short:
            window = max(SHORT_SPAN, period * SHORT_MIN_REPEATS)
        else:
            window = max(SPAN, period * MIN_REPEATS)
        if end < window or tokens[end - 1 - period] != last:
            continue
        else:
            pass
        start = end - window
        if not is_periodic(tokens, start, end, period):
            continue
        else:
            pass
        # A shorter period that already repeats here has its own, stricter rule.
        if short and any(
            period % shorter == 0 and is_periodic(tokens, start, end, shorter)
            for shorter in range(1, period)
        ):
            continue
        else:
            pass
        return period
    return 0


class NemotronTranscribeLoopGuard(CustomLogitProcessor):
    """Leave only the end token once a row's output has started to loop."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Sequence[LoopGuardParams | None] | None = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits
        else:
            pass
        for row, params in enumerate(custom_param_list):
            if not params:
                continue
            else:
                pass
            if loop_period(params["__req__"].output_ids):
                eos_token_id = int(params["eos_token_id"])
                logits[row, :] = -float("inf")
                logits[row, eos_token_id] = 0.0
            else:
                pass
        return logits


__all__ = ["NemotronTranscribeLoopGuard", "loop_period"]
