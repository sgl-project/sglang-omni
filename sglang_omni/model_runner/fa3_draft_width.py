# SPDX-License-Identifier: Apache-2.0
"""Lay out a DSpark draft batch on SGLang's FA3 backend at the draft's real width."""

from __future__ import annotations

from threading import Lock

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.dspark_components.dspark_config import (
    get_dspark_sample_from_anchor,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

FA3_DRAFT_WIDTH_LOCK = Lock()
fa3_draft_width_installed = False


def install_fa3_draft_width() -> None:
    """Make SGLang's FA3 backend lay out a DSpark draft batch at the draft's real width."""
    global fa3_draft_width_installed
    with FA3_DRAFT_WIDTH_LOCK:
        if fa3_draft_width_installed:
            return
        else:
            pass
        sglang_init = FlashAttentionBackend.__init__

        def init_with_draft_width(
            self: FlashAttentionBackend,
            model_runner: ModelRunner,
            skip_prefill: bool = False,
            speculative_step_id: int = 0,
            topk: int = 0,
            speculative_num_steps: int = 0,
            fa_impl_ver: int = 3,
        ) -> None:
            sglang_init(
                self,
                model_runner,
                skip_prefill,
                speculative_step_id,
                topk,
                speculative_num_steps,
                fa_impl_ver,
            )
            # note (wilsonzheng0327): SGLang (through 0.5.22) sizes a DSpark draft batch
            # as gamma rows per request, but a sample_from_anchor=False draft also feeds
            # the anchor row, so every request after the first would attend over shifted
            # rows. Its model runner already sizes logits this way; remove once the FA3
            # backend does too.
            if (
                model_runner.is_draft_worker
                and self.speculative_num_draft_tokens is not None
                and SpeculativeAlgorithm.from_string(
                    get_spec().speculative_algorithm
                ).is_dspark()
                and not get_dspark_sample_from_anchor(
                    model_runner.model_config.hf_config
                )
            ):
                self.speculative_num_draft_tokens = int(
                    get_spec().speculative_num_draft_tokens
                )
            else:
                pass

        FlashAttentionBackend.__init__ = init_with_draft_width
        fa3_draft_width_installed = True
