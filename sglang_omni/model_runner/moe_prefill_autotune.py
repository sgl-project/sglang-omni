# SPDX-License-Identifier: Apache-2.0
"""Opt-in FlashInfer MoE autotune pass at prefill-sized token counts.

SGLang's built-in FlashInfer autotune warmup (``BaseRunner.warmup`` ->
``_flashinfer_autotune``) runs a single decode-shaped dummy forward whose
token count equals the decode runner's max batch size (32 for the current
Qwen3-Omni talker stage). The trtllm-gen fused-MoE tuning buckets are
generated from
``tune_max_num_tokens = next_power_of_2(num_tokens)`` of the tuned forward
(SGLang passes ``next_power_of_2(a_q.shape[0])`` per call), so only buckets
up to that decode batch size ever get profiled. Runtime prefill / mixed-chunk
extend batches (hundreds to thousands of tokens) then miss the autotune cache
and fall back to the untuned default tactic (-1).

This module adds an opt-in second autotune pass: one dummy EXTEND forward at
a prefill-sized token count. FlashInfer's trtllm-gen MoE runner profiles its
full hybrid bucket set up to that token count in a single ``choose_one`` pass,
and, when cache reuse is enabled, the results merge into the same per-model
autotune cache JSON the decode warmup wrote.

Opt in with ``SGLANG_OMNI_MOE_AUTOTUNE_TALKER_TOKENS=<max_tokens>``. Zero or
unset preserves current startup behavior.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

MOE_AUTOTUNE_PREFILL_TOKENS_ENV = "SGLANG_OMNI_MOE_AUTOTUNE_PREFILL_TOKENS"
MOE_AUTOTUNE_TALKER_TOKENS_ENV = "SGLANG_OMNI_MOE_AUTOTUNE_TALKER_TOKENS"


def moe_autotune_prefill_tokens(
    env_var: str = MOE_AUTOTUNE_PREFILL_TOKENS_ENV,
) -> int:
    """Parse the opt-in token-count knob; zero means disabled."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; expected an integer, ignoring.",
            env_var,
            raw,
        )
        return 0


def maybe_autotune_prefill_moe(
    model_runner,
    env_var: str = MOE_AUTOTUNE_PREFILL_TOKENS_ENV,
) -> None:
    """Run one prefill-sized dummy EXTEND forward under FlashInfer autotune.

    This must run after normal model-runner warmup and on every TP rank at the
    same point because the dummy forward participates in TP collectives.
    """
    tokens = moe_autotune_prefill_tokens(env_var)
    if tokens <= 0:
        return

    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner.flashinfer_autotune import (
        flashinfer_autotune_context,
        should_run_flashinfer_autotune,
    )

    mr = model_runner
    if not should_run_flashinfer_autotune(mr):
        logger.info(
            "%s=%d set but FlashInfer autotune is not applicable to this "
            "runner (device/backend); skipping prefill MoE autotune.",
            env_var,
            tokens,
        )
        return

    runner = getattr(mr, "eager_runner", None)
    if runner is None:
        logger.warning(
            "%s=%d set but eager_runner is unavailable; skipping prefill MoE "
            "autotune.",
            env_var,
            tokens,
        )
        return

    # Mirror pp_parallel_deep_gemm_warmup's shape alignment: _dummy_run does
    # not pad like the real flow, so align to CP padding and attention TP.
    import math

    from sglang.srt.layers.cp.padding import get_cp_padding_align_size
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils.common import ceil_align, require_mlp_sync

    align = max(get_cp_padding_align_size(), 1)
    attn_tp_size = get_parallel().attn_tp_size
    if require_mlp_sync(mr.server_args) and attn_tp_size > 1:
        align = math.lcm(align, attn_tp_size)
    tokens = ceil_align(tokens, align)

    started_at = time.perf_counter()
    logger.info(
        "Prefill MoE autotune: dummy EXTEND forward at %d tokens (tp_rank=%d).",
        tokens,
        mr.ps.tp_rank,
    )
    buffers = runner._alloc_dummy_decode_buffers(tokens)
    canary = getattr(mr, "canary_manager", None)
    run_ctx = (
        canary.with_active_single_forward_manager(0) if canary is not None else None
    )
    # SGLang represents this synthetic token shape as a request batch. The
    # talker's nested backbone normally substitutes decode feedback for that
    # axis, but its buffers are capped at max_running_requests. Real prefill
    # supplies input_embeds and bypasses the feedback branch, so do the same
    # only for this dummy forward and restore the live decode state afterward.
    backbone = getattr(getattr(mr, "model", None), "model", None)
    has_feedback_mode = hasattr(backbone, "_cp_enabled")
    feedback_mode = getattr(backbone, "_cp_enabled", False)
    if has_feedback_mode:
        backbone._cp_enabled = False
    try:
        with flashinfer_autotune_context(mr, skip_logits=True):
            runner._dummy_run(
                batch_size=tokens,
                run_ctx=run_ctx,
                forward_mode_override=ForwardMode.EXTEND,
                buffers=buffers,
            )
    finally:
        if has_feedback_mode:
            backbone._cp_enabled = feedback_mode
    logger.info("Prefill MoE autotune done in %.2fs.", time.perf_counter() - started_at)
