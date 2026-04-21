# SPDX-License-Identifier: Apache-2.0
"""Shared mem_fraction_static precedence resolution for Qwen3 speech launchers.

Note (Ratish, Chenyang):

Used by the two Qwen3-Omni speech launchers that expose three flags
(--mem-fraction-static, --thinker-mem-fraction-static,
--talker-mem-fraction-static) side by side. Single-flag launchers keep
their ~8-line inline block; the 20-line 3-flag block is the only variant
that earns a shared helper.
"""
from __future__ import annotations

from sglang_omni.config.schema import PipelineConfig


def resolve_and_apply_speech_mem_fraction(
    config: PipelineConfig,
    *,
    global_mem_fraction_static: float | None,
    thinker_mem_fraction_static: float | None,
    talker_mem_fraction_static: float | None,
    thinker_stage: str = "thinker",
    talker_stage: str = "talker_ar",
) -> tuple[float | None, float | None]:
    """Validate flag ranges, resolve per-stage precedence, and apply overrides.

    Precedence: a stage-specific flag wins; otherwise the global flag is the
    fallback. Returns `(thinker_resolved, talker_resolved)` so the caller can
    log what was actually applied.
    """
    for flag_name, value in (
        ("--mem-fraction-static", global_mem_fraction_static),
        ("--thinker-mem-fraction-static", thinker_mem_fraction_static),
        ("--talker-mem-fraction-static", talker_mem_fraction_static),
    ):
        if value is not None and not 0.0 < value < 1.0:
            raise ValueError(f"{flag_name} must be > 0 and < 1, got {value}")

    thinker_resolved = (
        thinker_mem_fraction_static
        if thinker_mem_fraction_static is not None
        else global_mem_fraction_static
    )
    talker_resolved = (
        talker_mem_fraction_static
        if talker_mem_fraction_static is not None
        else global_mem_fraction_static
    )
    if thinker_resolved is not None:
        config.apply_server_args_overrides(
            stage_name=thinker_stage,
            overrides={"mem_fraction_static": thinker_resolved},
        )
    if talker_resolved is not None:
        config.apply_server_args_overrides(
            stage_name=talker_stage,
            overrides={"mem_fraction_static": talker_resolved},
        )
    return thinker_resolved, talker_resolved
