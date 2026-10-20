# SPDX-License-Identifier: Apache-2.0
"""Cross-stage state for YuE2."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

from .constants import (
    DEFAULT_ABC_MAX_TOKENS,
    DEFAULT_COT,
    DEFAULT_ODE_STEPS,
    DEFAULT_SEMANTIC_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_STYLE,
    DEFAULT_VAE_CORE_FRAMES,
    DEFAULT_VAE_HALO_FRAMES,
)


@dataclass
class Yue2State(DeclarativeStateBase):
    """Serializable request state passed from preprocessing to the synth stage."""

    style: str = wire(DEFAULT_STYLE, codec="str")
    lyrics: str = wire("", codec="str")
    cot: str = wire(DEFAULT_COT, codec="str")
    seed: int = wire(DEFAULT_SEED, codec="int")
    abc: str | None = wire(None, codec="raw")
    cfg_scale: float = wire(1.0, codec="float")
    abc_max_tokens: int = wire(DEFAULT_ABC_MAX_TOKENS, codec="int")
    semantic_max_tokens: int = wire(DEFAULT_SEMANTIC_MAX_TOKENS, codec="int")
    ode_steps: int = wire(DEFAULT_ODE_STEPS, codec="int")
    vae_core_frames: int = wire(DEFAULT_VAE_CORE_FRAMES, codec="int")
    vae_halo_frames: int = wire(DEFAULT_VAE_HALO_FRAMES, codec="int")
    finish_reason: str | None = wire(None, codec="str")


__all__ = ["Yue2State"]
