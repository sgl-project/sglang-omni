# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker session history and bounded unit request state."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@dataclass(kw_only=True)
class SpokenUnit:
    """The latest unit the model spoke in, the one unit a client stop can still drop from the session, with the sampler state from before it."""

    unit_index: int
    kept_token_count: int
    is_turn_ended_at_start: bool
    generated_history_at_start: list[int]


@dataclass(kw_only=True)
class MiniCPMOThinkerSessionState:
    sampling: MiniCPMODuplexSampling
    is_turn_ended: bool = True
    is_prefix_pending: bool = True
    is_interrupt_pending: bool = False
    force_listen_counter: int = 0
    generated_history: list[int] = field(default_factory=list)
    latest_spoken_unit: SpokenUnit | None = None
    # note (ruoyu): After a stop names a unit the client never received, the next unit keeps this many native tokens and refills the dropped unit with these ids before its own.
    kept_token_count: int | None = None
    refill_token_ids: list[int] = field(default_factory=list)


@dataclass(kw_only=True)
class DuplexUnitRequestData(SGLangARRequestData):
    """One bounded generated unit appended to an SGLang streaming session."""

    thinker_state: MiniCPMOThinkerSessionState
    # note (Junnan Li): Each entry is (token id, thinker hidden state, ends the turn).
    talker_conditions: list[tuple[int, torch.Tensor, bool]] = field(
        default_factory=list
    )
    generated_unit_ids: list[int] = field(default_factory=list)
    pending_unit_token: int | None = None
    is_listen_forced: bool = False
    is_interrupted: bool = False
    unit_index: int = 0
    is_turn_ended_at_start: bool = True
    generated_history_at_start: list[int] = field(default_factory=list)
    enforce_request_limits: bool = True
