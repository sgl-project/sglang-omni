# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker session history and bounded unit request state."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sglang_omni.audio.vad import StatelessVAD
from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.scheduling.sglang_backend.request_data import (
    EmbeddingSpan,
    SGLangARRequestData,
)


@dataclass(kw_only=True)
class ReplyUnit:
    """A unit since the latest reply began: its own ids and spans, so a cut before it can re-feed it as a listen unit, and while it is spoken, where its output begins and the sampler state from before it."""

    unit_index: int
    is_spoken: bool
    kept_token_count: int
    perception_token_ids: list[int]
    perception_embedding_spans: list[EmbeddingSpan]
    is_turn_ended_at_start: bool
    generated_history_at_start: list[int]


@dataclass(kw_only=True)
class MiniCPMOThinkerSessionState:
    sampling: MiniCPMODuplexSampling
    is_turn_ended: bool = True
    is_prefix_pending: bool = True
    is_interrupt_pending: bool = False
    # Note (ruoyu): After a client stop the model, hearing only silence, resumes the stopped answer, so every unit listens until one carries the user's voice.
    is_listen_held: bool = False
    # note (luojiaxuan): One detector per session, since silero carries its recurrent state from one frame to the next.
    speech_detector: StatelessVAD
    drain_units_left: int = 0
    force_listen_counter: int = 0
    generated_history: list[int] = field(default_factory=list)
    reply_units: list[ReplyUnit] = field(default_factory=list)
    is_latest_reply_ended: bool = False
    units_since_last_spoken: int = 0
    # note (ruoyu): After a cut names a unit the client never heard, the next unit keeps this many native tokens and refills the dropped units with these ids and spans before its own.
    kept_token_count: int | None = None
    refill_token_ids: list[int] = field(default_factory=list)
    refill_embedding_spans: list[EmbeddingSpan] = field(default_factory=list)


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
    hears_user: bool = False
    unit_index: int = 0
    perception_token_ids: list[int] = field(default_factory=list)
    perception_embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    is_turn_ended_at_start: bool = True
    generated_history_at_start: list[int] = field(default_factory=list)
    enforce_request_limits: bool = True
