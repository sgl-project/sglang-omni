# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@dataclass
class MiniCPMOThinkerSessionState:
    current_turn_ended: bool = True
    prefix_pending: bool = True
    force_listen_counter: int = 0
    generated_history: list[int] = field(default_factory=list)


@dataclass
class DuplexUnitRequestData(SGLangARRequestData):
    """One bounded generated unit appended to an SGLang streaming session."""

    thinker_state: MiniCPMOThinkerSessionState | None = None
    prefill_schema: list[Any] = field(default_factory=list)
    unit_pairs: list[tuple[int, torch.Tensor, bool]] = field(default_factory=list)
    generated_unit_ids: list[int] = field(default_factory=list)
    pending_unit_token: int | None = None
    forced_listen: bool = False
    sampling_config: dict[str, Any] = field(default_factory=dict)
    enforce_request_limits: bool = True
