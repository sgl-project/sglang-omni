# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker session history and bounded unit request state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.scheduling.sglang_backend.history_cut import HistoryCut
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@dataclass(kw_only=True)
class MiniCPMOThinkerSessionState:
    sampling: MiniCPMODuplexSampling
    is_turn_ended: bool = True
    is_prefix_pending: bool = True
    force_listen_counter: int = 0
    generated_history: list[int] = field(default_factory=list)
    # note (0xtoward): Session position of each retained finished unit's <unit>; the first ends the system prompt and reference audio, which always stay.
    unit_starts: list[int] = field(default_factory=list)

    def drop_units(
        self, drop_order: Iterable[int], next_unit_start: int, target_start: int
    ) -> HistoryCut | None:
        """Drop finished units in drop_order until the next unit starts at or below target_start.

        The newest finished unit stays: its outputs join the session ids only with the next unit.
        """
        dropped: set[int] = set()
        for index in drop_order:
            if next_unit_start <= target_start:
                break
            elif index + 1 < len(self.unit_starts):
                dropped.add(index)
                next_unit_start -= self.unit_starts[index + 1] - self.unit_starts[index]
            else:
                pass
        ranges: list[tuple[int, int]] = []
        for index in sorted(dropped):
            start, end = self.unit_starts[index], self.unit_starts[index + 1]
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], end)
            else:
                ranges.append((start, end))
        if not ranges:
            return None
        else:
            cut = HistoryCut(tuple(ranges))
            self.unit_starts = [
                cut.new_position(unit_start)
                for index, unit_start in enumerate(self.unit_starts)
                if index not in dropped
            ]
            return cut


@dataclass(kw_only=True)
class DuplexUnitRequestData(SGLangARRequestData):
    """One bounded generated unit appended to an SGLang streaming session."""

    thinker_state: MiniCPMOThinkerSessionState
    # note (0xtoward): Index of the unit's <unit> among its own token ids.
    unit_token_index: int
    # note (Junnan Li): Each entry is (token id, thinker hidden state, ends the turn).
    talker_conditions: list[tuple[int, torch.Tensor, bool]] = field(
        default_factory=list
    )
    generated_unit_ids: list[int] = field(default_factory=list)
    pending_unit_token: int | None = None
    is_listen_forced: bool = False
    enforce_request_limits: bool = True
