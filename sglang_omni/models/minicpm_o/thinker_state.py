# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker session history and bounded unit request state."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.scheduling.sglang_backend.ar_session import HistoryCut
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@dataclass(kw_only=True)
class MiniCPMOThinkerSessionState:
    sampling: MiniCPMODuplexSampling
    is_turn_ended: bool = True
    is_prefix_pending: bool = True
    force_listen_counter: int = 0
    generated_history: list[int] = field(default_factory=list)
    # note (0xtoward): Session position of each retained finished unit's <unit>; the first one ends the system prompt and reference audio.
    unit_starts: list[int] = field(default_factory=list)

    def drop_oldest_units(
        self, next_unit_start: int, target_start: int
    ) -> HistoryCut | None:
        """Drop the oldest finished units until the next unit starts at or below target_start.

        The newest finished unit stays: its generated tokens are not retained ids yet.
        """
        dropped_unit_count = 0
        while dropped_unit_count + 1 < len(self.unit_starts) and (
            next_unit_start
            - (self.unit_starts[dropped_unit_count] - self.unit_starts[0])
            > target_start
        ):
            dropped_unit_count += 1
        if dropped_unit_count == 0:
            return None
        else:
            cut = HistoryCut(
                ranges=((self.unit_starts[0], self.unit_starts[dropped_unit_count]),)
            )
            self.unit_starts = [
                unit_start - cut.length
                for unit_start in self.unit_starts[dropped_unit_count:]
            ]
            return cut

    def scoreable_unit_count(self, next_unit_start: int, protect_tokens: int) -> int:
        """How many finished units, oldest first, a scored cut chooses among.

        The newest finished unit and every unit starting within protect_tokens of the next unit stay out.
        """
        return sum(
            1
            for unit_start in self.unit_starts[:-1]
            if unit_start < next_unit_start - protect_tokens
        )

    def drop_scored_units(
        self, next_unit_start: int, target_start: int, unit_importance: list[float]
    ) -> HistoryCut | None:
        """Drop finished units, least important first, until the next unit starts at or below target_start.

        unit_importance covers the scoreable units, oldest first. When they do not free enough, the
        oldest of the newer finished units go next; the newest finished unit always stays.
        """
        unit_lengths = [
            end - start for start, end in zip(self.unit_starts, self.unit_starts[1:])
        ]
        drop_order = sorted(
            range(len(unit_importance)), key=lambda index: unit_importance[index]
        ) + list(range(len(unit_importance), len(unit_lengths)))
        needed_length = next_unit_start - target_start
        dropped_indices: set[int] = set()
        dropped_length = 0
        for index in drop_order:
            if dropped_length >= needed_length:
                break
            else:
                dropped_indices.add(index)
                dropped_length += unit_lengths[index]
        if not dropped_indices:
            return None
        else:
            ranges: list[tuple[int, int]] = []
            for index in sorted(dropped_indices):
                start, end = self.unit_starts[index], self.unit_starts[index + 1]
                if ranges and ranges[-1][1] == start:
                    ranges[-1] = (ranges[-1][0], end)
                else:
                    ranges.append((start, end))
            kept_starts: list[int] = []
            dropped_before = 0
            for index, unit_start in enumerate(self.unit_starts):
                if index in dropped_indices:
                    dropped_before += unit_lengths[index]
                else:
                    kept_starts.append(unit_start - dropped_before)
            self.unit_starts = kept_starts
            return HistoryCut(ranges=tuple(ranges))


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
