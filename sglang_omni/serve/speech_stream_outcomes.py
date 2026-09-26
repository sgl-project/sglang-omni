# SPDX-License-Identifier: Apache-2.0
"""Terminal state of recent raw PCM speech streams, kept for a follow-up GET.

A raw PCM stream sends its headers before generation ends and has no in-band
channel for trailing metadata, so the server keeps each finished stream's
terminal state by request id for ``GET /v1/audio/speech/{request_id}``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from sglang_omni.client.types import UsageInfo


@dataclass(frozen=True, kw_only=True)
class SpeechStreamOutcome:
    request_id: str
    finish_reason: str | None
    usage: UsageInfo | None

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict() if self.usage is not None else None,
        }


class SpeechStreamOutcomes:
    """Bounded FIFO of the most recent stream outcomes by request id."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max_entries
        self.outcomes_by_request_id: OrderedDict[str, SpeechStreamOutcome] = (
            OrderedDict()
        )

    def record(
        self, request_id: str, finish_reason: str | None, usage: UsageInfo | None
    ) -> None:
        self.outcomes_by_request_id[request_id] = SpeechStreamOutcome(
            request_id=request_id, finish_reason=finish_reason, usage=usage
        )
        while len(self.outcomes_by_request_id) > self.max_entries:
            self.outcomes_by_request_id.popitem(last=False)

    def get(self, request_id: str) -> SpeechStreamOutcome | None:
        return self.outcomes_by_request_id.get(request_id)
