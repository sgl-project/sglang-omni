# SPDX-License-Identifier: Apache-2.0
"""The published outputs of one Session under test, read back as labels, controls and turns."""

import time
from collections.abc import Callable

from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    CONTROL,
    SESSION_CONTROL,
    TEXT_DELTA,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
)

TURN_CONTROLS = (TURN_QUEUED, TURN_STARTED, TURN_DONE, TURN_EMPTY, TURN_INTERRUPTED)
WAIT_S = 5


def eventually(
    condition: Callable[[], bool], step: Callable[[], object] | None = None
) -> None:
    """Wait up to WAIT_S for `condition`, running `step` (a short sleep by default) between checks."""
    deadline = time.monotonic() + WAIT_S
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        if step is None:
            time.sleep(0.01)
        else:
            step()


def control_chunk(
    kind: str,
    t_start_ms: float = 0.0,
    *,
    seq: int = 0,
    turn: int | None = None,
    **fields: object,
) -> TimedChunk:
    """A session control event of type `kind` at `t_start_ms`; `fields` join its payload."""
    payload = {"type": kind, **fields}
    return TimedChunk(CONTROL, t_start_ms, 0, seq, payload, SESSION_CONTROL, turn=turn)


class RecordedOutputs:
    def __init__(self) -> None:
        self.outputs: list[TimedChunk] = []

    def labels(self) -> list[str]:
        return [control_type(o) or o.format for o in self.outputs]

    def controls(self) -> list[str]:
        return [control_type(o) for o in self.outputs if o.format == SESSION_CONTROL]

    def turn_controls(self) -> list[tuple[str, int]]:
        """(control type, turn) of every published turn control."""
        return [
            (control_type(o), o.turn)
            for o in self.outputs
            if control_type(o) in TURN_CONTROLS
        ]

    def turns(self) -> list[tuple[str, int]]:
        """(control type, turn) of every published turn control and text delta."""
        return [
            (control_type(o) or o.format, o.turn)
            for o in self.outputs
            if o.format == TEXT_DELTA or control_type(o) in TURN_CONTROLS
        ]

    def heard_ms(self) -> float:
        return sum(o.duration_ms for o in self.outputs if o.modality == AUDIO)

    def text(self) -> str:
        return "".join(
            o.payload["text"] for o in self.outputs if o.format == TEXT_DELTA
        )
