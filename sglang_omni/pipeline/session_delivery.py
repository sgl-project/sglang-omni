# SPDX-License-Identifier: Apache-2.0
"""Real-time delivery of reply audio from a Session to its client."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import replace

from sglang_omni.proto.session import OutputChunk, wire_size
from sglang_omni.proto.streams import (
    AUDIO_FORMATS,
    PCM_F32,
    TURN_DONE,
    TURN_INTERRUPTED,
    control_type,
    is_interrupted_reply,
)

# Note (Junnan Li): Once a held chunk is cut at the horizon, the next cut waits until at least this much more may leave.
MIN_SLICE_MS = 40

Budgeted = tuple[OutputChunk, int]


def split_audio(
    output: OutputChunk, size: int, head_ms: float
) -> tuple[Budgeted, Budgeted] | None:
    """The first head_ms of an audio chunk and the rest, cut on a sample boundary, or None when head_ms holds no whole sample; the two sizes add up to `size`."""
    payload = output.payload
    assert isinstance(payload, bytes)
    sample_bytes = 4 if output.format == PCM_F32 else 2
    cut = (
        int(len(payload) * head_ms / output.duration_ms) // sample_bytes * sample_bytes
    )
    if cut == 0:
        return None
    else:
        pass
    head_duration_ms = output.duration_ms * cut / len(payload)
    head = replace(
        output, duration_ms=head_duration_ms, payload=payload[:cut], eos=False
    )
    tail = replace(
        output,
        t_start_ms=output.t_start_ms + head_duration_ms,
        duration_ms=output.duration_ms - head_duration_ms,
        payload=payload[cut:],
    )
    head_size = min(wire_size(head.to_dict()), size)
    return (head, head_size), (tail, size - head_size)


class DeliveryPacer:
    """The outputs of one node on their way to the client, with the bytes each holds of the output budget.

    `offer` queues an output and `release` returns what may leave now, in
    order, contiguous audio of one turn joined into chunks of at most the
    lead; the deliverer keeps what it has not yet handed on in `unsent`.
    `wake` is set whenever the queue changes, and `wait` returns on it or
    once the first held chunk may leave.
    """

    def __init__(self, lead_ms: float, wake: asyncio.Event) -> None:
        self.lead_ms = lead_ms
        self.wake = wake
        self.items: deque[Budgeted] = deque()
        self.unsent: deque[Budgeted] = deque()
        # note (luojiaxuan): The client plays a reply from its first audio, so the release of a turn's first audio starts the reply's real time; audio before any turn's is not paced.
        self.reply_start_ms = 0.0
        self.reply_started_at = -math.inf
        self.clock_turn: int | None = None
        self.is_input_ended = False
        # Note (Junnan Li): The audio of the latest turn handed to the client, so an interrupt can say how much of its reply left the server.
        self.sent_turn: int | None = None
        self.sent_ms = 0.0
        # note (luojiaxuan): Where the released audio of released_turn ends on its reply's timeline; its turn.done leaves once the client has played that far.
        self.released_turn: int | None = None
        self.released_end_ms = 0.0

    def offer(self, output: OutputChunk, size: int) -> int:
        """Queue an output; returns the budgeted bytes a turn.interrupted dropped from its turn."""
        if control_type(output) == TURN_INTERRUPTED:
            dropped = self.purge(output.turn)
        else:
            dropped = 0
        self.items.append((output, size))
        self.wake.set()
        return dropped

    def purge(self, turn: int | None) -> int:
        """Drop the held and the unsent chunks of the turn; returns their budgeted bytes."""
        if turn is None:
            return 0
        else:
            dropped = 0
            for queue in (self.items, self.unsent):
                kept = [
                    (output, size)
                    for output, size in queue
                    if not is_interrupted_reply(output, turn)
                ]
                dropped += sum(size for _, size in queue) - sum(
                    size for _, size in kept
                )
                queue.clear()
                queue.extend(kept)
            self.wake.set()
            return dropped

    def sent(self, output: OutputChunk) -> None:
        """Count an output handed to the client toward its turn's delivered audio."""
        if output.kind == "data" and output.format in AUDIO_FORMATS:
            if output.turn != self.sent_turn:
                self.sent_turn, self.sent_ms = output.turn, 0.0
            else:
                pass
            self.sent_ms += output.duration_ms
        else:
            pass

    def delivered_ms(self, turn: int) -> float:
        """The audio of the turn handed to the client, from the reply's first sample."""
        if turn == self.sent_turn:
            return self.sent_ms
        else:
            return 0.0

    def end_input(self) -> None:
        self.is_input_ended = True
        self.wake.set()

    def horizon_ms(self, now: float) -> float:
        return self.reply_start_ms + (now - self.reply_started_at) * 1000 + self.lead_ms

    def take_audio(self, released: list[Budgeted], audio: Budgeted) -> None:
        output, _ = audio
        self.released_turn = output.turn
        self.released_end_ms = output.t_start_ms + output.duration_ms
        released.append(audio)

    def release(self, now: float) -> list[Budgeted]:
        self.wake.clear()
        released: list[Budgeted] = []
        held: deque[Budgeted] = deque()
        for output, size in self.items:
            is_reply_audio = output.kind == "data" and output.format in AUDIO_FORMATS
            if held and (is_reply_audio or control_type(output) == TURN_DONE):
                held.append((output, size))
            elif is_reply_audio and not self.is_input_ended:
                if output.turn != self.clock_turn:
                    self.clock_turn = output.turn
                    self.reply_start_ms = output.t_start_ms
                    self.reply_started_at = now
                else:
                    pass
                horizon_ms = self.horizon_ms(now)
                if output.t_start_ms >= horizon_ms:
                    held.append((output, size))
                elif (
                    output.t_start_ms + output.duration_ms > horizon_ms
                    and output.payload
                ):
                    split = split_audio(output, size, horizon_ms - output.t_start_ms)
                    if split is None:
                        held.append((output, size))
                    else:
                        head, tail = split
                        self.take_audio(released, head)
                        held.append(tail)
                else:
                    self.take_audio(released, (output, size))
            elif (
                control_type(output) == TURN_DONE
                and not self.is_input_ended
                and output.turn == self.released_turn
                and self.horizon_ms(now) - self.lead_ms < self.released_end_ms
            ):
                held.append((output, size))
            else:
                released.append((output, size))
        self.items = held
        # Note (Junnan Li): The input's end or a late release lets many held chunks out at once; contiguous audio of one turn leaves as one chunk per lead, the most pacing ever lets the client hold.
        coalesced: list[Budgeted] = []
        for output, size in released:
            previous, previous_size = coalesced[-1] if coalesced else (None, 0)
            if (
                previous is not None
                and output.kind == "data"
                and output.format in AUDIO_FORMATS
                and previous.kind == "data"
                and previous.format == output.format
                and previous.turn == output.turn
                and not previous.eos
                and math.isclose(
                    previous.t_start_ms + previous.duration_ms, output.t_start_ms
                )
                and previous.duration_ms + output.duration_ms <= self.lead_ms
            ):
                coalesced[-1] = (
                    replace(
                        previous,
                        duration_ms=previous.duration_ms + output.duration_ms,
                        payload=previous.payload + output.payload,
                        eos=output.eos,
                    ),
                    previous_size + size,
                )
            else:
                coalesced.append((output, size))
        return coalesced

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        if self.items and not self.wake.is_set():
            output, _ = self.items[0]
            if control_type(output) == TURN_DONE:
                due_ms = self.released_end_ms + self.lead_ms
            else:
                due_ms = output.t_start_ms + min(output.duration_ms, MIN_SLICE_MS)
            delay_s = (
                self.reply_started_at
                + (due_ms - self.reply_start_ms - self.lead_ms) / 1000
                - loop.time()
            )
            timer = loop.call_later(max(delay_s, 0), self.wake.set)
            await self.wake.wait()
            timer.cancel()
        else:
            await self.wake.wait()
