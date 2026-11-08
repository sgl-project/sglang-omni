# SPDX-License-Identifier: Apache-2.0
"""A simulated client player: how much of a reply's audio it has played by a given time."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ReplyAudioChunk:
    received_s: float
    sample_count: int


def played_sample_count(
    chunks: list[ReplyAudioChunk],
    *,
    sample_rate: int,
    start_buffer_ms: float,
    at_s: float,
) -> int:
    """Samples of one reply played by at_s at 1.0x; like the playground player, it starts once start_buffer_ms of audio is queued and buffers again after running dry."""
    start_buffer_samples = round(start_buffer_ms * sample_rate / 1000)
    received_samples = 0
    played_samples = 0
    started_s: float | None = None
    played_at_start_samples = 0
    for chunk in chunks:
        if chunk.received_s > at_s:
            break
        elif started_s is not None:
            played_samples = min(
                received_samples,
                played_at_start_samples
                + round((chunk.received_s - started_s) * sample_rate),
            )
            if played_samples == received_samples:
                started_s = None
            else:
                pass
        else:
            pass
        received_samples += chunk.sample_count
        if (
            started_s is None
            and received_samples - played_samples >= start_buffer_samples
        ):
            started_s = chunk.received_s
            played_at_start_samples = played_samples
        else:
            pass
    if started_s is not None:
        return min(
            received_samples,
            played_at_start_samples + round((at_s - started_s) * sample_rate),
        )
    else:
        return played_samples
