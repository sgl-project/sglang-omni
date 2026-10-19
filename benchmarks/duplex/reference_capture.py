# SPDX-License-Identifier: Apache-2.0
"""Normalize retained realtime capture formats without changing their clocks."""

from __future__ import annotations

import base64
import binascii
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import TextIO

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, JsonValue


class TraceFormat(str, Enum):
    PCM16 = "realtime-pcm16-v1"
    LEGACY_PCM16 = "realtime-legacy-pcm16-v1"


AUDIO_DELTA = {
    TraceFormat.PCM16: "response.output_audio.delta",
    TraceFormat.LEGACY_PCM16: "response.audio.delta",
}


def resolve_trace_format(trace_format: str | None) -> TraceFormat:
    if trace_format is None:
        return TraceFormat.PCM16
    else:
        return TraceFormat(trace_format)


class TraceRow(BaseModel):
    model_config = ConfigDict(strict=True)

    direction: str
    time_s: float
    event: dict[str, JsonValue]


@dataclass(kw_only=True)
class CaptureRecord:
    line_number: int
    row: TraceRow | None
    read_error: str | None = None
    source_start_s: float | None = None
    payload_error: str | None = None
    output_pcm: bytes | None = None
    output_rate: int | None = None
    declares_output_rate: bool = False


def decode_b64(value: JsonValue) -> bytes:
    if not isinstance(value, str):
        raise ValueError("payload is not a base64 string")
    else:
        pass
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64: {exc}") from exc


def read_trace(trace: TextIO) -> Iterator[CaptureRecord]:
    for line_number, line in enumerate(trace, 1):
        try:
            record = json.loads(line)
            direction, time_s, event = (
                record["direction"],
                record["time_s"],
                record["event"],
            )
            event.get("type")
            if type(time_s) not in (int, float) or not math.isfinite(time_s):
                yield CaptureRecord(
                    line_number=line_number,
                    row=None,
                    read_error=f"trace line {line_number} has malformed clock {time_s!r}",
                )
                continue
            else:
                pass
            row = TraceRow(
                direction=direction,
                time_s=time_s,
                event=event,
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            yield CaptureRecord(
                line_number=line_number,
                row=None,
                read_error=f"trace line {line_number} unreadable: {type(exc).__name__}",
            )
            continue
        else:
            yield CaptureRecord(line_number=line_number, row=row)


def parse_pcm16_trace(
    trace: TextIO,
    samples: NDArray[np.int16],
    packet_samples: int,
    trace_format: TraceFormat = TraceFormat.PCM16,
    profile_rate: int | None = None,
) -> Iterator[CaptureRecord]:
    """profile_rate is the recorded profile's output rate; the legacy format needs it
    because its session.updated declares no audio format."""
    index = 0
    output_rate = profile_rate if trace_format is TraceFormat.LEGACY_PCM16 else None
    for record in read_trace(trace):
        if record.row is None:
            yield record
            continue
        else:
            row = record.row
        event = row.event
        kind = event.get("type")
        try:
            if row.direction == "send" and kind == "input_audio_buffer.append":
                source = event.get("sglang") or {}
                if (
                    not isinstance(source, dict)
                    or source.get("seq") != index
                    or type(source.get("t_start_ms")) not in (int, float)
                ):
                    raise ValueError("append lacks matching sglang seq/t_start_ms")
                else:
                    pass
                pcm = decode_b64(event.get("audio"))
                record.source_start_s = source["t_start_ms"] / 1000
                expected = samples[
                    index * packet_samples : (index + 1) * packet_samples
                ]
                if len(pcm) % 2 or not np.array_equal(
                    np.frombuffer(pcm, "<i2"), expected
                ):
                    raise ValueError("serialized PCM16 differs from input.pcm")
                else:
                    pass
            elif (
                row.direction == "receive"
                and kind == "session.updated"
                and trace_format is TraceFormat.PCM16
            ):
                session = event.get("session") or {}
                audio = session.get("audio") or {}
                output = audio.get("output") or {}
                audio_format = output.get("format") or {}
                rate = (
                    audio_format.get("rate")
                    if audio_format.get("type") == "audio/pcm"
                    else None
                )
                output_rate = rate if type(rate) is int else None
                record.declares_output_rate = True
                record.output_rate = output_rate
            elif row.direction == "receive" and kind == AUDIO_DELTA[trace_format]:
                record.output_rate = output_rate
                record.output_pcm = decode_b64(event.get("delta"))
            else:
                pass
        except ValueError as exc:
            record.payload_error = str(exc)
        if row.direction == "send" and kind == "input_audio_buffer.append":
            index += 1
        else:
            pass
        yield record
