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
    FLOAT32 = "realtime-f32-v1"


def resolve_trace_format(engine: str, trace_format: str | None) -> TraceFormat:
    if trace_format is not None:
        return TraceFormat(trace_format)
    elif engine == "sglang":
        return TraceFormat.PCM16
    elif engine == "vllm":
        return TraceFormat.FLOAT32
    else:
        raise ValueError(f"--trace-format is required for engine {engine!r}")


class TraceRow(BaseModel):
    model_config = ConfigDict(strict=True)

    direction: str
    time_s: float
    event: dict[str, JsonValue]
    client_source: dict[str, JsonValue] | None = None


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
                client_source=record.get("client_source"),
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
    trace: TextIO, samples: NDArray[np.int16], packet_samples: int
) -> Iterator[CaptureRecord]:
    index = 0
    output_rate = None
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
            elif row.direction == "receive" and kind == "session.updated":
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
            elif row.direction == "receive" and kind == "response.output_audio.delta":
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


def parse_float32_trace(
    trace: TextIO, samples: NDArray[np.int16], packet_samples: int, sample_rate: int
) -> Iterator[CaptureRecord]:
    index = 0
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
                source = row.client_source or {}
                expected = samples[
                    index * packet_samples : (index + 1) * packet_samples
                ]
                valid = len(expected)
                if (
                    source.get("index"),
                    source.get("valid_samples"),
                    source.get("padded_samples"),
                ) != (index, valid, packet_samples - valid):
                    raise ValueError("append client_source index/valid/padded mismatch")
                elif (event.get("format"), event.get("sample_rate_hz")) != (
                    "pcm_f32le",
                    sample_rate,
                ):
                    raise ValueError("append is not pcm_f32le at 16 kHz")
                else:
                    pass
                start = source.get("start_s")
                if type(start) not in (int, float):
                    raise ValueError("append source start is not index * 80 ms")
                else:
                    pass
                audio = decode_b64(event.get("audio"))
                record.source_start_s = start
                if len(audio) != 4 * packet_samples:
                    raise ValueError("serialized frame is not 1280 float32 samples")
                else:
                    pass
                scaled = np.frombuffer(audio, "<f4").astype(np.float64) * 32768
                if (
                    not np.isfinite(scaled).all()
                    or not np.array_equal(scaled[:valid], expected.astype(np.float64))
                    or np.any(scaled[valid:])
                ):
                    raise ValueError(
                        "serialized float32 frame differs from input.pcm/zero padding"
                    )
                else:
                    pass
            elif row.direction == "receive" and kind == "response.output_audio.delta":
                if event.get("format") != "pcm16":
                    raise ValueError("audio delta is not pcm16")
                else:
                    pass
                rate = event.get("sample_rate_hz")
                record.output_rate = rate if type(rate) is int else None
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
