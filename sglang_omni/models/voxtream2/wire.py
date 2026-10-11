# SPDX-License-Identifier: Apache-2.0
"""Bounded JSON frames for a local, operator-owned model subprocess."""

import json
from typing import BinaryIO

MAX_FRAME_BYTES = 1024 * 1024
PROTOCOL_VERSION = 1


def read_frame(stream: BinaryIO) -> dict[str, object]:
    line = stream.readline(MAX_FRAME_BYTES + 1)
    if not line:
        raise EOFError("VoXtream2 model worker closed its output")
    else:
        pass
    if len(line) > MAX_FRAME_BYTES or not line.endswith(b"\n"):
        raise RuntimeError("Invalid or oversized VoXtream2 worker frame")
    else:
        pass
    value = json.loads(line)
    if not isinstance(value, dict):
        raise RuntimeError("VoXtream2 worker frame must be an object")
    else:
        pass
    return value


def write_frame(stream: BinaryIO, value: dict[str, object]) -> None:
    try:
        line = (
            json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "VoXtream2 invalid request: message cannot be serialized"
        ) from exc
    if len(line) > MAX_FRAME_BYTES:
        raise ValueError("VoXtream2 invalid request: worker message exceeds 1 MiB")
    else:
        pass
    stream.write(line)
    stream.flush()
