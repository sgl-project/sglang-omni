# SPDX-License-Identifier: Apache-2.0
"""Compare a recorded greedy PersonaPlex session with an offline request."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

import httpx
import numpy as np
import soundfile
from pydantic import BaseModel, JsonValue

from benchmarks.duplex.artifacts import (
    add_server_identity_args,
    replay_run,
    server_identity,
)
from benchmarks.duplex.reference_capture import TraceRow, decode_b64

SAMPLE_RATE = 24000


class OfflineAudio(BaseModel):
    data: str
    format: Literal["wav"]


class OfflineReply(BaseModel):
    text: str
    audio: OfflineAudio


def compare_pcm(session_pcm: bytes, offline_pcm: bytes) -> dict[str, JsonValue]:
    """Report exact PCM equality, lengths and the first differing sample."""
    session_samples = np.frombuffer(session_pcm, dtype="<i2")
    offline_samples = np.frombuffer(offline_pcm, dtype="<i2")
    shared_samples = min(len(session_samples), len(offline_samples))
    differences = np.flatnonzero(
        session_samples[:shared_samples] != offline_samples[:shared_samples]
    )
    if differences.size:
        first_difference = int(differences[0])
    elif len(session_samples) != len(offline_samples):
        first_difference = shared_samples
    else:
        first_difference = None
    return {
        "byte_identical": session_pcm == offline_pcm,
        "session_samples": len(session_samples),
        "offline_samples": len(offline_samples),
        "session_sha256": hashlib.sha256(session_pcm).hexdigest(),
        "offline_sha256": hashlib.sha256(offline_pcm).hexdigest(),
        "first_differing_sample": first_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-run", type=Path, required=True)
    parser.add_argument("--offline-url", required=True, help="Offline server base URL")
    parser.add_argument(
        "--offline-mode",
        choices=("default", "session-reference"),
        default="default",
        help="Record whether the offline server was launched with session_reference",
    )
    parser.add_argument(
        "--voice", required=True, help="Voice used by the recorded session"
    )
    parser.add_argument(
        "--instructions",
        type=Path,
        required=True,
        help="Role prompt used by the recorded session",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    add_server_identity_args(parser)
    arguments = parser.parse_args()
    offline_identity = server_identity(
        arguments.offline_url,
        revision=arguments.server_revision,
        model=arguments.model,
        model_revision=arguments.model_revision,
        runtime=arguments.runtime,
    )
    manifest = json.loads((arguments.session_run / "manifest.json").read_text())
    if manifest["profile"] != "personaplex":
        raise ValueError("Session recording must use the personaplex profile")
    else:
        pass
    replay = replay_run(arguments.session_run)
    if replay["summary"]["passed"] != replay["summary"]["selected"]:
        raise ValueError("Session recording failed protocol replay")
    else:
        pass
    input_pcm = (arguments.session_run / "input.pcm").read_bytes()
    session_chunks: list[bytes] = []
    session_text_parts: list[str] = []
    for line in (arguments.session_run / "continuous.jsonl").read_text().splitlines():
        row = TraceRow.model_validate_json(line)
        if (
            row.direction == "receive"
            and row.event["type"] == "response.output_audio.delta"
        ):
            session_chunks.append(decode_b64(row.event["delta"]))
        elif (
            row.direction == "receive"
            and row.event["type"] == "response.output_audio_transcript.delta"
        ):
            delta = row.event["delta"]
            if not isinstance(delta, str):
                raise ValueError("Transcript delta must be a string")
            else:
                session_text_parts.append(delta)
        else:
            pass
    session_pcm = b"".join(session_chunks)
    input_wav = io.BytesIO()
    soundfile.write(
        input_wav,
        np.frombuffer(input_pcm, dtype="<i2"),
        SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )
    request = {
        "prompt": "data:audio/wav;base64,"
        + base64.b64encode(input_wav.getvalue()).decode(),
        "return_logprob": False,
        "output_modalities": ["text", "audio"],
        "sampling_params": {"temperature": 0.0},
        "stage_params": {
            "preprocessing": {
                "voice": arguments.voice,
                "text_prompt": arguments.instructions.read_text(),
            },
            "lm": {"audio_temperature": 0.0},
        },
    }
    arguments.output.mkdir(parents=True, exist_ok=False)
    with httpx.Client(timeout=arguments.timeout_seconds) as client:
        request_started_seconds = perf_counter()
        response = client.post(
            arguments.offline_url.rstrip("/") + "/generate", json=request
        )
        request_seconds = perf_counter() - request_started_seconds
        response.raise_for_status()
    result = OfflineReply.model_validate_json(response.content)
    offline_wav = base64.b64decode(result.audio.data, validate=True)
    offline_samples, sample_rate = soundfile.read(
        io.BytesIO(offline_wav), dtype="int16"
    )
    if sample_rate != SAMPLE_RATE or offline_samples.ndim != 1:
        raise ValueError("Offline response must be mono 24 kHz audio")
    else:
        pass
    offline_pcm = offline_samples.astype("<i2").tobytes()
    session_text = "".join(session_text_parts)
    (arguments.output / "offline.wav").write_bytes(offline_wav)
    (arguments.output / "session.pcm").write_bytes(session_pcm)
    (arguments.output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (arguments.output / "response.json").write_bytes(response.content)
    report = {
        "offline_mode": arguments.offline_mode,
        "offline_request_seconds": request_seconds,
        "text": {
            "session": session_text,
            "offline": result.text,
            "equal": session_text == result.text,
        },
        "session_manifest": manifest,
        "offline_server": offline_identity,
        "session_replay": replay,
        "input_sha256": hashlib.sha256(input_pcm).hexdigest(),
        "full_reply": compare_pcm(session_pcm, offline_pcm),
        "caller_duration": compare_pcm(session_pcm[: len(input_pcm)], offline_pcm),
        "session_extra_samples": (len(session_pcm) - len(input_pcm)) // 2,
    }
    (arguments.output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["full_reply"], indent=2))
    if not report["full_reply"]["byte_identical"] or not report["text"]["equal"]:
        raise SystemExit(1)
    else:
        pass


if __name__ == "__main__":
    main()
