# SPDX-License-Identifier: Apache-2.0
"""Measure haru_server's realtime path through the public WebSocket API.

Haru is full-duplex, so the model's cost is not a request's latency but a frame's:
after a session starts, every 3200-sample input frame costs one model step, which
emits one 133 ms audio frame. This benchmark reports that step cost (ms/frame and
real-time factor) and the time to the session's first audio frame, and along the
way checks the receipts a full-duplex client relies on: one accepted per append,
a drained receipt whose consumed and accepted durations agree, a completed
response, and the same seed replaying the same audio.

The run needs only the native runtime: no Python model code, no MLX bindings.

    python -m sglang_omni_mlx.native.ci.benchmark_haru \
        --runtime-bin <dir with haru_server> \
        --data-root <root with models/higurashi-ai_Haru-4B> \
        --output <report.json>

The benchmark exits non-zero when a check fails, so it doubles as a smoke test of
a machine's realtime path; pass --no-check to only measure.
"""

from __future__ import annotations

import argparse
import array
import base64
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from websockets.sync.client import connect

from sglang_omni_mlx.native.tests.native_server import SupervisedServer

MODEL_REPO = "higurashi-ai/Haru-4B"
# One full-duplex frame is 3200 16-bit samples at 24 kHz.
SAMPLE_RATE = 24000
FRAME_SAMPLES = 3200
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
EVENT_TIMEOUT_SECONDS = 120
# Two sessions with one seed must replay the same audio; compare their digests.
REPEAT_SESSIONS = 2


@dataclass
class Session:
    """What one session sent, what came back, and how long it took."""

    index: int
    update_s: float
    first_audio_s: float
    input_frames: int = 0
    deltas: int = 0
    step_seconds: list[float] = field(default_factory=list)
    audio_seconds: float = 0.0
    response_s: float = 0.0
    transcript: str = ""
    audio_sha256: str = ""
    audio_rms: float = 0.0
    drained: dict = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return self.deltas

    @property
    def ms_per_frame(self) -> float:
        return 1000 * statistics.fmean(self.step_seconds) if self.step_seconds else 0.0

    @property
    def real_time_factor(self) -> float:
        # Seconds of compute per second of audio, as benchmarks/metrics/wer.py
        # reports it: below 1 runs faster than real time.
        return self.response_s / self.audio_seconds if self.audio_seconds else 0.0


def silence_frames(count: int) -> list[bytes]:
    return [b"\x00\x00" * FRAME_SAMPLES for _ in range(count)]


def frames_from_wav(path: Path, count: int) -> list[bytes]:
    """`count` frames of a 24 kHz mono 16-bit WAV, looped when it is shorter."""
    with wave.open(str(path)) as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (
            1,
            2,
            SAMPLE_RATE,
        ):
            raise SystemExit(
                f"{path} must be 24 kHz mono 16-bit PCM; the server frames at 24 kHz"
            )
        pcm = reader.readframes(reader.getnframes())
    width = FRAME_SAMPLES * 2
    starts = range(0, len(pcm) - width + 1, width)
    frames = [pcm[start : start + width] for start in starts]
    if not frames:
        raise SystemExit(f"{path} is shorter than one {width // 2}-sample frame")
    return [frames[index % len(frames)] for index in range(count)]


def receive_until(socket, stop_types: set[str]) -> list[dict]:
    """Read events until one of stop_types arrives."""
    received = []
    while True:
        event = json.loads(socket.recv(timeout=EVENT_TIMEOUT_SECONDS))
        received.append(event)
        if event["type"] in stop_types:
            return received


def bytes_pcm16(pcm: bytes) -> array.array:
    samples = array.array("h")
    samples.frombytes(pcm)
    return samples


def rms(samples: array.array) -> float:
    """Root-mean-square of PCM16 samples in the -1..1 range the model emits."""
    if not samples:
        return 0.0
    total = sum(float(sample) ** 2 for sample in samples)
    return math.sqrt(total / len(samples)) / 32768.0


def check_drained(drained: dict, accepted_end_ms: float) -> dict[str, bool]:
    return {
        "drained_after_end": True,
        "drained_reports_the_same_input": abs(
            float(drained.get("accepted_end_ms", -1)) - accepted_end_ms
        )
        < 1e-6,
        "consumed_matches_accepted": float(drained.get("consumed_ms", -1))
        == float(drained.get("accepted_end_ms", -2)),
        "nothing_discarded": float(drained.get("discarded_ms", -1)) == 0.0,
    }


def run_session(
    server: SupervisedServer,
    index: int,
    seed: int,
    inputs: list[bytes],
    tail: int,
) -> Session:
    """Stream `inputs` and then `tail` silent frames through one session."""
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime", max_size=None) as socket:
        created = json.loads(socket.recv(timeout=EVENT_TIMEOUT_SECONDS))
        assert created["type"] == "session.created", created
        update = {
            "type": "session.update",
            "event_id": f"bench-update-{index}",
            "session": {
                "output_modalities": ["audio"],
                "sglang": {"sampling": {"seed": seed}},
            },
        }
        update_started = time.perf_counter()
        socket.send(json.dumps(update))
        updated = receive_until(socket, {"session.updated"})[-1]
        assert updated["type"] == "session.updated", updated
        update_seconds = time.perf_counter() - update_started

        session = Session(
            index=index,
            update_s=update_seconds,
            first_audio_s=0.0,
            input_frames=len(inputs),
        )
        capabilities = updated["capabilities"]
        session.checks["realtime_capabilities"] = (
            capabilities["native_full_duplex"] is True
            and capabilities["interaction"] == "native"
        )
        session.checks["session_update_is_acknowledged"] = (
            updated.get("client_event_id") == update["event_id"]
        )
        # A session's update costs the prompt prefill, so time it apart from a frame.
        session.checks["update_below_prefill_budget"] = update_seconds < 60.0

        audio = hashlib.sha256()
        samples: list[bytes] = []
        transcript_parts: list[str] = []
        started = time.perf_counter()
        appends = 0
        frames = len(inputs) + tail
        # A turn ends in silence: the model answers once the user stops talking,
        # so the frames after the input are part of the turn.
        for pcm in inputs + silence_frames(tail):
            socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "event_id": f"bench-append-{index}-{appends}",
                        "audio": base64.b64encode(pcm).decode(),
                        "sglang": {"seq": appends, "t_start_ms": appends * 400.0},
                    }
                )
            )
            appends += 1
        # An input's end drains what the frames above produced, so the received
        # audio is read after it rather than before.
        socket.send(
            json.dumps(
                {
                    "type": "sglang.input_audio.end",
                    "event_id": f"bench-end-{index}",
                    "sglang": {"seq": appends, "t_start_ms": appends * 400.0},
                }
            )
        )

        delta_times: list[float] = []
        accepted_seqs: list[int] = []
        accepted_end_ms = 0.0
        drained: dict = {}
        done: dict = {}
        while not drained:
            event = json.loads(socket.recv(timeout=EVENT_TIMEOUT_SECONDS))
            kind = event["type"]
            if kind == "response.output_audio.delta":
                delta_times.append(time.perf_counter())
                pcm = base64.b64decode(event["delta"])
                audio.update(pcm)
                samples.append(pcm)
            elif kind == "response.output_audio_transcript.delta":
                transcript_parts.append(event["delta"])
            elif kind == "sglang.input_audio.accepted":
                accepted_seqs.append(event["seq"])
                accepted_end_ms = event["accepted_end_ms"]
            elif kind == "sglang.input_audio.ended":
                session.checks["input_end_is_acknowledged"] = (
                    abs(float(event["accepted_end_ms"]) - frames * FRAME_SECONDS * 1000)
                    < 1e-6
                    and event["tail_policy"] == "pad"
                )
            elif kind == "response.done":
                done = event
            elif kind == "sglang.input_audio.drained":
                drained = event
            elif kind == "error":
                raise RuntimeError(f"session {index} failed: {event}")
        session.response_s = time.perf_counter() - started
        session.drained = drained
        session.first_audio_s = delta_times[0] - started if delta_times else 0.0
        session.deltas = len(delta_times)
        session.step_seconds = [
            later - earlier for earlier, later in zip(delta_times, delta_times[1:])
        ]
        session.audio_seconds = session.deltas * FRAME_SECONDS
        session.audio_sha256 = audio.hexdigest()
        session.audio_rms = rms(bytes_pcm16(b"".join(samples)))
        session.transcript = "".join(transcript_parts)
        session.checks["audio_streams"] = session.deltas > 0
        session.checks["every_append_is_accepted"] = accepted_seqs == list(
            range(appends)
        )
        session.checks["accepted_end_matches_input"] = (
            abs(accepted_end_ms - frames * FRAME_SECONDS * 1000) < 1e-6
        )
        session.checks["response_completed"] = bool(done) and (
            done["response"]["status"] == "completed"
            and done["response"]["status_details"]["reason"] == "stop"
        )
        session.checks.update(check_drained(session.drained, accepted_end_ms))
        return session


def summary(sessions: list[Session]) -> dict:
    warm = [session for session in sessions if session.step_seconds]
    repeats = sessions[:REPEAT_SESSIONS]
    deterministic = len(repeats) == REPEAT_SESSIONS and (
        repeats[0].audio_sha256 == repeats[1].audio_sha256
        and repeats[0].transcript == repeats[1].transcript
    )
    return {
        "session_update_s": {
            "cold": sessions[0].update_s,
            "warm": (
                statistics.fmean(session.update_s for session in sessions[1:])
                if len(sessions) > 1
                else None
            ),
        },
        "first_audio_s": {
            "cold": sessions[0].first_audio_s,
            "warm": (
                statistics.fmean(session.first_audio_s for session in sessions[1:])
                if len(sessions) > 1
                else None
            ),
        },
        "ms_per_frame": (
            statistics.fmean(session.ms_per_frame for session in warm) if warm else None
        ),
        "real_time_factor": statistics.fmean(
            session.real_time_factor for session in sessions
        ),
        "audio_seconds": statistics.fmean(
            session.audio_seconds for session in sessions
        ),
        "transcripts": [session.transcript for session in sessions],
        "same_seed_replays_the_same_audio": deterministic,
    }


def device() -> str:
    named = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return named or platform.machine()


def report(sessions: list[Session], startup_s: float, arguments) -> dict:
    checks = {
        f"session{index}.{name}": passed
        for index, session in enumerate(sessions)
        for name, passed in session.checks.items()
    }
    return {
        "device": device(),
        "python": platform.python_version(),
        "model": arguments.model_repo,
        "quantize": arguments.quantize,
        "seed": arguments.seed,
        "frames_per_session": arguments.frames,
        "input": str(arguments.input_wav) if arguments.input_wav else "silence",
        "sessions": [
            {
                "index": session.index,
                "frames": session.frames,
                "input_frames": session.input_frames,
                "session_update_s": round(session.update_s, 4),
                "first_audio_s": round(session.first_audio_s, 4),
                "ms_per_frame": round(session.ms_per_frame, 2),
                "response_s": round(session.response_s, 4),
                "real_time_factor": round(session.real_time_factor, 4),
                "transcript": session.transcript,
                "audio_rms": round(session.audio_rms, 6),
                "audio_sha256": session.audio_sha256,
                "drained": session.drained,
            }
            for session in sessions
        ],
        "summary": summary(sessions),
        "startup_s": startup_s,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    parser.add_argument("--quantize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frames", type=int, default=24, help="input frames per session"
    )
    parser.add_argument(
        "--sessions", type=int, default=3, help="first is cold, the rest warm"
    )
    parser.add_argument(
        "--input-wav",
        type=Path,
        help="24 kHz mono 16-bit WAV to stream; silence when omitted",
    )
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=2.0,
        help="silence streamed after the input, so the model can take its turn",
    )
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--instructions-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-check", action="store_true", help="only measure, never fail"
    )
    arguments = parser.parse_args()

    flags = ["--quantize", str(arguments.quantize)]
    if arguments.reference_audio:
        flags += ["--reference-audio", str(arguments.reference_audio)]
    if arguments.instructions_file:
        flags += ["--instructions-file", str(arguments.instructions_file)]
    model_directory = (
        arguments.data_root / "models" / arguments.model_repo.replace("/", "_")
    )
    tail = round(arguments.tail_seconds / FRAME_SECONDS)
    server = SupervisedServer(
        arguments.runtime_bin / "haru_server", "haru", model_directory, tuple(flags)
    )
    startup_s = float(server.first_event["startup_s"])
    try:
        inputs = (
            frames_from_wav(arguments.input_wav, arguments.frames)
            if arguments.input_wav
            else silence_frames(arguments.frames)
        )
        sessions = [
            run_session(server, index, arguments.seed, inputs, tail)
            for index in range(arguments.sessions)
        ]
    finally:
        server.stop()
    fields = report(sessions, startup_s, arguments)
    text = json.dumps(fields, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.write_text(text + "\n")
    print(text, flush=True)
    if not fields["passed"] and not arguments.no_check:
        sys.exit(1)


if __name__ == "__main__":
    main()
