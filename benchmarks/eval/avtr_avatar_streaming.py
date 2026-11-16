# SPDX-License-Identifier: Apache-2.0
"""Per-chunk latency and Session capacity of the avatar stage over running AVTR-1 renderer workers, without the cascade.

Opens --sessions Sessions on the avatar hooks over a pool of the workers at
--renderer-url, feeds each the WAV as the user's audio in 200 ms packets at
real time and, every --reply-every-s, the WAV again at 24 kHz as a reply of
the agent, for --seconds. A chunk's latency runs from the append of the
packet that makes it ready to its last frame published, JPEG encoding
included; a chunk the stage held because its Session fell behind publishes
empty frames and is counted, not timed. The first --warm-up-s are left out.
Also reports each worker's render call (request, rendering, transfer and
JPEG encoding), how many chunks each worker rendered and how often a
Session's next chunk went to a different worker than its last one.

    python -m benchmarks.eval.avtr_avatar_streaming --renderer-url http://127.0.0.1:8020 \\
        --sessions 3 --audio question.wav --output n3.json
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

import httpx
import numpy as np
from scipy.signal import resample_poly

from sglang_omni.models.avtr.renderer import AVTR1Renderer
from sglang_omni.models.avtr.session import CHUNK_SAMPLES, AvatarHooks
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    PCM16,
    PCM_F32,
    SESSION_CONTROL,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    TURN_DONE,
)
from sglang_omni.scheduling.session import SessionContext

PACKET_MS = 200
PACKET_SAMPLES = SESSION_INPUT_SAMPLE_RATE_HZ * PACKET_MS // 1000
SPEECH_RATE_HZ = 24000
CHUNK_MS = CHUNK_SAMPLES * 1000 // SESSION_INPUT_SAMPLE_RATE_HZ
FRAMES_PER_CHUNK = 5


def stats(values: list[float]) -> dict | None:
    return (
        {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "n": len(values),
        }
        if values
        else None
    )


class TimedRenderer:
    """A worker of the pool that times each render call and remembers which worker returned each state."""

    def __init__(self, index: int, renderer: AVTR1Renderer, shared: dict) -> None:
        self.index = index
        self.renderer = renderer
        self.shared = shared

    def render(self, state, speech, listen, frame) -> bytes:
        began = time.perf_counter()
        next_state = self.renderer.render(state, speech, listen, frame)
        with self.shared["lock"]:
            self.shared["calls"].append(
                (self.index, (time.perf_counter() - began) * 1000)
            )
            if state:
                self.shared["drifts"] += self.shared["owner"].pop(state) != self.index
            else:
                pass
            self.shared["owner"][next_state] = self.index
        return next_state

    def is_healthy(self) -> bool:
        return self.renderer.is_healthy()


class Session:
    """One Session of the hooks: the times its chunks became ready and their frames were published."""

    def __init__(self, hooks: AvatarHooks, name: str) -> None:
        self.hooks = hooks
        self.identity = SessionIdentity(name)
        self.ready: dict[float, float] = {}
        self.frames: dict[float, list[tuple[float, int]]] = defaultdict(list)
        self.errors: list[str] = []
        self.context = SessionContext(
            session_identity=self.identity, cancelled=threading.Event(), emit=self.emit
        )
        hooks.open(self.identity, OmniRequest(None))
        hooks.attach_output_error_handler(
            self.identity, lambda error: self.errors.append(repr(error))
        )
        hooks.attach_output_stream(self.identity, self.emit)

    def emit(self, chunk: TimedChunk) -> None:
        start = chunk.t_start_ms // CHUNK_MS * CHUNK_MS
        self.frames[start].append((time.perf_counter(), len(chunk.payload)))

    def append(self, chunk: TimedChunk) -> None:
        self.hooks.append(
            chunk, StagePayload("x", OmniRequest(None), None), self.context
        )

    def user(self, index: int, pcm: bytes) -> None:
        self.append(
            TimedChunk(AUDIO, index * PACKET_MS, PACKET_MS, index, pcm, format=PCM16)
        )
        # The chunk starting where this packet ends is ready now.
        self.ready[(index + 1) * PACKET_MS] = time.perf_counter()

    def reply(self, turn: int, speech: np.ndarray) -> None:
        self.append(
            TimedChunk(
                AUDIO,
                0,
                speech.size * 1000 / SPEECH_RATE_HZ,
                0,
                speech.tobytes(),
                format=PCM_F32,
                turn=turn,
            )
        )
        self.append(
            TimedChunk(
                AUDIO, 0, 0, 0, {"type": TURN_DONE}, format=SESSION_CONTROL, turn=turn
            )
        )


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        assert (
            source.getframerate() == SESSION_INPUT_SAMPLE_RATE_HZ
            and source.getnchannels() == 1
        )
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--renderer-url", action="append", required=True)
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warm-up-s", type=float, default=5.0)
    parser.add_argument("--reply-every-s", type=float, default=8.0)
    parser.add_argument("--avatar", default="maria")
    parser.add_argument("--background", default="plain_white")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument(
        "--requests-per-renderer",
        type=int,
        default=1,
        help="chunks kept in flight at each worker (a session-batched service renders them in one batch)",
    )
    parser.add_argument(
        "--service-jpeg",
        action="store_true",
        help="ask the workers for JPEG frames instead of encoding their I420 here",
    )
    parser.add_argument(
        "--stagger",
        action="store_true",
        help="spread the Sessions' packets over the 200 ms instead of sending every Session's at once",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    user = read_wav(args.audio)
    speech = resample_poly(user.astype(np.float32) / 32768, 3, 2).astype("<f4")
    shared = {"lock": threading.Lock(), "calls": [], "owner": {}, "drifts": 0}
    renderers = [
        TimedRenderer(
            index,
            AVTR1Renderer(
                httpx.Client(base_url=url, timeout=10.0),
                args.avatar,
                args.background,
                width=args.width,
                height=args.height,
                jpeg_quality=args.jpeg_quality,
                service_jpeg=args.service_jpeg,
            ),
            shared,
        )
        for index, url in enumerate(
            url for url in args.renderer_url for _ in range(args.requests_per_renderer)
        )
    ]
    hooks = AvatarHooks(renderers, SPEECH_RATE_HZ)
    sessions = [Session(hooks, f"s{index}") for index in range(args.sessions)]
    packets = int(args.seconds * 1000 / PACKET_MS)
    reply_every = int(args.reply_every_s * 1000 / PACKET_MS)
    began = time.perf_counter()
    phase_ms = PACKET_MS / len(sessions) if args.stagger else 0.0
    for index in range(packets):
        for number, session in enumerate(sessions):
            time.sleep(
                max(
                    began
                    + (index * PACKET_MS + number * phase_ms) / 1000
                    - time.perf_counter(),
                    0,
                )
            )
            offset = (index * PACKET_SAMPLES + number * 4321) % max(
                user.size - PACKET_SAMPLES, 1
            )
            session.user(index, user[offset : offset + PACKET_SAMPLES].tobytes())
            if index % reply_every == reply_every // 2:
                session.reply(index, speech)
            else:
                pass
    time.sleep(2.0)
    for session in sessions:
        hooks.close(session.identity)
    for worker in hooks.workers:
        if worker is not None:
            worker.join(10)
        else:
            pass
    warm_up_ms = args.warm_up_s * 1000
    latency, first, held, missing = [], [], 0, 0
    per_session = []
    for session in sessions:
        session_latency = []
        for start, ready_at in session.ready.items():
            frames = session.frames.get(start, [])
            if start < warm_up_ms or start > (packets - 2) * PACKET_MS:
                continue
            elif len(frames) < FRAMES_PER_CHUNK:
                missing += 1
            elif not all(size for _, size in frames):
                held += 1
            else:
                session_latency.append((max(at for at, _ in frames) - ready_at) * 1000)
                first.append((min(at for at, _ in frames) - ready_at) * 1000)
        latency += session_latency
        per_session.append(stats(session_latency))
    calls = shared["calls"]
    result = {
        "sessions": args.sessions,
        "renderer_urls": args.renderer_url,
        "requests_per_renderer": args.requests_per_renderer,
        "service_jpeg": args.service_jpeg,
        "stagger": args.stagger,
        "seconds": args.seconds,
        "frame_size": [args.width, args.height],
        "chunk_latency_ms": stats(latency),
        "first_frame_of_chunk_ms": stats(first),
        "per_session_chunk_latency_ms": per_session,
        "real_time": bool(latency)
        and float(np.percentile(latency, 95)) < CHUNK_MS
        and held == 0,
        "held_chunks": held,
        "missing_chunks": missing,
        "render_call_ms": stats([ms for _, ms in calls]),
        "chunks_per_worker": dict(Counter(index for index, _ in calls)),
        "worker_changes": shared["drifts"],
        "jpeg_bytes_mean": float(
            np.mean(
                [
                    size
                    for s in sessions
                    for frames in s.frames.values()
                    for _, size in frames
                    if size
                ]
            )
        ),
        "errors": [error for session in sessions for error in session.errors],
    }
    args.output.write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "per_session_chunk_latency_ms"
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
