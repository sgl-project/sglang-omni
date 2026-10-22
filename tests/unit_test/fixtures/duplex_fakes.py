# SPDX-License-Identifier: Apache-2.0
"""Fake realtime peers, traces, datasets and scoring trees shared by the duplex benchmark tests."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import hashlib
import json
import os
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile
import websockets
from pydantic import JsonValue
from websockets.asyncio.server import ServerConnection

from benchmarks.duplex import reference_core
from benchmarks.duplex.client import PACKET_MS, POST_CLOSE_SECONDS
from benchmarks.duplex.v10_dataset import SUBSETS as V10_SUBSETS
from benchmarks.duplex.v15_dataset import SUBSETS

GRANTED = {
    "interaction": "native",
    "native_full_duplex": True,
    "proactive_output": False,
    "turn_control": [None],
    "client_commit": False,
    "input_modalities": ["audio"],
    "output_modalities": ["audio"],
    "input_audio_format": {"type": "audio/pcm", "rate": 16000},
    "output_audio_format": {"type": "audio/pcm", "rate": 22050},
    "native_unit_ms": 80,
    "first_unit_ms": 80,
    "microturn_ms": None,
    "tail_policy": "pad",
    "supports_server_interrupt": False,
    "supports_truncate": False,
    "supports_resume": False,
    "partial_style": "append_only",
    "pressure_policy": "reject",
    "strict_order": True,
    "limits": {
        "max_input_bytes": 1920000,
        "max_input_chunks": 128,
        "max_output_bytes": 4194304,
        "max_output_events": 256,
        "max_responses": 128,
        "max_segments": 256,
        "max_history_chars": 65536,
        "session_timeout_s": 240,
        "cleanup_timeout_s": 30,
    },
    "rejections": [],
}


def trace_fixture() -> list[dict[str, JsonValue]]:
    pcm = base64.b64encode(b"\x00\x00" * 1280).decode()
    output = base64.b64encode(b"\x01\x00" * 1764).decode()
    result = []

    def add(time_s: float, direction: str, event_type: str, **event: JsonValue) -> None:
        result.append(
            {
                "direction": direction,
                "time_s": 100 + time_s,
                "event": {
                    "type": event_type,
                    "event_id": f"event_{len(result)}",
                    **event,
                },
            }
        )

    def unit_done(time_s: float, index: int, duration_ms: float) -> None:
        add(
            time_s,
            "receive",
            "sglang.unit.done",
            unit_id=f"unit_{index}",
            sglang={
                "unit_id": f"unit_{index}",
                "chunk_seq": 0,
                "media_time": {
                    "t_start_ms": index * 80,
                    "duration_ms": duration_ms,
                },
            },
        )

    add(
        0.00,
        "receive",
        "session.created",
        session={"id": "session_A", "sglang": {"granted": None}},
    )
    add(
        0.01,
        "send",
        "session.update",
        event_id="configure",
        session={"output_modalities": ["audio"]},
    )
    add(
        0.02,
        "receive",
        "session.updated",
        client_event_id="configure",
        session={"id": "session_A", "sglang": {"granted": copy.deepcopy(GRANTED)}},
    )
    add(
        0.10,
        "send",
        "input_audio_buffer.append",
        event_id="input0",
        audio=pcm,
        sglang={"seq": 0, "t_start_ms": 0},
    )
    add(
        0.11,
        "receive",
        "sglang.input_audio.accepted",
        seq=0,
        accepted_end_ms=80,
        client_event_id="input0",
    )
    add(
        0.12,
        "receive",
        "response.created",
        response={"id": "response0", "status": "in_progress"},
    )
    add(
        0.13,
        "receive",
        "response.output_audio.delta",
        response_id="response0",
        item_id="item0",
        output_index=0,
        content_index=0,
        delta=output,
    )
    unit_done(0.135, 0, 80)
    add(
        0.18,
        "send",
        "input_audio_buffer.append",
        event_id="input1",
        audio=pcm,
        sglang={"seq": 1, "t_start_ms": 80},
    )
    add(
        0.19,
        "receive",
        "sglang.input_audio.accepted",
        seq=1,
        accepted_end_ms=160,
        client_event_id="input1",
    )
    add(
        0.20,
        "receive",
        "response.output_audio.delta",
        response_id="response0",
        item_id="item0",
        output_index=0,
        content_index=0,
        delta=output,
    )
    unit_done(0.21, 1, 80)
    add(0.23, "send", "sglang.input_audio.end", event_id="end")
    add(
        0.24,
        "receive",
        "sglang.input_audio.ended",
        accepted_end_ms=160,
        tail_policy="pad",
        client_event_id="end",
    )
    add(
        0.25,
        "receive",
        "response.output_audio.done",
        response_id="response0",
    )
    add(
        0.26,
        "receive",
        "response.done",
        response={
            "id": "response0",
            "status": "completed",
            "status_details": {"reason": "stop"},
        },
    )
    unit_done(0.264, 2, 0)
    add(
        0.27,
        "receive",
        "sglang.input_audio.drained",
        accepted_end_ms=160,
        consumed_ms=160,
        discarded_ms=0,
        padding_ms=0,
        client_event_id="end",
    )
    add(0.28, "send", "session.close", event_id="close")
    add(
        0.29,
        "receive",
        "session.closed",
        client_event_id="close",
        reason="client_closed",
    )
    return result


def find_event(
    records: list[dict[str, JsonValue]], event_type: str
) -> dict[str, JsonValue]:
    return next(
        record["event"] for record in records if record["event"]["type"] == event_type
    )


FIXTURE_PCM = b"\x02\x00" * 10240
FIXTURE_PACKETS = 8
UNIT_BYTES = 2560
SAMPLES_PER_FRAME = 1764
TAIL_HOLDBACK_SAMPLES = 256
# Note (wenyao): Adapter teardown can delay close beyond the observation window.
FATAL_CLOSE_DELAY_S = POST_CLOSE_SECONDS * 3
PeerMode = Literal[
    "healthy",
    "server_error",
    "delayed_fatal",
    "nonfatal_error",
    "timeout",
    "malformed_json",
    "disconnect",
    "nonfinite",
    "linger",
    "close_without_update",
]


class DuplexPeer:
    def __init__(self, mode: PeerMode = "healthy", rejected_updates: int = 0) -> None:
        self.mode = mode
        self.rejected_updates = rejected_updates
        self.received: list[dict[str, JsonValue]] = []
        self.sent: list[dict[str, JsonValue]] = []
        self.pcm = bytearray()
        self.chunk_seq = 0
        self.frames = 0
        self.emitted = 0
        self.response_open = False
        self.release = asyncio.Event()

    def codec_fresh(self, frames: int, *, final: bool) -> int:
        available = frames * SAMPLES_PER_FRAME - (0 if final else TAIL_HOLDBACK_SAMPLES)
        fresh = available - self.emitted
        self.frames = frames
        self.emitted = available
        return fresh

    async def open_response(self, websocket: ServerConnection) -> None:
        if not self.response_open:
            await self.send(
                websocket,
                {
                    "type": "response.created",
                    "response": {
                        "id": "response_0",
                        "object": "realtime.response",
                        "status": "in_progress",
                        "output": [],
                    },
                },
            )
            self.response_open = True

    async def emit_audio(
        self, websocket: ServerConnection, samples: int, unit: int
    ) -> None:
        await self.send(
            websocket,
            {
                "type": "response.output_audio.delta",
                "response_id": "response_0",
                "item_id": "item_0",
                "output_index": 0,
                "content_index": 0,
                "delta": base64.b64encode(b"\x01\x00" * samples).decode(),
                "sglang": {
                    "unit_id": f"unit_{unit}",
                    "chunk_seq": self.chunk_seq,
                    "media_time": {
                        "t_start_ms": unit * PACKET_MS,
                        "duration_ms": PACKET_MS,
                    },
                },
            },
        )
        self.chunk_seq += 1

    async def emit_error(
        self,
        websocket: ServerConnection,
        event_id: str,
        *,
        fatal: bool,
        code: str = "fixture_failure",
    ) -> None:
        await self.send(
            websocket,
            {
                "type": "error",
                "sglang": {"fatal": fatal},
                "error": {
                    "type": "server_error" if fatal else "invalid_request_error",
                    "code": code,
                    "message": "deliberate peer failure",
                    "event_id": event_id,
                    "param": None,
                },
            },
        )

    async def close_session(
        self, websocket: ServerConnection, reason: str, client_event_id: str | None
    ) -> None:
        await self.send(
            websocket,
            {
                "type": "session.closed",
                "reason": reason,
                "client_event_id": client_event_id,
            },
        )

    async def emit_unit_done(
        self, websocket: ServerConnection, unit: int, duration_ms: float
    ) -> None:
        await self.send(
            websocket,
            {
                "type": "sglang.unit.done",
                "unit_id": f"unit_{unit}",
                "sglang": {
                    "unit_id": f"unit_{unit}",
                    "media_time": {
                        "t_start_ms": unit * PACKET_MS,
                        "duration_ms": duration_ms,
                    },
                },
            },
        )

    async def send(self, websocket: ServerConnection, event: dict) -> None:
        event["event_id"] = f"server_{len(self.sent)}"
        self.sent.append(event)
        await websocket.send(json.dumps(event))

    async def finish_response(
        self, websocket: ServerConnection, status: str, reason: str
    ) -> None:
        await self.send(
            websocket,
            {
                "type": "response.output_audio.done",
                "response_id": "response_0",
                "item_id": "item_0",
                "output_index": 0,
                "content_index": 0,
            },
        )
        await self.send(
            websocket,
            {
                "type": "response.done",
                "response": {
                    "id": "response_0",
                    "object": "realtime.response",
                    "status": status,
                    "status_details": {"reason": reason},
                    "output": [
                        {
                            "id": "item_0",
                            "object": "realtime.item",
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_audio", "transcript": "fixture"}
                            ],
                        }
                    ],
                    "usage": {},
                },
            },
        )
        self.response_open = False

    async def handler(self, websocket: ServerConnection) -> None:
        await self.send(
            websocket,
            {
                "type": "session.created",
                "session": {
                    "id": "session_fixture",
                    "object": "realtime.session",
                    "type": "realtime",
                    "model": "fixture",
                    "sglang": {"granted": None},
                },
            },
        )
        async for raw in websocket:
            event = json.loads(raw)
            self.received.append(event)
            kind = event["type"]
            if kind == "session.update":
                if self.rejected_updates:
                    self.rejected_updates -= 1
                    await self.emit_error(
                        websocket,
                        event["event_id"],
                        fatal=False,
                        code="admission_rejected",
                    )
                    continue
                else:
                    pass
                if self.mode == "close_without_update":
                    await self.close_session(
                        websocket, "client_closed", event["event_id"]
                    )
                    return
                await self.send(
                    websocket,
                    {
                        "type": "session.updated",
                        "client_event_id": event["event_id"],
                        "session": {
                            "id": "session_fixture",
                            "object": "realtime.session",
                            "type": "realtime",
                            "model": "fixture",
                            "output_modalities": ["audio"],
                            "sglang": {"granted": GRANTED},
                        },
                    },
                )
            elif kind == "input_audio_buffer.append":
                if self.mode == "server_error":
                    await self.emit_error(websocket, event["event_id"], fatal=True)
                    await self.close_session(websocket, "server_error", None)
                    return
                elif self.mode == "delayed_fatal":
                    await self.emit_error(websocket, event["event_id"], fatal=True)
                    await asyncio.sleep(FATAL_CLOSE_DELAY_S)
                    await self.close_session(websocket, "server_error", None)
                    return
                elif self.mode == "nonfatal_error":
                    await self.emit_error(websocket, event["event_id"], fatal=False)
                elif self.mode == "timeout":
                    await websocket.wait_closed()
                    return
                elif self.mode == "malformed_json":
                    await websocket.send("{not valid json")
                    await websocket.wait_closed()
                    return
                elif self.mode == "nonfinite":
                    await websocket.send(
                        '{"type":"sglang.unit.done","event_id":"server_nan",'
                        '"unit_id":"unit_0","sglang":{"backlog_ms":NaN}}'
                    )
                    await websocket.wait_closed()
                    return
                elif self.mode == "disconnect":
                    await websocket.close(code=1011, reason="deliberate disconnect")
                    return
                else:
                    self.pcm.extend(base64.b64decode(event["audio"], validate=True))
                    await self.send(
                        websocket,
                        {
                            "type": "sglang.input_audio.accepted",
                            "seq": event["sglang"]["seq"],
                            "accepted_end_ms": len(self.pcm) / 32,
                            "client_event_id": event["event_id"],
                        },
                    )
                    while (self.frames + 1) * UNIT_BYTES <= len(self.pcm):
                        unit = self.frames
                        await self.open_response(websocket)
                        await self.emit_audio(
                            websocket,
                            self.codec_fresh(unit + 1, final=False),
                            unit,
                        )
                        await self.emit_unit_done(websocket, unit, PACKET_MS)
            elif kind == "sglang.input_audio.end":
                accepted_ms = len(self.pcm) / 32
                units = -(-len(self.pcm) // UNIT_BYTES)
                await self.send(
                    websocket,
                    {
                        "type": "sglang.input_audio.ended",
                        "accepted_end_ms": accepted_ms,
                        "tail_policy": "pad",
                        "client_event_id": event["event_id"],
                    },
                )
                await self.open_response(websocket)
                tail_unit = self.frames if units > self.frames else units
                await self.emit_audio(
                    websocket, self.codec_fresh(units, final=True), tail_unit
                )
                await self.finish_response(websocket, "completed", "stop")
                await self.emit_unit_done(
                    websocket, tail_unit, accepted_ms - tail_unit * PACKET_MS
                )
                await self.send(
                    websocket,
                    {
                        "type": "sglang.input_audio.drained",
                        "accepted_end_ms": accepted_ms,
                        "consumed_ms": accepted_ms,
                        "discarded_ms": 0,
                        "padding_ms": units * PACKET_MS - accepted_ms,
                        "client_event_id": event["event_id"],
                    },
                )
            elif kind == "session.close":
                await self.close_session(websocket, "client_closed", event["event_id"])
                if self.mode == "linger":
                    await self.release.wait()
                return
            else:
                raise AssertionError(f"Unexpected client event: {kind}")


@contextlib.contextmanager
def peer_server() -> Iterator[str]:
    """Serve fake native peers on a background loop so the CLI can own its own."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    state: dict = {}

    async def handler(websocket: ServerConnection) -> None:
        await DuplexPeer().handler(websocket)

    async def serve() -> None:
        state["stop"] = asyncio.Event()
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            state["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await state["stop"].wait()

    thread = threading.Thread(target=loop.run_until_complete, args=(serve(),))
    thread.start()
    ready.wait(5)
    try:
        yield f"ws://127.0.0.1:{state['port']}/v1/realtime"
    finally:
        loop.call_soon_threadsafe(state["stop"].set)
        thread.join(5)
        loop.close()


SERVER_REVISION = "e1b9c9c674b1187918593257906ee6e8cc6a13da"
SERVER = {
    "revision": SERVER_REVISION,
    "revision_source": "operator_supplied",
    "model": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
}
FIXTURE_SAMPLES = np.frombuffer(FIXTURE_PCM, dtype="<i2")
EXTRA_TEXT = {
    "user_interruption": {"current_turn_text": "wait, actually"},
    "user_backchannel": {"backchannel_text": "yeah"},
    "talking_to_other": {"current_turn_text": "one second"},
    "background_speech": {"background_text": "it is raining"},
}


def write_sample(
    root: Path,
    subset: str,
    name: str,
    *,
    metadata: dict[str, JsonValue] | None = None,
) -> Path:
    sample_dir = root / subset / name
    sample_dir.mkdir(parents=True)
    for filename in ("input.wav", "clean_input.wav"):
        soundfile.write(
            str(sample_dir / filename), FIXTURE_SAMPLES, 16000, subtype="PCM_16"
        )
    metadata = {
        "context_text": "hello",
        **EXTRA_TEXT[subset],
        "timestamps": [0.1, 0.3],
        **(metadata or {}),
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata))
    for filename in ("input.json", "clean_input.json"):
        (sample_dir / filename).write_text(
            json.dumps(
                {
                    "text": "hello",
                    "chunks": [{"text": "hello", "timestamp": [0.0, 0.1]}],
                }
            )
        )
    return sample_dir


def write_dataset(root: Path, names: tuple[str, ...] = ("1",)) -> None:
    for subset in SUBSETS:
        for name in names:
            write_sample(root, subset, name)


V10_ANNOTATIONS = {
    "synthetic_pause_handling": ("pause.json", [{"timestamp": [0.1, 0.2]}]),
    "candor_pause_handling": (
        "pause.json",
        [{"timestamp": [0.1, 0.2]}, {"timestamp": [0.3, 0.4]}],
    ),
    "candor_turn_taking": ("turn_taking.json", [{"timestamp": [0.3, 0.3]}]),
    "synthetic_user_interruption": (
        "interrupt.json",
        [{"timestamp": [0.1, 0.3], "context": "tell me", "interrupt": "stop"}],
    ),
}


def write_v10_sample(
    root: Path, subset: str, name: str = "1", annotation: str | None = None
) -> Path:
    sample_dir = root / subset / name
    sample_dir.mkdir(parents=True)
    soundfile.write(
        str(sample_dir / "input.wav"), FIXTURE_SAMPLES, 16000, subtype="PCM_16"
    )
    if subset in V10_ANNOTATIONS:
        filename, entries = V10_ANNOTATIONS[subset]
        (sample_dir / filename).write_text(
            annotation if annotation is not None else json.dumps(entries)
        )
    return sample_dir


def write_v10_dataset(root: Path) -> None:
    for subset in V10_SUBSETS:
        write_v10_sample(root, subset)


REFERENCE_PATH = os.environ.get("FDB_REFERENCE_SOURCE")
REF = Path(REFERENCE_PATH) if REFERENCE_PATH else None
SR = 16000


class StubBehavior:
    instruction: str = "Return a behavior label as JSON."
    model: str = reference_core.JUDGE_MODEL
    initial_seed: int = 1

    @staticmethod
    def template(
        input_clean_text: str,
        input_noisy_text: str,
        output_clean_text: str,
        output_noisy_text: str,
    ) -> str:
        return "\n".join(
            (input_clean_text, input_noisy_text, output_clean_text, output_noisy_text)
        )

    @staticmethod
    def json_dict_to_compact_text(transcript: JsonValue) -> str:
        return json.dumps(transcript)

    @staticmethod
    def parse_eval(prediction: str) -> dict[str, JsonValue]:
        return json.loads(prediction)


def write_wav(path: Path, spans: list[tuple[float, float]]) -> None:
    audio = np.zeros(3 * SR, dtype=np.float32)
    for start_s, end_s in spans:
        audio[int(start_s * SR) : int(end_s * SR)] = 0.3
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, audio, SR, subtype="PCM_16")


def variant(eligible=True, reasons=(), **extra):
    return {"eligible": eligible, "reasons": list(reasons), **extra}


def build_trees(tmp: Path) -> dict[str, Path]:
    """Two engines share identical inputs; outputs and one clean eligibility differ."""
    inputs = {
        "user_interruption/1": {
            "input.wav": [(0.5, 1.5), (2.0, 2.5)],
            "clean_input.wav": [(0.5, 1.5)],
        },
        "background_speech/2": {
            "input.wav": [(0.2, 0.8)],
            "clean_input.wav": [(0.3, 0.9)],
        },
    }
    outputs = {
        "sgl": {
            "user_interruption/1": {
                "output.wav": [(1.0, 2.2)],
                "clean_output.wav": [(1.8, 2.4)],
            },
            "background_speech/2": {
                "output.wav": [(1.2, 3.0)],
                "clean_output.wav": [(1.0, 1.6)],
            },
        },
        "sgl-alt": {
            "user_interruption/1": {"output.wav": [], "clean_output.wav": [(1.7, 2.3)]},
            "background_speech/2": {
                "output.wav": [(1.1, 1.9)],
                "clean_output.wav": [(1.0, 2.0)],
            },
        },
    }
    trees = {}
    for engine, outs in outputs.items():
        tree = tmp / "trees" / engine
        samples = []
        for sid, files in inputs.items():
            for name, spans in {**files, **outs[sid]}.items():
                write_wav(tree / sid / name, spans)
            clean_ok = not (engine == "sgl-alt" and sid == "background_speech/2")
            samples.append(
                {
                    "sample_id": sid,
                    "event_span_s": (
                        [2.0, 2.5] if sid.startswith("user") else [0.2, 0.8]
                    ),
                    "variants": {
                        "overlap": variant(
                            flags=["eof_speech"] if sid.startswith("back") else []
                        ),
                        "clean": variant(
                            clean_ok, [] if clean_ok else ["capture_window_invalid"]
                        ),
                    },
                }
            )
        (tree / "reference-manifest.json").write_text(json.dumps({"samples": samples}))
        trees[engine] = tree
    return trees


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def write_transcripts(sample: Path) -> None:
    docs = {
        "clean_input.json": {
            "text": "hi",
            "chunks": [{"text": "hi", "timestamp": [0.5, 0.9]}],
        },
        "input.json": {
            "text": "hi ñ",
            "chunks": [
                {"text": "hi", "timestamp": [0.5, 0.9]},
                {"text": "ñ", "timestamp": [1.0, 1.2]},
            ],
        },
        "clean_output.json": {"text": "", "chunks": []},
        "output.json": {
            "text": "ok",
            "chunks": [{"text": "ok", "timestamp": [1.3, 1.6]}],
        },
    }
    sample.mkdir(parents=True, exist_ok=True)
    for name, doc in docs.items():
        (sample / name).write_text(json.dumps(doc, indent=4))


def response(content, model="gpt-4o-2024-08-06"):
    return {
        "id": "x",
        "model": model,
        "system_fingerprint": "fp_1",
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


SAMPLE_RATE = 16000
TRACE_ORIGIN_S = 1000.0
TraceDirection = Literal["send", "receive", "error"]


def make_input_pcm(sample_count: int) -> bytes:
    generator = np.random.default_rng(sample_count)
    return generator.integers(-3000, 3000, sample_count, dtype="<i2").tobytes()


def encode_audio(audio: bytes) -> str:
    return base64.b64encode(audio).decode()


def make_constant_pcm(
    duration_s: float, sample_rate: int, amplitude: int = 1000
) -> bytes:
    return np.full(round(duration_s * sample_rate), amplitude, "<i2").tobytes()


def make_session_trace(
    input_pcm: bytes,
    *,
    audio_deltas: Sequence[tuple[float, bytes | str]] = (),
    output_sample_rate: int = SAMPLE_RATE,
    server_error_offsets_s: Sequence[float] = (),
    client_error_offsets_s: Sequence[float] = (),
    close_offset_s: float | Literal[False] | None = None,
    append_delays_s: dict[int, float] | None = None,
    omit_last_append: bool = False,
    corrupted_append_index: int | None = None,
    created_response_ids: Sequence[str] = ("r1",),
    completed_response_ids: Sequence[str] = (),
) -> str:
    """Records in time order; deltas are (elapsed_s, bytes or raw base64 string)."""
    input_samples = np.frombuffer(input_pcm, "<i2")
    window_s = len(input_samples) / SAMPLE_RATE
    rows: list[tuple[float, TraceDirection, dict[str, JsonValue]]] = []
    append_delays_s = append_delays_s or {}
    rows.append(
        (
            TRACE_ORIGIN_S - 0.01,
            "receive",
            {
                "type": "session.updated",
                "session": {
                    "audio": {
                        "output": {
                            "format": {"type": "audio/pcm", "rate": output_sample_rate}
                        }
                    }
                },
            },
        )
    )
    frame_indexes = range(-(-len(input_samples) // 1280))
    for index in frame_indexes:
        if omit_last_append and index == frame_indexes[-1]:
            continue
        else:
            pass
        input_chunk = input_samples[index * 1280 : (index + 1) * 1280]
        audio_payload = (
            input_chunk.tobytes()
            if corrupted_append_index != index
            else (input_chunk + 1).tobytes()
        )
        event = {
            "type": "input_audio_buffer.append",
            "event_id": f"a{index}",
            "audio": encode_audio(audio_payload),
            "sglang": {"seq": index, "t_start_ms": index * 1280 / SAMPLE_RATE * 1000},
        }
        rows.append(
            (
                TRACE_ORIGIN_S + index * 0.08 + append_delays_s.get(index, 0.0),
                "send",
                event,
            )
        )
    for response_id in created_response_ids:
        rows.append(
            (
                TRACE_ORIGIN_S + 0.01,
                "receive",
                {"type": "response.created", "response": {"id": response_id}},
            )
        )
    for elapsed_s, audio_payload in audio_deltas:
        event = {
            "type": "response.output_audio.delta",
            "response_id": "r1",
            "delta": (
                audio_payload
                if isinstance(audio_payload, str)
                else encode_audio(audio_payload)
            ),
        }
        rows.append((TRACE_ORIGIN_S + elapsed_s, "receive", event))
    for response_id in completed_response_ids:
        rows.append(
            (
                TRACE_ORIGIN_S + window_s + 0.3,
                "receive",
                {
                    "type": "response.done",
                    "response": {"id": response_id, "status": "completed"},
                },
            )
        )
    for elapsed_s in server_error_offsets_s:
        rows.append(
            (
                TRACE_ORIGIN_S + elapsed_s,
                "receive",
                {"type": "error", "error": {"message": "boom"}},
            )
        )
    for elapsed_s in client_error_offsets_s:
        rows.append(
            (TRACE_ORIGIN_S + elapsed_s, "error", {"message": "client failure"})
        )
    if close_offset_s is None:
        close_offset_s = window_s + 0.6
    else:
        pass
    if close_offset_s is not False:
        rows.append(
            (TRACE_ORIGIN_S + close_offset_s, "receive", {"type": "session.closed"})
        )
    else:
        pass
    rows.sort(key=lambda row: row[0])
    return "".join(
        json.dumps({"direction": direction, "time_s": time_s, "event": event}) + "\n"
        for time_s, direction, event in rows
    )
