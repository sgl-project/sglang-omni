# SPDX-License-Identifier: Apache-2.0
"""Streaming latency and Session capacity of the ELLSA 2 motion node.

Drives the node's Session hooks with N concurrent Sessions, each a test-set
clip: the clip's target audio arrives first as one reply (as a TTS faster
than real time sends it) and its partner audio follows as the user's audio,
in 200 ms units sent in real time. Session starts are spread evenly over one
0.8 s block. Per block it records when the block's last user unit was sent
and when its motion was emitted.

A run passes when every Session gets every block and the p95 of the
per-block latency (emit - last unit sent) is at most one block period,
800 ms: a node that needs more than a period per round of N blocks falls
behind without bound.

    python -m benchmarks.eval.ellsa2_streaming --checkpoint /path/to/ELLSA2/checkpoints \\
        --data /path/to/ELLSA2/data --sessions 1 2 4 6 8 --output-dir outputs/streaming
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import torch

from sglang_omni.models.ellsa2.generator import BLOCK_SAMPLES
from sglang_omni.models.ellsa2.loading import load_motion_models
from sglang_omni.models.ellsa2.session import MotionHooks
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    PCM16,
    PCM_F32,
    SESSION_INPUT_SAMPLE_RATE_HZ,
)
from sglang_omni.scheduling.session import SessionContext

UNIT_MS = 200
UNIT_SAMPLES = SESSION_INPUT_SAMPLE_RATE_HZ * UNIT_MS // 1000
BLOCK_S = BLOCK_SAMPLES / SESSION_INPUT_SAMPLE_RATE_HZ
UNITS_PER_BLOCK = BLOCK_SAMPLES // UNIT_SAMPLES
REPLY_TURN = 1


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")


def percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


class StreamedSession:
    """One Session of the run: its clip, the wall time each user unit was sent and each motion block arrived."""

    def __init__(
        self, hooks: MotionHooks, name: str, target: np.ndarray, partner: np.ndarray
    ) -> None:
        self.hooks = hooks
        self.identity = SessionIdentity(name)
        self.target = target
        self.partner = partner
        self.sent_s: list[float] = []
        self.arrivals: list[tuple[float, TimedChunk]] = []
        self.errors: list[BaseException] = []
        self.done = threading.Event()
        self.blocks = len(partner) // BLOCK_SAMPLES
        hooks.open(self.identity, OmniRequest(None))
        hooks.attach_output_stream(self.identity, self.arrive)
        hooks.attach_output_error_handler(self.identity, self.errors.append)
        self.context = SessionContext(
            session_identity=self.identity,
            cancelled=threading.Event(),
            emit=self.arrive,
        )

    def arrive(self, chunk: TimedChunk) -> None:
        self.arrivals.append((time.perf_counter(), chunk))
        if len(self.arrivals) == self.blocks:
            self.done.set()
        else:
            pass

    def append(self, chunk: TimedChunk) -> None:
        self.hooks.append(
            chunk, StagePayload(self.identity.id, OmniRequest(None), None), self.context
        )

    def stream(self, start_s: float) -> None:
        """The reply at once, then each user unit when its last sample would have been recorded."""
        time.sleep(max(start_s - time.perf_counter(), 0))
        self.append(
            TimedChunk(
                AUDIO,
                0.0,
                0.0,
                0,
                (self.target.astype(np.float32) / 32768).tobytes(),
                format=PCM_F32,
                turn=REPLY_TURN,
            )
        )
        units = self.blocks * UNITS_PER_BLOCK
        for unit in range(units):
            due = start_s + (unit + 1) * UNIT_MS / 1000
            time.sleep(max(due - time.perf_counter(), 0))
            samples = self.partner[unit * UNIT_SAMPLES : (unit + 1) * UNIT_SAMPLES]
            self.sent_s.append(time.perf_counter())
            self.append(
                TimedChunk(
                    AUDIO,
                    unit * UNIT_MS,
                    UNIT_MS,
                    unit,
                    samples.tobytes(),
                    format=PCM16,
                    eos=unit == units - 1,
                )
            )

    def measure(self) -> dict:
        start_s = self.sent_s[0] - UNIT_MS / 1000
        latency, behind_audio, late = [], [], []
        reply_start_s = self.arrivals[0][0]
        for index, (arrived_s, chunk) in enumerate(self.arrivals):
            ready_s = self.sent_s[(index + 1) * UNITS_PER_BLOCK - 1]
            latency.append((arrived_s - ready_s) * 1000)
            # Motion shown on arrival against reply audio played on the input timeline.
            behind_audio.append(
                (arrived_s - (start_s + chunk.t_start_ms / 1000)) * 1000
            )
            # Audio held until the reply's first motion block, then played on: how late each block's motion is for its audio.
            late.append((arrived_s - (reply_start_s + index * BLOCK_S)) * 1000)
        return {"latency_ms": latency, "behind_audio_ms": behind_audio, "late_ms": late}


def run(
    hooks: MotionHooks, clips: list[tuple[str, np.ndarray, np.ndarray]], sessions: int
) -> dict:
    streamed = [
        StreamedSession(hooks, f"s{index}", *clips[index % len(clips)][1:])
        for index in range(sessions)
    ]
    first_s = time.perf_counter() + 0.5
    threads = [
        threading.Thread(
            target=session.stream, args=(first_s + index * BLOCK_S / sessions,)
        )
        for index, session in enumerate(streamed)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    timeout_s = time.perf_counter() + 120
    for session in streamed:
        session.done.wait(max(timeout_s - time.perf_counter(), 0))
    for session in streamed:
        hooks.close(session.identity)
    complete = all(len(s.arrivals) == s.blocks and not s.errors for s in streamed)
    measured = [s.measure() for s in streamed if s.arrivals]
    latency = [value for m in measured for value in m["latency_ms"]]
    result = {
        "sessions": sessions,
        "complete": complete,
        "blocks": sum(len(s.arrivals) for s in streamed),
        "errors": [repr(error) for s in streamed for error in s.errors],
        "latency_ms": percentiles(latency),
        "behind_audio_ms": percentiles(
            [v for m in measured for v in m["behind_audio_ms"]]
        ),
        "late_ms": percentiles([v for m in measured for v in m["late_ms"]]),
    }
    result["passed"] = complete and result["latency_ms"]["p95"] <= BLOCK_S * 1000
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--style", default="r2d_00073")
    parser.add_argument("--sessions", type=int, nargs="+", default=[1])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--clips", type=int, default=16)
    parser.add_argument(
        "--encoder-attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument(
        "--cudnn-attention",
        action="store_true",
        help="let the backbone's attention use cuDNN, as the reference and ellsa2_parity do",
    )
    parser.add_argument(
        "--kv-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="the serving path: the ready blocks of all Sessions as one batch, each block's steps decoding from a KV cache; --no-kv-batch runs the reference's path, one block at a time",
    )
    parser.add_argument(
        "--cuda-graphs", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reuse-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with --kv-batch, keep each Session's keys and values across blocks; --no-reuse-history recomputes the whole history every block",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (args.data / "testset/manifest.jsonl").read_text().splitlines()
        if line
    ][: args.clips]
    clips = [
        (
            row["group_id"],
            read_wav(args.data / row["audio20_target"]),
            read_wav(args.data / row["audio20_partner"]),
        )
        for row in rows
    ]
    models = load_motion_models(
        model_path=args.checkpoint,
        data_path=str(args.data),
        style=args.style,
        encoder_attn_implementation=args.encoder_attn_implementation,
        cudnn_attention=args.cudnn_attention,
        device=torch.device("cuda:0"),
    )
    hooks = MotionHooks(
        models,
        SESSION_INPUT_SAMPLE_RATE_HZ,
        kv_batch=args.kv_batch,
        max_batch=max(args.sessions),
        cuda_graphs=args.cuda_graphs,
        reuse_history=args.reuse_history,
    )
    hooks.warm_up_serving_thread()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for sessions in args.sessions:
        for repeat in range(args.repeats):
            result = {
                "repeat": repeat,
                "kv_batch": args.kv_batch,
                "reuse_history": args.reuse_history,
                **run(hooks, clips, sessions),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output_dir / "streaming.json").write_text(
                json.dumps(results, indent=1)
            )
        if not any(r["passed"] for r in results if r["sessions"] == sessions):
            break
        else:
            pass


if __name__ == "__main__":
    main()
