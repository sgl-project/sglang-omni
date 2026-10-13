# SPDX-License-Identifier: Apache-2.0
"""One AURA Omni turn offline: the user's speech and a video in, the answer out.

Qwen3-ASR transcribes the speech, AURA answers from the transcript and the
video or stays silent, and Qwen3-TTS speaks the answer.

    python examples/run_aura_omni.py \\
        --audio tests/data/query_to_draw.wav \\
        --video tests/data/draw.mp4 \\
        --out reply.wav

The default placement puts ASR, AURA and TTS on three cards. On 24 GB Intel Arc
Pro B60 cards pass --config examples/configs/aura_omni_xpu_b60.yaml, which also
sizes AURA's KV pool to fit.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="aurateam/AURA", help="AURA checkpoint")
    parser.add_argument("--config", default=None, help="pipeline config file")
    parser.add_argument("--audio", default=None, help="the user's speech")
    parser.add_argument("--video", default=None, help="the video AURA watches")
    parser.add_argument("--text", default=None, help="typed text after the speech")
    parser.add_argument("--voice", default=None, help="Qwen3-TTS CustomVoice speaker")
    parser.add_argument("--out", default="reply.wav", help="where to write the reply")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=1200.0,
        help="seconds to wait for the stages to load",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    from sglang_omni.client import Client, GenerateRequest, Message
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    if args.config is not None:
        config = ConfigManager.from_file(args.config).config
    else:
        config = ConfigManager.from_model_path(args.model_path).config
    content: list[dict[str, object]] = []
    if args.video is not None:
        content.append(
            {"type": "video_url", "video_url": {"url": str(Path(args.video).resolve())}}
        )
    else:
        pass
    if args.audio is not None:
        content.append(
            {"type": "audio_url", "audio_url": {"url": str(Path(args.audio).resolve())}}
        )
    else:
        pass
    if args.text is not None:
        content.append({"type": "text", "text": args.text})
    else:
        pass
    metadata: dict[str, object] = {}
    if args.voice is not None:
        metadata["audio_config"] = {"voice": args.voice}
    else:
        pass

    runner = MultiProcessPipelineRunner(config)
    started = time.perf_counter()
    await runner.start(timeout=args.startup_timeout)
    print(f"pipeline ready in {time.perf_counter() - started:.0f}s")
    try:
        client = Client(runner.coordinator)
        request = GenerateRequest(
            model=config.name,
            messages=[Message(role="user", content=content)],
            output_modalities=["text", "audio"],
            stream=False,
            metadata=metadata,
        )
        started = time.perf_counter()
        result = await client.completion(
            request, request_id="aura-omni-1", audio_format="wav"
        )
        print(f"reply in {time.perf_counter() - started:.1f}s")
    finally:
        await runner.stop()

    if not result.text:
        print("AURA stayed silent")
        return 0
    else:
        pass
    print(f"text: {result.text}")
    blob = result.audio.data if result.audio else None
    wav = base64.b64decode(blob) if isinstance(blob, str) else (blob or b"")
    if not wav:
        print("no audio in the reply")
        return 1
    else:
        pass
    Path(args.out).write_bytes(wav)
    print(f"audio written to {args.out}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
