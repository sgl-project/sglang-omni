# SPDX-License-Identifier: Apache-2.0
"""One-shot voice chat with NVIDIA-NemotronLabs-VoiceChat-11B.

Feeds a recording of the user's turn and writes back the agent's reply as text
and audio. The model is frame-locked at 12.5 Hz: it emits one text token per
80 ms of input, most of them markers for the stretches where it is listening
rather than speaking, and the talker renders the speaking ones into audio.

    python examples/run_nemotron_voicechat.py \\
        --model-path /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \\
        --audio /path/to/NVIDIA-NemotronLabs-VoiceChat-11B/turn_taking.wav \\
        --out reply.wav

The four GPU stages share one card, so pick a free one with CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import time
import wave
from pathlib import Path

OUTPUT_SAMPLE_RATE = 22_050


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="checkpoint directory")
    parser.add_argument(
        "--audio", required=True, help="the user's turn, any sample rate"
    )
    parser.add_argument("--out", default="reply.wav", help="where to write the reply")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for the stages to load",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    from sglang_omni.client import Client, GenerateRequest, SamplingParams
    from sglang_omni.models.nemotron_voicechat.config import (
        NemotronVoiceChatPipelineConfig,
    )
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    config = NemotronVoiceChatPipelineConfig(model_path=args.model_path)
    runner = MultiProcessPipelineRunner(config)

    started = time.perf_counter()
    await runner.start(timeout=args.startup_timeout)
    print(f"pipeline ready in {time.perf_counter() - started:.0f}s")

    try:
        client = Client(runner.coordinator)
        request = GenerateRequest(
            model=config.name,
            prompt={"audio_path": args.audio},
            sampling=SamplingParams(temperature=0.0),
            output_modalities=["text", "audio"],
            stream=False,
        )
        started = time.perf_counter()
        # PCM rather than the default WAV: the reply arrives as one blob here,
        # and a container would have to be unwrapped before writing it out.
        result = await client.completion(
            request, request_id="voicechat-1", audio_format="pcm"
        )
        print(f"reply in {time.perf_counter() - started:.1f}s")
    finally:
        await runner.stop()

    print(f"text: {result.text or ''}")

    blob = result.audio.data if result.audio else None
    pcm = base64.b64decode(blob) if isinstance(blob, str) else (blob or b"")
    if not pcm:
        print("no audio in the reply")
        return 1

    out = Path(args.out)
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(OUTPUT_SAMPLE_RATE)
        wav.writeframes(pcm)
    seconds = len(pcm) / 2 / OUTPUT_SAMPLE_RATE
    print(f"audio: {seconds:.2f}s written to {out}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
