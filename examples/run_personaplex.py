# SPDX-License-Identifier: Apache-2.0
"""One-shot conversation with NVIDIA PersonaPlex 7B.

Feeds a recording of the caller's side and writes back what the agent said,
as text and as 24 kHz audio of the same length. The model is frame-locked at
12.5 Hz and decides for itself when to speak; a voice prompt and a role prompt
set who it is.

    python examples/run_personaplex.py \\
        --model-path nvidia/personaplex-7b-v1 \\
        --audio input_assistant.wav \\
        --voice NATF2 \\
        --text-prompt "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way." \\
        --out reply.wav

The GPU stages share one card, so pick a free one with CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import time
import wave
from pathlib import Path

OUTPUT_SAMPLE_RATE = 24_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, help="checkpoint directory or repo id"
    )
    parser.add_argument(
        "--audio", required=True, help="the caller's side, any sample rate"
    )
    parser.add_argument("--out", default="reply.wav", help="where to write the reply")
    parser.add_argument(
        "--voice",
        default=None,
        help="packaged voice name (NATF0..3, NATM0..3, VARF0..4, VARM0..4), a .pt "
        "file, or a recording; default NATF2, empty string for no voice prompt",
    )
    parser.add_argument(
        "--text-prompt", default=None, help="the role prompt; default: assistant"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="text sampling (0 = greedy)"
    )
    parser.add_argument("--top-k", type=int, default=None, help="text top-k")
    parser.add_argument(
        "--audio-temperature",
        type=float,
        default=None,
        help="code sampling (0 = greedy)",
    )
    parser.add_argument("--audio-top-k", type=int, default=None, help="code top-k")
    parser.add_argument(
        "--seed", type=int, default=None, help="make both draws reproducible"
    )
    parser.add_argument(
        "--greedy", action="store_true", help="temperature 0 for text and codes"
    )
    parser.add_argument(
        "--out-text", default=None, help="also write the reply text here"
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900.0,
        help="seconds to wait for the stages to load",
    )
    return parser.parse_args()


def _extra_params(args: argparse.Namespace) -> dict:
    params: dict = {}
    if args.voice is not None:
        params["voice"] = args.voice
    if args.text_prompt is not None:
        params["text_prompt"] = args.text_prompt
    if args.greedy:
        params["audio_temperature"] = 0.0
    for key in ("audio_temperature", "audio_top_k", "seed", "top_k"):
        value = getattr(args, key)
        if value is not None:
            params[key] = value
    return params


def _explicit_fields(args: argparse.Namespace) -> list[str]:
    fields = []
    if args.greedy or args.temperature is not None:
        fields.append("temperature")
    if args.top_k is not None:
        fields.append("top_k")
    return fields


async def run(args: argparse.Namespace) -> int:
    from sglang_omni.client import Client, GenerateRequest, SamplingParams
    from sglang_omni.models.personaplex.config import PersonaPlexPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY

    config = PersonaPlexPipelineConfig(model_path=args.model_path)
    runner = MultiProcessPipelineRunner(config)

    started = time.perf_counter()
    await runner.start(timeout=args.startup_timeout)
    print(f"pipeline ready in {time.perf_counter() - started:.0f}s")

    temperature = 0.0 if args.greedy else args.temperature
    try:
        client = Client(runner.coordinator)
        request = GenerateRequest(
            model=config.name,
            prompt={"audio_path": args.audio},
            sampling=(
                SamplingParams(temperature=temperature)
                if temperature is not None
                else SamplingParams()
            ),
            extra_params=_extra_params(args),
            metadata={EXPLICIT_GENERATION_PARAMS_KEY: _explicit_fields(args)},
            output_modalities=["text", "audio"],
            stream=False,
        )
        started = time.perf_counter()
        # Note (wilsonzheng0327): PCM, not the default WAV, so the blob can be written
        # without unwrapping a container.
        result = await client.completion(
            request, request_id="personaplex-1", audio_format="pcm"
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

    if args.out_text:
        Path(args.out_text).write_text((result.text or "") + "\n")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
