# SPDX-License-Identifier: Apache-2.0
"""Text-first pipeline for Ming-Omni (audio input → text output).

Usage::

    # Text-only (no audio input):
    python examples/run_ming_omni_text_first.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --prompt "你好，请介绍一下你自己。"

    # With audio input:
    python examples/run_ming_omni_text_first.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --audio-path /path/to/audio.wav \
        --prompt "请描述这段音频的内容。"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from sglang_omni.config import build_pipeline_runner
from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig
from sglang_omni.proto import OmniRequest

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="inclusionAI/Ming-flash-omni-2.0",
        help="Hugging Face model id or local path",
    )
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument(
        "--cpu-offload-gb",
        type=int,
        default=80,
        help="GB of model weights to offload to CPU (default: 80 for Ming-flash-omni-2.0)",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set SGLang mem_fraction_static for the thinker stage. "
            "If omitted, SGLang chooses automatically."
        ),
    )
    parser.add_argument(
        "--relay-backend", type=str, default="shm", choices=["nixl", "shm"]
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    overrides = {}
    if args.cpu_offload_gb:
        overrides["cpu_offload_gb"] = args.cpu_offload_gb

    config = MingOmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )
    if overrides:
        config.apply_server_args_overrides(stage_name="thinker", overrides=overrides)
    if args.mem_fraction_static is not None:
        if not 0.0 < args.mem_fraction_static < 1.0:
            raise ValueError(
                f"--mem-fraction-static must be > 0 and < 1, got {args.mem_fraction_static}"
            )
        config.apply_server_args_overrides(
            stage_name="thinker",
            overrides={"mem_fraction_static": args.mem_fraction_static},
        )
    runner = build_pipeline_runner(config)

    await runner.start()
    try:
        # Build message content
        if args.audio_path:
            content = [
                {"type": "audio_url", "audio_url": {"url": args.audio_path}},
                {"type": "text", "text": args.prompt},
            ]
        else:
            content = args.prompt

        request = {
            "messages": [
                {"role": "user", "content": content},
            ],
            "audios": [args.audio_path] if args.audio_path else [],
        }
        result = await runner.coordinator.submit(
            "ming-omni-text-first",
            OmniRequest(
                inputs=request,
                params={
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                },
            ),
        )
        print(result)
    finally:
        await runner.stop()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
