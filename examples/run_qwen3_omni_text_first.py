# SPDX-License-Identifier: Apache-2.0
"""Text-first split pipeline for Qwen3-Omni."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from _mem_fraction_cli import (
    add_mem_fraction_static_args,
    apply_mem_fraction_static_args,
)

from sglang_omni.config import build_pipeline_runner
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.proto import OmniRequest

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Hugging Face model id",
    )
    parser.add_argument("--prompt", type=str, default="Describe this input.")
    parser.add_argument("--thinker-max-seq-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--use-audio-in-video", action="store_true")
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    parser.add_argument(
        "--relay-backend", type=str, default="nixl", choices=["nixl", "shm"]
    )
    add_mem_fraction_static_args(
        parser,
        global_target_help="the thinker stage",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=int,
        default=0,
        help="GB of model weights to offload to CPU",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    overrides = {}
    if args.cpu_offload_gb:
        overrides["cpu_offload_gb"] = args.cpu_offload_gb

    config = Qwen3OmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )
    if overrides:
        config.apply_thinker_server_args_overrides(overrides)
    apply_mem_fraction_static_args(config, args)
    # Override thinker_max_seq_len in stage executor args if provided
    if args.thinker_max_seq_len is not None:
        for stage in config.stages:
            if stage.name == "thinker":
                stage.executor.args["thinker_max_seq_len"] = args.thinker_max_seq_len
    runner = build_pipeline_runner(config)

    await runner.start()
    try:
        images = [args.image_path] if args.image_path else []
        videos = [args.video_path] if args.video_path else []
        audios = [args.audio_path] if args.audio_path else []
        request = {
            "messages": [
                {"role": "user", "content": args.prompt},
            ],
            "images": images,
            "videos": videos,
            "video_fps": args.video_fps,
            "use_audio_in_video": args.use_audio_in_video,
            "audios": audios,
            "audio_target_sr": args.audio_target_sr,
        }
        result = await runner.coordinator.submit(
            "qwen3-omni-text-first",
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
