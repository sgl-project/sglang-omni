# SPDX-License-Identifier: Apache-2.0
"""Text-first split pipeline for Qwen3-Omni."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from sglang_omni.config import PipelineRunner, compile_pipeline
from sglang_omni.config.compiler import _acquire_ipc_namespace_lock
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
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
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
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    config = Qwen3OmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )
    ipc_namespace_lock = _acquire_ipc_namespace_lock(config)
    coordinator, stages = compile_pipeline(
        config,
        ipc_namespace=ipc_namespace_lock.ipc_namespace if ipc_namespace_lock else None,
    )
    runner = PipelineRunner(
        coordinator,
        stages,
        ipc_namespace_lock=ipc_namespace_lock,
    )

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
        result = await coordinator.submit(
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
