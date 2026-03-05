# SPDX-License-Identifier: Apache-2.0
"""
Text-first split pipeline for Qwen3-Omni.


Usage:
# image only
python bench_sglang_omni.py --image-path ../../tests/data/cars.jpg

# video only
python bench_sglang_omni.py --video-path ../../tests/data/draw.mp4

# audio only
python bench_sglang_omni.py --audio-path ../../tests/data/cough.wav

# all modalities
python bench_sglang_omni.py --image-path ../../tests/data/cars.jpg --video-path ../../tests/data/draw.mp4 --audio-path ../../tests/data/cough.wav
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

import torch

from sglang_omni.config import PipelineRunner, compile_pipeline
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.proto import OmniRequest

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def get_timestamp():
    torch.cuda.synchronize()
    return time.time()


class Timer:

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None

    @property
    def elapsed_time(self):
        return self.end_time - self.start_time

    def __enter__(self):
        self.start_time = get_timestamp()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = get_timestamp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--prompt", type=str, default="What is in the content of these files?"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--thinker-max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--use-audio-in-video", action="store_true")
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    config = Qwen3OmniPipelineConfig(
        model_path=args.model_path,
        relay_backend="nixl",
    )
    coordinator, stages = compile_pipeline(config)
    runner = PipelineRunner(coordinator, stages)

    await runner.start()

    # form request
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
        "use_audio_in_video": False,
        "audios": audios,
        "audio_target_sr": args.audio_target_sr,
    }

    with Timer("Run inference") as inference_timer:
        try:
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
    print(f"Inference latency: {inference_timer.elapsed_time} seconds")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
