# SPDX-License-Identifier: Apache-2.0
"""Text-first split pipeline for Qwen3-Omni."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from sglang_omni.config import PipelineRunner, compile_pipeline
from sglang_omni.models.qwen3_omni import create_text_first_pipeline_config
from sglang_omni.proto import OmniRequest

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Hugging Face model id",
    )
    parser.add_argument("--prompt", type=str, default="Describe this input.")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--preprocessing-device", type=str, default="cpu")
    parser.add_argument("--image-device", type=str, default="cuda:0")
    parser.add_argument("--audio-device", type=str, default="cuda:0")
    parser.add_argument("--thinker-device", type=str, default="cuda:0")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--use-audio-in-video", action="store_true")
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    config = create_text_first_pipeline_config(
        model_id=args.model_id,
        preprocessing_device=args.preprocessing_device,
        image_device=args.image_device,
        audio_device=args.audio_device,
        thinker_device=args.thinker_device,
        thinker_max_seq_len=args.thinker_max_seq_len,
        dtype=args.dtype,
    )
    coordinator, stages = compile_pipeline(config)
    runner = PipelineRunner(coordinator, stages)

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
