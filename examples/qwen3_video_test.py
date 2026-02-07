# SPDX-License-Identifier: Apache-2.0
"""Text-first split pipeline for Qwen3-Omni."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

logging.getLogger("sglang_omni").setLevel(logging.DEBUG)
logging.getLogger("sglang_omni.pipeline").setLevel(logging.DEBUG)
logging.getLogger("sglang_omni.engines").setLevel(logging.DEBUG)
logging.getLogger("sglang_omni.relay").setLevel(logging.DEBUG)
logging.getLogger("transformers").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

from sglang_omni.config import PipelineRunner, compile_pipeline
from sglang_omni.models.qwen3_omni import create_text_first_pipeline_config
from sglang_omni.proto import OmniRequest

video_path = "/sgl-workspace/sglang-omni/tests/data/draw.mp4"
image_path = "/sgl-workspace/sglang-omni/tests/data/cars.jpg"
model_path = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        type=str,
        default=model_path,
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--prompt", type=str, default="What do you see and hear in the video?"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--preprocessing-device", type=str, default="cpu")
    parser.add_argument("--image-device", type=str, default="cuda:0")
    parser.add_argument("--audio-device", type=str, default="cuda:0")
    parser.add_argument("--thinker-device", type=str, default="cuda:0")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    parser.add_argument("--video-path", type=str, default=video_path)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--use-audio-in-video", action="store_true")
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("Starting Pipeline Initialization")
    logger.info("=" * 60)
    logger.info(f"Model ID: {args.model_id}")
    logger.info(f"Image device: {args.image_device}")
    logger.info(f"Audio device: {args.audio_device}")
    logger.info(f"Thinker device: {args.thinker_device}")

    logger.info("Step 1: Creating pipeline config...")
    config = create_text_first_pipeline_config(
        model_id=args.model_id,
        preprocessing_device=args.preprocessing_device,
        image_device=args.image_device,
        audio_device=args.audio_device,
        thinker_device=args.thinker_device,
        thinker_max_seq_len=args.thinker_max_seq_len,
        dtype=args.dtype,
    )
    logger.info("✓ Pipeline config created")

    logger.info("Step 2: Compiling pipeline...")
    coordinator, stages = compile_pipeline(config)
    logger.info(f"✓ Pipeline compiled, {len(stages)} stages created")

    logger.info("Step 3: Creating PipelineRunner...")
    runner = PipelineRunner(coordinator, stages)
    logger.info("✓ PipelineRunner created")

    logger.info(
        "Step 4: Starting pipeline (this may take a while for model loading)..."
    )
    await runner.start()
    logger.info("✓ Pipeline started successfully")

    try:
        logger.info("Step 5: Preparing request...")
        images = [args.image_path] if args.image_path else []
        audios = [args.audio_path] if args.audio_path else []
        videos = [args.video_path] if args.video_path else []
        request = {
            "messages": [
                {"role": "user", "content": args.prompt},
            ],
            "videos": videos,
            "audio_target_sr": args.audio_target_sr,
            "video_fps": args.video_fps,
            "use_audio_in_video": args.use_audio_in_video,
        }
        if images:
            request["images"] = images
        if audios:
            request["audios"] = audios
        logger.info(
            "Request prepared: %s images, %s videos, %s audios",
            len(images),
            len(videos),
            len(audios),
        )

        logger.info("Step 6: Submitting request (this may take a while)...")
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
        logger.info("=" * 60)
        logger.info("PIPELINE RESULT:")
        logger.info("=" * 60)
        print(result)
        logger.info("=" * 60)
    finally:
        logger.info("Step 7: Stopping pipeline...")
        await runner.stop()
        logger.info("✓ Pipeline stopped")


def main() -> None:
    logger.info("=" * 60)
    logger.info("Pipeline Test Script Starting")
    logger.info("=" * 60)
    args = parse_args()
    logger.info(f"Arguments parsed: model_id={args.model_id}")
    asyncio.run(main_async(args))
    logger.info("=" * 60)
    logger.info("Pipeline Test Script Finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
