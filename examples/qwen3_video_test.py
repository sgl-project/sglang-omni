# SPDX-License-Identifier: Apache-2.0
"""Text-first split pipeline for Qwen3-Omni."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

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

import os

from sglang_omni.config import PipelineRunner, compile_pipeline
from sglang_omni.models.qwen3_omni import create_text_first_pipeline_config
from sglang_omni.proto import OmniRequest

# Auto-detect paths based on current environment
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)


video_path = os.path.join(_project_root, "tests/data/draw.mp4")
image_path = os.path.join(_project_root, "tests/data/cars.jpg")

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
    parser.add_argument("--frontend-device", type=str, default="cpu")
    parser.add_argument("--image-device", type=str, default="cuda:0")
    parser.add_argument("--audio-device", type=str, default="cuda:3")
    parser.add_argument("--thinker-device", type=str, default="cuda:3")
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    parser.add_argument("--video-path", type=str, default=video_path)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--use-audio-in-video", action="store_true")
    parser.add_argument(
        "--enable-overlap",
        action="store_true",
        help="Enable intra-stage CPU/GPU overlap",
    )
    parser.add_argument(
        "--max-pending", type=int, default=4, help="Max pending requests for overlap"
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    timing_results = {}

    logger.info("=" * 60)
    logger.info("Starting Pipeline Initialization")
    logger.info("=" * 60)
    logger.info(f"Model ID: {args.model_id}")
    logger.info(f"Image device: {args.image_device}")
    logger.info(f"Audio device: {args.audio_device}")
    logger.info(f"Thinker device: {args.thinker_device}")
    logger.info(f"Intra-stage overlap: {args.enable_overlap}")

    logger.info("Step 1: Creating pipeline config...")
    t0 = time.time()
    config = create_text_first_pipeline_config(
        model_id=args.model_id,
        frontend_device=args.frontend_device,
        image_device=args.image_device,
        audio_device=args.audio_device,
        thinker_device=args.thinker_device,
        thinker_max_seq_len=args.thinker_max_seq_len,
        dtype=args.dtype,
        enable_intra_stage_overlap=args.enable_overlap,
        max_pending=args.max_pending,
    )
    t1 = time.time()
    timing_results["1_create_config"] = t1 - t0
    logger.info(f"✓ Pipeline config created (took {t1 - t0:.3f}s)")

    logger.info("Step 2: Compiling pipeline...")
    t0 = time.time()
    coordinator, stages = compile_pipeline(config)
    t1 = time.time()
    timing_results["2_compile_pipeline"] = t1 - t0
    logger.info(
        f"✓ Pipeline compiled, {len(stages)} stages created (took {t1 - t0:.3f}s)"
    )

    logger.info("Step 3: Creating PipelineRunner...")
    t0 = time.time()
    runner = PipelineRunner(coordinator, stages)
    t1 = time.time()
    timing_results["3_create_runner"] = t1 - t0
    logger.info(f"✓ PipelineRunner created (took {t1 - t0:.3f}s)")

    logger.info(
        "Step 4: Starting pipeline (this may take a while for model loading)..."
    )
    t0 = time.time()
    await runner.start()
    t1 = time.time()
    timing_results["4_start_pipeline"] = t1 - t0
    logger.info(f"✓ Pipeline started successfully (took {t1 - t0:.3f}s)")

    try:
        logger.info("Step 5: Preparing request...")
        t0 = time.time()
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
        t1 = time.time()
        timing_results["5_prepare_request"] = t1 - t0
        logger.info(
            "Request prepared: %s images, %s videos, %s audios (took %.3fs)",
            len(images),
            len(videos),
            len(audios),
            t1 - t0,
        )

        logger.info("Step 6: Submitting request (this may take a while)...")
        t0 = time.time()
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
        t1 = time.time()
        timing_results["6_submit_request"] = t1 - t0

        logger.info("=" * 60)
        logger.info("PIPELINE RESULT:")
        logger.info("=" * 60)
        print(result)
        logger.info("=" * 60)
        logger.info(f"Request processing took {t1 - t0:.3f}s")

    finally:
        logger.info("Step 7: Stopping pipeline...")
        t0 = time.time()
        await runner.stop()
        t1 = time.time()
        timing_results["7_stop_pipeline"] = t1 - t0
        logger.info(f"✓ Pipeline stopped (took {t1 - t0:.3f}s)")

    # Write timing report
    logger.info("=" * 60)
    logger.info("TIMING SUMMARY:")
    logger.info("=" * 60)
    total_time = sum(timing_results.values())
    for step, duration in timing_results.items():
        percentage = (duration / total_time) * 100
        logger.info(f"{step}: {duration:.3f}s ({percentage:.1f}%)")
    logger.info(f"Total time: {total_time:.3f}s")
    logger.info("=" * 60)


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
