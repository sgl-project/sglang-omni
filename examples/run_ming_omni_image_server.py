# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for Ming-Omni with image generation.

Each stage runs in its own process with dedicated GPU placement.
Supports text + image responses via the OpenAI chat completions API.

Usage::

    python examples/run_ming_omni_image_server.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --dit-model-path /path/to/zimage-or-sd3

    # Custom GPU placement:
    python examples/run_ming_omni_image_server.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --dit-model-path /path/to/zimage \
        --dit-type zimage \
        --gpu-thinker 0 --gpu-img-gen 1

    # Then test:
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "ming-omni",
            "messages": [{"role": "user", "content": "Draw a cat under moonlight"}],
            "max_tokens": 256,
            "modalities": ["text", "image"],
            "image_generation": {
                "size": "1024x1024",
                "num_inference_steps": 28,
                "guidance_scale": 7.0
            }
        }'
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import os

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="inclusionAI/Ming-flash-omni-2.0",
        help="Hugging Face model id or local path for the thinker",
    )
    parser.add_argument(
        "--dit-model-path",
        type=str,
        required=True,
        help="Path to the diffusion model (SD3 or Z-Image)",
    )
    parser.add_argument(
        "--dit-type",
        type=str,
        default="zimage",
        choices=["sd3", "zimage"],
        help="Diffusion model type",
    )

    # GPU placement
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-img-gen", type=int, default=1)

    # Pipeline
    parser.add_argument(
        "--relay-backend", type=str, default="shm", choices=["nixl", "shm"]
    )

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", type=str, default="ming-omni")

    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    import uvicorn

    from sglang_omni.client import Client
    from sglang_omni.models.ming_omni.config import MingOmniImagePipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.serve.openai_api import create_app

    gpu_placement = {
        "thinker": args.gpu_thinker,
        "img_gen": args.gpu_img_gen,
    }

    config = MingOmniImagePipelineConfig(
        model_path=args.model_path,
        dit_type=args.dit_type,
        dit_model_path=args.dit_model_path,
        relay_backend=args.relay_backend,
        gpu_placement=gpu_placement,
    )

    runner = MultiProcessPipelineRunner(config)
    logger.info("Starting Ming-Omni image pipeline (multiprocess)...")
    await runner.start(timeout=600)
    logger.info("Pipeline ready.")

    try:
        client = Client(runner.coordinator)
        app = create_app(client, model_name=args.model_name)

        server_config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
        server = uvicorn.Server(server_config)
        await server.serve()
    finally:
        logger.info("Shutting down pipeline...")
        await runner.stop()
        logger.info("Pipeline stopped.")


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
