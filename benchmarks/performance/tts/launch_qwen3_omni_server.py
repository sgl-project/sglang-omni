#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch a Qwen3-Omni speech server with configurable IPC base path.

Allows running multiple server instances in parallel by giving each
a unique IPC namespace (avoids socket path collisions).

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python launch_qwen3_omni_server.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --port 8000 --instance-id 0
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--instance-id",
        type=int,
        default=0,
        help="Unique instance ID for IPC path isolation",
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--model-name", type=str, default="qwen3-omni")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    import uvicorn

    from sglang_omni.client import Client
    from sglang_omni.config.schema import EndpointsConfig
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.serve.openai_api import create_app

    # Each instance gets its own IPC base path
    ipc_base = f"/tmp/sglang_omni_inst{args.instance_id}"
    os.makedirs(ipc_base, exist_ok=True)

    config = Qwen3OmniSpeechPipelineConfig(
        model_path=args.model_path,
        name=f"qwen3-omni-{args.instance_id}",
        endpoints=EndpointsConfig(base_path=ipc_base),
        gpu_placement={
            "thinker": 0,
            "talker_ar": 1,
            "code_predictor": 1,
            "code2wav": 0,
        },
    )

    runner = MultiProcessPipelineRunner(config)
    logger.info(
        "Starting instance %d (port=%d, ipc=%s)...",
        args.instance_id,
        args.port,
        ipc_base,
    )
    await runner.start(timeout=600)
    logger.info("Instance %d pipeline ready.", args.instance_id)

    try:
        client = Client(runner.coordinator)
        app = create_app(client, model_name=args.model_name)
        server_config = uvicorn.Config(
            app, host=args.host, port=args.port, log_level="info"
        )
        server = uvicorn.Server(server_config)
        await server.serve()
    finally:
        logger.info("Shutting down instance %d...", args.instance_id)
        await runner.stop()


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
