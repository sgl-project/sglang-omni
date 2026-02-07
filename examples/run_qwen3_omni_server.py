# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for Qwen3-Omni.

Usage::

    python examples/run_qwen3_omni_server.py \
        --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --thinker-device cuda:0 \
        --image-device cuda:1 \
        --audio-device cuda:1 \
        --port 8000

Then test with::

    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "qwen3-omni",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 256,
            "stream": true
        }'
"""

from __future__ import annotations

import argparse
import logging
import os

from sglang_omni.models.qwen3_omni import create_text_first_pipeline_config
from sglang_omni.serve import launch_server

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Model
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Hugging Face model id",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")

    # Device placement
    parser.add_argument("--preprocessing-device", type=str, default="cpu")
    parser.add_argument("--image-device", type=str, default="cuda:0")
    parser.add_argument("--audio-device", type=str, default="cuda:0")
    parser.add_argument("--thinker-device", type=str, default="cuda:0")
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)

    # Pipeline options
    parser.add_argument(
        "--relay-type",
        type=str,
        default="shm",
        choices=["shm", "nccl", "nixl"],
        help="Relay type for inter-stage data transfer",
    )

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for /v1/models (default: pipeline name)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build pipeline config
    config = create_text_first_pipeline_config(
        model_id=args.model_id,
        preprocessing_device=args.preprocessing_device,
        image_device=args.image_device,
        audio_device=args.audio_device,
        thinker_device=args.thinker_device,
        thinker_max_seq_len=args.thinker_max_seq_len,
        dtype=args.dtype,
        relay_type=args.relay_type,
    )

    # Launch: compile pipeline + start stages + start OpenAI server
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
