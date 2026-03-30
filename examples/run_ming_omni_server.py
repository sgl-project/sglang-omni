# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for Ming-Omni (text output).

Usage::

    python examples/run_ming_omni_server.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --port 8000

Then test with::

    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "ming-omni",
            "messages": [{"role": "user", "content": "你好！"}],
            "max_tokens": 256,
            "stream": true
        }'
"""

from __future__ import annotations

import argparse
import logging
import os

from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig
from sglang_omni.serve import launch_server

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        default="inclusionAI/Ming-flash-omni-2.0",
        help="Hugging Face model id or local path",
    )

    # Pipeline options
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
    parser.add_argument(
        "--relay-backend",
        type=str,
        default="shm",
        choices=["shm", "nixl"],
        help="Relay backend for inter-stage data transfer",
    )

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-name",
        type=str,
        default="ming-omni",
        help="Model name for /v1/models (default: ming-omni)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = MingOmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
