# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for Qwen3-Omni with text only output.

Usage::

    python examples/run_qwen3_omni_server.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
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

from _mem_fraction_cli import (
    add_mem_fraction_static_args,
    apply_mem_fraction_static_args,
)

from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.serve import launch_server

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Hugging Face model id or local path",
    )
    parser.add_argument("--thinker-max-seq-len", type=int, default=None)
    parser.add_argument(
        "--cpu-offload-gb",
        type=int,
        default=0,
        help="GB of model weights to offload to CPU",
    )

    # Pipeline options
    parser.add_argument(
        "--relay-backend",
        type=str,
        default="shm",
        choices=["shm", "nccl", "nixl"],
        help="Relay type for inter-stage data transfer",
    )
    add_mem_fraction_static_args(
        parser,
        global_target_help="the thinker stage",
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

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
