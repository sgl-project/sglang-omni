# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for MiniCPM-V (vision-only).

Supports both MiniCPM-V 2.6 and 4.5 architectures:
- 2.6: SigLIP-400M + MiniCPM-3.0 LLM + Perceiver Resampler
- 4.5: SigLIP2-400M + Qwen3-8B LLM + 3D-Resampler

Usage::

    # MiniCPM-V 2.6
    python examples/run_minicpm_v_server.py \\
        --model-path openbmb/MiniCPM-V-2_6 \\
        --port 8000

    # MiniCPM-V 4.5
    python examples/run_minicpm_v_server.py \\
        --model-path openbmb/MiniCPM-V-4_5 \\
        --port 8000

Then test with::

    # Text-only request
    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "minicpm-v",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 256,
            "stream": true
        }'

    # Vision request with base64 image
    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "minicpm-v",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                        {"type": "text", "text": "Describe this image."}
                    ]
                }
            ],
            "max_tokens": 512
        }'
"""

from __future__ import annotations

import argparse
import logging
import os

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        default="openbmb/MiniCPM-V-2_6",
        help="Hugging Face model id or local path (supports 2.6 and 4.5)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16)",
    )

    # Device placement
    parser.add_argument(
        "--preprocessing-device",
        type=str,
        default="cpu",
        help="Device for preprocessing stage",
    )
    parser.add_argument(
        "--image-device",
        type=str,
        default="cuda:0",
        help="Device for image encoder (SigLIP/SigLIP2 + Resampler)",
    )
    parser.add_argument(
        "--llm-device",
        type=str,
        default="cuda:0",
        help="Device for LLM backbone (MiniCPM-3.0 or Qwen3-8B)",
    )

    # Pipeline options
    parser.add_argument(
        "--llm-max-seq-len",
        type=int,
        default=8192,
        help="Maximum sequence length for LLM (default: 8192, use 32768 for 4.5)",
    )
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
        default="minicpm-v",
        help="Model name for /v1/models endpoint",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sglang_omni.models.minicpm_v.components.common import (
        get_minicpm_version,
        is_minicpm_v45,
    )
    from sglang_omni.models.minicpm_v.config import (
        MiniCPMV45PipelineConfig,
        MiniCPMVPipelineConfig,
    )
    from sglang_omni.serve import launch_server

    # Detect model version and select appropriate config
    version = get_minicpm_version(args.model_path)
    logger.info(f"Detected MiniCPM-V version: {version}")

    # Use version-specific pipeline config
    if is_minicpm_v45(args.model_path):
        logger.info("Using MiniCPM-V 4.5 pipeline (Qwen3-8B backbone)")
        config = MiniCPMV45PipelineConfig(model_path=args.model_path)
    else:
        logger.info("Using MiniCPM-V 2.6 pipeline (MiniCPM-3.0 backbone)")
        config = MiniCPMVPipelineConfig(model_path=args.model_path)

    # Launch OpenAI-compatible server
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
