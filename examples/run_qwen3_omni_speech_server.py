# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for Qwen3-Omni with speech output.

Each stage runs in its own process with dedicated GPU placement.
Supports text + audio responses via the OpenAI chat completions API.

Usage::

    python examples/run_qwen3_omni_speech_server.py

    # Custom GPU placement:
    python examples/run_qwen3_omni_speech_server.py \
        --gpu-thinker 0 --gpu-talker 1 --gpu-code-predictor 2

    # Then test:
    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "qwen3-omni",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 64,
            "stream": true,
            "modalities": ["text", "audio"]
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
        "--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )

    # GPU placement
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=1)
    parser.add_argument("--gpu-code-predictor", type=int, default=2)
    parser.add_argument("--gpu-code2wav", type=int, default=0)
    parser.add_argument("--gpu-image-encoder", type=int, default=0)
    parser.add_argument("--gpu-audio-encoder", type=int, default=0)

    # Pipeline
    parser.add_argument(
        "--relay-backend", type=str, default="shm", choices=["nixl", "shm"]
    )
    parser.add_argument(
        "--thinker-max-seq-len",
        type=int,
        default=8192,
        help=(
            "Context length for the thinker stage. The same value is routed "
            "to preprocessing and Talker context guards."
        ),
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set SGLang mem_fraction_static for both Qwen AR stages "
            "(thinker and talker). If omitted, SGLang chooses automatically."
        ),
    )
    parser.add_argument(
        "--thinker-mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set SGLang mem_fraction_static only for the thinker stage. "
            "Overrides --mem-fraction-static for thinker."
        ),
    )
    parser.add_argument(
        "--talker-mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set SGLang mem_fraction_static only for the talker stage. "
            "Overrides --mem-fraction-static for talker."
        ),
    )

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", type=str, default="qwen3-omni")

    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    import uvicorn
    from _launcher_mem_fraction import resolve_and_apply_speech_mem_fraction

    from sglang_omni.client import Client
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.serve.openai_api import create_app

    # Build GPU placement from CLI args
    gpu_placement = {
        "thinker": args.gpu_thinker,
        "talker_ar": args.gpu_talker,
        "code_predictor": args.gpu_code_predictor,
        "code2wav": args.gpu_code2wav,
    }

    config = Qwen3OmniSpeechPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
        gpu_placement=gpu_placement,
    )
    config.apply_server_args_overrides(
        stage_name="thinker",
        overrides={"thinker_max_seq_len": args.thinker_max_seq_len},
    )
    thinker_mem_fraction_static, talker_mem_fraction_static = (
        resolve_and_apply_speech_mem_fraction(
            config,
            global_mem_fraction_static=args.mem_fraction_static,
            thinker_mem_fraction_static=args.thinker_mem_fraction_static,
            talker_mem_fraction_static=args.talker_mem_fraction_static,
        )
    )
    logger.info(
        f"Speech server config: thinker_gpu={args.gpu_thinker} "
        f"talker_gpu={args.gpu_talker} "
        f"code_predictor_gpu={args.gpu_code_predictor} "
        f"code2wav_gpu={args.gpu_code2wav} "
        f"thinker_mem_fraction_static="
        f"{'auto' if thinker_mem_fraction_static is None else thinker_mem_fraction_static} "
        f"talker_mem_fraction_static="
        f"{'auto' if talker_mem_fraction_static is None else talker_mem_fraction_static} "
        f"thinker_max_seq_len={args.thinker_max_seq_len}"
    )

    runner = MultiProcessPipelineRunner(config)
    logger.info("Starting 9-stage speech pipeline (multiprocess)...")
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
