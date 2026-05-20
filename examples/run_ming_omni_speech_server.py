# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for Ming-Omni with speech output.

Each stage runs in its own process with dedicated GPU placement.
Supports text + audio responses via the OpenAI chat completions API.

Usage::

    python examples/run_ming_omni_speech_server.py \
        --model-path inclusionAI/Ming-flash-omni-2.0

    # Custom GPU placement:
    python examples/run_ming_omni_speech_server.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --gpu-thinker 0 --gpu-talker 1

    # Then test:
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "ming-omni",
            "messages": [{"role": "user", "content": "你好！"}],
            "max_tokens": 256,
            "stream": true,
            "modalities": ["text", "audio"]
        }'
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from typing import Any

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
        help="Hugging Face model id or local path",
    )

    # GPU placement
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=1)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help=(
            "Tensor parallel size for the thinker stage. "
            "--gpu-thinker is interpreted as the first visible GPU rank."
        ),
    )

    # Pipeline
    parser.add_argument(
        "--relay-backend", type=str, default="shm", choices=["nixl", "shm"]
    )
    parser.add_argument(
        "--voice", type=str, default="DB30", help="Voice ID for the talker"
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set SGLang mem_fraction_static for the thinker stage. "
            "If omitted, SGLang chooses automatically."
        ),
    )
    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", type=str, default="ming-omni")

    return parser.parse_args()


def _validate_fraction(flag_name: str, value: float | None) -> None:
    if value is not None and not 0.0 < value < 1.0:
        raise ValueError(f"{flag_name} must be > 0 and < 1, got {value}")


def _apply_stage_factory_updates(
    config: Any,
    *,
    stage_name: str,
    updates: dict[str, object] | None = None,
    server_arg_updates: dict[str, object] | None = None,
) -> None:
    for stage in config.stages:
        if stage.name != stage_name:
            continue
        factory_args = dict(stage.factory_args or {})
        if updates:
            factory_args.update(updates)
        if server_arg_updates:
            overrides = dict(factory_args.get("server_args_overrides") or {})
            overrides.update(server_arg_updates)
            factory_args["server_args_overrides"] = overrides
        stage.factory_args = factory_args
        return
    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def _set_stage_gpu(config: Any, stage_name: str, gpu_id: int) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.gpu = int(gpu_id)
            return
    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def _set_thinker_tp(config: Any, *, start_gpu: int, tp_size: int) -> None:
    if tp_size < 1:
        raise ValueError(f"--tp-size must be >= 1, got {tp_size}")
    for stage in config.stages:
        if stage.name == "thinker":
            stage.tp_size = int(tp_size)
            stage.parallelism = stage.parallelism.model_copy(
                update={"tp": int(tp_size)}
            )
            if tp_size == 1:
                stage.gpu = int(start_gpu)
            else:
                stage.gpu = list(range(int(start_gpu), int(start_gpu) + int(tp_size)))
            return
    raise ValueError("Stage 'thinker' not found in config")


def _launch_speech_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig
    from sglang_omni.serve import launch_server

    _validate_fraction("--mem-fraction-static", args.mem_fraction_static)

    config = MingOmniSpeechPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )
    _set_thinker_tp(
        config,
        start_gpu=args.gpu_thinker,
        tp_size=int(args.tp_size),
    )
    _set_stage_gpu(config, "talker", args.gpu_talker)
    config._validate_talker_gpu_not_in_thinker_tp_range()

    server_arg_updates: dict[str, object] = {}
    if args.tp_size and args.tp_size > 1:
        server_arg_updates["disable_custom_all_reduce"] = True
    if args.mem_fraction_static is not None:
        server_arg_updates["mem_fraction_static"] = args.mem_fraction_static
    if server_arg_updates:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates=server_arg_updates,
        )
    _apply_stage_factory_updates(
        config,
        stage_name="talker",
        updates={"voice": args.voice},
    )

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    _launch_speech_server(args)


if __name__ == "__main__":
    main()
