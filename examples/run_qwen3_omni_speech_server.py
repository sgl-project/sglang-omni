# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for Qwen3-Omni with speech output.

Each stage runs in its own process with dedicated GPU placement.
Supports text + audio responses via the OpenAI chat completions API.

Usage::

    python examples/run_qwen3_omni_speech_server.py

    # Custom GPU placement:
    python examples/run_qwen3_omni_speech_server.py \
        --gpu-thinker 0 --gpu-talker 1 --gpu-code2wav 1

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
        "--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )

    # GPU placement
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=1)
    parser.add_argument("--gpu-code-predictor", type=int, default=None)
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
            stage.gpu = gpu_id
            return
    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def _launch_speech_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
    from sglang_omni.serve import launch_server

    for flag_name, value in (
        ("--mem-fraction-static", args.mem_fraction_static),
        ("--thinker-mem-fraction-static", args.thinker_mem_fraction_static),
        ("--talker-mem-fraction-static", args.talker_mem_fraction_static),
    ):
        _validate_fraction(flag_name, value)

    gpu_code_predictor = (
        args.gpu_code_predictor
        if args.gpu_code_predictor is not None
        else args.gpu_talker
    )
    if gpu_code_predictor != args.gpu_talker:
        raise ValueError(
            "Qwen3 speech pipeline does not expose a separate code_predictor "
            "stage. Use the same GPU for --gpu-code-predictor and --gpu-talker."
        )

    config = Qwen3OmniSpeechPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )

    _set_stage_gpu(config, "image_encoder", args.gpu_image_encoder)
    _set_stage_gpu(config, "audio_encoder", args.gpu_audio_encoder)
    _set_stage_gpu(config, "thinker", args.gpu_thinker)
    _set_stage_gpu(config, "talker_ar", args.gpu_talker)
    _set_stage_gpu(config, "code2wav", args.gpu_code2wav)

    thinker_mem_fraction = (
        args.thinker_mem_fraction_static
        if args.thinker_mem_fraction_static is not None
        else args.mem_fraction_static
    )
    talker_mem_fraction = (
        args.talker_mem_fraction_static
        if args.talker_mem_fraction_static is not None
        else args.mem_fraction_static
    )

    if thinker_mem_fraction is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            server_arg_updates={"mem_fraction_static": thinker_mem_fraction},
        )
    if talker_mem_fraction is not None:
        _apply_stage_factory_updates(
            config,
            stage_name="talker_ar",
            server_arg_updates={"mem_fraction_static": talker_mem_fraction},
        )

    if args.thinker_max_seq_len is not None:
        thinker_seq_len_updates: dict[str, object] = {
            "thinker_max_seq_len": int(args.thinker_max_seq_len)
        }
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            updates=thinker_seq_len_updates,
        )
        _apply_stage_factory_updates(
            config,
            stage_name="preprocessing",
            updates=thinker_seq_len_updates,
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
