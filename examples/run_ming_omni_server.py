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
import multiprocessing as mp
import os
from typing import Any

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
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for thinker",
    )
    parser.add_argument(
        "--gpu-audio-encoder",
        type=int,
        default=None,
        help="GPU id for the audio encoder stage.",
    )
    parser.add_argument(
        "--gpu-image-encoder",
        type=int,
        nargs="+",
        default=None,
        help=(
            "GPU id(s) for the image encoder stage. "
            "For --image-encoder-tp N, pass N GPU ids."
        ),
    )
    parser.add_argument(
        "--image-encoder-tp",
        type=int,
        default=1,
        help="Tensor parallel size for image encoder (default: 1)",
    )
    parser.add_argument(
        "--thinker-only",
        action="store_true",
        help=(
            "Launch a pure-text smoke pipeline without audio/image encoders. "
            "Default keeps the full v0-parity multimodal pipeline."
        ),
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g., fp8) for thinker model",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=int,
        default=80,
        help="GB of model weights to offload to CPU (default: 80 for Ming-flash-omni-2.0)",
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
    parser.add_argument(
        "--relay-backend",
        type=str,
        default="shm",
        choices=["shm", "nccl", "nixl"],
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


def _configure_thinker_only_pipeline(config: Any) -> None:
    stages = {stage.name: stage for stage in config.stages}
    preprocessing = stages["preprocessing"]
    aggregate = stages["mm_aggregate"]

    preprocessing.next = "mm_aggregate"
    preprocessing.project_payload = {
        "mm_aggregate": (
            "sglang_omni.models.ming_omni.stages."
            "project_preprocessing_to_mm_aggregate"
        )
    }
    aggregate.wait_for = ["preprocessing"]
    config.stages = [
        stage
        for stage in config.stages
        if stage.name not in {"audio_encoder", "image_encoder"}
    ]


def _launch_text_server(args: argparse.Namespace) -> None:
    from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig
    from sglang_omni.serve import launch_server

    _validate_fraction("--mem-fraction-static", args.mem_fraction_static)

    config = MingOmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )

    if getattr(args, "thinker_only", False):
        if args.gpu_audio_encoder is not None or args.gpu_image_encoder is not None:
            raise ValueError(
                "--gpu-audio-encoder/--gpu-image-encoder cannot be used "
                "with --thinker-only"
            )
        _configure_thinker_only_pipeline(config)

    server_arg_updates: dict[str, object] = {}
    if args.tp_size and args.tp_size > 1:
        thinker = next(stage for stage in config.stages if stage.name == "thinker")
        tp_size = int(args.tp_size)
        thinker.tp_size = tp_size
        thinker.parallelism = thinker.parallelism.model_copy(update={"tp": tp_size})
        thinker.gpu = list(range(tp_size))
        server_arg_updates["disable_custom_all_reduce"] = True
    if args.gpu_audio_encoder is not None:
        _set_stage_gpu(config, "audio_encoder", args.gpu_audio_encoder)
    image_encoder_tp = getattr(args, "image_encoder_tp", 1)
    if image_encoder_tp < 1:
        raise ValueError("--image-encoder-tp must be >= 1")
    if image_encoder_tp > 1 and getattr(args, "thinker_only", False):
        raise ValueError("--thinker-only cannot be used with --image-encoder-tp > 1")
    if image_encoder_tp > 1:
        if args.gpu_image_encoder is None:
            raise ValueError(
                "--gpu-image-encoder must be specified when --image-encoder-tp > 1"
            )
        if len(args.gpu_image_encoder) != image_encoder_tp:
            raise ValueError(
                f"--gpu-image-encoder requires exactly {image_encoder_tp} GPU ids "
                f"(matching --image-encoder-tp), got {len(args.gpu_image_encoder)}"
            )
        if len(set(args.gpu_image_encoder)) != len(args.gpu_image_encoder):
            raise ValueError("--gpu-image-encoder GPU ids must be unique")
        img_stage = next(
            stage for stage in config.stages if stage.name == "image_encoder"
        )
        img_stage.tp_size = image_encoder_tp
        img_stage.parallelism = img_stage.parallelism.model_copy(
            update={"tp": image_encoder_tp}
        )
        img_stage.gpu = args.gpu_image_encoder
    elif args.gpu_image_encoder is not None:
        _set_stage_gpu(config, "image_encoder", args.gpu_image_encoder[0])
    if args.quantization:
        server_arg_updates["quantization"] = args.quantization
    if args.cpu_offload_gb:
        server_arg_updates["cpu_offload_gb"] = int(args.cpu_offload_gb)
    if args.mem_fraction_static is not None:
        server_arg_updates["mem_fraction_static"] = args.mem_fraction_static

    _apply_stage_factory_updates(
        config,
        stage_name="thinker",
        updates={"thinker_max_seq_len": int(args.thinker_max_seq_len)},
        server_arg_updates=server_arg_updates or None,
    )

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


def main() -> None:
    args = parse_args()
    _launch_text_server(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
