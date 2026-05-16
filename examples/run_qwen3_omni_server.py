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
from typing import Any

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
        "--encoder-mem-reserve",
        type=float,
        default=None,
        help=(
            "GPU-memory fraction kept OUT of SGLang's static pool (model weights "
            "+ KV cache) and left free for the co-located vision/audio encoder's "
            "weights and activations on the thinker GPU.\n"
            "Behavior across the four flag combinations of --mem-fraction-static "
            "and --encoder-mem-reserve:\n"
            "  (1) neither flag passed: SGLang auto-selects mem_fraction_static "
            "and the default reserve 0.05 is subtracted;\n"
            "  (2) only --encoder-mem-reserve X: SGLang auto-selects "
            "mem_fraction_static and X is subtracted;\n"
            "  (3) only --mem-fraction-static X: X is used verbatim and the "
            "default reserve is ignored;\n"
            "  (4) both flags: rejected at CLI as mutually exclusive.\n"
            "Default 0.05 is tuned for single-request / short-video workloads; "
            "raise to 0.15-0.20 for high-concurrency long-video or long-audio "
            "workloads."
        ),
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


def _check_mem_flag_mutex(
    mem_fraction_static: float | None,
    encoder_mem_reserve: float | None,
) -> None:
    """Reject passing both --mem-fraction-static and --encoder-mem-reserve."""
    if mem_fraction_static is not None and encoder_mem_reserve is not None:
        raise ValueError(
            "--mem-fraction-static and --encoder-mem-reserve are mutually "
            "exclusive: --mem-fraction-static pins the pool size directly "
            "and the reserve only subtracts from SGLang's auto-selected "
            "value. Pass only one."
        )


def _validate_fraction(flag_name: str, value: float | None) -> None:
    if value is not None and not 0.0 < value < 1.0:
        raise ValueError(f"{flag_name} must be > 0 and < 1, got {value}")


def _validate_encoder_mem_reserve(value: float | None) -> None:
    if value is not None and not 0.0 <= value < 1.0:
        raise ValueError(f"--encoder-mem-reserve must be in [0, 1), got {value}")


def _apply_stage_factory_updates(
    config: Any,
    *,
    stage_name: str,
    updates: dict[str, object],
    server_arg_updates: dict[str, object] | None = None,
) -> None:
    for stage in config.stages:
        if stage.name != stage_name:
            continue

        factory_args = dict(stage.factory_args or {})
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


def _launch_text_server(args: argparse.Namespace) -> None:
    _check_mem_flag_mutex(args.mem_fraction_static, args.encoder_mem_reserve)
    _validate_fraction("--mem-fraction-static", args.mem_fraction_static)
    _validate_encoder_mem_reserve(args.encoder_mem_reserve)

    config = Qwen3OmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
    )

    stage_updates: dict[str, object] = {}
    preprocessing_updates: dict[str, object] = {}
    if args.thinker_max_seq_len is not None:
        thinker_max_seq_len = int(args.thinker_max_seq_len)
        stage_updates["thinker_max_seq_len"] = thinker_max_seq_len
        preprocessing_updates["thinker_max_seq_len"] = thinker_max_seq_len
    if args.encoder_mem_reserve is not None:
        stage_updates["encoder_mem_reserve"] = args.encoder_mem_reserve

    server_arg_updates: dict[str, object] = {}
    if args.cpu_offload_gb:
        server_arg_updates["cpu_offload_gb"] = int(args.cpu_offload_gb)
    if args.mem_fraction_static is not None:
        server_arg_updates["mem_fraction_static"] = args.mem_fraction_static

    if stage_updates or server_arg_updates:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            updates=stage_updates,
            server_arg_updates=server_arg_updates or None,
        )
    if preprocessing_updates:
        _apply_stage_factory_updates(
            config,
            stage_name="preprocessing",
            updates=preprocessing_updates,
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
    main()
