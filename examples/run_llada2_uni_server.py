# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for LLaDA2-Uni with text only output.

Usage::

    python examples/run_llada2_uni_server.py \
        --model-path inclusionAI/LLaDA2.0-Uni \
        --port 8000

Then test with::

    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "llada2-uni",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 256
        }'
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model-path",
        type=str,
        default="inclusionAI/LLaDA2.0-Uni",
        help="Hugging Face model id or local path",
    )
    parser.add_argument("--thinker-max-seq-len", type=int, default=None)
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set SGLang mem_fraction_static for the thinker stage. "
            "If omitted, SGLang chooses automatically."
        ),
    )

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for /v1/models (default: pipeline name)",
    )

    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

    from sglang_omni.models.llada2_uni.config import LLaDA2UniPipelineConfig
    from sglang_omni.serve import launch_server

    if (
        args.mem_fraction_static is not None
        and not 0.0 < args.mem_fraction_static < 1.0
    ):
        raise ValueError(
            f"--mem-fraction-static must be > 0 and < 1, got {args.mem_fraction_static}"
        )

    config = LLaDA2UniPipelineConfig(model_path=args.model_path)

    stage_updates: dict[str, object] = {}
    if args.thinker_max_seq_len is not None:
        stage_updates["thinker_max_seq_len"] = int(args.thinker_max_seq_len)

    server_arg_updates: dict[str, object] = {}
    if args.mem_fraction_static is not None:
        server_arg_updates["mem_fraction_static"] = args.mem_fraction_static

    if stage_updates or server_arg_updates:
        _apply_stage_factory_updates(
            config,
            stage_name="thinker",
            updates=stage_updates,
            server_arg_updates=server_arg_updates or None,
        )

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
