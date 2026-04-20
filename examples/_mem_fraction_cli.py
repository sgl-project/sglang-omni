# SPDX-License-Identifier: Apache-2.0
"""Shared argparse helpers for mem_fraction_static launchers."""

from __future__ import annotations

import argparse
from typing import Any


def add_mem_fraction_static_args(
    parser: argparse.ArgumentParser,
    *,
    global_target_help: str,
    include_thinker: bool = False,
    include_talker: bool = False,
) -> None:
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=(
            f"Set SGLang mem_fraction_static for {global_target_help}. "
            "This controls SGLang's weights + KV cache memory budget. "
            "If omitted, SGLang chooses the value automatically."
        ),
    )
    if include_thinker:
        parser.add_argument(
            "--thinker-mem-fraction-static",
            type=float,
            default=None,
            help=(
                "Set SGLang mem_fraction_static only for the thinker stage. "
                "Overrides --mem-fraction-static for thinker."
            ),
        )
    if include_talker:
        parser.add_argument(
            "--talker-mem-fraction-static",
            type=float,
            default=None,
            help=(
                "Set SGLang mem_fraction_static only for the talker stage. "
                "Overrides --mem-fraction-static for talker."
            ),
        )


def apply_mem_fraction_static_args(config: Any, args: argparse.Namespace) -> None:
    config.apply_mem_fraction_static_overrides(
        mem_fraction_static=getattr(args, "mem_fraction_static", None),
        thinker_mem_fraction_static=getattr(args, "thinker_mem_fraction_static", None),
        talker_mem_fraction_static=getattr(args, "talker_mem_fraction_static", None),
    )


def get_applied_mem_fraction_static(config: Any, stage_name: str | None) -> str:
    if stage_name is None:
        return "n/a"
    for stage in config.stages:
        if stage.name == stage_name:
            value = (stage.executor.args.get("server_args_overrides") or {}).get(
                "mem_fraction_static"
            )
            return "auto" if value is None else str(value)
    raise ValueError(f"Unknown stage {stage_name!r}")
