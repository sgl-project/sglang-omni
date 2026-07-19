# SPDX-License-Identifier: Apache-2.0
"""Resolve launcher values from an SGLang Omni pipeline config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_static_factory_args


def resolve_max_total_tokens(
    config_path: str | Path,
    max_total_tokens_override: int | None = None,
    *,
    require_single_sglang_engine: bool = False,
) -> int | None:
    """Return the effective generation-stage KV cap, or None when unpinned."""

    pipeline_config = ConfigManager.from_file(str(config_path)).config
    config_type = type(pipeline_config)
    stage_name = config_type.generation_sglang_role_to_stage().get("generation")
    if stage_name is None:
        raise ValueError(f"{config_type.__name__} does not declare a generation stage")

    sglang_stage_names = {
        *config_type.mem_fraction_role_to_stage().values(),
        *config_type.talker_sglang_role_to_stage().values(),
        *config_type.generation_sglang_role_to_stage().values(),
    }
    if require_single_sglang_engine and sglang_stage_names != {stage_name}:
        raise ValueError(
            "KV verification requires CONFIG with one SGLang engine stage; "
            f"found {sorted(sglang_stage_names)}"
        )

    stage = next(
        (stage for stage in pipeline_config.stages if stage.name == stage_name),
        None,
    )
    if stage is None:
        raise ValueError(
            f"generation stage {stage_name!r} is missing from the pipeline"
        )

    value = max_total_tokens_override
    if value is None:
        factory_args = resolve_stage_static_factory_args(stage, pipeline_config)
        value = factory_args.get("server_args_overrides", {}).get("max_total_tokens")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "the generation stage must define a positive integer max_total_tokens"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="SGLang Omni pipeline config")
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--require-single-sglang-engine", action="store_true")
    args = parser.parse_args()
    try:
        value = resolve_max_total_tokens(
            args.config,
            args.max_total_tokens,
            require_single_sglang_engine=args.require_single_sglang_engine,
        )
        if value is not None:
            print(value)
    except (OSError, KeyError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
