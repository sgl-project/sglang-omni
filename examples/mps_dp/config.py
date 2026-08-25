# SPDX-License-Identifier: Apache-2.0
"""Resolve launcher values from an SGLang Omni pipeline config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)

# Note (Jiaxin Deng): the machine-readable weight-share support registry:
# pipeline configs whose documented launch.sh command passed shared-DP
# end-to-end validation (boot, attach, concurrent correctness, teardown),
# paired with the SGLang engine architecture each config serves. Unit tests
# enforce the implication chain, not an identity: a launcher-supported config
# implies its architecture is gate-enabled, a shipped example config exists,
# and the docs mark it supported. An architecture may back several configs;
# an architecture audit alone does not add a row here.
WEIGHT_SHARE_VALIDATED_CONFIGS: dict[str, str] = {
    "HiggsTtsPipelineConfig": "HiggsMultimodalQwen3ForConditionalGeneration",
    "MossTTSLocalPipelineConfig": "MossTTSLocalSGLangModel",
    "MossTTSPipelineConfig": "MossTTSDelaySGLangModel",
    "MossTranscribeDiarizePipelineConfig": (
        "MossTranscribeDiarizeForConditionalGeneration"
    ),
    "Qwen3ASRPipelineConfig": "Qwen3ASRForConditionalGeneration",
    "WhisperASRPipelineConfig": "WhisperForConditionalGeneration",
    "FunASRPipelineConfig": "FunAsrNanoForConditionalGeneration",
}


def resolve_max_total_tokens(
    config_path: str | Path,
    max_total_tokens_override: int | None = None,
    *,
    require_single_sglang_engine: bool = False,
    weight_share: bool = False,
) -> tuple[str, int | None]:
    """Return the generation stage name and its KV cap (None when unpinned)."""

    pipeline_config = ConfigManager.from_file(str(config_path)).config
    config_type = type(pipeline_config)
    if weight_share and config_type.__name__ not in WEIGHT_SHARE_VALIDATED_CONFIGS:
        raise ValueError(
            f"weight sharing is not supported for {config_type.__name__}: it has "
            "not passed end-to-end validation on this launcher (an architecture "
            "audit alone does not enable sharing). Validated configs: "
            f"{', '.join(sorted(WEIGHT_SHARE_VALIDATED_CONFIGS))}"
        )
    engine_stage_names = [
        stage.name
        for stage in pipeline_config.stages
        if config_type.stage_config_cls(stage.name).engine_stage
    ]
    if not engine_stage_names:
        raise ValueError(
            f"{config_type.__name__} does not declare an SGLang engine stage; "
            "the mps_dp launcher only drives pipelines with exactly one SGLang "
            "generation engine"
        )
    if require_single_sglang_engine and len(engine_stage_names) != 1:
        raise ValueError(
            "KV verification requires CONFIG with one SGLang engine stage; "
            f"found {sorted(engine_stage_names)}"
        )
    # In pipeline order, the generation engine comes first; every launcher-
    # validated config has exactly one engine stage anyway.
    stage_name = engine_stage_names[0]

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
        # Both kwarg channels can carry server args; per-key, the stage's own
        # engine block outranks the author's stage_factory_kwargs, matching
        # how the stage worker overlays them at construction.
        overrides = dict(
            resolve_stage_factory_kwargs(stage, pipeline_config).get(
                "server_args_overrides"
            )
            or {}
        )
        overrides.update(
            resolve_stage_typed_kwargs(stage).get("server_args_overrides") or {}
        )
        value = overrides.get("max_total_tokens")
    if value is None:
        return stage_name, None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "the generation stage must define a positive integer max_total_tokens"
        )
    return stage_name, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="SGLang Omni pipeline config")
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--require-single-sglang-engine", action="store_true")
    parser.add_argument("--weight-share", action="store_true")
    parser.add_argument(
        "--print-stage",
        action="store_true",
        help=(
            "Print 'STAGE VALUE' instead of VALUE, so the launcher can build "
            "the dotted serve flag (--STAGE.engine.max_total_tokens)."
        ),
    )
    args = parser.parse_args()
    try:
        stage_name, value = resolve_max_total_tokens(
            args.config,
            args.max_total_tokens,
            require_single_sglang_engine=args.require_single_sglang_engine,
            weight_share=args.weight_share,
        )
        if value is not None:
            print(f"{stage_name} {value}" if args.print_stage else value)
    except (OSError, KeyError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
