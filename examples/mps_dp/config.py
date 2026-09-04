# SPDX-License-Identifier: Apache-2.0
"""Resolve launcher values from an SGLang Omni pipeline config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import (
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.utils.gpu_memory import format_bytes_gib, get_gpu_device_info

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


def _resolve_generation_stage(
    config_path: str | Path,
    *,
    require_single_sglang_engine: bool = False,
    weight_share: bool = False,
) -> tuple[Any, Any, Any]:
    """Return the pipeline config, its type, and the generation stage."""

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
    return pipeline_config, config_type, stage


def _resolve_mps_memory_budget(
    config_path: str | Path,
    gpu_id: int,
    replicas: int,
    *,
    allow_missing_budget: bool,
    weight_share: bool = False,
) -> dict[str, int | str] | None:
    if replicas <= 0:
        raise ValueError("replicas must be a positive integer")

    _, config_type, stage = _resolve_generation_stage(
        config_path, weight_share=weight_share
    )
    kv_cache_bytes = stage.engine.kv_cache_bytes if stage.engine is not None else None
    total_reserve_bytes = stage.total_reserve_bytes
    if kv_cache_bytes is None and not allow_missing_budget:
        raise ValueError(
            f"{config_type.__name__} generation stage {stage.name!r} must define "
            "positive engine.kv_cache_bytes for MPS byte-budget preflight"
        )
    if kv_cache_bytes is None and total_reserve_bytes is None:
        return None

    device_info = get_gpu_device_info(gpu_id)
    if device_info.total_memory_bytes is None:
        name = device_info.name or "unknown GPU"
        raise ValueError(
            f"GPU {gpu_id} ({name}) total VRAM metadata is unavailable; cannot "
            "preflight byte budgets"
        )
    total_memory_bytes = device_info.total_memory_bytes
    gpu_name = device_info.name or "unknown GPU"

    budget: dict[str, int | str] = {
        "gpu_id": gpu_id,
        "gpu_name": gpu_name,
        "gpu_device_id": (
            "unknown" if device_info.device_id is None else device_info.device_id
        ),
        "replicas": replicas,
        "weight_share": "1" if weight_share else "0",
        "total_vram_bytes": total_memory_bytes,
        "total_vram_gib": format_bytes_gib(total_memory_bytes),
    }

    # Note (Jiaxin Deng): the pass criteria only ever multiply numbers the user
    # wrote; the KV bound holds with WEIGHT_SHARE too (KV pools stay private).
    if kv_cache_bytes is not None:
        total_kv_bytes = replicas * kv_cache_bytes
        if total_kv_bytes > total_memory_bytes:
            raise ValueError(
                "MPS byte-budget preflight exceeds physical VRAM: "
                f"GPU {gpu_id} ({gpu_name}) has "
                f"{format_bytes_gib(total_memory_bytes)}, but {replicas} "
                f"replicas x kv_cache_bytes={format_bytes_gib(kv_cache_bytes)} "
                f"= {format_bytes_gib(total_kv_bytes)} of KV pools alone. Lower "
                "engine.kv_cache_bytes or reduce the replica count."
            )
        budget["per_replica_kv_cache_bytes"] = kv_cache_bytes
        budget["per_replica_kv_cache_gib"] = format_bytes_gib(kv_cache_bytes)
        budget["total_kv_cache_bytes"] = total_kv_bytes
        budget["total_kv_cache_gib"] = format_bytes_gib(total_kv_bytes)

    if total_reserve_bytes is None:
        if replicas >= 2:
            even_split = total_memory_bytes // replicas
            print(
                "warning: total_reserve_bytes is not set; the "
                f"{replicas}-replica total footprint (weights, activations, CUDA "
                "graphs) is NOT validated, only the KV pool lower bound. Size "
                "each replica against roughly the even split of GPU "
                f"{gpu_id} ({format_bytes_gib(even_split)}) and set "
                "total_reserve_bytes explicitly to enable the full check.",
                file=sys.stderr,
            )
        return budget

    budget["per_replica_total_reserve_bytes"] = total_reserve_bytes
    budget["per_replica_total_reserve_gib"] = format_bytes_gib(total_reserve_bytes)
    if weight_share:
        # One full reservation covers replica 0 (weights resident once); every
        # further replica still owns a private KV pool.
        shared_floor_bytes = total_reserve_bytes
        if kv_cache_bytes is not None:
            shared_floor_bytes += (replicas - 1) * kv_cache_bytes
        if shared_floor_bytes > total_memory_bytes:
            raise ValueError(
                "MPS byte-budget preflight exceeds physical VRAM: "
                f"GPU {gpu_id} ({gpu_name}) has "
                f"{format_bytes_gib(total_memory_bytes)}, but one replica's "
                "total_reserve_bytes="
                f"{format_bytes_gib(total_reserve_bytes)} plus {replicas - 1} "
                "further private KV pool(s) totals "
                f"{format_bytes_gib(shared_floor_bytes)}. Lower "
                "total_reserve_bytes or engine.kv_cache_bytes, or "
                "reduce the replica count."
            )
        print(
            "warning: WEIGHT_SHARE=1 - MPS byte-budget preflight only checked "
            "the KV pool lower bound and one replica's reservation "
            f"({format_bytes_gib(total_reserve_bytes)}) against GPU {gpu_id} "
            f"VRAM ({format_bytes_gib(total_memory_bytes)}). The {replicas}-way "
            "total is NOT validated, because the shared weight size is unknown "
            "until a replica boots. Use autodp.sh to size shared-weight DP "
            "from a measured footprint.",
            file=sys.stderr,
        )
        return budget

    requested_total_bytes = replicas * total_reserve_bytes
    budget["requested_total_bytes"] = requested_total_bytes
    budget["requested_total_gib"] = format_bytes_gib(requested_total_bytes)
    if requested_total_bytes > total_memory_bytes:
        raise ValueError(
            "MPS byte-budget preflight exceeds physical VRAM: "
            f"GPU {gpu_id} ({gpu_name}) has "
            f"{format_bytes_gib(total_memory_bytes)}, but {replicas} replicas x "
            f"total_reserve_bytes={format_bytes_gib(total_reserve_bytes)} "
            f"requested={format_bytes_gib(requested_total_bytes)}. Lower "
            "total_reserve_bytes or reduce the replica count."
        )
    return budget


def _serialize_mps_memory_budget_manifest(budget: dict[str, int | str]) -> str:
    return "\n".join(f"mps_budget_{key}={value}" for key, value in budget.items())


def resolve_max_total_tokens(
    config_path: str | Path,
    max_total_tokens_override: int | None = None,
    *,
    require_single_sglang_engine: bool = False,
    weight_share: bool = False,
) -> int | None:
    """Return the effective generation-stage KV cap, or None when unpinned."""

    pipeline_config, _, stage = _resolve_generation_stage(
        config_path,
        require_single_sglang_engine=require_single_sglang_engine,
        weight_share=weight_share,
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
    if (
        value is not None
        and stage.engine is not None
        and stage.engine.kv_cache_bytes is not None
    ):
        raise ValueError(
            "engine.kv_cache_bytes and max_total_tokens both pin the "
            "generation stage's KV capacity; keep exactly one (a lower token "
            "cap would silently shrink the byte-derived pool)"
        )
    if value is None:
        return stage.name, None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "the generation stage must define a positive integer max_total_tokens"
        )
    return stage.name, value


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
    parser.add_argument("--print-mps-memory-budget", action="store_true")
    parser.add_argument("--print-kv-cache-bytes", action="store_true")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--replicas", type=int)
    args = parser.parse_args()
    try:
        if args.print_kv_cache_bytes:
            _, _, stage = _resolve_generation_stage(
                args.config, weight_share=args.weight_share
            )
            if stage.engine is not None and stage.engine.kv_cache_bytes is not None:
                print(stage.engine.kv_cache_bytes)
        elif args.print_mps_memory_budget:
            if args.gpu_id is None or args.replicas is None:
                parser.error(
                    "--print-mps-memory-budget requires both --gpu-id and " "--replicas"
                )
            budget = _resolve_mps_memory_budget(
                args.config,
                gpu_id=args.gpu_id,
                replicas=args.replicas,
                allow_missing_budget=True,
                weight_share=args.weight_share,
            )
            if budget is not None:
                print(_serialize_mps_memory_budget_manifest(budget))
        else:
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
