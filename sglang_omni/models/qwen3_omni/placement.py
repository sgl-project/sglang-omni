# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni placement policy."""

from __future__ import annotations

from sglang_omni.config import PipelineConfig, StageConfig, StagePlacementPlan
from sglang_omni.models.qwen3_omni.config import (
    THINKER_DECODE_STAGE,
    THINKER_PREFILL_STAGE,
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniTextPDPipelineConfig,
)

_SPEECH_STAGE_ORDER = (
    "preprocessing",
    "image_encoder",
    "audio_encoder",
    "thinker",
    "decode",
    "talker_ar",
    "code2wav",
)
_SPEECH_STAGE_SET = set(_SPEECH_STAGE_ORDER)
_COLOCATED_BUDGET_STAGES = {
    "image_encoder",
    "audio_encoder",
    "thinker",
    "talker_ar",
    "code2wav",
}
_AR_STAGES = ("thinker", "talker_ar")
# Stages another model package adds to the same server are not Qwen's; only Qwen's own stage names take part in the check.
_QWEN_STAGE_SET = _SPEECH_STAGE_SET | {"mm_aggregate"}
TEXT_PD_STAGE_SET = {
    "preprocessing",
    "image_encoder",
    "audio_encoder",
    THINKER_PREFILL_STAGE,
    THINKER_DECODE_STAGE,
    "decode",
}
PD_MATCHING_ENGINE_DEFAULTS: dict[str, int | str] = {
    "page_size": 1,
    "kv_cache_dtype": "auto",
    "dtype": "auto",
}


class Qwen3OmniPlacementPolicy:
    """Validate Qwen-specific placement semantics outside generic planning."""

    def validate(self, config: PipelineConfig, plan: StagePlacementPlan) -> None:
        stage_map = {stage.name: stage for stage in config.stages}

        if "code_predictor" in stage_map:
            raise ValueError(
                "Qwen code predictor is part of talker_ar and must not be a stage"
            )
        else:
            pass

        is_text_pd = (
            isinstance(config, Qwen3OmniTextPDPipelineConfig)
            or THINKER_PREFILL_STAGE in stage_map
            or THINKER_DECODE_STAGE in stage_map
        )
        if is_text_pd:
            self.validate_text_pd(stage_map, plan)
            return
        else:
            pass

        has_speech_stage = "talker_ar" in stage_map or "code2wav" in stage_map
        if has_speech_stage:
            self.validate_speech_topology(stage_map)
        else:
            pass

        if not has_speech_stage:
            return
        else:
            pass

        if isinstance(config, Qwen3OmniSpeechColocatedPipelineConfig):
            self.validate_colocated_qwen_replicas(plan)
            self.validate_colocated_qwen_parallelism(stage_map)
            self.validate_colocated_qwen_topology(plan)
            self.validate_colocated_qwen_runtime(stage_map)
            return
        else:
            pass

        thinkers = plan.instances_of("thinker")
        talkers = plan.instances_of("talker_ar")
        has_replicated_ar_stage = len(thinkers) > 1 or len(talkers) > 1
        for thinker in thinkers:
            for talker in talkers:
                if not has_replicated_ar_stage and (
                    thinker.tp_size != 1 or talker.tp_size != 1
                ):
                    continue
                else:
                    pass
                if not set(thinker.gpu_ids).intersection(talker.gpu_ids):
                    continue
                else:
                    pass
                raise ValueError(
                    f"Qwen thinker and talker_ar ({talker.stage_name!r}) may "
                    f"share a GPU only with {Qwen3OmniSpeechColocatedPipelineConfig.__name__} or its subclasses"
                )

    def validate_colocated_qwen_replicas(self, plan: StagePlacementPlan) -> None:
        # Note (kaige): read the expanded plan rather than the pre-expansion
        # stage config, because replica counts now live on the process.
        replicated = sorted(
            name
            for name, instances in plan.replica_instances.items()
            if len(instances) > 1
        )
        if replicated:
            raise ValueError(
                "Qwen colocated speech does not support process replicas; "
                f"got replicated stage(s) {replicated}"
            )
        else:
            pass

    def validate_colocated_qwen_parallelism(self, stage_map) -> None:
        for stage_name in _AR_STAGES:
            stage = stage_map.get(stage_name)
            if stage is not None and stage.tp_size != 1:
                raise ValueError(
                    f"Qwen Phase 1 colocation does not support {stage_name} TP"
                )
            else:
                pass

    def validate_speech_topology(self, stage_map) -> None:
        names = set(stage_map) & _QWEN_STAGE_SET
        if names != _SPEECH_STAGE_SET:
            missing = sorted(_SPEECH_STAGE_SET - names)
            extra = sorted(names - _SPEECH_STAGE_SET)
            raise ValueError(
                "Qwen speech must use the seven configured stages; "
                f"missing={missing}, extra={extra}"
            )
        else:
            pass

    def validate_colocated_qwen_topology(self, plan: StagePlacementPlan) -> None:
        gpu_ids: set[int] = set()
        invalid: list[str] = []
        for stage_name in sorted(_COLOCATED_BUDGET_STAGES):
            placement = plan.stages.get(stage_name)
            if placement is None or len(placement.gpu_ids) != 1:
                invalid.append(stage_name)
                continue
            else:
                pass
            gpu_ids.add(placement.gpu_ids[0])

        if invalid:
            raise ValueError(
                "Qwen colocated speech requires exactly one GPU id for " f"{invalid}"
            )
        else:
            pass
        if len(gpu_ids) != 1:
            raise ValueError(
                "Qwen colocated speech requires image_encoder, audio_encoder, "
                "thinker, talker_ar, and code2wav to share one GPU"
            )
        else:
            pass

    def validate_colocated_qwen_runtime(self, stage_map) -> None:
        # Note (Jiaxin Deng): the schema forbids fraction next to kv_cache_bytes,
        # so requiring the fraction would outlaw every colocated byte budget.
        missing_budgets = [
            stage_name
            for stage_name in sorted(_COLOCATED_BUDGET_STAGES)
            if (
                stage_map[stage_name].gpu_memory_fraction is None
                and (
                    stage_map[stage_name].engine is None
                    or stage_map[stage_name].engine.kv_cache_bytes is None
                )
            )
        ]
        if missing_budgets:
            raise ValueError(
                "Qwen colocated speech requires gpu_memory_fraction or "
                f"engine.kv_cache_bytes for {missing_budgets}"
            )
        else:
            pass

        for stage_name in _AR_STAGES:
            stage = stage_map[stage_name]
            total_fraction = stage.gpu_memory_fraction
            mem_fraction = (
                stage.engine.mem_fraction_static if stage.engine is not None else None
            )
            if mem_fraction is None:
                continue
            else:
                pass
            if abs(mem_fraction - total_fraction) > 1e-3:
                raise ValueError(
                    f"Qwen colocated speech stage {stage_name} sets conflicting "
                    "memory fractions: gpu_memory_fraction="
                    f"{total_fraction:.3f}, mem_fraction_static={mem_fraction:.3f}"
                )
            else:
                pass

    def validate_text_pd(
        self,
        stage_map: dict[str, StageConfig],
        plan: StagePlacementPlan,
    ) -> None:
        if "talker_ar" in stage_map or "code2wav" in stage_map:
            raise ValueError(
                "Qwen3-Omni text-pd does not support speech stages "
                "(talker_ar/code2wav)"
            )
        else:
            pass

        pd_stage_names = set(stage_map) & (TEXT_PD_STAGE_SET | _QWEN_STAGE_SET)
        if pd_stage_names != TEXT_PD_STAGE_SET:
            missing = sorted(TEXT_PD_STAGE_SET - pd_stage_names)
            extra = sorted(pd_stage_names - TEXT_PD_STAGE_SET)
            raise ValueError(
                "Qwen3-Omni text-pd must use the six configured stages; "
                f"missing={missing}, extra={extra}"
            )
        else:
            pass

        prefill_stage = stage_map[THINKER_PREFILL_STAGE]
        decode_stage = stage_map[THINKER_DECODE_STAGE]
        if prefill_stage.kv_transfer_to != THINKER_DECODE_STAGE:
            raise ValueError(
                f"Qwen3-Omni text-pd requires {THINKER_PREFILL_STAGE}.kv_transfer_to "
                f"== {THINKER_DECODE_STAGE!r}, got {prefill_stage.kv_transfer_to!r}"
            )
        else:
            pass

        for stage in (prefill_stage, decode_stage):
            if stage.tp_size != 1:
                raise ValueError(
                    f"Qwen3-Omni text-pd stage {stage.name!r} requires tp_size == 1, "
                    f"got {stage.tp_size}"
                )
            else:
                pass

        prefill_instances = plan.instances_of(THINKER_PREFILL_STAGE)
        decode_instances = plan.instances_of(THINKER_DECODE_STAGE)
        if not prefill_instances or not decode_instances:
            raise ValueError(
                "Qwen3-Omni text-pd requires GPU placement for both "
                f"{THINKER_PREFILL_STAGE!r} and {THINKER_DECODE_STAGE!r}"
            )
        else:
            pass
        for prefill in prefill_instances:
            for decode in decode_instances:
                shared_gpus = set(prefill.gpu_ids).intersection(decode.gpu_ids)
                if shared_gpus:
                    raise ValueError(
                        f"Qwen3-Omni text-pd requires {prefill.stage_name!r} and "
                        f"{decode.stage_name!r} on distinct GPUs; "
                        f"shared={sorted(shared_gpus)}"
                    )
                else:
                    pass

        prefill_overrides = (
            prefill_stage.engine.overrides() if prefill_stage.engine is not None else {}
        )
        decode_overrides = (
            decode_stage.engine.overrides() if decode_stage.engine is not None else {}
        )
        for stage_name, overrides in (
            (THINKER_PREFILL_STAGE, prefill_overrides),
            (THINKER_DECODE_STAGE, decode_overrides),
        ):
            if overrides.get("disable_radix_cache") is False:
                raise ValueError(
                    f"Qwen3-Omni text-pd stage {stage_name!r} requires "
                    "disable_radix_cache=True"
                )
            else:
                pass
            if overrides.get("enable_mixed_chunk") is True:
                raise ValueError(
                    f"Qwen3-Omni text-pd stage {stage_name!r} requires "
                    "enable_mixed_chunk=False"
                )
            else:
                pass
            spec_algorithm = overrides.get("speculative_algorithm")
            has_spec = spec_algorithm is not None and str(
                spec_algorithm
            ).strip().lower() not in {"", "none"}
            if has_spec or overrides.get("speculative_draft_model_path") is not None:
                raise ValueError(
                    f"Qwen3-Omni text-pd stage {stage_name!r} does not support "
                    "speculative decoding"
                )
            else:
                pass

        for key, default_value in PD_MATCHING_ENGINE_DEFAULTS.items():
            prefill_value = prefill_overrides.get(key)
            decode_value = decode_overrides.get(key)
            normalized_prefill = (
                default_value if prefill_value is None else prefill_value
            )
            normalized_decode = default_value if decode_value is None else decode_value
            if normalized_prefill != normalized_decode:
                raise ValueError(
                    f"Qwen3-Omni text-pd requires matching engine.{key} on "
                    f"{THINKER_PREFILL_STAGE!r} ({normalized_prefill!r}) and "
                    f"{THINKER_DECODE_STAGE!r} ({normalized_decode!r})"
                )
            else:
                pass
