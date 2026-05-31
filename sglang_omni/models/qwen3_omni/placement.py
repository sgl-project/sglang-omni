# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni placement policy."""

from __future__ import annotations

from sglang_omni.config import PipelineConfig, StagePlacementPlan

_SPEECH_STAGE_ORDER = (
    "preprocessing",
    "image_encoder",
    "audio_encoder",
    "mm_aggregate",
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
_COLOCATED_CONFIG_CLASS = "Qwen3OmniSpeechColocatedPipelineConfig"


class Qwen3OmniPlacementPolicy:
    """Validate Qwen-specific placement semantics outside generic planning."""

    def validate(self, config: PipelineConfig, plan: StagePlacementPlan) -> None:
        stage_map = {stage.name: stage for stage in config.stages}

        if "code_predictor" in stage_map:
            raise ValueError(
                "Qwen code predictor is part of talker_ar and must not be a stage"
            )

        has_speech_stage = "talker_ar" in stage_map or "code2wav" in stage_map
        if has_speech_stage:
            self._validate_speech_topology(stage_map)

        if not has_speech_stage:
            return

        if type(config).__name__ == _COLOCATED_CONFIG_CLASS:
            self._validate_colocated_qwen_parallelism(stage_map)
            self._validate_colocated_qwen_topology(plan)
            self._validate_colocated_qwen_runtime(stage_map)
            return

        thinker = plan.stages.get("thinker")
        talker = plan.stages.get("talker_ar")
        if thinker is None or talker is None:
            return
        if thinker.tp_size != 1 or talker.tp_size != 1:
            return
        if not set(thinker.gpu_ids).intersection(talker.gpu_ids):
            return

        raise ValueError(
            "Qwen thinker and talker_ar may share a GPU only with "
            f"{_COLOCATED_CONFIG_CLASS}"
        )

    def _validate_colocated_qwen_parallelism(self, stage_map) -> None:
        for stage_name in _AR_STAGES:
            stage = stage_map.get(stage_name)
            if stage is not None and stage.tp_size != 1:
                raise ValueError(
                    f"Qwen Phase 1 colocation does not support {stage_name} TP"
                )

    def _validate_speech_topology(self, stage_map) -> None:
        names = set(stage_map)
        if names != _SPEECH_STAGE_SET:
            missing = sorted(_SPEECH_STAGE_SET - names)
            extra = sorted(names - _SPEECH_STAGE_SET)
            raise ValueError(
                "Qwen speech must use the eight configured stages; "
                f"missing={missing}, extra={extra}"
            )

    def _validate_colocated_qwen_topology(self, plan: StagePlacementPlan) -> None:
        gpu_ids: set[int] = set()
        invalid: list[str] = []
        for stage_name in sorted(_COLOCATED_BUDGET_STAGES):
            placement = plan.stages.get(stage_name)
            if placement is None or len(placement.gpu_ids) != 1:
                invalid.append(stage_name)
                continue
            gpu_ids.add(placement.gpu_ids[0])

        if invalid:
            raise ValueError(
                "Qwen colocated speech requires exactly one GPU id for " f"{invalid}"
            )
        if len(gpu_ids) != 1:
            raise ValueError(
                "Qwen colocated speech requires image_encoder, audio_encoder, "
                "thinker, talker_ar, and code2wav to share one GPU"
            )

    def _validate_colocated_qwen_runtime(self, stage_map) -> None:
        missing_budgets = [
            stage_name
            for stage_name in sorted(_COLOCATED_BUDGET_STAGES)
            if (
                stage_map[stage_name].runtime.resources.total_gpu_memory_fraction
                is None
            )
        ]
        if missing_budgets:
            raise ValueError(
                "Qwen colocated speech requires total_gpu_memory_fraction for "
                f"{missing_budgets}"
            )

        for stage_name in _AR_STAGES:
            stage = stage_map[stage_name]
            total_fraction = stage.runtime.resources.total_gpu_memory_fraction
            mem_fraction = stage.runtime.sglang_server_args.mem_fraction_static
            if mem_fraction is None:
                continue
            if abs(mem_fraction - total_fraction) > 1e-3:
                raise ValueError(
                    f"Qwen colocated speech stage {stage_name} sets conflicting "
                    "memory fractions: total_gpu_memory_fraction="
                    f"{total_fraction:.3f}, mem_fraction_static={mem_fraction:.3f}"
                )
