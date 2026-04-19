# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang_omni.config.schema import (
    ExecutorConfig,
    MemFractionOverrideStages,
    PipelineConfig,
    StageConfig,
)

_NOOP_FACTORY = "sglang_omni.pipeline.mp_runner._noop_executor_factory"
_NOOP_GET_NEXT = "sglang_omni.pipeline.mp_runner._noop_get_next"


def _make_stage(name: str, *, args: dict | None = None) -> StageConfig:
    return StageConfig(
        name=name,
        executor=ExecutorConfig(factory=_NOOP_FACTORY, args=args or {}),
        get_next=_NOOP_GET_NEXT,
    )


def _make_pipeline(
    *,
    thinker: str | None = "thinker",
    talker: str | None = "talker",
    thinker_args: dict | None = None,
    talker_args: dict | None = None,
) -> PipelineConfig:
    stages = [_make_stage("preprocessing")]
    if thinker is not None:
        stages.append(_make_stage(thinker, args=thinker_args))
    if talker is not None and talker != thinker:
        stages.append(_make_stage(talker, args=talker_args))

    return PipelineConfig(
        model_path="dummy",
        entry_stage="preprocessing",
        stages=stages,
        mem_fraction_override_stages=MemFractionOverrideStages(
            thinker=thinker,
            talker=talker,
        ),
    )


class TestMemFractionStaticOverrides(unittest.TestCase):
    def test_global_override_applies_to_all_targets_and_preserves_other_overrides(
        self,
    ) -> None:
        config = _make_pipeline(
            thinker_args={"server_args_overrides": {"cpu_offload_gb": 80}},
            talker_args={"server_args_overrides": {"enable_dp_attention": True}},
        )

        config.apply_mem_fraction_static_overrides(mem_fraction_static=0.88)

        thinker_overrides = config.stages[1].executor.args["server_args_overrides"]
        talker_overrides = config.stages[2].executor.args["server_args_overrides"]

        self.assertEqual(thinker_overrides["cpu_offload_gb"], 80)
        self.assertEqual(thinker_overrides["mem_fraction_static"], 0.88)
        self.assertTrue(talker_overrides["enable_dp_attention"])
        self.assertEqual(talker_overrides["mem_fraction_static"], 0.88)

    def test_stage_specific_overrides_take_precedence_over_global(self) -> None:
        config = _make_pipeline()

        config.apply_mem_fraction_static_overrides(
            mem_fraction_static=0.88,
            thinker_mem_fraction_static=0.83,
            talker_mem_fraction_static=0.91,
        )

        thinker_overrides = config.stages[1].executor.args["server_args_overrides"]
        talker_overrides = config.stages[2].executor.args["server_args_overrides"]

        self.assertEqual(thinker_overrides["mem_fraction_static"], 0.83)
        self.assertEqual(talker_overrides["mem_fraction_static"], 0.91)

    def test_invalid_stage_specific_override_does_not_partially_mutate_config(
        self,
    ) -> None:
        config = _make_pipeline(talker=None)

        with self.assertRaisesRegex(
            ValueError,
            "--talker-mem-fraction-static requires a pipeline with a 'talker'",
        ):
            config.apply_mem_fraction_static_overrides(
                mem_fraction_static=0.88,
                talker_mem_fraction_static=0.91,
            )

        thinker_args = config.stages[1].executor.args
        self.assertNotIn("server_args_overrides", thinker_args)

    def test_global_override_requires_declared_targets(self) -> None:
        config = _make_pipeline(thinker=None, talker=None)

        with self.assertRaisesRegex(
            ValueError,
            "--mem-fraction-static requires a pipeline with a supported",
        ):
            config.apply_mem_fraction_static_overrides(mem_fraction_static=0.88)

    def test_global_override_applies_to_single_declared_target(self) -> None:
        config = _make_pipeline(
            thinker="thinker",
            talker=None,
            thinker_args={"server_args_overrides": {"cpu_offload_gb": 80}},
        )

        config.apply_mem_fraction_static_overrides(mem_fraction_static=0.88)

        thinker_overrides = config.stages[1].executor.args["server_args_overrides"]

        self.assertEqual(thinker_overrides["cpu_offload_gb"], 80)
        self.assertEqual(thinker_overrides["mem_fraction_static"], 0.88)

    def test_override_targets_must_reference_distinct_stages(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "mem_fraction_override_stages thinker and talker must reference different stages",
        ):
            _make_pipeline(thinker="shared", talker="shared")
