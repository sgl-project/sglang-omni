# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline config GPU validation (Ming + Qwen3)."""
from __future__ import annotations

import unittest


class TestMingOmniSpeechGPUValidation(unittest.TestCase):
    def test_default_tp_construction_rejects_colliding_gpu_placement(self):
        from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

        with self.assertRaises(ValueError) as ctx:
            MingOmniSpeechPipelineConfig(
                model_path="test/model",
                gpu_placement={"thinker": 0, "talker": 0},
            )
        self.assertIn("collides", str(ctx.exception).lower())

    def test_tp2_default_gpus_rejected(self):
        from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

        config = MingOmniSpeechPipelineConfig(model_path="test/model")
        with self.assertRaises(ValueError) as ctx:
            config.apply_server_args_overrides(
                stage_name="thinker",
                overrides={"tp_size": 2},
            )
        self.assertIn("collides", str(ctx.exception).lower())

    def test_tp2_talker_gpu2_accepted(self):
        from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

        config = MingOmniSpeechPipelineConfig(
            model_path="test/model",
            gpu_placement={"thinker": 0, "talker": 2},
        )
        config.apply_server_args_overrides(
            stage_name="thinker",
            overrides={"tp_size": 2},
        )
        self.assertEqual(config.gpu_placement["talker"], 2)

    def test_tp1_default_accepted(self):
        from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

        config = MingOmniSpeechPipelineConfig(
            model_path="test/model",
        )
        self.assertEqual(config.gpu_placement["thinker"], 0)
        self.assertEqual(config.gpu_placement["talker"], 1)


try:
    from sglang_omni.models.qwen3_omni.config import (  # noqa: F401
        Qwen3OmniSpeechPipelineConfig,
    )

    _qwen3_available = True
except ImportError:
    _qwen3_available = False


@unittest.skipUnless(_qwen3_available, "qwen3_omni config not importable (missing av?)")
class TestQwen3OmniSpeechGPUValidation(unittest.TestCase):
    def test_default_placement_accepted(self) -> None:
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        config = Qwen3OmniSpeechPipelineConfig(model_path="test/model")
        self.assertEqual(config.gpu_placement["thinker"], 0)
        self.assertEqual(config.gpu_placement["talker_ar"], 1)
        self.assertEqual(config.gpu_placement["code_predictor"], 1)
        self.assertEqual(config.gpu_placement["code2wav"], 1)

    def test_construction_rejects_speech_stage_on_thinker_gpu(self) -> None:
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        for stage_name in ("talker_ar", "code_predictor", "code2wav"):
            with self.subTest(stage_name=stage_name):
                gpu_placement = {
                    "thinker": 0,
                    "talker_ar": 1,
                    "code_predictor": 1,
                    "code2wav": 1,
                    stage_name: 0,
                }
                with self.assertRaisesRegex(ValueError, "collides"):
                    Qwen3OmniSpeechPipelineConfig(
                        model_path="test/model",
                        gpu_placement=gpu_placement,
                    )

    def test_tp2_default_gpus_rejected_on_override_path(self) -> None:
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        config = Qwen3OmniSpeechPipelineConfig(model_path="test/model")
        with self.assertRaisesRegex(ValueError, "collides"):
            config.apply_server_args_overrides(
                stage_name="thinker",
                overrides={"tp_size": 2},
            )

    def test_tp2_non_colliding_gpus_reaches_unsupported_tp_guard(self) -> None:
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        config = Qwen3OmniSpeechPipelineConfig(
            model_path="test/model",
            gpu_placement={
                "thinker": 0,
                "talker_ar": 2,
                "code_predictor": 2,
                "code2wav": 2,
            },
        )
        with self.assertRaisesRegex(NotImplementedError, "not supported yet"):
            config.apply_server_args_overrides(
                stage_name="thinker",
                overrides={"tp_size": 2},
            )

    def test_talker_ar_tp2_reaches_unsupported_tp_guard(self) -> None:
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        config = Qwen3OmniSpeechPipelineConfig(
            model_path="test/model",
            gpu_placement={
                "thinker": 0,
                "talker_ar": 2,
                "code_predictor": 2,
                "code2wav": 2,
            },
        )
        with self.assertRaisesRegex(NotImplementedError, "not supported yet"):
            config.apply_server_args_overrides(
                stage_name="talker_ar",
                overrides={"tp_size": 2},
            )

    def test_talker_ar_tp_override_runs_placement_check_before_tp_guard(
        self,
    ) -> None:
        """Scaling talker_ar must re-run placement validation, not fall through
        to the unsupported-TP guard. Pins the forward-compat contract: once
        the NotImplementedError is removed, the placement check still gates.
        """
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        config = Qwen3OmniSpeechPipelineConfig(
            model_path="test/model",
            gpu_placement={
                "thinker": 0,
                "talker_ar": 2,
                "code_predictor": 2,
                "code2wav": 2,
            },
        )
        with self.assertRaisesRegex(ValueError, "collides"):
            config.apply_server_args_overrides(
                stage_name="talker_ar",
                overrides={"tp_size": 3},
            )


if __name__ == "__main__":
    unittest.main()
