# SPDX-License-Identifier: Apache-2.0
"""Tests for Ming speech pipeline GPU validation."""
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


if __name__ == "__main__":
    unittest.main()
