# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline config GPU validation (Ming + Qwen3)."""
from __future__ import annotations

import unittest


class TestMingOmniSpeechGPUValidation(unittest.TestCase):
    def test_tp2_default_gpus_rejected(self):
        from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

        with self.assertRaises(ValueError) as ctx:
            MingOmniSpeechPipelineConfig(
                model_path="test/model",
                gpu_placement={"thinker": 0, "talker": 1},
                server_args_overrides={"tp_size": 2},
            )
        self.assertIn("collides", str(ctx.exception).lower())

    def test_tp2_talker_gpu2_accepted(self):
        from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

        config = MingOmniSpeechPipelineConfig(
            model_path="test/model",
            gpu_placement={"thinker": 0, "talker": 2},
            server_args_overrides={"tp_size": 2},
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
    def test_tp2_default_gpus_rejected(self):
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        with self.assertRaises(ValueError) as ctx:
            Qwen3OmniSpeechPipelineConfig(
                model_path="test/model",
                server_args_overrides={"tp_size": 2},
            )
        self.assertIn("collides", str(ctx.exception).lower())

    def test_tp2_speech_on_gpu2_accepted(self):
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        config = Qwen3OmniSpeechPipelineConfig(
            model_path="test/model",
            gpu_placement={
                "thinker": 0,
                "talker_ar": 2,
                "code_predictor": 2,
                "code2wav": 2,
            },
            server_args_overrides={"tp_size": 2},
        )
        self.assertEqual(config.gpu_placement["talker_ar"], 2)

    def test_tp1_default_accepted(self):
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        config = Qwen3OmniSpeechPipelineConfig(
            model_path="test/model",
        )
        self.assertEqual(config.gpu_placement["thinker"], 0)
        self.assertEqual(config.gpu_placement["talker_ar"], 1)


if __name__ == "__main__":
    unittest.main()
