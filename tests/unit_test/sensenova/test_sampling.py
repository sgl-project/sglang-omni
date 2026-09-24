# SPDX-License-Identifier: Apache-2.0
import unittest

from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1Sampling


class TestSenseNovaSampling(unittest.TestCase):
    def test_defaults_match_source_pipeline(self):
        options = SenseNovaU1Sampling.from_params({})
        self.assertEqual((options.width, options.height), (2048, 2048))
        self.assertEqual(options.num_inference_steps, 50)
        self.assertEqual(options.guidance_scale, 4.0)
        self.assertEqual(options.seed, 42)

    def test_supported_overrides(self):
        options = SenseNovaU1Sampling.from_params(
            {"width": 1024, "height": 768, "num_inference_steps": 4, "seed": 0}
        )
        self.assertEqual((options.width, options.height, options.seed), (1024, 768, 0))

    def test_rejects_invalid_or_unsupported_options(self):
        for params in (
            {"width": 513},
            {"height": 0},
            {"num_inference_steps": -1},
            {"seed": True},
            {"guidance_scale": float("nan")},
            {"think_mode": True},
            {"n": 2},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                SenseNovaU1Sampling.from_params(params)


if __name__ == "__main__":
    unittest.main()
