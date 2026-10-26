# SPDX-License-Identifier: Apache-2.0
import unittest

from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
)


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
            {"n": 0},
            {"n": 11},
            {"n": True},
            {"n": 2.0},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                SenseNovaU1Sampling.from_params(params)

    def test_image_edit_defaults_and_overrides(self):
        defaults = SenseNovaU1ImageEditSampling.from_params({})
        self.assertEqual((defaults.width, defaults.height), (2048, 2048))
        self.assertEqual(defaults.num_inference_steps, 50)
        self.assertEqual(defaults.guidance_scale, 4.0)
        self.assertEqual(defaults.img_cfg_scale, 1.0)
        self.assertEqual(defaults.seed, 42)
        self.assertIsNone(defaults.input_max_pixels)
        self.assertTrue(defaults.do_resize)
        self.assertFalse(defaults.size_explicit)

        options = SenseNovaU1ImageEditSampling.from_params(
            {"width": 512, "height": 768, "img_cfg_scale": 2.0}
        )
        self.assertEqual((options.width, options.height), (512, 768))
        self.assertEqual(options.img_cfg_scale, 2.0)

    def test_multiple_t2i_outputs(self):
        self.assertEqual(SenseNovaU1Sampling.from_params({"n": 4}).n, 4)

    def test_seed_range_reserves_every_output(self) -> None:
        """A seed accepted by sampling overflowed the RNG after TP dispatch."""
        for parameters in (
            {"seed": 2**64 - 1},
            {"seed": 2**64 - 10, "n": 10},
        ):
            with self.subTest(parameters=parameters):
                self.assertEqual(
                    SenseNovaU1Sampling.from_params(parameters).seed, parameters["seed"]
                )
        for parameters in (
            {"seed": 2**64},
            {"seed": 2**64 - 1, "n": 2},
            {"seed": 2**64 - 9, "n": 10},
            {"seed": 2**64 - 1, "stage_params": {"generate": {"n": 2}}},
            {"stage_params": {"generate": {"seed": 2**64}}},
        ):
            with (
                self.subTest(parameters=parameters),
                self.assertRaisesRegex(ValueError, "seed"),
            ):
                SenseNovaU1Sampling.from_params(parameters)
        for parameters in (
            {"seed": 2**64},
            {"stage_params": {"generate": {"seed": 2**64}}},
        ):
            with (
                self.subTest(parameters=parameters),
                self.assertRaisesRegex(ValueError, "seed"),
            ):
                SenseNovaU1ImageEditSampling.from_params(parameters)
        self.assertEqual(
            SenseNovaU1ImageEditSampling.from_params({"seed": 2**64 - 1}).seed,
            2**64 - 1,
        )

    def test_image_edit_rejects_invalid_options(self):
        for params in (
            {"img_cfg_scale": float("inf")},
            {"img_cfg_scale": -1},
            {"input_max_pixels": 512 * 512 - 1},
            {"do_resize": "false"},
            {"size_explicit": 1},
            {"n": 2},
            {"think_mode": True},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                SenseNovaU1ImageEditSampling.from_params(params)

    def test_stage_overrides_apply_to_both_sampling_paths(self) -> None:
        """Stage options were honored by execution but ignored by other consumers."""
        parameters = {
            "num_inference_steps": 50,
            "stage_params": {
                "generate": {"num_inference_steps": 4, "width": 512},
                "output": {"width": 1024},
            },
        }
        for sampling_type in (SenseNovaU1Sampling, SenseNovaU1ImageEditSampling):
            with self.subTest(sampling_type=sampling_type):
                options = sampling_type.from_params(parameters)
                self.assertEqual(options.num_inference_steps, 4)
                self.assertEqual(options.width, 512)
        self.assertEqual(parameters["num_inference_steps"], 50)

    def test_invalid_stage_overrides_are_rejected(self) -> None:
        """Malformed generate overrides must not silently execute default options."""
        for parameters in (
            {"stage_params": "invalid"},
            {"stage_params": {"generate": "invalid"}},
            {"stage_params": {"generate": []}},
            {"stage_params": {"generate": False}},
            {"stage_params": {"generate": {"num_inference_steps": -1}}},
        ):
            for sampling_type in (SenseNovaU1Sampling, SenseNovaU1ImageEditSampling):
                with self.subTest(parameters=parameters, sampling_type=sampling_type):
                    with self.assertRaises(ValueError):
                        sampling_type.from_params(parameters)

    def test_null_generate_overrides_preserve_top_level_options(self) -> None:
        """A null stage entry must behave like an omitted entry, not fail validation."""
        for sampling_type in (SenseNovaU1Sampling, SenseNovaU1ImageEditSampling):
            with self.subTest(sampling_type=sampling_type):
                options = sampling_type.from_params(
                    {"width": 512, "stage_params": {"generate": None}}
                )
                self.assertEqual(options.width, 512)


if __name__ == "__main__":
    unittest.main()
