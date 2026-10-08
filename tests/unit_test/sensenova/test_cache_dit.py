# SPDX-License-Identifier: Apache-2.0
"""Controller regressions using an injected Cache-DiT backend (no GPU needed)."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang_omni.models.sensenova_u1.cache_dit import (
    SenseNovaCacheDit,
    decoder_attention_type,
    decoder_layers,
)
from sglang_omni.models.sensenova_u1.sampling import (
    SenseNovaU1ImageEditSampling,
    SenseNovaU1Sampling,
    cache_dit_batch_key,
    image_guidance_branch_count,
    resolve_cache_dit_params,
)


class TestCacheDitParameters(unittest.TestCase):
    def test_disabled_defaults_ignore_invalid_parameters(self):
        for raw in ({"enable_taylorseer": True}, {"Fn_compute_blocks": -1}, []):
            for enabled in (None, False):
                with self.subTest(raw=raw, enabled=enabled):
                    self.assertEqual(
                        resolve_cache_dit_params(
                            {"enable_cache_dit": enabled, "cache_dit_params": raw},
                            False,
                        ),
                        (False, None),
                    )

    def test_inherited_enabled_default_validates_parameters(self):
        for enabled in (None, True):
            with self.subTest(enabled=enabled), self.assertRaises(ValueError):
                resolve_cache_dit_params(
                    {
                        "enable_cache_dit": enabled,
                        "cache_dit_params": {"enable_taylorseer": True},
                    },
                    True,
                )

    def test_deferred_parameters_survive_sampling_and_are_hashable(self):
        raw = {"unknown": {"nested": [1, 2]}}
        options = SenseNovaU1Sampling.from_params({"cache_dit_params": raw})
        self.assertEqual(options.cache_dit_params, raw)
        hash(cache_dit_batch_key(options.cache_dit_params))
        self.assertEqual(
            cache_dit_batch_key({"a": 1, "b": 2}),
            cache_dit_batch_key({"b": 2, "a": 1}),
        )

    def test_three_branch_edit_ignores_parameters(self):
        for enabled in (None, False, True):
            with self.subTest(enabled=enabled):
                options = SenseNovaU1ImageEditSampling.from_params(
                    {
                        "guidance_scale": 4.0,
                        "img_cfg_scale": 2.0,
                        "enable_cache_dit": enabled,
                        "cache_dit_params": {"enable_taylorseer": True},
                    }
                )
                self.assertIsNone(options.cache_dit_params)

    def test_guidance_branch_matrix(self):
        for cfg, img_cfg, count in (
            (1, 1, 1),
            (0.5, 1, 2),
            (4, 1, 2),
            (0.5, 0.5, 2),
            (4, 4, 2),
            (1, 2, 3),
            (4, 2, 3),
        ):
            with self.subTest(cfg=cfg, img_cfg=img_cfg):
                self.assertEqual(image_guidance_branch_count(cfg, img_cfg), count)


class TestCacheDitLifecycle(unittest.TestCase):
    def setUp(self):
        self.native = [SimpleNamespace(attention_type="full_attention")]
        self.transformer = SimpleNamespace(
            layers=self.native, config=SimpleNamespace(num_hidden_layers=1)
        )
        self.controller = SenseNovaCacheDit(
            enabled_by_default=True, default_params={"residual_diff_threshold": 0.1}
        )
        self.backend = SimpleNamespace(
            BlockAdapter=Mock(side_effect=lambda **kw: SimpleNamespace(**kw)),
            DBCacheConfig=Mock(side_effect=lambda **kw: kw),
            ForwardPattern=SimpleNamespace(Pattern_3=3),
            enable_cache=Mock(side_effect=self.enable),
            disable_cache=Mock(side_effect=self.disable),
            refresh_context=Mock(),
        )
        self.module_patch = patch.dict("sys.modules", {"cache_dit": self.backend})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def enable(self, adapter, **kwargs):
        adapter.transformer.layers = [SimpleNamespace(cached=True)]

    def disable(self, adapter):
        adapter.transformer.layers = adapter.blocks

    def prepare(self, **kwargs):
        options = {
            "enabled": None,
            "params": None,
            "steps": 50,
            "branch_count": 1,
            "cfg_interval": (0.0, 1.0),
        }
        options.update(kwargs)
        self.controller.prepare(self.transformer, **options)

    def test_refresh_updates_steps_and_retains_effective_parameters(self):
        self.prepare(params={"Fn_compute_blocks": 4})
        adapter = self.controller.adapter
        self.prepare(params={"Fn_compute_blocks": 4}, steps=30)
        self.assertIs(self.controller.adapter, adapter)
        self.backend.enable_cache.assert_called_once()
        self.backend.refresh_context.assert_called_once_with(
            self.transformer,
            cache_config={
                "num_inference_steps": 30,
                "Fn_compute_blocks": 4,
                "residual_diff_threshold": 0.1,
            },
        )

    def test_cfg_and_parameter_changes_remount(self):
        self.prepare()
        self.prepare(branch_count=2)
        self.assertTrue(self.controller.adapter.has_separate_cfg)
        self.prepare(branch_count=2, params={"residual_diff_threshold": 0.2})
        self.prepare(branch_count=1)
        self.assertFalse(self.controller.adapter.has_separate_cfg)
        self.assertEqual(self.backend.enable_cache.call_count, 4)
        self.assertEqual(self.backend.disable_cache.call_count, 3)
        self.assertEqual(self.controller.adapter.blocks_name, "layers")

    def test_disabled_schedules_unmount_before_parameter_validation(self):
        for overrides in (
            {"enabled": False},
            {"branch_count": 3},
            {"branch_count": 2, "cfg_interval": (0.2, 0.8)},
        ):
            with self.subTest(overrides=overrides):
                self.prepare()
                self.prepare(params={"enable_taylorseer": True}, **overrides)
                self.assertIs(self.transformer.layers, self.native)
                self.assertIsNone(self.controller.adapter)
                self.assertNotIn(
                    "_sensenova_cache_dit_native_layers", self.transformer.__dict__
                )

    def test_inherited_disabled_request_skips_invalid_parameters(self):
        self.controller.enabled_by_default = False
        self.prepare(params=["invalid"])
        self.backend.enable_cache.assert_not_called()
        self.controller.enabled_by_default = True
        with self.assertRaises(ValueError):
            self.prepare(params={"enable_taylorseer": True})

    def test_failed_mount_retains_recovery_and_original_exception(self):
        def fail_enable(adapter, **kwargs):
            self.enable(adapter, **kwargs)
            raise RuntimeError("original mount failure")

        self.backend.enable_cache.side_effect = fail_enable
        self.backend.disable_cache.side_effect = RuntimeError("cleanup failure")
        with (
            self.assertLogs(level="ERROR"),
            self.assertRaisesRegex(RuntimeError, "original mount failure"),
        ):
            self.prepare()
        self.assertIs(self.transformer.layers, self.native)
        self.assertTrue(self.controller.cleanup_required)
        self.assertIsNotNone(self.controller.adapter)
        self.backend.disable_cache.side_effect = self.disable
        self.backend.enable_cache.side_effect = self.enable
        self.prepare()
        self.assertFalse(self.controller.cleanup_required)
        self.assertIsNot(self.transformer.layers, self.native)

    def test_unmount_failure_can_be_retried(self):
        self.prepare()
        self.backend.disable_cache.side_effect = RuntimeError("cleanup failure")
        with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
            self.controller.unmount()
        self.assertIs(self.transformer.layers, self.native)
        self.assertTrue(self.controller.cleanup_required)
        self.backend.disable_cache.side_effect = self.disable
        self.prepare(enabled=False)
        self.assertFalse(self.controller.cleanup_required)
        self.assertIsNone(self.controller.adapter)

    def test_config_failure_does_not_leave_native_alias(self):
        self.backend.DBCacheConfig.side_effect = ValueError("invalid config")
        with self.assertRaisesRegex(ValueError, "invalid config"):
            self.prepare()
        self.assertIs(self.transformer.layers, self.native)
        self.assertIsNone(self.controller.adapter)
        self.assertNotIn(
            "_sensenova_cache_dit_native_layers", self.transformer.__dict__
        )

    def test_only_pure_denoising_uses_cached_layers(self):
        self.prepare(branch_count=2)
        for update, non_image, image in (
            (True, False, True),
            (False, True, True),
            (False, False, False),
        ):
            self.assertIs(
                decoder_layers(
                    self.transformer,
                    update_cache=update,
                    has_non_image_tokens=non_image,
                    has_image_tokens=image,
                ),
                self.native,
            )
        self.assertIs(
            decoder_layers(
                self.transformer,
                update_cache=False,
                has_non_image_tokens=False,
                has_image_tokens=True,
            ),
            self.transformer.layers,
        )
        self.assertEqual(
            decoder_attention_type(self.transformer, self.transformer.layers[0]),
            "full_attention",
        )


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("cache_dit"),
    "requires torch and cache-dit for real residual isolation",
)
class TestRealCacheDitIsolation(unittest.TestCase):
    def test_two_guidance_branches_keep_distinct_residuals(self):
        import torch

        class Block(torch.nn.Module):
            attention_type = "full_attention"

            def __init__(self):
                super().__init__()
                self.calls = 0
                self.scale = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)

            def forward(self, hidden_states, **kwargs):
                self.calls += 1
                return hidden_states * self.scale

        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Block() for _ in range(4)])
                self.config = SimpleNamespace(num_hidden_layers=4)

            def forward(self, hidden_states):
                for layer in self.layers:
                    hidden_states = layer(hidden_states)
                return hidden_states

        transformer = Transformer().eval()
        native = transformer.layers
        controller = SenseNovaCacheDit(enabled_by_default=True, default_params=None)
        controller.prepare(
            transformer,
            enabled=True,
            steps=6,
            branch_count=2,
            cfg_interval=(0.0, 1.0),
            params={
                "Fn_compute_blocks": 1,
                "Bn_compute_blocks": 1,
                "max_warmup_steps": 1,
                "residual_diff_threshold": 1.0,
                "max_continuous_cached_steps": 3,
            },
        )
        self.addCleanup(controller.unmount)
        with torch.inference_mode():
            for _ in range(6):
                for value in (1.0, 10.0):
                    inputs = torch.full((1, 2, 4), value)
                    torch.testing.assert_close(transformer(inputs), inputs * 16)
        # Without this check an integration that never caches could pass.
        self.assertLess(native[1].calls, 12)


if __name__ == "__main__":
    unittest.main()
