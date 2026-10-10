# SPDX-License-Identifier: Apache-2.0
"""Controller regressions using an injected Cache-DiT backend (no GPU needed)."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from pydantic import JsonValue

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
    resolve_cache_dit_defaults,
    resolve_cache_dit_params,
)


class TestCacheDitParameters(unittest.TestCase):
    def test_disabled_defaults_ignore_invalid_parameters(self) -> None:
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

    def test_inherited_enabled_default_validates_parameters(self) -> None:
        for enabled in (None, True):
            with self.subTest(enabled=enabled), self.assertRaises(ValueError):
                resolve_cache_dit_params(
                    {
                        "enable_cache_dit": enabled,
                        "cache_dit_params": {"enable_taylorseer": True},
                    },
                    True,
                )

    def test_deferred_parameters_survive_sampling_and_are_hashable(self) -> None:
        raw = {"unknown": {"nested": [1, 2]}}
        options = SenseNovaU1Sampling.from_params({"cache_dit_params": raw})
        self.assertEqual(options.cache_dit_params, raw)
        hash(cache_dit_batch_key(options.cache_dit_params))
        self.assertEqual(
            cache_dit_batch_key({"a": 1, "b": 2}),
            cache_dit_batch_key({"b": 2, "a": 1}),
        )

    def test_zero_front_blocks_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            resolve_cache_dit_params(
                {"cache_dit_params": {"Fn_compute_blocks": 0}}, True
            )

    def test_multi_output_preserves_cache_settings(self) -> None:
        params = {"Fn_compute_blocks": 2}
        options = SenseNovaU1Sampling.from_params(
            {"n": 3, "enable_cache_dit": True, "cache_dit_params": params}
        )
        self.assertEqual(options.n, 3)
        self.assertTrue(options.enable_cache_dit)
        self.assertEqual(options.cache_dit_params, params)

    def test_three_branch_edit_ignores_parameters(self) -> None:
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

    def test_guidance_branch_matrix(self) -> None:
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
    def setUp(self) -> None:
        self.native: list[SimpleNamespace] = [
            SimpleNamespace(attention_type="full_attention") for _ in range(32)
        ]
        self.transformer: SimpleNamespace = SimpleNamespace(
            layers=self.native,
            config=SimpleNamespace(num_hidden_layers=len(self.native)),
        )
        self.controller: SenseNovaCacheDit = SenseNovaCacheDit(
            enabled_by_default=True, default_params={"residual_diff_threshold": 0.1}
        )
        self.backend: SimpleNamespace = SimpleNamespace(
            BlockAdapter=Mock(side_effect=lambda **kw: SimpleNamespace(**kw)),
            DBCacheConfig=Mock(
                side_effect=lambda **kw: SimpleNamespace(
                    **{"Fn_compute_blocks": 8, "Bn_compute_blocks": 0, **kw}
                )
            ),
            ForwardPattern=SimpleNamespace(Pattern_3=3),
            enable_cache=Mock(side_effect=self.enable),
            disable_cache=Mock(side_effect=self.disable),
            refresh_context=Mock(),
        )
        module_patch = patch(
            "sglang_omni.models.sensenova_u1.cache_dit.cache_dit", self.backend
        )
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def enable(
        self, adapter: SimpleNamespace, *, cache_config: SimpleNamespace
    ) -> None:
        # note (Codex): Match the backend callback; the fake only swaps layers.
        adapter.transformer.layers = [SimpleNamespace(cached=True)]

    def disable(self, adapter: SimpleNamespace) -> None:
        adapter.transformer.layers = adapter.blocks

    def prepare(
        self,
        *,
        enabled: bool | None = None,
        params: JsonValue = None,
        steps: int = 50,
        branch_count: int = 1,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.controller.prepare(
            self.transformer,
            enabled=enabled,
            params=params,
            steps=steps,
            branch_count=branch_count,
            cfg_interval=cfg_interval,
        )

    def test_refresh_updates_steps_and_retains_effective_parameters(self) -> None:
        self.prepare(params={"Fn_compute_blocks": 4})
        adapter = self.controller.adapter
        self.prepare(params={"Fn_compute_blocks": 4}, steps=30)
        self.assertIs(self.controller.adapter, adapter)
        self.backend.enable_cache.assert_called_once()
        self.backend.refresh_context.assert_called_once_with(
            self.transformer,
            cache_config=SimpleNamespace(
                num_inference_steps=30,
                Fn_compute_blocks=4,
                Bn_compute_blocks=0,
                residual_diff_threshold=0.1,
            ),
        )

    def test_block_counts_use_backend_defaults(self) -> None:
        for params in (
            {"Bn_compute_blocks": 25},
            {"Fn_compute_blocks": 31, "Bn_compute_blocks": 2},
            {"Fn_compute_blocks": 33},
        ):
            with (
                self.subTest(params=params),
                self.assertRaisesRegex(ValueError, "decoder layer count"),
            ):
                self.prepare(params=params)
        self.backend.enable_cache.assert_not_called()
        for params in (
            {"Bn_compute_blocks": 24},
            {"Fn_compute_blocks": 1, "Bn_compute_blocks": 31},
        ):
            with self.subTest(params=params):
                self.prepare(params=params)
        self.assertEqual(self.backend.enable_cache.call_count, 2)

    def test_disabled_server_defaults_survive_request_enable(self) -> None:
        enabled, params = resolve_cache_dit_defaults(
            False, {"residual_diff_threshold": 0.01}
        )
        self.controller = SenseNovaCacheDit(
            enabled_by_default=enabled, default_params=params
        )
        self.prepare()
        self.backend.enable_cache.assert_not_called()
        self.prepare(enabled=True)
        config = self.backend.enable_cache.call_args.kwargs["cache_config"]
        self.assertEqual(config.residual_diff_threshold, 0.01)

    def test_cfg_and_parameter_changes_remount(self) -> None:
        self.prepare()
        self.prepare(branch_count=2)
        self.assertTrue(self.controller.adapter.has_separate_cfg)
        self.prepare(branch_count=2, params={"residual_diff_threshold": 0.2})
        self.prepare(branch_count=1)
        self.assertFalse(self.controller.adapter.has_separate_cfg)
        self.assertEqual(self.backend.enable_cache.call_count, 4)
        self.assertEqual(self.backend.disable_cache.call_count, 3)
        self.assertEqual(self.controller.adapter.blocks_name, "layers")

    def test_disabled_schedules_unmount_before_parameter_validation(self) -> None:
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
                    "sensenova_cache_dit_native_layers", self.transformer.__dict__
                )

    def test_inherited_disabled_request_skips_invalid_parameters(self) -> None:
        self.controller.enabled_by_default = False
        self.prepare(params=["invalid"])
        self.backend.enable_cache.assert_not_called()
        self.controller.enabled_by_default = True
        with self.assertRaises(ValueError):
            self.prepare(params={"enable_taylorseer": True})

    def test_failed_mount_retains_recovery_and_original_exception(self) -> None:
        def fail_enable(
            adapter: SimpleNamespace, *, cache_config: SimpleNamespace
        ) -> None:
            self.enable(adapter, cache_config=cache_config)
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

    def test_unmount_failure_can_be_retried(self) -> None:
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

    def test_config_failure_does_not_leave_native_alias(self) -> None:
        self.backend.DBCacheConfig.side_effect = ValueError("invalid config")
        with self.assertRaisesRegex(ValueError, "invalid config"):
            self.prepare()
        self.assertIs(self.transformer.layers, self.native)
        self.assertIsNone(self.controller.adapter)
        self.assertNotIn("sensenova_cache_dit_native_layers", self.transformer.__dict__)

    def test_only_pure_denoising_uses_cached_layers(self) -> None:
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
    importlib.util.find_spec("cache_dit"),
    "requires torch and cache-dit for real residual isolation",
)
class TestRealCacheDitIsolation(unittest.TestCase):
    def test_implicit_front_blocks_cannot_overlap_back_blocks(self) -> None:
        import cache_dit

        defaults = cache_dit.DBCacheConfig()
        num_layers = defaults.Fn_compute_blocks + 2
        transformer = torch.nn.Module()
        transformer.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(num_layers)]
        )
        transformer.config = SimpleNamespace(num_hidden_layers=num_layers)
        controller = SenseNovaCacheDit(enabled_by_default=True, default_params=None)
        with self.assertRaisesRegex(ValueError, "decoder layer count"):
            controller.prepare(
                transformer,
                enabled=True,
                params={"Bn_compute_blocks": 3},
                steps=6,
                branch_count=1,
                cfg_interval=(0.0, 1.0),
            )

    def test_two_guidance_branches_keep_distinct_residuals(self) -> None:
        class Block(torch.nn.Module):
            attention_type = "full_attention"

            def __init__(self) -> None:
                super().__init__()
                self.calls: int = 0
                self.scale: torch.nn.Parameter = torch.nn.Parameter(
                    torch.tensor(2.0), requires_grad=False
                )

            def forward(
                self, hidden_states: torch.Tensor, **kwargs: torch.Tensor
            ) -> torch.Tensor:
                self.calls += 1
                return hidden_states * self.scale

        class Transformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers: torch.nn.ModuleList = torch.nn.ModuleList(
                    [Block() for _ in range(4)]
                )
                self.config: SimpleNamespace = SimpleNamespace(num_hidden_layers=4)

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        self.assertLess(native[1].calls, 12)
