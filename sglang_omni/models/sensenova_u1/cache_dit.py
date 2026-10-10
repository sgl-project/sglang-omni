# SPDX-License-Identifier: Apache-2.0
"""SenseNova request-scoped Cache-DiT integration."""

from __future__ import annotations

import logging
from typing import Protocol

import torch
from pydantic import JsonValue

try:
    import cache_dit
except ModuleNotFoundError as exc:
    if exc.name != "cache_dit":
        raise
    else:
        cache_dit = None

from sglang_omni.models.sensenova_u1.sampling import resolve_cache_dit_params

logger = logging.getLogger(__name__)


class CacheDitAdapter(Protocol):
    transformer: torch.nn.Module
    blocks: torch.nn.ModuleList


class SenseNovaCacheDit:
    def __init__(
        self,
        *,
        enabled_by_default: bool,
        default_params: dict[str, int | float] | None,
    ) -> None:
        self.enabled_by_default: bool = enabled_by_default
        self.default_params: dict[str, int | float] = (
            {} if default_params is None else dict(default_params)
        )
        self.transformer: torch.nn.Module | None = None
        self.adapter: CacheDitAdapter | None = None
        self.active_key: tuple[tuple[str, int | float], ...] | None = None
        self.cleanup_required: bool = False

    def prepare(
        self,
        transformer: torch.nn.Module,
        *,
        enabled: bool | None,
        params: JsonValue,
        steps: int,
        branch_count: int,
        cfg_interval: tuple[float, float],
    ) -> None:
        if self.cleanup_required:
            self.unmount()
        else:
            pass
        requested = self.enabled_by_default if enabled is None else enabled
        if requested and branch_count > 2:
            logger.warning("Cache-DiT is disabled for three-branch image guidance")
            requested = False
        elif requested and branch_count > 1 and cfg_interval != (0.0, 1.0):
            logger.warning("Cache-DiT requires full-interval image guidance")
            requested = False
        else:
            pass

        if not requested:
            self.unmount()
            return
        else:
            pass
        _, request_params = resolve_cache_dit_params(
            {"enable_cache_dit": True, "cache_dit_params": params}, True
        )
        _, effective_params = resolve_cache_dit_params(
            {"cache_dit_params": self.default_params | (request_params or {})}, True
        )
        effective_params = effective_params or {}
        if cache_dit is None:
            raise RuntimeError("Cache-DiT is required when enabled")
        else:
            pass

        key = (
            *sorted(effective_params.items()),
            ("separate_cfg", int(branch_count > 1)),
        )
        if self.active_key is not None and key != self.active_key:
            self.unmount()
        else:
            pass
        if self.active_key is None:
            if self.adapter is not None:
                self.unmount()
            else:
                pass
            self.mount(transformer, steps, effective_params, branch_count > 1)
            self.active_key = key
        else:
            cache_dit.refresh_context(
                transformer,
                cache_config=cache_dit.DBCacheConfig(
                    num_inference_steps=steps, **effective_params
                ),
            )

    def mount(
        self,
        transformer: torch.nn.Module,
        steps: int,
        params: dict[str, int | float],
        has_separate_cfg: bool,
    ) -> None:
        layers = transformer.layers
        num_layers = transformer.config.num_hidden_layers
        first_blocks = params.get("Fn_compute_blocks", 1)
        back_blocks = params.get("Bn_compute_blocks", 0)
        if first_blocks + back_blocks > num_layers:
            raise ValueError(
                "SenseNova Cache-DiT Fn_compute_blocks and Bn_compute_blocks "
                "must not exceed the decoder layer count"
            )
        else:
            pass
        attention_types = {
            layer.attention_type for layer in layers[:num_layers]
        }
        if len(attention_types) != 1 or None in attention_types:
            raise ValueError(
                "SenseNova Cache-DiT requires a uniform decoder attention type"
            )
        else:
            pass

        adapter = cache_dit.BlockAdapter(
            transformer=transformer,
            blocks=layers,
            blocks_name="layers",
            forward_pattern=cache_dit.ForwardPattern.Pattern_3,
            has_separate_cfg=has_separate_cfg,
        )
        config = cache_dit.DBCacheConfig(num_inference_steps=steps, **params)
        object.__setattr__(transformer, "_sensenova_cache_dit_native_layers", layers)
        transformer._sensenova_cache_dit_attention_type: str = next(
            iter(attention_types)
        )
        self.transformer = transformer
        self.adapter = adapter
        self.cleanup_required = True
        try:
            cache_dit.enable_cache(adapter, cache_config=config)
        except Exception:
            try:
                self.unmount()
            except Exception:
                logger.exception(
                    "Failed to roll back a partial SenseNova Cache-DiT mount"
                )
            raise
        self.cleanup_required = False

    def unmount(self) -> None:
        if self.adapter is None:
            self.cleanup_required = False
            return
        else:
            pass
        try:
            cache_dit.disable_cache(self.adapter)
        except Exception:
            try:
                self.restore_native_layers()
            except Exception:
                logger.exception("Failed to restore native SenseNova decoder layers")
            self.cleanup_required = True
            raise

        try:
            self.restore_native_layers()
        except Exception:
            self.cleanup_required = True
            raise
        transformer = self.transformer
        if transformer is not None:
            self.clear_native_layers(transformer)
        else:
            pass
        self.transformer = None
        self.adapter = None
        self.active_key = None
        self.cleanup_required = False

    def restore_native_layers(self) -> None:
        if self.transformer is None:
            raise RuntimeError("SenseNova Cache-DiT has no transformer to restore")
        else:
            pass
        native_layers = self.transformer.__dict__.get(
            "_sensenova_cache_dit_native_layers"
        )
        if native_layers is None:
            raise RuntimeError("SenseNova Cache-DiT lost its native decoder layers")
        else:
            pass
        self.transformer.layers = native_layers
        if self.transformer.layers is not native_layers:
            raise RuntimeError(
                "SenseNova Cache-DiT failed to restore native decoder layers"
            )
        else:
            pass

    @staticmethod
    def clear_native_layers(transformer: torch.nn.Module) -> None:
        transformer.__dict__.pop("_sensenova_cache_dit_native_layers", None)
        transformer.__dict__.pop("_sensenova_cache_dit_attention_type", None)


def decoder_layers(
    model: torch.nn.Module,
    *,
    update_cache: bool,
    has_non_image_tokens: bool,
    has_image_tokens: bool,
) -> torch.nn.ModuleList:
    native_layers = model.__dict__.get("_sensenova_cache_dit_native_layers")
    if native_layers is not None and (
        update_cache or has_non_image_tokens or not has_image_tokens
    ):
        return native_layers
    else:
        return model.layers


def decoder_attention_type(model: torch.nn.Module, layer: torch.nn.Module) -> str:
    native_layers = model.__dict__.get("_sensenova_cache_dit_native_layers")
    if native_layers is None or any(
        layer is native_layer for native_layer in native_layers
    ):
        return layer.attention_type
    else:
        return model._sensenova_cache_dit_attention_type
