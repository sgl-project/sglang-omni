# SPDX-License-Identifier: Apache-2.0
"""Mount Cache-DiT on the decoder layers that run SenseNova's denoising steps."""
from __future__ import annotations

import functools
import logging
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import torch
from cache_dit import ForwardPattern
from pydantic import JsonValue
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    _CUSTOM_BLOCK_ADAPTER_SPECS,
    CACHE_DIT_REQUEST_PARAM_KEYS,
    CacheDitConfig,
    CustomBlockAdapterSpec,
    cache_dit_overrides_key,
    disable_cache_on_transformer,
    enable_cache_on_transformer,
    refresh_context_on_transformer,
    resolve_cache_dit_request_overrides,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from torch import nn
from transformers import AutoConfig
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from sglang_omni.models.sensenova_u1.native.options import DEFAULT_CFG_INTERVAL
from sglang_omni.vendor.sensenova_u1.image_edit import it2i_guidance_branch_count
from sglang_omni.vendor.sensenova_u1.modeling_qwen3 import Qwen3Model

logger = logging.getLogger(__name__)

# SenseNova runs plain DBCache only. cache-dit 1.3.0 has no DMD, secondary settings only apply to
# dual-transformer models, and TaylorSeer and step computation masks are not validated on SenseNova.
UNSUPPORTED_REQUEST_KEYS = frozenset(
    {
        "secondary",
        "enable_dmd",
        "dmd_history",
        "dmd_rank",
        "dmd_ridge",
        "dmd_svd_precision",
        "enable_taylorseer",
        "taylorseer_order",
        "scm_preset",
        "scm_policy",
        "scm_compute_bins",
        "scm_cache_bins",
    }
)
SEPARATE_GUIDANCE_BRANCH_COUNT = 2


class SenseNovaGenerationBlocks(nn.Module):
    """Expose the decoder layers of a denoising step to Cache-DiT.

    The name must not start with a family name in cache-dit's adapter registry, which matches class names by prefix.
    """

    def __init__(self, layers: Sequence[nn.Module]) -> None:
        super().__init__()
        attention_types = {layer.attention_type for layer in layers}
        assert (
            len(attention_types) == 1
        ), f"SenseNova Cache-DiT needs one attention type across decoder layers, got {sorted(attention_types)}"
        self.blocks = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values: Cache | None,
        use_cache: bool | None,
        cache_position: torch.LongTensor | None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        return hidden_states


@dataclass(frozen=True, kw_only=True)
class CacheDitMountKey:
    overrides: Hashable
    separate_cfg: bool
    max_warmup_steps: int


def environment_cache_dit_params() -> dict[str, JsonValue]:
    """Return the server's DBCache settings under their request key names."""
    return {
        "Fn_compute_blocks": envs.SGLANG_CACHE_DIT_FN,
        "Bn_compute_blocks": envs.SGLANG_CACHE_DIT_BN,
        "max_warmup_steps": envs.SGLANG_CACHE_DIT_WARMUP,
        "residual_diff_threshold": envs.SGLANG_CACHE_DIT_RDT,
        "max_continuous_cached_steps": envs.SGLANG_CACHE_DIT_MC,
    }


def request_cache_dit_overrides(raw_params: JsonValue) -> dict[str, JsonValue]:
    """Check the keys of a request's cache_dit_params and return its overrides."""
    if isinstance(raw_params, dict):
        unknown = sorted(set(raw_params) - CACHE_DIT_REQUEST_PARAM_KEYS)
        unsupported = sorted(set(raw_params) & UNSUPPORTED_REQUEST_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown cache_dit_params keys: {unknown}. Valid keys: "
                f"{sorted(CACHE_DIT_REQUEST_PARAM_KEYS - UNSUPPORTED_REQUEST_KEYS)}."
            )
        elif unsupported:
            raise ValueError(
                "SenseNova Cache-DiT supports DBCache settings only and does not "
                f"support cache_dit_params keys {unsupported}"
            )
        else:
            pass
    else:
        pass
    return resolve_cache_dit_request_overrides(raw_params)


def require_integer(params: dict[str, JsonValue], name: str, minimum: int) -> None:
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"Cache-DiT {name} must be an integer of at least {minimum}, got {value!r}"
        )
    else:
        pass


def check_cache_dit_params(params: dict[str, JsonValue]) -> None:
    """Check effective DBCache settings, whichever of the server or the request set them."""
    require_integer(params, "Fn_compute_blocks", 1)
    require_integer(params, "Bn_compute_blocks", 0)
    require_integer(params, "max_warmup_steps", 0)
    # note (Wes): cache-dit reads -1 as no limit on consecutive cached steps.
    require_integer(params, "max_continuous_cached_steps", -1)
    threshold = params["residual_diff_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            f"Cache-DiT residual_diff_threshold must be a number between 0 and 1, got {threshold!r}"
        )
    else:
        pass


def check_environment_cache_dit_params() -> None:
    """Fail startup on server Cache-DiT settings that no request could run with."""
    enabled_extensions = [
        name
        for name, enabled in (
            ("SGLANG_CACHE_DIT_DMD", envs.SGLANG_CACHE_DIT_DMD),
            ("SGLANG_CACHE_DIT_TAYLORSEER", envs.SGLANG_CACHE_DIT_TAYLORSEER),
            (
                "SGLANG_CACHE_DIT_SCM_PRESET",
                str(envs.SGLANG_CACHE_DIT_SCM_PRESET).strip().lower()
                not in ("", "none"),
            ),
            (
                "SGLANG_CACHE_DIT_SCM_COMPUTE_BINS",
                bool((envs.SGLANG_CACHE_DIT_SCM_COMPUTE_BINS or "").strip()),
            ),
            (
                "SGLANG_CACHE_DIT_SCM_CACHE_BINS",
                bool((envs.SGLANG_CACHE_DIT_SCM_CACHE_BINS or "").strip()),
            ),
        )
        if enabled
    ]
    if enabled_extensions:
        raise ValueError(
            "SenseNova Cache-DiT supports DBCache settings only; unset "
            f"{', '.join(enabled_extensions)}"
        )
    else:
        check_cache_dit_params(environment_cache_dit_params())


def resolve_cache_dit_config(
    overrides: dict[str, JsonValue], num_inference_steps: int
) -> CacheDitConfig:
    """Apply request overrides over the server settings, one key at a time."""
    params = environment_cache_dit_params() | overrides
    check_cache_dit_params(params)
    return CacheDitConfig(
        enabled=True,
        Fn_compute_blocks=params["Fn_compute_blocks"],
        Bn_compute_blocks=params["Bn_compute_blocks"],
        # note (Wes): cache-dit 1.3.0 rebuilds a list of every warmup step on each step; warmup past the last step changes nothing.
        max_warmup_steps=min(params["max_warmup_steps"], num_inference_steps),
        residual_diff_threshold=params["residual_diff_threshold"],
        max_continuous_cached_steps=params["max_continuous_cached_steps"],
        num_inference_steps=num_inference_steps,
    )


def check_compute_blocks(
    first_blocks: int, last_blocks: int, decoder_layer_count: int
) -> None:
    if first_blocks + last_blocks > decoder_layer_count:
        raise ValueError(
            f"Cache-DiT Fn_compute_blocks={first_blocks} plus Bn_compute_blocks={last_blocks} "
            f"exceeds the {decoder_layer_count} decoder layers"
        )
    else:
        pass


@functools.cache
def checkpoint_decoder_layer_count(model_path: str, revision: str | None) -> int:
    checkpoint_config = AutoConfig.from_pretrained(model_path, revision=revision)
    return int(checkpoint_config.llm_config.num_hidden_layers)


def guidance_branch_count(
    *, is_edit: bool, cfg_scale: float, img_cfg_scale: float
) -> int:
    if is_edit:
        return it2i_guidance_branch_count(cfg_scale, img_cfg_scale)
    elif cfg_scale > 1:
        return SEPARATE_GUIDANCE_BRANCH_COUNT
    else:
        return 1


def unsupported_guidance(
    branch_count: int, cfg_interval: tuple[float, float]
) -> str | None:
    """Describe guidance whose per-step model calls do not alternate the way cache-dit counts them."""
    if branch_count > SEPARATE_GUIDANCE_BRANCH_COUNT:
        return (
            "editing with three guidance branches, where img_cfg_scale is neither 1 "
            "nor equal to guidance_scale"
        )
    elif branch_count == SEPARATE_GUIDANCE_BRANCH_COUNT and tuple(
        cfg_interval
    ) != tuple(DEFAULT_CFG_INTERVAL):
        return f"guidance on only part of the steps, with cfg_interval other than {DEFAULT_CFG_INTERVAL}"
    else:
        return None


def resolve_cache_dit_switch(
    request_switch: bool | None,
    *,
    branch_count: int,
    cfg_interval: tuple[float, float],
) -> bool:
    """Return whether a request runs with Cache-DiT.

    Guidance the cache cannot follow fails a request that set enable_cache_dit=true,
    and runs uncached when the switch comes from SGLANG_CACHE_DIT_ENABLED.
    """
    enabled = (
        envs.SGLANG_CACHE_DIT_ENABLED if request_switch is None else request_switch
    )
    problem = unsupported_guidance(branch_count, cfg_interval) if enabled else None
    if problem is None:
        return enabled
    elif request_switch is None:
        logger.info(f"Running a request without Cache-DiT: it uses {problem}")
        return False
    else:
        raise ValueError(
            f"Cache-DiT does not support {problem}; omit enable_cache_dit or set it to false"
        )


class SenseNovaCacheDit:
    """Keep the Cache-DiT mount matched to each request, as native denoising stages do."""

    def __init__(self) -> None:
        self.view: SenseNovaGenerationBlocks | None = None
        self.active_key: CacheDitMountKey | None = None

    def prepare(
        self,
        decoder: Qwen3Model,
        batch: Req,
        *,
        branch_count: int,
        cfg_interval: tuple[float, float],
    ) -> None:
        sampling_params = batch.sampling_params
        requested = resolve_cache_dit_switch(
            sampling_params.enable_cache_dit,
            branch_count=branch_count,
            cfg_interval=cfg_interval,
        )
        if requested:
            overrides = request_cache_dit_overrides(sampling_params.cache_dit_params)
            config = resolve_cache_dit_config(overrides, int(batch.num_inference_steps))
            desired_key = CacheDitMountKey(
                overrides=cache_dit_overrides_key(overrides),
                separate_cfg=branch_count == SEPARATE_GUIDANCE_BRANCH_COUNT,
                max_warmup_steps=config.max_warmup_steps,
            )
        else:
            config = None
            desired_key = None
        if self.active_key is not None and self.active_key != desired_key:
            self.unmount(decoder)
        else:
            pass
        if config is None or desired_key is None:
            pass
        elif self.active_key is not None:
            # note (Wes): a refresh starts a new cache context, so a cancelled or failed run leaves nothing behind.
            refresh_context_on_transformer(
                self.view,
                config.num_inference_steps,
                steps_computation_mask=config.steps_computation_mask,
                steps_computation_policy=config.steps_computation_policy,
            )
        elif batch.is_warmup:
            pass
        else:
            self.mount(decoder, config, desired_key)

    def mount(
        self, decoder: Qwen3Model, config: CacheDitConfig, key: CacheDitMountKey
    ) -> None:
        check_compute_blocks(
            config.Fn_compute_blocks,
            config.Bn_compute_blocks,
            decoder.config.num_hidden_layers,
        )
        _CUSTOM_BLOCK_ADAPTER_SPECS.setdefault(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            SenseNovaGenerationBlocks.__name__,
            CustomBlockAdapterSpec(
                blocks_attr="blocks", forward_pattern=ForwardPattern.Pattern_3
            ),
        )
        if self.view is None:
            self.view = SenseNovaGenerationBlocks(
                decoder.layers[: decoder.config.num_hidden_layers]
            )
        else:
            pass
        # note (Wes): cache-dit 1.3.0 flags transformer-only mounts on a class shared by the whole process,
        # so a process holds one mount at a time and unmount must run before the next mount.
        enable_cache_on_transformer(
            self.view,
            config,
            model_name="sensenova_u1",
            has_separate_cfg=key.separate_cfg,
        )
        decoder.set_generation_blocks(self.view)
        self.active_key = key
        logger.info(
            f"SenseNova Cache-DiT mounted with Fn={config.Fn_compute_blocks} "
            f"Bn={config.Bn_compute_blocks} warmup={config.max_warmup_steps} "
            f"rdt={config.residual_diff_threshold} mc={config.max_continuous_cached_steps} "
            f"separate_cfg={key.separate_cfg}"
        )

    def unmount(self, decoder: Qwen3Model) -> None:
        assert self.view is not None, "SenseNova Cache-DiT has no mounted view"
        decoder.set_generation_blocks(None)
        disable_cache_on_transformer(self.view)
        self.active_key = None
