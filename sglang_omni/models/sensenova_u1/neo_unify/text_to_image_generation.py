"""Text to image generation."""

import os
from typing import Literal

import torch.utils.checkpoint
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.sensenova_u1.neo_unify.generation_math import (
    SenseNovaStageTimer,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_state import GenerationState
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    KV_DIAGNOSTIC_DIR,
    KV_TRANSFER_DIR,
    USE_SRT_KV_TRANSFER,
)
from sglang_omni.models.sensenova_u1.neo_unify.text_image_denoise import (
    text_image_denoise,
)
from sglang_omni.models.sensenova_u1.neo_unify.text_image_prefix import (
    text_image_prefix,
)
from sglang_omni.models.sensenova_u1.neo_unify.thinking_types import ThinkingBackend
from sglang_omni.models.sensenova_u1.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

# Modified for SGLang; see this directory's README.md for upstream source.


@torch.no_grad()
def t2i_generate(
    self: GenerationState,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[str],
    cfg_scale: float = 1,
    timestep_shift: float = 1,
    enable_timestep_shift: bool = True,
    cfg_norm: Literal["none", "global", "channel", "cfg_zero_star"] = "none",
    image_size: tuple[int, int] = (256, 256),
    num_steps: int = 30,
    IMG_START_TOKEN: str = "<img>",
    IMG_END_TOKEN: str = "</img>",
    IMG_CONTEXT_TOKEN: str = "<IMG_CONTEXT>",
    method: Literal["euler"] = "euler",
    cfg_interval: tuple[float, float] = (0, 1),
    batch_size: int = 1,
    t_eps: float = 0.02,
    think_mode: bool = False,
    seed: int | list[int] = 0,
    max_think_tokens: int = 1024,
    profile_stages: bool = False,
    thinking_backend: ThinkingBackend | None = None,
) -> torch.Tensor | tuple[torch.Tensor, str | list[str]]:
    profiler = SenseNovaStageTimer(profile_stages, self.device)
    self.last_profile_timings_ms = {}
    self.last_thinking_backend = None
    self.last_srt_kv_transfer_used = False
    self.last_srt_kv_transfer_timings = {}
    self.last_srt_kv_transfer_timings_list = []
    self.last_srt_kv_transferred_prefixes = []
    assert self.concat_time_token_num == 0
    assert cfg_norm in ["cfg_zero_star", "global", "none", "channel"]
    prompts = prompt if isinstance(prompt, list) else [prompt]
    if not prompts:
        raise ValueError("prompt batch must not be empty")
    else:
        pass
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    else:
        pass
    if isinstance(prompt, list) and len(prompts) != batch_size:
        raise ValueError(
            f"batch_size={batch_size} does not match {len(prompts)} prompts"
        )
    else:
        pass
    use_direct_srt_prefix = (
        os.environ.get(USE_SRT_KV_TRANSFER) == "1"
        and not os.environ.get(KV_DIAGNOSTIC_DIR)
        and os.environ.get(KV_TRANSFER_DIR)
        and thinking_backend is not None
        and hasattr(thinking_backend, "transfer_prefix_kv")
    )
    compact_mode = getattr(self, "sensenova_compact_mode", False)
    if compact_mode and len(prompts) > 2:
        raise RuntimeError("SenseNova compact mode supports at most two T2I prompts")
    else:
        pass
    if compact_mode and not use_direct_srt_prefix:
        raise RuntimeError(
            "SenseNova compact mode requires the managed SRT prefix KV transfer "
            "backend"
        )
    else:
        pass
    self.notify_layer_offload_phase("prefix")
    merge_size = int(1 / self.downsample_ratio)

    self.config.t_eps = t_eps
    think_text = ""
    self.last_think_token_count = 0
    self.last_think_token_counts = [0] * len(prompts)
    self.last_thinking_backends = [None] * len(prompts)
    needs_cfg = cfg_scale > 1

    think_content = (
        "<think>\n" if think_mode else "<think>\n\n</think>\n\n" + IMG_START_TOKEN
    )
    query_conditions = [
        self.build_t2i_query(
            item,
            system_message=SYSTEM_MESSAGE_FOR_GEN,
            append_text=think_content,
        )
        for item in prompts
    ]
    query_condition = (
        query_conditions[0] if len(query_conditions) == 1 else query_conditions
    )
    query_uncondition = (
        self.build_t2i_query("", append_text=IMG_START_TOKEN) if needs_cfg else None
    )

    (
        input_ids_condition,
        indexes_condition,
        attention_mask_condition_prefix,
        condition_key_valid_mask,
        condition_prefix_lengths,
    ) = self.build_t2i_text_inputs(tokenizer, query_condition)
    if query_uncondition is not None:
        (
            input_ids_uncondition,
            indexes_uncondition,
            attention_mask_uncondition_prefix,
            _,
            uncondition_prefix_lengths,
        ) = self.build_t2i_text_inputs(tokenizer, query_uncondition)
    else:
        input_ids_uncondition = indexes_uncondition = None
        uncondition_prefix_lengths = None

    token_h = image_size[1] // (self.patch_size * merge_size)
    token_w = image_size[0] // (self.patch_size * merge_size)

    indexes_image_condition = self.build_t2i_image_indexes(
        token_h,
        token_w,
        (
            int(condition_prefix_lengths[0].item())
            if len(prompts) == 1
            else condition_prefix_lengths
        ),
        device=input_ids_condition.device,
    )
    indexes_image_uncondition = (
        self.build_t2i_image_indexes(
            token_h,
            token_w,
            int(uncondition_prefix_lengths[0].item()),
            device=input_ids_uncondition.device,
        )
        if indexes_uncondition is not None
        else None
    )
    profiler.mark("input_prepare")

    prefix = text_image_prefix(
        IMG_START_TOKEN,
        attention_mask_condition_prefix,
        compact_mode,
        condition_key_valid_mask,
        condition_prefix_lengths,
        indexes_condition,
        indexes_image_condition,
        input_ids_condition,
        max_think_tokens,
        profile_stages,
        profiler,
        prompts,
        self,
        think_mode,
        think_text,
        thinking_backend,
        token_h,
        token_w,
        tokenizer,
        use_direct_srt_prefix,
    )
    condition_key_valid_mask = prefix.condition_key_valid_mask
    condition_prefix_lengths = prefix.condition_prefix_lengths
    device = prefix.device
    dtype = prefix.dtype
    indexes_image_condition = prefix.indexes_image_condition
    past_key_values_condition = prefix.past_key_values_condition
    think_text = prefix.think_text
    use_direct_srt_prefix = prefix.use_direct_srt_prefix
    del prefix
    past_key_values_uncondition = None
    if input_ids_uncondition is not None:
        if use_direct_srt_prefix:
            (
                past_key_values_uncondition,
                uncondition_transfer_timings,
            ) = self.try_import_srt_text_prefix(
                thinking_backend,
                input_ids_uncondition,
                uncondition_prefix_lengths[0],
                "CFG",
            )
            if past_key_values_uncondition is not None:
                self.last_srt_kv_transferred_prefixes.append("uncondition")
                self.last_srt_kv_transfer_timings["uncondition"] = (
                    uncondition_transfer_timings
                )
                for row_timings in self.last_srt_kv_transfer_timings_list:
                    row_timings["uncondition"] = uncondition_transfer_timings
            elif compact_mode:
                raise RuntimeError(
                    "SenseNova compact mode did not receive the CFG prefix KV"
                )
            else:
                pass
        else:
            pass
        if past_key_values_uncondition is None:
            past_key_values_uncondition, _ = self.t2i_prefix_forward(
                input_ids_uncondition,
                indexes_uncondition,
                attention_mask_uncondition_prefix,
            )
        else:
            pass
    else:
        pass
    profiler.mark("cfg_prefill")

    del input_ids_condition, indexes_condition, attention_mask_condition_prefix
    if input_ids_uncondition is not None:
        del (
            input_ids_uncondition,
            indexes_uncondition,
            attention_mask_uncondition_prefix,
        )
    else:
        pass
    self.notify_layer_offload_phase("denoise")

    image_prediction = text_image_denoise(
        batch_size,
        cfg_interval,
        cfg_norm,
        cfg_scale,
        condition_key_valid_mask,
        condition_prefix_lengths,
        device,
        dtype,
        enable_timestep_shift,
        image_size,
        indexes_image_condition,
        indexes_image_uncondition,
        merge_size,
        needs_cfg,
        num_steps,
        past_key_values_condition,
        past_key_values_uncondition,
        profiler,
        seed,
        self,
        timestep_shift,
        token_h,
        token_w,
        uncondition_prefix_lengths,
    )
    profiler.mark("denoise_loop")

    self.last_think_content = think_text
    self.last_profile_timings_ms = profiler.finish()
    if think_mode:
        return image_prediction, think_text
    else:
        pass
    return image_prediction
